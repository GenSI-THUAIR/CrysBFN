from typing import Any, Optional, Union, Dict
from func_timeout import FunctionTimedOut, func_timeout
import hydra
from p_tqdm import p_map,p_umap,t_map
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
import torch.nn.functional as F
from absl import logging
import time
import os
import glob
import wandb
from torch.optim import Optimizer
from copy import deepcopy
from overrides import overrides
from tqdm import tqdm
import crysbfn.evaluate
from crysbfn.pl_modules.crysbfn_csp_plmodel import CrysBFN_CSP_PL_Model
from pytorch_lightning.utilities import rank_zero_only
from tqdm import trange
from hydra.core.hydra_config import HydraConfig

from crysbfn.pl_modules.crysbfn_plmodel import CrysBFN_PL_Model

class Queue:
    def __init__(self, max_len=50):
        self.items = [1]
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


class Gradient_clip(Callback):
    # gradient clupping for
    def __init__(self, Q=Queue(3000), maximum_allowed_norm=1e15,use_queue_clip=False) -> None:
        super().__init__()
        # self.max_norm = max_norm
        self.gradnorm_queue = Q
        self.maximum_allowed_norm = maximum_allowed_norm
        self.use_queue_clip = use_queue_clip

    @overrides
    def on_after_backward(self, trainer, pl_module) -> None:
        # zero graidents if they are not finite
        if not all([torch.isfinite(t.grad if t.grad is not None else torch.tensor(0.)).all() for t in pl_module.parameters()]):
            hydra.utils.log.info("Gradients are not finite number")
            pl_module.zero_grad()
            return
        if not self.use_queue_clip:
            parameters = [p for p in pl_module.parameters() if p.grad is not None]
            device = parameters[0].grad.device
            parameters = [p for p in parameters if p.grad is not None]
            norm_type = 2.0
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
            wandb.log({"grad_norm": total_norm})
            return
        minimum_queue_length = 10
        optimizer = trainer.optimizers[0]
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), max_norm=max_grad_norm, norm_type=2.0
        )
        if len(self.gradnorm_queue) < minimum_queue_length:
            self.gradnorm_queue.add(float(grad_norm))
        elif float(grad_norm) > self.maximum_allowed_norm:
            optimizer.zero_grad()
            hydra.utils.log.info(
                f"Too large gradient with value {grad_norm:.1f}, NO UPDATE!"
            )
        elif float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))
        # if float(grad_norm) > max_grad_norm:
        #     hydra.utils.log.info(
        #         f"Clipped gradient with value {grad_norm:.1f} "
        #         f"while allowed {max_grad_norm:.1f}",
        #     )
        wandb.log({"grad_norm": grad_norm})
        # pl_module.log_dict(
        #     {"grad_norm": grad_norm},
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=False,
        #     # batch_size=pl_module.hparams.data.datamodule.batch_size.train,
        # )



class EMACallback(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        ema_device: Optional[Union[torch.device, str]] = None,
        pin_memory=True,
    ):
        super().__init__()
        self.decay = decay
        self.ema_device: str = (
            f"{ema_device}" if ema_device else None
        )  # perform ema on different device from the model
        self.ema_pin_memory = (
            pin_memory if torch.cuda.is_available() else False
        )  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.

        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs
    ) -> None:
        # Update EMA weights
        start_step = 1500
        if trainer.global_step > start_step:
            if not self._ema_state_dict_ready and pl_module.global_rank == 0:
                self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
                if self.ema_device:
                    self.ema_state_dict = {
                        k: tensor.to(device=self.ema_device)
                        for k, tensor in self.ema_state_dict.items()
                    }

                if self.ema_device == "cpu" and self.ema_pin_memory:
                    self.ema_state_dict = {
                        k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()
                    }

                self._ema_state_dict_ready = True
            with torch.no_grad():
                for key, value in self.get_state_dict(pl_module).items():
                    ema_value = self.ema_state_dict[key]
                    ema_value.copy_(
                        self.decay * ema_value + (1.0 - self.decay) * value,
                        non_blocking=True,
                    )
        
    def load_ema(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # print('here is the EMA call back start!!!')
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))

        # trainer.strategy.broadcast(self.ema_state_dict, 0)

        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), (
            f"There are some keys missing in the ema static dictionary broadcasted. "
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        )
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    def unload_ema(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # print('here is the EMA call back end!!!')
        
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    @overrides
    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ):
    # ) -> dict:
        checkpoint["ema_state_dict"] = self.ema_state_dict
        checkpoint["_ema_state_dict_ready"] = self._ema_state_dict_ready
        return None
        return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    @overrides
    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
        # callback_state: Dict[str, Any]  older pytorchlightning interface 
    ) -> None:
        if checkpoint is None:
        # if callback_state is None:
            self._ema_state_dict_ready = False
            print('no ema state!!!')
        else:
            self._ema_state_dict_ready = checkpoint["_ema_state_dict_ready"]
            # self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
            self.ema_state_dict = checkpoint["ema_state_dict"]
            # self.ema_state_dict = callback_state["ema_state_dict"]
            print('load ema state!!!')
            if self.ema_device:
                    self.ema_state_dict = {
                        k: tensor.to(device=self.ema_device)
                        for k, tensor in self.ema_state_dict.items()
                    }
            # pl_module.load_state_dict(self.ema_state_dict, strict=False)


        
                    