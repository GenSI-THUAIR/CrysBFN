## [ICLR 2025 Spotlight] A Periodic Bayesian Flow for Material Generation (CrysBFN) 
This is the official implementation code for ICLR 2025 Spotlight paper CrysBFN.

[\[paper\]](arxiv.org/pdf/2502.02016) 
[\[website\]](https://t.co/a4x4qlROH7)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Here is the visualization of the proposed periodic Bayesian flow:


![image](./asset/heatmap.png)

And here is the visualization of the unified BFN generation framework  
![image](./asset/model.png)
<!-- Here is an animation of the generation process.
![GIF](./asset/generation_animation.gif) -->

## Install
### 1. Set up environment variables
Firstly please set up dot environment variables in .env file.
- `PROJECT_ROOT`: path to the folder that contains this repo. e.g. /data/wuhl/CrysBFN
- `HYDRA_JOBS`: path to a folder to store hydra outputs. This is the directory where we store checkpoints. e.g. /data/wuhl/CrysBFN/hydra
- `WABDB`: path to a folder to store wabdb outputs e.g. /data/wuhl/CrysBFN/wandb

### 2. Install with Mamba
We recommend using [Mamba](https://github.com/conda-forge/miniforge) or conda (with libmamba solver) to build the python environment. It may take several minutes to solve the environment—please wait patiently.
```
conda env create -f environment.yml
conda activate crysbfn
```

## Training, Sampling and Evaluation
We use shell scripts in `scripts` to manage all pipelines. Hyper-parameters can be set in those shell script files. Scripts to launch experiments can be found in `scripts/csp_scripts` and `scripts/gen_scripts` for crystal structure prediction task and de novo generation task.
### Training

For launching a de novo generation task training experiment, please use the following code:
```
bash ./scripts/gen_scripts/mp20_exps.sh
```
Every first run on each dataset requires longer time (< 1 hour) for preparing the cache processed data.
For launching a crystal structure prediction task training experiment, please use the following code:
```
bash ./scripts/csp_scripts/mp20_exps.sh
```
### Sampling and Evaluating
After training, please modify the MODEL_PATH variable as the hydra directory of the training experiment. Then, use the below code to generate and evaluating samples.
```
bash scripts/csp_scripts/eval_mp20.sh
```
### Toy Example
We provide toy examples with minimal components illustrating how BFNs work in `./toy_example`.

## Citation
If you find this repo or our paper useful, please cite our paper :\)
```
@misc{wu2025periodicbayesianflowmaterial,
        title={A Periodic Bayesian Flow for Material Generation}, 
        author={Hanlin Wu and Yuxuan Song and Jingjing Gong and Ziyao Cao and Yawen Ouyang and Jianbing Zhang and Hao Zhou and Wei-Ying Ma and Jingjing Liu},
        year={2025},
        eprint={2502.02016},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2502.02016}, 
  }
```

## Acknowledgement 
The main structure of this repository is mainly based on [CDVAE](https://github.com/txie-93/cdvae). The environment configuration file is modified after environment.yml in [FlowMM](https://github.com/txie-93/cdvae).