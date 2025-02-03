## [ICLR 2025] A Periodic Bayesian Flow for Material Generation (CrysBFN) 
This is the official implementation code for ICLR 2025  [\[paper\]](https://openreview.net/pdf?id=Lz0XW99tE0) 

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
- `PROJECT_ROOT`: path to the folder that contains this repo
- `HYDRA_JOBS`: path to a folder to store hydra outputs
- `WABDB`: path to a folder to store wabdb outputs

### 2. Install with Mamba
We recommend using [Mamba](https://github.com/conda-forge/miniforge) or conda (with libmamba solver) to build the python environment. 
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
For launching a crystal structure prediction task training experiment, please use the following code:
```
bash ./scripts/gen_scripts/mp20_exps.sh
```
### Sampling and Evaluating
After training, please modify the MODEL_PATH variable as the hydra directory of the training experiment. Then, use the below code to generate and evaluating samples.
```
bash scripts/csp_scripts/eval_mp20.sh
```
### Toy Example
We provide toy examples with minimal components illustrating how BFNs work in `./toy_example`.

## Acknowledgement 
The main structure of this repository is mainly based on [CDVAE](https://github.com/txie-93/cdvae). The environment configuration file is modified after environment.yml in [FlowMM](https://github.com/txie-93/cdvae).