# glm-hmm
Code to run GLM-HMM analysis on dmdm data using Zoe Ashwood's work and SWC's HPC. The code is heavily adapted from [Ashwood's repo](https://github.com/zashwood/glm-hmm). 

Code is originally made to reproduce figures in ["Mice alternate between discrete strategies
 during perceptual decision-making"](https://www.biorxiv.org/content/10.1101/2020.10.19.346353v4.full.pdf) from Ashwood, Roy, Stone, IBL, Urai, Churchland, Pouget and Pillow (2020).  Note: while this code reproduces the figures/analyses for their paper, the easiest way to get started applying the GLM-HMM to your own data, is with [this notebook](https://github.com/zashwood/ssm/blob/master/notebooks/2b%20Input%20Driven%20Observations%20(GLM-HMM).ipynb). 

## Installation
If you are new to SWC's HPC (like I was), [this page](https://howto.neuroinformatics.dev/programming/SSH-SWC-cluster.html) is very helpful.

To retrieve the code by cloning the repository:
```sh
git clone https://github.com/sumiya-kuroda/glm-hmm.git
```

Create a conda environment from the environment yaml file by running 
```sh
conda env create -f environment.yml
```
Note: this may take a while as ibllib relies on OpenCV, and installing this can take a while.  

We use version 0.0.1 of the Bayesian State Space Modeling framework from Scott Linderman's lab to perform GLM-HMM inference. Within the `glmhmm` environment, install the forked version of the `ssm` package available [here](https://github.com/zashwood/ssm). This is a lightly modified version of the master branch of the ssm package available at [https://github.com/lindermanlab/ssm](https://github.com/lindermanlab/ssm). It is modified so as to handle violation trials as described in Section 4 of Ashwood paper. In order to install this version of `ssm`, follow the instructions provided there, namely:     
```sh
git clone https://github.com/zashwood/ssm.git

cd ssm
pip install numpy cython
pip install -e .
```

## Getting started
When you ssh to the HPC, you will first enter the login node. No computation-heavy analysis should be done on this node. Here, we use jupyter to make things easier, but you can play around with it. Here is what I do:
```sh
srun --job-name=jupyter -p gpu --gres=gpu:1 --mem=32G -n 4 --pty bash -l
```

Then, activate the conda environment by running 
 ```sh
 module load miniconda
 module load matlab

 conda activate glmhmm
```

Launch Jupyter as
```sh
export XDG_RUNTIME_DIR="" # See https://swc-neuro.slack.com/archives/CHLGTQVLL/p1560422985006700
jupyter lab --port=2024
```
Following [this procedure](https://github.com/pierreglaser/jupyter-slurm-setup-instructions) will make you able to connect to the Jupter notebook on HPC with [`http://127.0.0.1:2024/lab`](http://127.0.0.1:2024/lab) using your web browser. I personally use VSCode, so I will simply run `ssh -q -N ${jupyter_host_name} -L 2024:localhost:2024` to port forwarding ([ref](https://swc-neuro.slack.com/archives/C0116D5V7SA/p1645618426952349)). I then follow [this step](https://github.com/microsoft/vscode-jupyter/discussions/13145) and simply copy and paste the jupyter URL within HPC.

## Run GLM-HMM
Code is ordered so that the dataset is preprocessed into the desired format and fitting parameters are efficiently initialized. 

### 1_preprocess_data 
Within this directory, run the scripts in the order indicated by the number at the beginning of the file name. For dmdm dataset, run `0_convert_behavior_dataset.sh` first. This script converts raw behavior data saved in .mat to Ashwood's GLM-HMM-friendly .npy format. You also need to be cautious that, even for the dmdm dataset, animals' behaviors are stored slightly different between projects. This shell scirpt therefore requires both a path to .mat file as well as which format the .mat file uses. This makes `data` folder with the behavior converted and saved in .npy.

Then run `1_begin_processing.py` and `2_create_design_mat.py` to obtain the design matrix used as input for all of the models. If you want to reproduce Ashwood's results on IBL dataset, follow the steps on her repo. 

### 2_fit_models
Next, you can fit the GLM, lapse and GLM-HMM models discussed in the paper using the code contained in "2_fit_models". As discussed in the paper, the GLM should be run first as the GLM fit is used to initialize the global GLM-HMM (the model that is fit with data from all animals). The lapse model fits, while not used for any initialization purposes, should be run next so as to be able to perform model comparison with the global and individual GLM-HMMs. The global GLM-HMM should be run next, as it is used to initialize the models for all individual animals.  Finally GLM-HMMs can be fit to the data from individual animals using the code in the associated directory. 
          
### 3_make_figures
Assuming that you have downloaded and preprocessed the datasets, and that you have fit all models on these datasets,  you can reproduce the figures of our paper corresponding to the IBL dataset by running the code contained in "3_make_figures".  In order to produce Figures 5 and 7, replace the IBL URL in the preprocessing pipeline scripts, with the URLs for the [Odoemene et al. (2018)](https://doi.org/10.14224/1.38944) and [Urai et al. (2017)](https://doi.org/10.6084/m9.figshare.4300043) datasets, and rerun the GLM, lapse and GLM-HMM models on these datasets before running the provided figure plotting code.


## Troubleshooting
- To create symbolic link `ln -s ../ssm ssm`
- Cannot find the kernel? See [here](https://www.mk-tech20.com/vscode-conda/)
- You want to use dask? Check [this](https://github.com/pierreglaser/hpc-tutorial/tree/main) and [this](https://github.com/dask/dask-labextension). [This](https://marketplace.visualstudio.com/items?itemName=joyceerhl.vscode-dask) is for VSCode users.
- .mat file is in v7.3? You can use [this python module](https://github.com/skjerns/mat7.3) to load, but I found this is extremely slow for the size of .mat file we have.
- If you get annoyed with VSCode's `Pylance(reportMissingModuleSource)` warning, reopening .ipynb solves the issue.
- Python is [here](/nfs/nhome/live/skuroda/.conda/envs/glmhmm/bin/python3.7). VSCode asks you to specify the path to it.