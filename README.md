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
Note: this may take a while as ibllib relies on OpenCV, and installing this can take a while. This is only needed for the analysis of IBL dataset, so it is completely optional whether you install them or not.

We use version 0.0.1 of the Bayesian State Space Modeling framework from Scott Linderman's lab to perform GLM-HMM inference. Within the `glmhmm` environment, install the forked version of the `ssm` package available [here](https://github.com/sumiya-kuroda/ssm/tree/sumiya-hpc). This is a lightly modified version of the master branch of the ssm package available at [https://github.com/lindermanlab/ssm](https://github.com/lindermanlab/ssm). It is modified so as to handle violation trials as described in Section 4 of Ashwood paper, as well as GLM weights for dmdm dataset. In order to install this version of `ssm`, follow the instructions provided there, namely:     
```sh
git clone https://github.com/sumiya-kuroda/ssm.git

cd ssm
git checkout sumiya-hpc # Change the branch
pip install numpy cython
pip install -e .
```

## Getting started
When you ssh to the HPC, you will first enter the login node. No computation-heavy analysis should be done on this node. We use Jupyter and `dask` to get the advantage of HPC ([Best Practices](https://docs.dask.org/en/stable/delayed-best-practices.html)). But you can play around with it. Here is what I do:
```sh
srun --job-name=jupyter -p gpu --gres=gpu:1 --mem=32G -n 4 --pty bash -l
```
This will let you login to a compute node. Then, activate the conda environment by the command below. (This will also load Matlab needed for some preprocessing.)
 ```sh
 module load miniconda
 module load matlab

 conda activate glmhmm
```

Launch Jupyter as
```sh
export XDG_RUNTIME_DIR="" # See https://swc-neuro.slack.com/archives/CHLGTQVLL/p1560422985006700
jupyter lab --port=2025
```
Following [this procedure](https://github.com/pierreglaser/jupyter-slurm-setup-instructions) will make you able to connect to the Jupter notebook on HPC with [`http://127.0.0.1:2025/lab`](http://127.0.0.1:2024/lab) using your web browser. I personally use VSCode, so I will simply run this command below from another HPC terminal to port forwarding ([ref](https://swc-neuro.slack.com/archives/C0116D5V7SA/p1645618426952349)).
```sh
ssh -q -N ${jupyter_host_name} -L 2025:localhost:2025
``` 
I then follow [this step](https://github.com/microsoft/vscode-jupyter/discussions/13145) and simply copy and paste the jupyter URL within HPC. We will only use Jupyter when we run GLMs and GLM-HMMs on preprocessed data. The first prerpocessing step does not require Jupyter, and you can simply run on a decent computing node. (Again, not login node!)

## Run GLM-HMM
Code is ordered so that the dataset is preprocessed into the desired format and fitting parameters are efficiently initialized. 

### 1_preprocess_data 
This creates `data` folder. Within this directory, run the scripts in the order indicated by the number at the beginning of the file name. For dmdm dataset, run `0_convert_behavior_dataset.sh` first. This script converts raw behavior data saved in `.mat` to Ashwood's GLM-HMM-friendly `.npy` format. You also need to be cautious that, even for the dmdm dataset, animals' behaviors are stored slightly different between projects. This shell scirpt therefore requires both a path to `.mat` file as well as which format the `.mat` file uses. Example usage:
```
. ./0_convert_behavior_dataset.sh -i ${path_to_raw_data} -f ${file_format}
```

Then run `1_begin_processing.py` and `2_create_design_mat.py` to obtain the design matrix used as input for all of the models. If you want to reproduce Ashwood's results on IBL dataset, follow the steps on her repo. If you want to customize the design matrix for GLMs, edit `design_mat_utils.py`. Example usage:
```
python 1_begin_processing.py ${filename}
python 2_create_design_mat.py ${filename} -r ${min_number_of_sessions}
```

### 2_fit_models
This creates `results` folder. With the processed dataset, you can fit the GLM, lapse and GLM-HMM models discussed in the paper using the code contained in this folder. Codes for dmdm dataset are organized separately under `dmdm/` from Ashwood's original codes for IBL. As discussed in the paper, the GLM should be run first as the GLM fit is used to initialize the global GLM-HMM (the model that is fit with data from all animals). The dmdm dataset has two observations on animal's behavior: outcomes and reaction times. Therefore, we fit these observatond independently using two separate GLMs. Also, because the dmdm task has more discrete outcomes than two binary choices, GLM needs to consider that complexity of the task when fitting and optimizing weights. We therefore use the [random utility maximization nested logit (RUMNL)](https://journals.sagepub.com/doi/10.1177/1536867X0200200301) model to basically ask GLM to calculate conditional probabilities for each outcome, by introducing a nested structure of outcomes. Reaction times are fitted with gaussian GLM.

After GLM fitting, you should be able to run GLM-HMM. The global GLM-HMM should be run next, as it explores all the parameter space and find a possible parameter candidate for fitting GLM-HMM to individual animal. As Ashwood described in her paper, we use EM algorithm here, which takes quite a lot of computational resources. EM algorithm does not guarantee that we will find global maxima, so we need to fit with several iterations (which also increases required computation resources). Thus HPC usage is highly recommended. 

Finally GLM-HMMs can be fit to the data from individual animals. We also use Maximum A Priori (MAP) Estimation here, as the number of trials available for each animal is very limited. (See [here](https://github.com/zashwood/ssm/blob/e9b408b79e2ab22a05ca93c3a1f78a7dae461992/notebooks/2b%20Input%20Driven%20Observations%20(GLM-HMM).ipynb) for more explanation.) This involves two additional hyperparameters: alpha and sigma. We use k-fold cross-validation here to find the best pairs of them. The Jupyter notebooks for GLM-HMM also generate some basic figures that will make easy to interpret the results.

It is completely optional to fit the lapse model to the dataset. While not used for any initialization purposes, this allows model comparison with the global and individual GLM-HMMs. 
          
### 3_make_figures
This creates `figures` folder. Assuming that you have downloaded and preprocessed the datasets, and that you have fit all models on these datasets, you can start some exploratory analysis and reproduce some figures. `3_make_figures/dmdm/write_results.py` will save the outputs so that you can run other analysis on trials characterized by GLM-HMM. Enjoy!


## Troubleshooting
- To create symbolic link `ln -s ../ssm ssm`
- Cannot find the kernel? See [here](https://www.mk-tech20.com/vscode-conda/)
- You want to improve your dask user experience? Check [this JupyterLab extension](https://github.com/dask/dask-labextension) and [this VSCode extension](https://marketplace.visualstudio.com/items?itemName=joyceerhl.vscode-dask).
- .mat file is in v7.3? You can use [this python module](https://github.com/skjerns/mat7.3) to load, but I found this is extremely slow for the size of .mat file we have.
- Python is [here](/nfs/nhome/live/skuroda/.conda/envs/glmhmm/bin/python3.7). VSCode asks you to specify the path to it.