# Study of "AE-FLOW: Autoencoders with Normalizing Flows for Anomaly Detection"

*The main branch conforms to the required structure. Please refer to the 'andre' branch if any errors are encountered while running scripts.*

This repository contains our work for a mini-project for the DL2 cours at the University of Amsterdam (https://uvadl2c.github.io/, 2023), aiming to firstly replicate the work "AE-FLOW: Autoencoders with Normalizing Flows for Anomaly Detection" by Yuzhong Zhao, Qiaoqiao Ding and Xiaoqun Zhang (2023), and secondly to extend this work by determining it's generalization of it's finding to other data-sets and uncertainty quantification using Deep Ensembles.

## Getting started
The directory 'src/script/job_files' contains all files needed to install the environment using our supplied 'src/script/job_files/environment.yml' file, and run the experiments. Using these will be enough to run the repository on a SLURM-based cluster such as LISA. For setting/defining parameters both with regard to model-setup and running experiments we currently refer to the train.py file.

Please install any missing requirements on your own.

## Data
The chest_xray dataset is included in the repo, however other datasets are too large for git so please download them manually if you want to run these. They should be downloaded into the dataset folder. Links to the datasets can be found;
OCT2017: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
ISIC: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
MIIC: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/WBLTFI
BRATS: http://braintumorsegmentation.org/

## Running locally
Multiple files can be used to run the model (located in src):
* train.py: used for running the autoencoder, Fastflow, AE-FLOW models. Please see the main function for details on arguments.
* generate_ue_results.py: can be ran out of the box. It will evaluate the deep ensemble and provide results. No models are required to run this file, as the model results and thresholds are pickled and available in the repository.
* generate_experiment_files.py: generates the job files for the final experiments. No practical use outside of our experiment.

## Deliverables
Please refer to the blogpost.md for our further research efforts for the course
