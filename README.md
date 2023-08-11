<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![LinkedIn][linkedin-shield]][linkedin-url]

# lulc
This repository has been created for the <b>sanchez2023classification</b> paper.

## Table of contents
* [Getting started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Usage](#usage)
* [Pretraining SSL models](#pretraining-ssl-models)
* [Downstream tasks: BSU and SC](#downstream-tasks-bsu-and-sc)
* [Code examples](#code-examples)
* [License](#license)

## Getting started

### Prerequisites
Anaconda distribution is recommended. You can install it following the [official installation guide](https://docs.anaconda.com/anaconda/install/linux/).

Check if Anaconda is installed:
```
conda --version
conda -V
```

### Installation
The environment.yml file contains all the necessary packages to use this project inside the environment with the name `lulc2-conda` provided. You can create a conda environment from the [environment2.yml](environment2.yml) file provided as follows:
```
conda env create -f environment2.yml
```

### Usage
Activate the conda environment:
```
conda activate lulc2-conda
```

Now you can run any Python script.

## Pretraining SSL models
Four SSL models are considered: Barlow Twins, MoCov2, SimCLRv2, and SimSiam. The ResNet18 is selected as the backbone of each network. The pretraining on the Sentinel2GlobalLULC pure-pixels dataset is launched via a SLURM job script by running:
```
./ssl_pretraining_slurm_launch_loop.sh <option>
```
For the `<option>` argument, four types of experiments can be selected: `RayTune`, `DDP`, `Imbalanced`, or `Balanced`.

If `RayTune` is selected, the script generates a csv file including the best configurations sorted according to the lowest training loss with the following format: `ray_tune_<backbone>_<model>.csv`. This file must be included in the path `./input/best_configs/` for the other types of experiments to start with the pseudo-optimal hyperparameters found.

## Downstream tasks: BSU and SC
```
sbatch finetuning_slurm.sh
```

This script runs the `finetuning_run_localhost.sh`. It should be configured with only one or two `train_rates` to launch several Slurm jobs.

Upon completion of the jobs, several files will be generated (one per seed) inside the output folder. The mean and std values per trial can be generated using the script `compute_mean_std_from_csv.py` (see the `-h` for help). Then, the csv generated can be plotted using the script `plot_final_graphs.py`.

To generate the tables with all values, the script `take_acc_values_from_csv_v2.py` is used.

## Code examples
Download the ImageNet dataset in the background on a remote server:
```
nohup wget <url> --no-check-certificate & exit
```

To check if the process is indeed running after exiting from shell and logging again:
```
top -u <username>
```

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://linkedin.com/in/sfandres
