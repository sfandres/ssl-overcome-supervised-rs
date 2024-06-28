<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![ORCID][orcid-shield]][orcid-url] [![GoogleScholar][google-scholar-shield]][google-scholar-url] [![GitHub][github-shield]][github-url] [![LinkedIn][linkedin-shield]][linkedin-url]

# Self-Supervised Learning on Small In-Domain Datasets Can Overcome Supervised Learning in Remote Sensing
This repository contains the official implementation of the paper <i>[Self-Supervised Learning on Small In-Domain Datasets Can Overcome Supervised Learning in Remote Sensing][paper-doi]</i>, published in the IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

[![DOI](https://zenodo.org/badge/doi/10.1109/JSTARS.2024.3421622.svg)](https://doi.org/10.1109/JSTARS.2024.3421622)

If you use this code or find our work useful, please consider citing our paper:
```
@article{sanchez2024ssl,
    title={Self-supervised learning on small in-domain datasets can overcome supervised learning in remote sensing},
    author={Sanchez-Fernandez, Andres J and Moreno-Alvarez, Sergio and Rico-Gallego, Juan A and Tabik, Siham},
    journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
    year={2024},
    publisher={IEEE}
}
```

The paper was authored by [Andres J. Sanchez-Fernandez][orcid-url] (University of Extremadura), [Sergio Moreno-Álvarez](https://orcid.org/0000-0002-1858-9920) (National University of Distance Education), [Juan A. Rico-Gallego](https://orcid.org/0000-0002-4264-7473) (CénitS-COMPUTAEX), and [Siham Tabik](https://orcid.org/0000-0003-4093-5356) (University of Granada).

## Table of contents
* [Getting started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Usage](#usage)
* [Pretraining SSL models](#pretraining-ssl-models)
* [Fine-tuning on downstream tasks](#fine-tuning-on-downstream-tasks)
* [License](#license)

## Getting started

### Prerequisites
Anaconda distribution is recommended. You can install it following the [official installation guide](https://docs.anaconda.com/anaconda/install/linux/).

Check if Anaconda is installed:
```
conda -V
```

### Installation
The [environment.yml](environment.yml) file contains all the packages needed to use this project. You can create your own environment from the file provided as follows:
```
conda env create -f environment.yml
```

Check that the new environment has been installed correctly:
```
conda env list
```

### Usage
Activate the conda environment:
```
conda activate ssl-os-rs
```

Now you can run any Python script.

## Pretraining SSL models
Four SSL models are considered: Barlow Twins, MoCov2, SimCLR, and SimSiam. The ResNet18 is selected as the backbone of each network. The pretraining on the Sentinel2GlobalLULC pure-pixels dataset is launched via a SLURM job script by running:
```
./ssl_pretraining_slurm_launch_loop.sh <option>
```
For the `<option>` argument, four types of experiments can be selected: `RayTune`, `DDP`, `Imbalanced` (default), or `Balanced`.

If `RayTune` is selected, the script generates a csv file including the best configurations sorted according to the lowest training loss with the following format: `ray_tune_<backbone>_<model>.csv`. This file must be included in the path `./input/best_configs/` for the other types of experiments to start with the pseudo-optimal hyperparameters found.

## Fine-tuning on downstream tasks
```
sbatch finetuning_slurm.sh
```

This script runs the [finetuning_run_localhost.sh](finetuning_run_localhost.sh) for the target downstream tasks: LULC fraction estimation and scene classification. It should be configured with only one or two `train_rates` to launch several Slurm jobs. The Python script accepts the desired number of samples per class as input. Upon completion of the jobs, several files will be generated (one per seed) inside the output folder.

* The mean and std values per trial can be generated using the script [sc_1_compute_mean_std_from_csv.py](scripts/sc_1_compute_mean_std_from_csv.py) (see the `-h` for help) as follows:
```
python3 sc_1_compute_mean_std_from_csv.py -i <parent_folder_of_the_csv_files> -o <desired_output_folder>
```
where `parent_folder_of_the_csv_files` should target the `multiclass/` and then `multilabel/` folders following the structure below:
```
csv_results/
├── multiclass/
│   ├── multiclass_tr=0.010_resnet18_BarlowTwins_bd=False_tl=FT_iw=random_s=05_lr=0.001_m=0.9_wd=0.0_do=None.csv
│   ├── multiclass_tr=0.010_resnet18_BarlowTwins_bd=False_tl=FT_iw=random_s=42_lr=0.001_m=0.9_wd=0.0_do=None.csv
│   ├── ...
├── multilabel/
│   ├── multilabel_tr=0.010_resnet18_BarlowTwins_bd=False_tl=FT_iw=random_s=05_lr=0.01_m=0.9_wd=1e-05_do=None.csv
│   ├── multilabel_tr=0.010_resnet18_BarlowTwins_bd=False_tl=FT_iw=random_s=42_lr=0.01_m=0.9_wd=1e-05_do=None.csv
│   ├── ...
```

* The output files generated by the previous script should be manually arranged in folders as follows:
```
both_mean_std_csv_files/
├── multiclass/
│   ├── 001p/
│      ├── pp_mean_multiclass_tr=0.010_resnet18_BarlowTwins_bd=False_tl=FT_iw=random.csv
│      ├── ...
│      ├── pp_std_multiclass_tr=0.010_resnet18_BarlowTwins_bd=False_tl=FT_iw=random.csv
│      ├── ...
├── multilabel/
│   ├── 001p/
│      ├── pp_mean_multilabel_tr=0.010_resnet18_BarlowTwins_bd=False_tl=FT_iw=random.csv
│      ├── ...
│      ├── pp_std_multilabel_tr=0.010_resnet18_BarlowTwins_bd=False_tl=FT_iw=random.csv
│      ├── ...
```

* The generated csv file can be plotted using the script [sc_2_plot_final_graphs_v2.py](scripts/sc_2_plot_final_graphs_v2.py). This script requires inputting the parent folder (`multiclass/` or `multilabel/`) and adjusting the hard-coded `x` variable to the current number of percentages available. It searches for the best results obtained in the validation dataset and then generates a **new *dataframe* with the final results used for the graphs**, as well as the **final line graphs** showing the training ratios versus the results of the desired final metric:
```
python3 scripts/sc_2_plot_final_graphs_v2.py -i ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/02_avg_csv_files/multiclass/ -o ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/ -m f1_macro -sf pdf
python3 scripts/sc_2_plot_final_graphs_v2.py -i ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/02_avg_csv_files/multilabel/ -o ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/ -m rmse -sf pdf
```

* To obtain the bar plots related to the F1 and RMSE results per class ([sc_3_plot_final_bar_graphs_v2.py](scripts/sc_3_plot_final_bar_graphs_v2.py)), run the following commands:
```
python3 scripts/sc_3_plot_final_bar_graphs_v2.py -i ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/exp_multiclass_best_results_means.csv -o ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/ -r Random -sf pdf
python3 scripts/sc_3_plot_final_bar_graphs_v2.py -i ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/exp_multiclass_best_results_means.csv -o ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/ -r ImageNet -sf pdf
python3 scripts/sc_3_plot_final_bar_graphs_v2.py -i ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/exp_multilabel_best_results_means.csv -o ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/ -r Random -sf pdf
python3 scripts/sc_3_plot_final_bar_graphs_v2.py -i ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/exp_multilabel_best_results_means.csv -o ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/ -r ImageNet -sf pdf
```

* To generate the tables for LaTeX ([sc_4_generate_latex_tables](scripts/sc_4_generate_latex_tables.py)), we use:
```
python3 scripts/sc_4_generate_latex_tables.py -i /home/sfandres/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/exp_multiclass_best_results_means.csv -o ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/ -v
python3 scripts/sc_4_generate_latex_tables.py -i /home/sfandres/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/exp_multilabel_best_results_means.csv -o ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/ -v
```

* Finally, to create the class-wise difference graphs ([sc_5_plot_discussion_bar_graphs.py](scripts/sc_5_plot_discussion_bar_graphs.py)), we run the following code:
```
python3 scripts/sc_5_plot_discussion_bar_graphs.py -i ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/exp_multiclass_best_results_means.csv -o ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/ -r ImageNet -v -sf pdf
python3 scripts/sc_5_plot_discussion_bar_graphs.py -i ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/exp_multilabel_best_results_means.csv -o ~/Documents/Experiments/SSL-BSU/02_v3_R1_Fine-tuning_new_results_val_test/03_dfs_final_results/ -r ImageNet -v -sf pdf
```

## License

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by]. See the [LICENSE](LICENSE) file for details.

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[orcid-shield]: https://img.shields.io/badge/orcid-A6CE39?style=for-the-badge&logo=orcid&logoColor=white
[orcid-url]: https://orcid.org/0000-0001-6743-3570
[google-scholar-shield]: https://img.shields.io/badge/Google%20Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white
[google-scholar-url]: https://scholar.google.es/citations?user=AYtHK3EAAAAJ&hl=en
[github-shield]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
[github-url]: https://github.com/sfandres94
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://linkedin.com/in/sfandres
[paper-doi]: https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4609443
