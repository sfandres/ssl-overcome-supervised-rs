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
Continue!

## Code examples
The examples are organized in folders:
* [CD22_23-P09_quijote](CD22_23-P09_quijote) contains several examples that introduce PySpark using 'El Quijote' as a case study.
* [CD22_23-P10_covid19](CD22_23-P10_covid19) presents the solution for the Covid-19 exercise, which focuses on database management using PySpark.

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://linkedin.com/in/sfandres
