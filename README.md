# Adversarial Attacks on the DINO-ViT Fruit Quality Assessment Model

This fork contains the required changes to assess the adversarial robustness of the models. 

As the original work employs a 2-step approach comprised of a pre-trained PyTorch vision transformer for feature extractor and a scikit-learn linear classifier, I build an identical end-to-end PyTorch model to craft adversarial examples with the [Foolbox](https://github.com/bethgelab/foolbox) library. The adversarial examples are then evaluated against the surrogate and the original model.

I had to adapt the existing code in some places, the script for attacking the model and the results can be found in the `adversarial_attacks` folder.

The results of this study were presented at the [Twenty-third International Working Seminar on Production Economics 2024](https://www.uibk.ac.at/wipl/productioneconomics2024/), and are currently under review for a journal publication.

---
# Overview
Code for the paper *Facilitated machine learning for image-based fruit quality assessment* published in the [Journal of Food Engineering](https://www.sciencedirect.com/science/article/pii/S0260877422004551?via%3Dihub).

A preprint version was published earlier on [arXiv](https://arxiv.org/abs/2207.04523).

# Appendix

For additional illustrations see the appendix file: [appendix.pdf](appendix.pdf)

# Source Code

## Python setup

The code was tested with python version 3.8 and 3.10. Make sure to install all packages in [requirements.txt](requirements.txt) and to have CUDA-compatible GPU available to be able to run all experiments.

## Datasets

The data sets used in this research are owned by the respective authors and are therefore not shared in this repository.
If you like to use them, please reach out to the authors.

In order to reproduce these experiments, place the files in the `datasets/data/` folder in accordance with the depicted folder structure.

## Run experiments

If you want to run all experiments at once, please refer to the [run_all_experiments.py](run_all_experiments.py) file.
These scripts save interim results in the `results/` folder.

Basline experiments are logged using Weights&Biases. To run these, you need an account there.

Please note that this might take several hours and your machine should be set up with a CUDA-compatible GPU.

## Plots and tables

Tables and figures are generated in the notebook `tables_and_figures.ipynb`.
It relies on precomputed data that is saved in the `results` folder by the `run_all_experiments.py` script.
