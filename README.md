# Causal Structural Hypothesis Testing

This repository contains the code for the paper ["Causal Structural Hypothesis Testing and Data Generation Models"](https://arxiv.org/abs/2210.11275) by Jeffrey Jiang, Omead Pooladzandi, Sunay Bhat, Gregory Pottie presented at the [NeuraIPS 2022 SyntheticData4ML Workshop](https://neurips.cc/virtual/2022/58649). 

## Overview

The paper introduces CSHTEST and CSVHTEST, novel architectures for causal model hypothesis testing and data generation. These models use non-parametric, structural causal knowledge and approximate a causal model's functional relationships using deep neural networks. The architectures are tested on extensive simulated DAGs, a synthetic pendulum dataset, and a real-world medical trauma dataset. This codebase contains the necessary files to run the simulated datasets. 

## Repository Structure

- `models.py`: Contains the implementations of the CSHTEST, CSVHTEST, and other baseline models.
- `sim_validate.py`: Script for running simulations to validate the CSHTEST and CSVHTEST models.
- `sim_cvhgen.py`: Script for running simulations to validate the CSHTEST and CSVHTEST models including pendulum dataset.
- `Analysis.ipynb`: Example Jupyter Notebook for conducting analysis of the simulations and experiments performed in the project.
- `utils.py`: Utility functions for training, testing, and model selection.
- `data_utils.py`: Data loading and preprocessing utilities.
- `dag_utils.py`: Functions for simulating DAGs, modifying DAGs, and computing accuracy metrics.
- `preconditioned_stochastic_gradient_descent.py`: Implementation of the Preconditioned Stochastic Gradient Descent optimizer (see PSGD link inb references)

## Dependencies

- Python 3.x
- PyTorch
- numpy
- pandas
- igraph
- seaborn
- matplotlib
- tqdm
- python-igraph
- numpy
- adabelief-pytorch


## Usage

To run the simulations for validating the CSHTEST and CSVHTEST models, use the `sim_validate.py` script:

```bash
python sim_validate.py --graph_size 4 --graph_edges 4 --dag_type linear --model_type CGen --epochs 100
```
Nonlinear DAG of a larger size
```bash
python sim_validate.py --graph_size 5 --graph_edges 5 --dag_type 'nonlinear'
```
Pendulum dataset DAG
```bash
python sim_cvhgen.py --true_dag 'pendulum'
```


## References

- [CausalVAE](https://arxiv.org/abs/2004.08697) (pendulum dataset)
- [PSGD](https://github.com/opooladz/Preconditioned-Stochastic-Gradient-Descent)
