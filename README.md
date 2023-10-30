# Lottery Ticket Hypothesis

This is a PyTorch implementation of the paper [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635).

This repository is based on [rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch),
but **more readable**.

Changes:

- We use [hydra](https://hydra.cc/) to manage configurations.
- We remove all global variables.
- We use a `dict` to represent mask for each layer instead of a `list`.
- We rename some variables to make them more readable, for example, `end_iters` to `num_epochs`, `iter_` to `epoch_idx`, etc.

Any suggestions are welcome.

## Setup Python environment

create an environment

```bash
conda create -n lottery python=3.10
conda activate lottery
pip install -r requirements.txt
```

## Project layout

```bash
data/ # datasets
config/ # config files
src/ # source code
    models/ # models
    utils.py # utility functions
scripts/ # entry points
results/ # results
```

## Usage

```bash
python3 scripts/main.py --prune_type=lt --arch_type=fc1 --dataset=mnist --prune_percent=10 --prune_iterations=35
```

## Acknowledgement

- [rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch),

