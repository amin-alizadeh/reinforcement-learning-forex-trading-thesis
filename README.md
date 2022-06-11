# Experimental code for Reinforcement Learning for Financial Portfolio Management: A study of Neural Networks for Reinforcement Learning on currency exchange market
The msater's thesis can be found at https://www.theseus.fi/handle/10024/502022

## Unzip raw data files
The data files are located at `./data/minutes/`. You can unzip them all by running `make`.

## Running experiments
The experiments are grouped under [`./experiments/`](tree/master/experiments). 
Each experimnent has a `Makefile` with 3 targets:
1. train (*default*)
1. run
1. best

### train
Runs the training with the provided configurations that are passed as command line arguments. The training generates 2 sets of outputs:
1. The latest model
1. The model with the best output
To train the model run
```bash
make train
```
These 2 can be used later for testing using either the model from the full run (after all iterations are completed) or the model that scored the best (highest reward after an episode).
### run
This target uses the latest model that was generated after the completion of the last episode. To execute this target, you can run:
```bash
make run
```
### best
This target uses the model that scored the highest in an episode. To execute this target, you can run:
```bash
make best
```
