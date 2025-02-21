# Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD

This repo stores the codes for the ICLR 2025 paper "Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD".

The code is modified from the supplementary materials of 
[On the Generalization of Neural Networks Trained with SGD: Information-Theoretical Bounds and Implications](https://openreview.net/forum?id=4i23Bjlh4Y9)


## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Run Experiments

Before starting, one can optionally set the following environmental variable to only estimate bounds after the last epoch for experiments on non-base hyperparameters, in order to save time:
```bash
export ARGS_FOR_REPEATED="--bound-freq -1"
```

To run the experiments on MNIST and CIFAR10:
```bash
TASK=mnist bash scripts/all.sh --batch-size-for-validation 4096 --dont-repeat
TASK=cifar10 bash scripts/all.sh --batch-size-for-validation 512 --dont-repeat
TASK=cifar10_vit bash scripts/all.sh --batch-size-for-validation 512 --dont-repeat
```
Once completed, the results can be found in `runs/`.

### Run a Subset of Experiments 

Run `scripts/weight_decay.sh`, `scripts/label_corruption.sh` or `scripts/lrbs.sh`, etc. instead of `scripts/all.sh` to run experiments with the corresponding hyperparameter subset varied instead of all experiments.

# Citations

```bibtex
@inproceedings{
    peng2025leveraging,
    title={Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for {SGD}},
    author={Ze Peng and Jian Zhang and Yisen Wang and Lei Qi and Yinghuan Shi and Yang Gao},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=pSdE7PIA64}
}
```