# CB2

Official implementation of Cutting the Black Box: conceptual interpretation of a latent layer with multi-criteria decision aid presented at IJCAI 2024.

## Installation

`Python 3.10` and `PyTorch 2.6.0+cu124` are recommended.

To set up the project, clone the repository and install the `cb2` package in a virtual environment:

    # Create and activate a virtual environment
    python -m venv .cb2_venv
    source .cb2_venv/bin/activate

    # Install the package
    pip install .

## Quickstart

A complete command-line interface (CLI) is provided for training the models in the `experiments` folder. 

For instance for a local training:

```bash
 python train.py device='cpu' \
    model=cb2-HCI dataset=cifar-10 \
    r_ae=1 r_align=100 r_pred=100 \
    epochs=15 black_box=resnet20

```

Or for a training on a slurm-based cluster:

```bash
python train.py --multirun hydra/launcher=submitit_slurm \
    hydra.launcher.partition=gpu_nodes \
    hydra.launcher.gpus_per_task=1
    dataset=cifar-100 epochs=50 \
    random_seed=1,2,3,4,5,6,7,8,9,10 

```
Refer to the configuration files in the `experiment/` folder for details about the available parameters.

## Reference

    @article{atienza2024cb2,
        title = {Cutting the Black Box: Conceptual Interpretation of a Deep Neural Net with Multi-Modal Embeddings and Multi-Criteria Decision Aid},
        author = {Atienza, Nicolas and Bresson, Roman and Rousselot, Cyriaque and Caillou, Philippe and Cohen, Johanne and Labreuche, Christophe and Sebag, Michele},
        journal={IJCAI},
        year={2024}
    }
