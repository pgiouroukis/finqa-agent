# setup

We will use `micromamba` (installation [guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)) for package-management. Please also add `conda-forge` for setting-up environments.

    micromamba config append channels conda-forge

Create a new environment. (`[VERSION]` use 3.11.8).

    micromamba create -n [PROJECT_NAME] python=[VERSION]

Activate environment.

    micromamba activate [PROJECT_NAME]

You will need to install these:

    micromamba install -c conda-forge python=[VERSION]

Install `poetry` for dependency management.

    pipx install poetry

Install all dependencies.

    poetry install
