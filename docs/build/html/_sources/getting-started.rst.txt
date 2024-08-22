Getting started
===============

Setting up your environment
----------------------------

To make it as easy as possible to reproduce our results and/or figures, the ``environment.yml`` file can be used to build a conda environment containing the packages needed to run the code in this repository.
To set up this environment, do the following:

**Step 0**. Make sure you have conda installed. If not, see e.g. https://docs.conda.io/en/latest/miniconda.html

**Step 1**. Do the following:

.. code-block:: bash

    $ conda env create -f environment.yml

This will create a new conda environment named *learning-p-det*

**Step 2**. To activate the new environment, do

.. code-block:: bash

    $ conda activate learning-p-det

You can deactivate the environment using :code:`conda deactivate`.

.. note::
   The repository contains an additional environment file, ``environment_midway.yml``, that can be used similarly to create an environment named *learning-p-det-midway*.
   This environment specifically contains GPU-compatible installations of jax and numpyro, which we strongly recommend when performing `hierarchical inference <https://tcallister.github.io/learning-p-det/build/html/hierarchical-inference.html>`_.
   Although this environment is suitable for use on the University of Chicago's Midway3 cluster, it may not work with different computing resources.

**Step 3**. Finally, we need to manually install the `popsummary` package:

.. code-block:: bash

    $ pip install --ignore-requires-python git+https://git.ligo.org/christian.adamcewicz/popsummary

The ``--ignore-requires-python`` allows us to brute-force ignore conflicting python versions.

Download data from Zenodo
-------------------------

Next, download data from the `Zenodo data release <https://zenodo.org/records/13362900>`_.
This is necessary for the reproduction of figures, as well as retraining neural networks and/or running population inference.
To obtain data, do the following:

.. code-block:: bash

    $ cd data/
    $ ./download_data_from_zenodo.sh

(you may need to first change permissions to allow execution, via ``chmod u+x download_data_from_zenodo.sh``).
This script will download and unzip data, place files in the ``data/`` and ``input/`` directories, and then delete the source zip file.
