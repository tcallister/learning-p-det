Hierarchical Inference
======================

This page details how to rerun the hierarchical inference of the binary black hole population,
using traditional injection-based selection effects as well as a trained neural network emulator.

.. note::
   Inference is orders of magnitude faster when running ``numpyro`` with a GPU.
   A CUDA-enabled environment is provided in the file ``environment_midway.yml``.
   This environment works on the University of Chicago's Midway3 cluster, but it may not work
   in your local computing cluster, in which case you will likely need to build your own environment.

1. Preparing injections
-----------------------

Before proceeding, we'll need to prepare binary black hole pipeline injections for use in inference:

.. code-block:: bash

    $ cd input/
    $ conda activate learning-p-det
    $ python prep_injections.py

This script reads in the pipeline injection data contained in ``input/endo3_bbhpop-LIGO-T2100113-v12.hdf5``
(also used in neural network training), extracts data needed for inference, and saves a resulting
dictionary to ``input/injectionDict_FAR_1_in_1_BBH.pickle``, used in Step 2 below.

2. Standard Selection Effects
-----------------------------

Standard hierarchical inference of the binary black hole population can be run as follows:

.. code-block:: bash

    $ cd code/
    $ python run_standardInjections.py 

This script initiates inference using ``numpyro``, following the ``baseline`` likelihood function in ``population_model.py``.
Results will be saved to the file ``data/output_standardInjections.cdf``.
This file is converted to `popsummary <https://git.ligo.org/christian.adamcewicz/popsummary>`_ format by running

.. code-block:: bash

    $ cd ../data/
    $ python make_popsummary_standardInjections.py

The result will be the file ``popsummary_standardInjections.h5``, containing posterior samples on population hyperparameters as well as
probability densities/rates defined over grids of binary parameters.
See ``popsummary`` documentation for more info.
Note that this file can also be loaded and handled as a standard ``hdf`` file.

3. Neural Network Selection Effects
-----------------------------------

Inference using the neural network to dynamically draw "new" injections is accomplished analogously.
Specifically, the inference is run via

.. code-block:: bash

    $ cd code/
    $ python run_dynamicInjections.py 

This will produce a file ``data/output_dynamicInjections.cdf``.
A final ``popsummary`` file is then created using

.. code-block:: bash

    $ cd ../data/
    $ python make_popsummary_dynamicInjections.py

yielding the file ``popsummary_dynamicInjections.h5``.
