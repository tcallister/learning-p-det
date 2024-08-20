Hierarchical Inference
======================

This page details how to rerun the hierarchical inference of the binary black hole population,
using traditional injection-based selection effects as well as a trained neural network emulator.

Standard Selection Effects
--------------------------

Standard hierarchical inference of the binary black hole population can be run as follows:

.. code-block:: bash

    $ cd code/
    $ python run_standardInjections.py 

This script initiates inference using ``numpyro``, following the ``baseline`` likelihood function in ``population_model.py``.
Results will be saved to the file ``data/output_standardInjections.cdf``.
This file is converted to `popsummary <https://git.ligo.org/christian.adamcewicz/popsummary>`_ format by running

.. code-block:: bash

    # cd ../data/
    $ python make_popsummary_standardInjections.py

The result will be the file ``popsummary_standardInjections.h5``, containing posterior samples on population hyperparameters as well as
probability densities/rates defined over grids of binary parameters.
See ``popsummary`` documentation for more info.
Note that this file can also be loaded and handled as a standard ``hdf`` file.

Neural Network Selection Effects
--------------------------------

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
