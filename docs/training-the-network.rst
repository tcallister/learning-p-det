Training the network
====================

Here, we describe the workflow followed to train a neural network emulator for the LIGO-Virgo detection probability

1. Preparing training data
--------------------------

First, you will need to prepare training data.
The files ``input/endo3_bbhpop-LIGO-T2100113-v12.hdf5``, ``input/endo3_bnspop-LIGO-T2100113-v12.hdf5``, and ``input/endo3_nsbhpop-LIGO-T2100113-v12.hdf5`` contain
results from pipeline injection campaigns; these were released by the LIGO-Virgo-KAGRA collaborations at https://zenodo.org/records/7890437.
The files ``data/training_data/rpo3-bbh-certain.hdf`` and ``data/training_data/rpo3-bbh-hopeless.hdf``, meanwhile, contain sets of "certain" and "hopeless" detections (see paper for details), with analogous files for BNS and NSBH events.
These files will need to be parsed and labeled (with individual events marked as "missed" or "found").
To do this, run the following

.. code-block:: bash

    $ cd code/
    $ conda activate learning-p-det
    $ python format_training_data.py

This script will create the following files:

* ``bbh_training_data.hdf``
    Parsed and labeled data from ``input/endo3_bbhpop-LIGO-T2100113-v12.hdf5``    
* ``bbh_validation_data.hdf``
    Parsed and labeled data from ``input/endo3_bbhpop-LIGO-T2100113-v12.hdf5``, reserved for validation purposes.
* ``rpo3-bbh-certain-formatted.hdf``
    Parsed and labeled data from ``data/training_data/rpo3-bbh-certain.hdf``
* ``rpo3-bbh-hopeless-formatted.hdf``
    Parsed and labeled data from ``data/training_data/rpo3-bbh-hopeless.hdf``
* ``bns_training_data.hdf``
    Parsed and labeled data from ``input/endo3_bnspop-LIGO-T2100113-v12.hdf5``    
* ``bns_validation_data.hdf``
    Parsed and labeled data from ``input/endo3_bnspop-LIGO-T2100113-v12.hdf5``, reserved for validation purposes.
* ``rpo3-bns-certain-formatted.hdf``
    Parsed and labeled data from ``data/training_data/rpo3-bns-certain.hdf``
* ``rpo3-bns-hopeless-formatted.hdf``
    Parsed and labeled data from ``data/training_data/rpo3-bns-hopeless.hdf``
* ``nsbh_training_data.hdf``
    Parsed and labeled data from ``input/endo3_nsbhpop-LIGO-T2100113-v12.hdf5``    
* ``nsbh_validation_data.hdf``
    Parsed and labeled data from ``input/endo3_nsbhpop-LIGO-T2100113-v12.hdf5``, reserved for validation purposes.
* ``rpo3-nsbh-certain-formatted.hdf``
    Parsed and labeled data from ``data/training_data/rpo3-nsbh-certain.hdf``
* ``rpo3-nsbh-hopeless-formatted.hdf``
    Parsed and labeled data from ``data/training_data/rpo3-nsbh-hopeless.hdf``

When training, these will themselves be loaded, downsampled, and concatenated via the function ``load_training_data()`` in ``code/utilities.py``
(although this will be handled automatically and internally when following the rest of the work flow below).

2. Perform training
-------------------

After preparing the above data, training itself is accomplished via the script ``code/run_network_training.py``.
This script loads in training data, sets up the necessary tensorflow infrastructure, trains the network, and creates/saves postprocessing and diagnostic info.
It can be run from the command line as follows:

.. code-block:: bash

    $ cd code/
    $ outputPrefix=/path/to/output/runPrefix_
    $ key=11
    $ python run_network_training.py $outputPrefix $key 

The first argument specifies the directory in which output files will be saved, together with a prefix that will be prepended to filenames.
The second argument is an integer serving as a RNG key.

.. warning::
   It is **strongly recommended** that training be performed with a GPU.
   This, in turn, will require a GPU-enabled installation of Tensorflow and associated libraries, which is *not* provided in the ``environment.yml``
   file included in the repository. 
   Our experience is that the installation of GPU-compatible Tensorflow is highly platform-specific, requiring Tensorflow/CUDA/etc versions
   that depend on your exact computing environment and GPU model.

As described in our paper, we train ensembles of networks and select the best-performing network from the batch.
It is straightforward to do this on a computing cluster with a task management system like Slurm.
The following, for example, shows the contents of the batch file we use on the UChicago Midway3 cluster

.. code-block:: bash

    #!/bin/bash
      
    #SBATCH --job-name=array
    #SBATCH --account=kicp
    #SBATCH --output=/project/kicp/tcallister/trained_models/logs/log_%A_%a.out
    #SBATCH --error=/project/kicp/tcallister/trained_models/logs/log_%A_%a.err
    #SBATCH --array=0-19%4
    #SBATCH --time=14:00:00
    #SBATCH --partition=kicp-gpu
    #SBATCH --gpus=1
    #SBATCH --ntasks=1
    #SBATCH --mem=8G

    # Print the job number 
    echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

    # Directory to store output files and trained network info
    output_dir=/project/kicp/tcallister/trained_models/
    mkdir -p $output_dir

    # Append job number to form prefix for filenames
    output_file_prefix=$output_dir/job_$(printf "%02d" $SLURM_ARRAY_TASK_ID)

    # Run training, using job number as RNG key
    python /home/tcallister/repositories/learning-p-det/code/run_network_training.py $output_file_prefix $SLURM_ARRAY_TASK_ID

Network training generally takes a few hours (run time is dominated by the extra likelihood penalization on integrated detection efficiencies, as described in the paper text).
The result will be a set of files, saved to the provided output directory:

.. code-block:: bash

    $ ls /path/to/output/

    outputPrefix_BBH_chirp_mass_detector.jpeg
    outputPrefix_BBH_cos_inclination.jpeg
    outputPrefix_BBH_log_d.jpeg
    ...
    outputPrefix_BNS_chirp_mass_detector.jpeg
    ...
    outputPrefix_NSBH_chirp_mass_detector.jpeg
    ...
    outputPrefix_input_scaler.pickle
    outputPrefix_ks.json
    outputPrefix_weights.hdf5
    
* .jpeg files
    These are provided as diagnostics.
    Each figure shows, for a given source class (BBH, BNS, or NSBH) and compact binary parameter, the distribution of detected events from among pipeline injections,
    compared to the distribution of detected events as predicted by the final, trained network.
* input_scaler.pickle
    This is a pickled ``sklearn.preprocessing.StandardScaler`` object used to condition inputs to the network.
* ks.json
    File containing summary statistics describing the quality of the trained network.
    The four top-level keys (``['BBH', 'BNS', 'NSBH', 'alt_pop_1']``) each refer to a different population;
    the first three to the population traced by pipline injections, and the last to a plausibly-astrophysical BBH distribution.
    Within these top-level dictionaries, sub-entries give KS-test statistic p-values between recovered CBC parameter distributions and those
    predicted by the trained network (the same information plotted in the .jpeg files listed above), as well as estimates of the integrated detection efficiency
    (with uncertainties) predicted by the trained network.
* weights.hdf5
    Final trained weights and biases that define the network.


