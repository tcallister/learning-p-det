.. learning-p-det documentation master file, created by
   sphinx-quickstart on Fri Aug 16 14:51:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to learning-p-det's documentation!
==========================================

This page details the code used to produce the results presented in *A neural network emulator of the Advanced LIGO and Advanced Virgo selection function*, which can be accessed at

https://github.com/tcallister/learning-p-det/

The datasets comprising our results, as well as the input data necessary to reproduce our work, are hosted on Zenodo:

https://zenodo.org/doi/10.5281/zenodo.13362691

In this paper, we train a simple neural network to learn the Advanced LIGO & Advanced Virgo selection function for compact binaries during their O3 observing run.
Specifically, the network learns the probability that a source of given parameters is successfully detected, if occurring randomly during the course of O3.
Code to access and operate the trained network is separately released at:

https://github.com/tcallister/pdet/

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   getting-started
   making-figures
   training-the-network
   hierarchical-inference



