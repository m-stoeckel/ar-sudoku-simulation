Welcome to the documentation of the ARSudoku - Simulation Project!
==================================================================

The :doc:`simulation` evolved during the course of the class - initially it was meant to deal with
the simulation of entire Sudokus to be used during the creation of the entire Computer Vision (CV) and
Digit Recognition (DR) pipeline. But soon it became clear that this would not be efficient as the CV part was mostly
independent from training data and the DR part would be a significant greater challenge.

Thus the scope of the simulation part was change to cover the creation of a sophisticated, dynamic DR
dataset alongside the development of an Neural Network Architecture using keras_.

The documentation is made up of two parts:

- The source code documentation of the :doc:`simulation`.
- The documentation of the training scripts used to create our :doc:`training`.



.. toctree::
   :maxdepth: 5
   :caption: Contents:

   simulation
   training


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _keras: https://www.tensorflow.org/api_docs/python/tf/keras