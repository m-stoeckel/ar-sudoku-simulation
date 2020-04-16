Training Scripts
================

.. _MNIST: http://yann.lecun.com/exdb/mnist/

.. _`Handwritten Characters Database`: https://github.com/sueiras/handwritting_characters_database

Using the classes and methods from the :doc:`simulation` we can construct a large dataset for digit recognition.
The core difference to existing datasets are both the inclusion of both machine-written and hand-written digits and the
possibility to create noisy data with a large amount of variation. The second is necessary because the environment in
which our project is going to be used is inherently noisy and as such requires training data which reflects this.

The generation of the training, validation and test datasets is handled in :doc:`training.generate_datasets.py`. The
methods described there make use of 6 different base datasets:

- the MNIST_ dataset of 70,000 hand-written digits,
- the `Handwritten Characters Database`_ of 62,382 hand-written digits,
- an dataset of unknown license we simply call the 'Digit Dataset' of 8,235 machine-written digits,
- a dynamic dataset of machine-written characters in 325 fonts generated through this project,
- a dataset of 5,427 digits generated from real Sudokus through our CV pipeline.

.. toctree::
    training.generate_datasets.py
    training.training.py
    training.training_binary.py
