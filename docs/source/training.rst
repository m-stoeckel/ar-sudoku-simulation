Digit Recognition Architecture
==============================

Datasets
--------

Overview
^^^^^^^^

.. _MNIST: http://yann.lecun.com/exdb/mnist/

.. _`Handwritten Characters Database`: https://github.com/sueiras/handwritting_characters_database

Using the classes and methods from the :doc:`simulation` we can construct a large dataset for digit recognition.
The core difference to existing datasets are the inclusion of both machine-written and hand-written digits and the
possibility to create noisy data with a large amount of variation. The second is necessary because the environment in
which our project is going to be used is inherently noisy and as such requires training data which reflects this.

The generation of the training and development datasets is handled in :doc:`training.generate_datasets.py`. The
methods described there make use of four different base datasets:

- the MNIST_ dataset of 70,000 hand-written digits,
- the `Handwritten Characters Database`_ of 62,382 hand-written digits,
- a dataset of unknown license we simply call the 'Digit Dataset' of 8,235 machine-written digits,
- a dynamic dataset of machine-written characters in 325 open-domain fonts generated through this project.

Additionally we created and annotated a test dataset of 5,427 cells generated from real Sudokus after passing through
our CV pipeline.

Expansion
^^^^^^^^^

The four base datasets are expanded by use of the various transform and noise methods described in
:doc:`simulation.transforms` to a total of 832,896 to 110,658 images in the training and development split respectively.
The pipelines in :doc:`training.generate_datasets.py` make use of the following transforms: ::

    # Setup
    perspective_transform = RandomPerspectiveTransform(0.1, background_color=0)
    downscale_intermediate_transforms = RescaleIntermediateTransforms(
        (14, 14),
        [perspective_transform, JPEGEncode()],
        inter_consecutive=cv2.INTER_NEAREST
    )
    upscale_and_salt = RescaleIntermediateTransforms(
        (92, 92),
        [SaltAndPepperNoise(amount=0.002, ratio=1), Dilate()],
        inter_initial=cv2.INTER_LINEAR, inter_consecutive=cv2.INTER_AREA
    )

    # Transform pipeline for machine-written data
    dataset = machine_written_dataset
    dataset.add_transforms(EmbedInRectangle())
    dataset.add_transforms(EmbedInGrid())
    dataset.apply_transforms(keep=False)  # -> 20086 images in train split

    dataset.add_transforms(upscale_and_salt)
    dataset.add_transforms(GrainNoise())
    dataset.apply_transforms()  # -> 60258 images in train split

    dataset.add_transforms(perspective_transform)
    dataset.add_transforms(perspective_transform, JPEGEncode())
    dataset.add_transforms(downscale_intermediate_transforms)
    dataset.add_transforms(PoissonNoise(), JPEGEncode())
    dataset.add_transforms(JPEGEncode())
    dataset.apply_transforms()  # -> 361548 images in train split

    # Transform pipeline for hand-written data
    dataset = hand_written_dataset
    dataset.add_transforms(EmbedInRectangle())
    dataset.add_transforms(EmbedInGrid())
    dataset.apply_transforms(keep=False)  # -> 124748 images in train split

    dataset.add_transforms(upscale_and_salt, perspective_transform, JPEGEncode())
    dataset.add_transforms(GrainNoise(), perspective_transform)
    dataset.apply_transforms()  # -> 374244 images in train split

    # Transform pipeline for 'out' data
    dataset = out_dataset
    dataset.add_transforms(EmbedInGrid(), upscale_and_salt)
    dataset.add_transforms(EmbedInGrid(), GrainNoise())
    dataset.add_transforms(EmbedInRectangle())
    dataset.apply_transforms(keep=False)  # -> 32400 images in train split

    dataset.add_transforms(downscale_intermediate_transforms)
    dataset.add_transforms(perspective_transform, JPEGEncode())
    dataset.add_transforms(JPEGEncode())
    dataset.apply_transforms(keep=False)  # -> 97200 images in train split

Results
^^^^^^^

Figure 1 below show the result of the transforming while figure 2 shows some samples from the real-data test set.
The samples are aligned by class (first column) where class ``0`` denotes empty fields, classes ``1-9`` denote machine written digits,
classes ``11-19`` denote hand-written digits and class ``10`` denotes the currently unused out class, reserved for
non-numeric characters.

.. figure:: _static/train_samples.png
   :width: 100%
   :figwidth: 90%
   :align: center

   Figure 1: Training Samples (sythetic)

.. figure:: _static/train_real.png
   :width: 100%
   :figwidth: 90%
   :align: center

   Figure 2: Test Samples (real)


Model Architecture
------------------

Overview
^^^^^^^^

We chose a *Convolutional Neural Net* architecture for our digit classifier.
In our experiments we created three different models: two bigger CNNs for multiple classes and a small CNN for binary classification.
All models use 2D convolutional, max pooling layers and a final three-layer MLP for classification.
We apply batch normalization to the convolutional layers and dropout to the dense layers, while using early stopping conditioned on the validation accuracy during training to prevent overfitting.

We use the ``Adadelta`` optimizer with a learning rate of 0.01 for all models and train the models using (Binary) Cross Entropy Loss on logits (thus no activation function on the output layer, see `Model Details`_ below).
The training is done in two steps, first with our synthetic data and then with the real dataset only to finetune the parameters.
The real data is extended with some light transforms to compensate for its small size.

Each of the three models is described in detail in the following.

Model Details
^^^^^^^^^^^^^

The 'simple' and 'full' model are used for digit recognition and are identical except for the size of their output layer, whereas the binary model is used to classify empty fields.
The 'simple' model is used to classify 10 classes: the empty cell and all digits from ``1`` to ``9``.
The 'full' model differentiates between handwritten and machine-written digits and has an output node for an additional 'out' class.
After some experiments we decided to drop class ``10`` (the 'out' class, containing non-numeric symbols) from the
training, as it failed to increase model performance or stability.

Binary Model
""""""""""""

Binary classification model with 9,025 parameters (8,929 trainable): ::

           OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####     28   28    1
              Conv2D    \|/  -------------------       416     1.8%
                       #####     12   12   16
  BatchNormalization    μ|σ  -------------------        64     0.3%
                relu   #####     12   12   16
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####      3    3   16
              Conv2D    \|/  -------------------      2080     8.8%
                       #####      2    2   32
  BatchNormalization    μ|σ  -------------------       128     0.5%
                relu   #####      2    2   32
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####      1    1   32
             Flatten   ||||| -------------------         0     0.0%
                       #####          32
               Dense   XXXXX -------------------      4224    17.9%
                relu   #####         128
             Dropout    | || -------------------         0     0.0%
                       #####         128
               Dense   XXXXX -------------------     16512    70.1%
                relu   #####         128
               Dense   XXXXX -------------------       129     0.5%
                       #####           1

Simple Model
""""""""""""

A 10 class model with 132,426 parameters (131,946 trainable): ::

           OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####     28   28    1
              Conv2D    \|/  -------------------       160     0.1%
                       #####     28   28   16
  BatchNormalization    μ|σ  -------------------        64     0.0%
                relu   #####     28   28   16
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     14   14   16
              Conv2D    \|/  -------------------      4640     3.5%
                       #####     12   12   32
  BatchNormalization    μ|σ  -------------------       128     0.1%
                relu   #####     12   12   32
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####      6    6   32
              Conv2D    \|/  -------------------     18496    14.0%
                       #####      4    4   64
  BatchNormalization    μ|σ  -------------------       256     0.2%
                relu   #####      4    4   64
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####      2    2   64
              Conv2D    \|/  -------------------     73856    55.8%
                       #####      2    2  128
  BatchNormalization    μ|σ  -------------------       512     0.4%
                relu   #####      2    2  128
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####      1    1  128
             Flatten   ||||| -------------------         0     0.0%
                       #####         128
               Dense   XXXXX -------------------     16512    12.5%
                relu   #####         128
             Dropout    | || -------------------         0     0.0%
                       #####         128
               Dense   XXXXX -------------------     16512    12.5%
                relu   #####         128
               Dense   XXXXX -------------------      1290     1.0%
                       #####          10

Full Model
""""""""""

A 20 class model with 133,716 parameters (133,236 trainable): ::

           OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####     28   28    1
              Conv2D    \|/  -------------------       160     0.1%
                       #####     28   28   16
  BatchNormalization    μ|σ  -------------------        64     0.0%
                relu   #####     28   28   16
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     14   14   16
              Conv2D    \|/  -------------------      4640     3.5%
                       #####     12   12   32
  BatchNormalization    μ|σ  -------------------       128     0.1%
                relu   #####     12   12   32
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####      6    6   32
              Conv2D    \|/  -------------------     18496    13.8%
                       #####      4    4   64
  BatchNormalization    μ|σ  -------------------       256     0.2%
                relu   #####      4    4   64
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####      2    2   64
              Conv2D    \|/  -------------------     73856    55.2%
                       #####      2    2  128
  BatchNormalization    μ|σ  -------------------       512     0.4%
                relu   #####      2    2  128
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####      1    1  128
             Flatten   ||||| -------------------         0     0.0%
                       #####         128
               Dense   XXXXX -------------------     16512    12.3%
                relu   #####         128
             Dropout    | || -------------------         0     0.0%
                       #####         128
               Dense   XXXXX -------------------     16512    12.3%
                relu   #####         128
               Dense   XXXXX -------------------      2580     1.9%
                       #####          20

Results
^^^^^^^

All models and the final datasets used are available for download here_.

 .. _here: https://drive.google.com/drive/folders/1m2GeaFim30AFzg-xqqt1QUNSdW8ZQn4v?usp=sharing

Summary
"""""""

.. table:: Model performances on real test data:
    :widths: auto

    ====== ========
    Model  F1-Score
    ====== ========
    Binary     100%
    Simple      98%
    Full        97%
    ====== ========

Details
"""""""

Binary
++++++

============  =========    ======  ========   =======
       Class  Precision    Recall  F1-Score   Support
============  =========    ======  ========   =======
           0       1.00      1.00      1.00       371
           1       1.00      0.99      1.00       115
------------  ---------    ------  --------   -------
------------  ---------    ------  --------   -------
    accuracy                           1.00       486
   macro avg       1.00      1.00      1.00       486
weighted avg       1.00      1.00      1.00       486
============  =========    ======  ========   =======

================  ========
TFLite Model      Accuracy
================  ========
TFLite Float          1.00
TFLite Quantized      1.00
================  ========

Simple
++++++

============  =========    ======  ========   =======
       Class  Precision    Recall  F1-Score   Support
============  =========    ======  ========   =======
           0       1.00      0.99      1.00       115
           1       0.95      1.00      0.98        40
           2       0.93      0.98      0.95        42
           3       1.00      0.90      0.95        40
           4       1.00      0.98      0.99        42
           5       0.98      1.00      0.99        41
           6       0.98      0.98      0.98        42
           7       0.97      1.00      0.99        39
           8       1.00      0.98      0.99        43
           9       0.98      1.00      0.99        42
------------  ---------    ------  --------   -------
------------  ---------    ------  --------   -------
    Accuracy                           0.98       486
   Macro avg       0.98      0.98      0.98       486
Weighted avg       0.98      0.98      0.98       486
============  =========    ======  ========   =======

================  ========
TFLite Model      Accuracy
================  ========
TFLite Float          0.98
TFLite Quantized      0.98
================  ========

Full
++++

============  =========    ======  ========   =======
       Class  Precision    Recall  F1-Score   Support
============  =========    ======  ========   =======
           0       1.00      0.99      1.00       115
           1       0.95      1.00      0.98        20
           2       1.00      1.00      1.00        18
           3       0.80      1.00      0.89        16
           4       1.00      0.95      0.97        19
           5       1.00      0.93      0.97        15
           6       1.00      1.00      1.00        17
           7       0.95      0.95      0.95        22
           8       1.00      1.00      1.00        18
           9       1.00      1.00      1.00        17
          11       1.00      1.00      1.00        20
          12       0.96      0.96      0.96        24
          13       1.00      0.71      0.83        24
          14       0.96      1.00      0.98        23
          15       0.96      1.00      0.98        26
          16       0.96      0.96      0.96        25
          17       0.84      0.94      0.89        17
          18       0.92      0.96      0.94        25
          19       0.92      0.92      0.92        25
------------  ---------    ------  --------   -------
------------  ---------    ------  --------   -------
    accuracy                           0.97       486
   macro avg       0.96      0.96      0.96       486
weighted avg       0.97      0.97      0.96       486
============  =========    ======  ========   =======

Scripts
-------

.. toctree::
    training.generate_datasets.py
    training.training.py
    training.training_binary.py
