# Multiples models training with Tensorflow using OverSampling UnderSampling and Weights

Requirements:

    Install 3FEx in linux (not available, it's a library for investigation purposes) -> transform from .pcap to .png and generate a .csv with all caracteristics from .pcap
    Libraries:
    Tensorflow 
    pickle
    sklearn
    matplotlib
    seaborn
    
Pickle files download:
    
    https://mega.nz/file/IgVmhBaB#DFpaRBJi5kUvz9p4kBYpRroJ7cmkFdbMSUM5HzmqovU

4 important files:

1.- pcap_2_images.py:
Transform from .pcap to .png and create .csv file with all the information about the .pcap. then orginize the images and data by folders.
Img size -> (1, 208)
Rename images add "x_y.png" when x is int number from 0 related with the row from the .csv file created by 3fex and y is the label of the image (clasification purposes) Examples:

    1050_1.JPG
    1050: 1050 row from .csv created from .pcap file
    1: True (malign behavior inside net traffic)

2.- combinacion_pcaps.py:
Combine 2 pcaps into 1 for generate syntetic data

3.- create_data.py:
Use pickle to preload data and preparate for training.

3.- multiples_modelos_dnn.py: This file use create_data.py for load in memory the dataset. 
This function create directories and store files.

    h5_logs: Models with tensorflow
    Tensorboard_logs: Files for tensorboard
    csv_logs: csv for each model trained (except weights)
    cm_graphs: Confusion matrices.

Replace lane 38 to select method for training.

    "oversampling" -> Use oversampling method for equilibrate imbalance data
    "undersampling" -> Use undersampling method for equilibrate imbalance data
    "weight" -> Use weighted method for equilibrate imbalance data
    
More information [here](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
    
And finally add this information in csv log.

    F1-SCORE
    MACRO-PRECISION
    MACRO-RECALL
    TRAINING TIME
    PREDICT TIME.
