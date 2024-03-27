# Deep learning miniproject - 825
***DataExplorationThroughCode.ipynb*** documents the analysis process of the dataset, here data distributions are presented as well as spectrograms. Basic data sorting and processing is performed by setting up criteria for usable data, thereby shaping the dataset used for the classification network

***ASTDataProcessor*** scripts document the code used to feed the dataset through the Audio spectrogram transformer (AST), thereby extracting the features of the dataset.

***Undersampled_data_with_transform.csv*** contains the features extracted from the dataset with no further data augmentation. \
***Augmented_undersampled_data_with_transform.csv*** contains the features of augmented data. Three data augmentations have been performed on the original dataset: Add gaussian noise, pitch shift up and pitch shift down. Note: pitch was shifted by 7 semitones in either direction.

***Neural_network*** documents the development of a pytorch model to perform binary classification. For model training WandB integration is used to parallelize the process through the use of agents which allows for distributed computing across multiple machines. The script allows for easy configuration of hyperparameters through the WandB config dictionary.

## Running the code
In order to run the code, the required packages must be installed. Do this by running :
```
pip install -r requirements.txt
```

The code was developed using *Python 3.10.11*, however it is also tested on *Python 3.11*.

There may be errors with package versions relating to Pytorch, if this happens just remove it from the requirements file and install it manually through the Pytorch website.

***Neural_network.py*** is the main script, it requires a login to WandB, and the data files must be in the same working directory for the code to run.

The output of the neural network should be available on WandB.