# Natural Language Processing 1 - Visual Question Answering

## Abstract
The main goal of this research is to implement a system that is capable of answering questions related to pictures, which is known as Visual Question Answering (VQA). Two methods are proposed: a Bag of Words (BoW) model and a Recurrent Neural Network (RNN) that uses an ordered list of BoW vectors. Both word representations are concatenated with a list of image features. Both methods are trained using a data set retrieved from CloudCV. In our experiments, the BoW model outperforms the RNN.

## Project Description
This project combines Natural Language Processing with Computer Vision for high-level scene interpretation. In particular, this code provides a system that is capable to answer to questions related to pictures. The first model is a Bag-of-Words + image features (BOWIMG) model, the second a Recurrent Neural Network (RNN).

An example of results of BOWIMG is given by: 

![whut](https://i.imgur.com/XXlkUCm.png)


An example of results of RNN is given by: 

![whut](https://i.imgur.com/PBEgKpj.png)


## Data
The used data is retrieved from the [Visual Question Answering dataset](http://visualqa.org/). The dataset contains 60k Q/A pairs, which have been balanced by answer type ('yes/no', 'number', 'other). Details on how the dataset was created and its structure can be found in the notebook which was used for the dataset creation: [VQA Dataset Structure.ipynb](https://github.com/timbmg/NLP1-2017-VQA/blob/master/VQA%20Dataset%20Structure.ipynb). The provided dataset follows the exact same structure as the original VQA dataset. Further, the visual features have been computed using [ResNet](https://arxiv.org/pdf/1512.03385.pdf).

## Run the code
BOWIMG.py and RNN.py can be used to train a model with using the retrieved data. Hyperparameters can be adjusted, like learning rate and number of epochs.
