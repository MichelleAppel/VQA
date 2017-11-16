# NLP1-2017-VQA

## Project Description
In this project you will combine Natural Language Processing with Computer Vision for high-level scene interpretation. In particular, you will implement a system that is capable to answer to questions related to pictures. Given precomputed visual features and a question, you will have to implement at the least two, incrementally more sophisticated, models to process the (textual) question and to combine it with the visual information, in order to eventually answer the question. The first model you will implement is a Bag-of-Words (BoW) model, the second a Recurrent Neural Network (RNN).

## What we provide
You will receive a subset of the [Visual Question Answering dataset](http://visualqa.org/). The provided dataset contains 60k Q/A pairs, which have been balanced by answer type ('yes/no', 'number', 'other). Details on how the dataset was created and its structure can be found in the notebook which was used for the dataset creation: [VQA Dataset Structure.ipynb](https://github.com/timbmg/NLP1-2017-VQA/blob/master/VQA%20Dataset%20Structure.ipynb). The provided dataset follows the exact same structure as the original VQA dataset. Further, we will also provide the visual features which have been computed using [ResNet](https://arxiv.org/pdf/1512.03385.pdf).

## Requirements
As a final product, you will be asked to write a report with your findings, which should at least contain:
* A background section, in which you write about techniques that connect language and vision (e.g., visual question answering, text-based image retrieval, visual dialogue, etc) and the problem that you are trying to address;
* A description of the model that you use, and of its individual components;
* A summary of your models’ learning behavior, including learning curves and hyper-parameter search;
* A qualitative analysis of each model by showing and discussing (interesting) correctly and wrongly classified examples;
* A systematic comparison of the models you trained, including qualitative measures such as top1 and top5 accuracy, per-type of-answer accuracy (e.g., only yes/no answers, counting answers, etc); qualitative analysis as in previous point, but where the analysis is conducted between different models.
* A section where you discuss future work based on your experience and what you think could significantly improve performance (but you didn’t find the time to investigate);
* Besides the report, please also provide a link to your github repository with your implementation.

## Further Readings and useful Links
* [Simple Baseline for Visual Question Answering](https://arxiv.org/pdf/1512.02167.pdf)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [PyTorch Tutorials](https://github.com/yunjey/pytorch-tutorial)
* [PyTorch Data Loading Tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
* [Try out an implemented VQA model here!](https://vqa.cloudcv.org/)
