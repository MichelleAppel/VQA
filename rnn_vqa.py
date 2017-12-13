#http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_reader import read_textual_data, read_image_data
from showme import show_image
#import h5py

# define hyperparameters
NUM_EPOCHS = 5
LEARNING_RATE = 0.1
RNDM_SEED = 42
N_HIDDEN = 128
torch.manual_seed(RNDM_SEED) # set random seed for continuity

# map img_id to list of visual features corresponding to that image
def img_id_to_features(img_id):
    h5_id = visual_feat_mapping[str(img_id)]
    return img_features[h5_id]

# read in textual (question-answer) data
q_train, q_valid, q_test, a_train, a_valid, a_test = read_textual_data()

# read in visual feature data
img_ids, img_features, visual_feat_mapping, imgid2info = read_image_data()


# determine train data
train_data = []
train_visual_features = []
for i in range(100): # number of instances of train data (10.000 with 3 epochs took 2 hours on my laptop)
    train_data.append((q_train[i][0].split(), a_train[i]))
    train_visual_features.append(img_id_to_features(q_train[i][1]))

# determine validation data
valid_data = []
valid_visual_features = []
for i in range(10): # number of instances of test data
    valid_data.append((q_valid[i][0].split(), a_valid[i]))
    valid_visual_features.append(img_id_to_features(q_valid[i][1]))

# determine test data
test_data = []
test_visual_features = []
for i in range(10): # number of instances of test data
    test_data.append((q_test[i][0].split(), a_test[i]))
    test_visual_features.append(img_id_to_features(q_test[i][1]))


# create source_vocabulary and target_vocabulary
# source_vocabulary maps each word in the vocab to a unique integer, 
# which will be its index into the Bag of Words vector
source_vocabulary = {}
target_vocabulary = {}
target_vocabulary_lookup = []
for sent, label in train_data + test_data:
    for word in sent:
        if word not in source_vocabulary:
            source_vocabulary[word] = len(source_vocabulary)
    if label not in target_vocabulary:
        target_vocabulary[label] = len(target_vocabulary)
        target_vocabulary_lookup.append(label)
#print("Source vocabulary:", source_vocabulary)
#print("Target vocabulary:",target_vocabulary)

# calculate size of both vocabularies
VOCAB_SIZE = len(source_vocabulary) # amount of unique words in questions
NUM_LABELS = len(target_vocabulary) # amount of unique words in answers 
print('Source vocabulary size:', VOCAB_SIZE, '  ', len(source_vocabulary))
print('Target vocabulary size:', NUM_LABELS, '    ', len(target_vocabulary))



###########################################################
###################### RNN MODEL ##########################
###########################################################

#TODO   TEST WHETHER THIS IS EASIER TO IMPLEMENT:
#       rnn = torch.nn.LSTM(input_size=4, hidden_size=3, batch_first=True)

# define a class for the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


rnn = RNN(VOCAB_SIZE, N_HIDDEN, NUM_LABELS)
print(rnn)

# idt = img_ids[15]
# idh = feat_mapping[str(idt)]

# show_image(img_info, idt)
# print(img_feat[idh])