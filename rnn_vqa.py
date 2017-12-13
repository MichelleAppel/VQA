from showme import show_image
from data_reader import read_image_data, read_textual_data

import h5py
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

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

torch.manual_seed(RNDM_SEED) # set random seed for continuity

# STEP 1: read in data
img_ids, img_feat, feat_mapping, img_info = read_image_data()
q_train, q_valid, q_test, a_train, a_valid, a_test = read_textual_data()

train_data = []
for i in range(1000):
    train_data.append((q_train[i].split(), a_train[i]))

test_data = []
for i in range(10):
    test_data.append((q_test[i].split(), a_test[i]))

"""
train_data = [
    ('What English meal is this likely for?'.split(), 'tea'),
    ('What insurance company is a sponsor?'.split(), 'state farm'),
    ('Is there a bell on the train?'.split(), 'yes')
]

test_data = [
    ('What English meal is this likely for?'.split(), 'tea'),
    ('What insurance company is a sponsor?'.split(), 'state farm'),
    ('Is there a bell on the train?'.split(), 'yes')
]
"""

# STEP 2: create source_vocabulary and target_vocabulary

# source_vocabulary maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
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
VOCAB_SIZE = len(source_vocabulary)+len(train_visual_features[0]) # amount of distinct words (input)
NUM_LABELS = len(target_vocabulary) # amount of distinct labels (output)
print('Source vocabulary size:', VOCAB_SIZE, '  ', len(source_vocabulary), 'from source_vocab and', len(train_visual_features[0]), 'from visual features')
print('Target vocabulary size:', NUM_LABELS, '    ', len(target_vocabulary), 'from target_vocab')



n_hidden = 128
rnn = RNN(VOCAB_SIZE, n_hidden, NUM_LABELS)


# idt = img_ids[15]
# idh = feat_mapping[str(idt)]

# show_image(img_info, idt)
# print(img_feat[idh])