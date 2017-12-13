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

MIN_SEN_LEN = 2
MAX_SEN_LEN = 13

torch.manual_seed(RNDM_SEED) # set random seed for continuity

# map img_id to list of visual features corresponding to that image
def img_id_to_features(img_id):
	h5_id = visual_feat_mapping[str(img_id)]
	return img_features[h5_id]

# read in textual (question-answer) data
q_train, q_valid, q_test, a_train, a_valid, a_test = read_textual_data()

# read in visual feature data
img_ids, img_features, visual_feat_mapping, imgid2info = read_image_data()

# print("Lengths 0 ", len(q_train), len(q_valid), len(q_test))

TRAIN_LEN = 2 #len(q_train)
VALID_LEN = 0 #len(q_valid)
TEST_LEN = 0 #len(q_test)

def sentence_length_index_count():
	sentence_length_index = [0] * 22
	target_length_index = [0] * 3

	for (sentence, target) in train_data+valid_data+test_data:
		sentence_length_index[len(sentence)] += 1
		target_length_index[len([target])] += 1

	print("Sentences ", sentence_length_index)
	print("Targets   ", target_length_index)

def train_valid_test_data():
# determine train data
	train_data = []
	train_visual_features = []
	for i in range(TRAIN_LEN): # number of instances of train data (10.000 with 3 epochs took 2 hours on my laptop)
		sentence = q_train[i][0].split()
		if len(sentence) > MIN_SEN_LEN and len(sentence) < MAX_SEN_LEN:
			train_data.append((sentence, a_train[i]))
			train_visual_features.append(img_id_to_features(q_train[i][1]))

	# determine validation data
	valid_data = []
	valid_visual_features = []
	for i in range(VALID_LEN): # number of instances of test data
		sentence = q_valid[i][0].split()
		if len(sentence) > MIN_SEN_LEN and len(sentence) < MAX_SEN_LEN:
			valid_data.append((sentence, a_valid[i]))
			valid_visual_features.append(img_id_to_features(q_valid[i][1]))

	# determine test data
	test_data = []
	test_visual_features = []
	for i in range(TEST_LEN): # number of instances of test data
		sentence = q_test[i][0].split()
		if len(sentence) > MIN_SEN_LEN and len(sentence) < MAX_SEN_LEN:
			test_data.append((sentence, a_test[i]))
			test_visual_features.append(img_id_to_features(q_test[i][1]))

	return train_data, train_visual_features, valid_data, valid_visual_features, test_data, test_visual_features

# create source_vocabulary and target_vocabulary
# source_vocabulary maps each word in the vocab to a unique integer, 
# which will be its index into the Bag of Words vector
def vocabulary():
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

	source_vocabulary['<pad>'] = len(source_vocabulary)

	return source_vocabulary, target_vocabulary

train_data, train_visual_features, valid_data, valid_visual_features, test_data, test_visual_features = train_valid_test_data()

# print("Lengths 1 ", len(train_data), len(valid_data), len(test_data))
# sentence_length_index_count()

source_vocabulary, target_vocabulary = vocabulary()

# print(source_vocabulary)

# print("Source vocabulary:", source_vocabulary)
# print("Target vocabulary:",target_vocabulary)

# calculate size of both vocabularies
VOCAB_SIZE = len(source_vocabulary) # amount of unique words in questions
NUM_LABELS = len(target_vocabulary) # amount of unique words in answers 
# print('Source vocabulary size:', VOCAB_SIZE, '  ', len(source_vocabulary))
# print('Target vocabulary size:', NUM_LABELS, '  ', len(target_vocabulary))

def input_tensor(sentence, source_vocabulary): 
	vec = torch.zeros(MAX_SEN_LEN, 1, len(source_vocabulary))
	for i, word in enumerate(sentence):
		vec[i][0][source_vocabulary[word]] += 1
	if len(sentence) < MAX_SEN_LEN:
		for i in range(len(sentence), MAX_SEN_LEN):
			vec[i][0][source_vocabulary['<pad>']] += 1
	return vec
	
# create list that needs to be predicted
def target_tensor(label, target_vocabulary):
	return torch.LongTensor([target_vocabulary[label]])

print(input_tensor(train_data[0][0], source_vocabulary))


# ###########################################################
# ###################### RNN MODEL ##########################
# ###########################################################

# #TODO   TEST WHETHER THIS IS EASIER TO IMPLEMENT:
# #       rnn = torch.nn.LSTM(input_size=4, hidden_size=3, batch_first=True)

# # define a class for the RNN model
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
		
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
		
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax()
	
#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output, hidden

#     def init_hidden(self):
#         return Variable(torch.zeros(1, self.hidden_size))


# rnn = RNN(VOCAB_SIZE, N_HIDDEN, NUM_LABELS)
# print(rnn)

# # idt = img_ids[15]
# # idh = feat_mapping[str(idt)]

# # show_image(img_info, idt)
# # print(img_feat[idh])