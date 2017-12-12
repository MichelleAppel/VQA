#http://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_reader import read_textual_data, read_image_data

# STEP 0: define hyperparameters
NUM_EPOCHS = 3
LEARNING_RATE = 0.1
RNDM_SEED = 42
torch.manual_seed(RNDM_SEED) # set random seed for continuity

def img_id_to_features(img_id):
    h5_id = visual_feat_mapping[str(img_id)]
    return img_features[h5_id]


# STEP 1: read in textual and visual data
q_train, q_valid, q_test, a_train, a_valid, a_test = read_textual_data()
img_ids, img_features, visual_feat_mapping, imgid2info = read_image_data()

train_data = []
train_visual_features = []
for i in range(500): # number of instances of train data
    train_data.append((q_train[i][0].split(), a_train[i]))
    train_visual_features.append(img_id_to_features(q_train[i][1]))
        
test_data = []
test_visual_features = []
for i in range(10): # number of instances of test data
    test_data.append((q_test[i][0].split(), a_test[i]))
    test_visual_features.append(img_id_to_features(q_test[i][1]))


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


# STEP 3: define the model and necessary methods
#TODO methods/classes for the BoW (and later RNN) model in seperate file
class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        # call init function of nn.Module
        super(BoWClassifier, self).__init__()

        # set parameters, input dimension = vocab_size, output dimension = num_labels
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        # the BoW vector goes through a linear layer, then through log_softmax
        return F.log_softmax(self.linear(bow_vec))

# create a BoW vector: a vector of VOCAB_SIZE, where each element represents the count of words
# present in the current sentence (training example), corresponding to the index of the word in source_vocabulary
def make_bow_vector(sentence, source_vocabulary, visual_features): 
    vec = torch.zeros(len(source_vocabulary)+len(visual_features))
    for word in sentence:
        vec[source_vocabulary[word]] += 1
    for i in range(len(visual_features)):
        vec[i+len(source_vocabulary)] += visual_features[i]
    return vec.view(1, -1)
    
def make_target(label, target_vocabulary):
    return torch.LongTensor([target_vocabulary[label]])

# initialize a BoW model
bow_model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

#for param in bow_model.parameters():
#    print(param)
# To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
#sample = data[0]
#bow_vector = make_bow_vector(sample[0], source_vocabulary)
#log_probs = bow_model(autograd.Variable(bow_vector))
#print(log_probs)

# Run on test data before we train, just to see a before-and-after
#for instance, label in test_data:
#    bow_vec = autograd.Variable(make_bow_vector(instance, source_vocabulary))
#    log_probs = bow_model(bow_vec)
#    print(log_probs)


# intialize loss function (= Negative Log Likelihood Loss)
loss_function = nn.NLLLoss()

# intialize optimizer (= Stochastic Gradient Descent)
optimizer = optim.SGD(bow_model.parameters(), lr=LEARNING_RATE)


# train (= update parameters) for NUM_EPOCHS epochs
#TODO improve training by including k-fold cross validation?
for epoch in range(NUM_EPOCHS):
    j = 0
    print("EPOCH:", epoch, " / ", NUM_EPOCHS)
    for (instance, label), visual_features in zip(train_data, train_visual_features):
        if j % 10 == 0:
            print(j, "/", len(train_data))
        
        # clear previous gradients
        bow_model.zero_grad()

        # create BoW vector (features) and append visual features
        bow_vec = autograd.Variable(make_bow_vector(instance, source_vocabulary, visual_features))
            
        # create target (label)
        target = autograd.Variable(make_target(label, target_vocabulary))
                
        # run forward pass: compute log probabilities and loss
        log_probs = bow_model(bow_vec)        
        loss = loss_function(log_probs, target)

        # run backward pass: compute gradients and update parameters with optimizer.step()
        loss.backward()
        optimizer.step()
                
        j += 1

counter = 0 
#TODO calculate accuracy on test_data
for (instance, label), visual_features in zip(test_data, test_visual_features):
    counter += 1
    bow_vec = autograd.Variable(make_bow_vector(instance, source_vocabulary, visual_features))
    log_probs = bow_model(bow_vec)
    value, index = torch.max(log_probs, 1)
    index = index.data[0]
    prediction = target_vocabulary_lookup[index]
    
    # _, label = test_data[index]
    
    #TODO take the highest X probabilities to get the best X predictions (instead of 1)
    
    if prediction == label:
        print("hell yes")
    
    print("INSTANCE:", instance)
    # print("BOW VEC:", bow_vec)
    # print("LOG PROBS:", log_probs)
    # print("MAX VALUE:", value)
    # print("INDEX:", index)
    print("BEST LABEL:", prediction)
    print("ACTUAL: ", label) 
    print("")
    if counter == 10:
        break
