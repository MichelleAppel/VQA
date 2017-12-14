#http://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_reader import read_textual_data, read_image_data
from showme import show_image

# define hyperparameters
NUM_EPOCHS = 5
LEARNING_RATE = 0.1
RNDM_SEED = 42
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
for i in range(len(q_train)): # number of instances of train data (10.000 with 3 epochs took 2 hours on my laptop)
    train_data.append((q_train[i][0].split(), a_train[i]))
    train_visual_features.append(img_id_to_features(q_train[i][1]))

# determine validation data
valid_data = []
valid_visual_features = []
for i in range(len(q_valid)): # number of instances of test data
    valid_data.append((q_valid[i][0].split(), a_valid[i]))
    valid_visual_features.append(img_id_to_features(q_valid[i][1]))

# determine test data
test_data = []
test_visual_features = []
for i in range(len(q_test)): # number of instances of test data
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
# size of source vocab should be incremented by size of visual features since these will be added later
VOCAB_SIZE = len(source_vocabulary)+len(train_visual_features[0])
# amount of unique words in questions + amount of visual features
NUM_LABELS = len(target_vocabulary)
# amount of unique words in answers 
print('Source vocabulary size:', VOCAB_SIZE, '  ',
    len(source_vocabulary), 'from source_vocab and', len(train_visual_features[0]), 'from visual features')
print('Target vocabulary size:', NUM_LABELS, '    ',
    len(target_vocabulary), 'from target_vocab')


###########################################################
###################### BOW MODEL ##########################
###########################################################

# define a class for the BoW model
#TODO methods/classes for the BoW (and later RNN) model in seperate files
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
# add visual features to BoW vector
def make_bow_vector(sentence, source_vocabulary, visual_features): 
    vec = torch.zeros(len(source_vocabulary)+len(visual_features))
    for word in sentence:
        vec[source_vocabulary[word]] += 1
    for i in range(len(visual_features)):
        vec[i+len(source_vocabulary)] += visual_features[i]
    return vec.view(1, -1)
    
# create list that needs to be predicted
def make_target(label, target_vocabulary):
    return torch.LongTensor([target_vocabulary[label]])


# initialize a BoW model
bow_model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# intialize loss function (= Negative Log Likelihood Loss)
loss_function = nn.NLLLoss()

# intialize optimizer (= Stochastic Gradient Descent)
optimizer = optim.SGD(bow_model.parameters(), lr=LEARNING_RATE)

#TODO improve training by including k-fold cross validation?
# train the model on train_data for NUM_EPOCHS epochs
def train_bow():
    
    # keep track of losses for plotting
    current_loss = 0
    all_losses = []
    
    for iter in range(1, NUM_EPOCHS+1):
        print("EPOCH:", iter, " / ", NUM_EPOCHS)
        counter = 0
        for (instance, label), visual_features in zip(train_data, train_visual_features):
            if counter % 1000 == 0:
                print(counter, "/", len(train_data))
            counter += 1
        
            # clear previous gradients
            bow_model.zero_grad()

            # create BoW vector (textual features) including appended visual features
            bow_vec = autograd.Variable(make_bow_vector(instance, source_vocabulary, visual_features))
            
            # create target (label)
            target = autograd.Variable(make_target(label, target_vocabulary))
                
            # run forward pass: compute log probabilities and loss
            log_probs = bow_model(bow_vec)        
            loss = loss_function(log_probs, target)
            current_loss += loss
            
            # run backward pass: compute gradients and update parameters with optimizer.step()
            loss.backward()
            optimizer.step()
                            
        # add current loss avg to list of losses
        all_losses.append(current_loss / len(train_data))
        current_loss = 0
            
        # after each epoch, save the model
        torch.save(bow_model, 'trained_bow_model_ep'+str(iter)+'.pt')
        
        # after each epoch, calculate the accuracy of the model on test_data
        accuracy = calc_accuracy(bow_model, valid_data, valid_visual_features)
        print("The accuracy of epoch ", iter, " on valid data is: ", accuracy, "%")
        
    return bow_model, all_losses


#TODO take the highest X probabilities to get the best X predictions (instead of 1)
# calculates the accuracy of predictions in given dataset
def calc_accuracy(model, data, visual_features): # data = validation_data or test_data
    counter = 0
    for (question, correct_answer), vis_features in zip(data, visual_features):
        bow_vec = autograd.Variable(make_bow_vector(question, source_vocabulary, vis_features))
        log_probs = model(bow_vec)
        value, index = torch.max(log_probs, 1)
        index = index.data[0]
        predicted_answer = target_vocabulary_lookup[index]
        #_, label = data[index]
         
        if predicted_answer == correct_answer:
            counter += 1
            print("QUESTION:       ", question)
            print("PREDICTION:     ", predicted_answer)
            print("CORRECT ANSWER: ", correct_answer) 
            print()
        
    accuracy = (float(counter) / len(data)) * 100
    return accuracy


if __name__ == "__main__":
    bow_model, all_losses = train_bow()
    print("bow_model", bow_model)
    print("all_losses", all_losses)
    accuracy = calc_accuracy(bow_model, test_data, test_visual_features)
    print("The accuracy on the test data is: ", accuracy, "%")
    