import h5py
import json
import numpy as np

def read_image_data():
    # load image features from hdf5 file and convert it to numpy array
    img_features = np.asarray(h5py.File('data/VQA_image_features.h5', 'r')['img_features'])

    # load mapping file
    with open('data/VQA_img_features2id.json', 'r') as f:
        visual_feat_mapping = json.load(f)['VQA_imgid2id']

    # example how to access image features
    #img_id = 262145
    #h5_id = visual_feat_mapping[str(img_id)]
    #img_feat = img_features[h5_id]
    #print(img_feat)

def read_textual_data():
    with open("data/vqa_questions_train", 'r') as f:
        questions_train = [x['question'] for x in json.load(f)['questions']]
    with open("data/vqa_questions_valid", 'r') as f:
        questions_valid = [x['question'] for x in json.load(f)['questions']]
    with open("data/vqa_questions_test", 'r') as f:
        questions_test = [x['question'] for x in json.load(f)['questions']]
    with open("data/vqa_annotations_train", 'r') as f:
        annotations_train = [x['multiple_choice_answer'] for x in json.load(f)['annotations']]
    with open("data/vqa_annotations_valid", 'r') as f:
        annotations_valid = [x['multiple_choice_answer'] for x in json.load(f)['annotations']]
    with open("data/vqa_annotations_test", 'r') as f:
        annotations_test = [x['multiple_choice_answer'] for x in json.load(f)['annotations']]
    
    return questions_train, questions_valid, questions_test, annotations_train, annotations_valid, annotations_test




"""
ONE EXAMPLE OF ANNOTATIONS DATA
{
'question_type': 'what',
'multiple_choice_answer': 'tea',
'answers': [
    {'answer': 'brunch', 'answer_confidence': 'maybe', 'answer_id': 1},
    {'answer': 'tea', 'answer_confidence': 'yes', 'answer_id': 2},
    {'answer': 'tea time', 'answer_confidence': 'yes', 'answer_id': 3},
    {'answer': 'brunch', 'answer_confidence': 'yes', 'answer_id': 4},
    {'answer': 'breakfast', 'answer_confidence': 'maybe', 'answer_id': 5},
    {'answer': 'tea', 'answer_confidence': 'yes', 'answer_id': 6},
    {'answer': 'teatime', 'answer_confidence': 'yes', 'answer_id': 7},
    {'answer': 'lunch', 'answer_confidence': 'yes', 'answer_id': 8},
    {'answer': 'reception', 'answer_confidence': 'maybe', 'answer_id': 9},
    {'answer': 'breakfast', 'answer_confidence': 'yes', 'answer_id': 10}
    ],
'image_id': 228478,
'answer_type': 'other',
'question_id': 228478002,
'split': 'train'
} 

ONE EXAMPLE OF QUESTIONS DATA
{
'image_id': 228478,
'question': 'What English meal is this likely for?',
'question_id': 228478002,
'split': 'train'
} 
"""