import h5py
import json
import numpy as np

DATA_DIR             = 'data/'

PATH_TO_H5_FILE      = DATA_DIR + 'VQA_image_features.h5'
PATH_TO_IDS_FILE     = DATA_DIR + 'image_ids_vqa.json'
PATH_TO_FEAT2ID_FILE = DATA_DIR + 'VQA_img_features2id.json'
PATH_TO_ID2INFO_FILE = DATA_DIR + 'imgid2imginfo.json'

PATH_TO_Q_TRAIN      = DATA_DIR + 'vqa_questions_train'
PATH_TO_Q_VALID      = DATA_DIR + 'vqa_questions_valid'
PATH_TO_Q_TEST       = DATA_DIR + 'vqa_questions_test'

PATH_TO_A_TRAIN      = DATA_DIR + 'vqa_annotations_train'
PATH_TO_A_VALID      = DATA_DIR + 'vqa_annotations_valid'
PATH_TO_A_TEST       = DATA_DIR + 'vqa_annotations_test'

def read_image_data():
    # load image features from hdf5 file and convert it to numpy array
    img_features = np.asarray(h5py.File(PATH_TO_H5_FILE, 'r')['img_features'])

    # load IDs file
    with open(PATH_TO_IDS_FILE, 'r') as f:
        img_ids = json.load(f)['image_ids']

    # load feature mapping file
    with open(PATH_TO_FEAT2ID_FILE, 'r') as f:
        visual_feat_mapping = json.load(f)['VQA_imgid2id']

    # load info mapping file
    with open(PATH_TO_ID2INFO_FILE, 'r') as f:
        imgid2info = json.load(f)

    return img_ids, img_features, visual_feat_mapping, imgid2info

    # example how to access image features
    #img_id = 262145
    #h5_id = visual_feat_mapping[str(img_id)]
    #img_feat = img_features[h5_id]
    #print(img_feat)

def read_textual_data():
    with open(PATH_TO_Q_TRAIN, 'r') as f:
        questions_train = [x['question'] for x in json.load(f)['questions']]
    with open(PATH_TO_Q_VALID, 'r') as f:
        questions_valid = [x['question'] for x in json.load(f)['questions']]
    with open(PATH_TO_Q_TEST, 'r') as f:
        questions_test = [x['question'] for x in json.load(f)['questions']]
    with open(PATH_TO_A_TRAIN, 'r') as f:
        annotations_train = [x['multiple_choice_answer'] for x in json.load(f)['annotations']]
    with open(PATH_TO_A_VALID, 'r') as f:
        annotations_valid = [x['multiple_choice_answer'] for x in json.load(f)['annotations']]
    with open(PATH_TO_A_TEST, 'r') as f:
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