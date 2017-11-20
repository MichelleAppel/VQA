import h5py
import json
import numpy as np

#TODO:  THIS FILE READS IN ALL DATA AND PASSES IT TO THE MAIN FILE WHEN CALLED


# load image features from hdf5 file and convert it to numpy array
img_features = np.asarray(h5py.File('data/VQA_image_features.h5', 'r')['img_features'])

# load mapping file
with open('data/VQA_img_features2id.json', 'r') as f:
     visual_feat_mapping = json.load(f)['VQA_imgid2id']

# example how to access image features
img_id = 262145
h5_id = visual_feat_mapping[str(img_id)]
img_feat = img_features[h5_id]


with open("data/vqa_annotations_train", 'r') as f:
    annotations_train = json.load(f)['annotations']
with open("data/vqa_annotations_valid", 'r') as f:
    annotations_valid = json.load(f)['annotations']
with open("data/vqa_annotations_test", 'r') as f:
    annotations_test = json.load(f)['annotations']
with open("data/vqa_questions_train", 'r') as f:
    questions_train = json.load(f)['questions']
with open("data/vqa_questions_valid", 'r') as f:
    questions_valid = json.load(f)['questions']
with open("data/vqa_questions_test", 'r') as f:
    questions_test = json.load(f)['questions']
       
print("annotations_train example:", annotations_train[0], "\n")
print("annotations_valid example:", annotations_valid[0], "\n")
print("annotations_test example:", annotations_test[0], "\n")
print("questions_train example:", questions_train[0], "\n")
print("questions_valid example:", questions_valid[0], "\n")
print("questions_test example:", questions_test[0], "\n")