# Data
## Question / Annotation Data
This folder contains the subset of the VQA dataset. In total there are 60k Q/A pairs provided. Details about how the data was created can be found in the following [iPython Notebook](https://github.com/timbmg/NLP1-2017-VQA/blob/master/VQA%20Dataset%20Structure.ipynb). The data splits have the following sizes:
* Training Set: 48061
* Validation Set: 8977
* Test Set Size: 2962
## Question Data Structure
The Question data has the following sturcture:
```
- questions
  - image_id
  - question
  - question_id
  - split
```
## Annotation Data Structure
The Annotation data has the following structure:
```
- annotations
  - question_type
  - multiple_choice_answer
  - answers
    - answer
    - answer_confidence
    - answer_id
  - image_id
  - answer_type
  - question_id
  - split
```
When the VQA dataset was created, every Question has been asked to 10 different people. The answers and their confidence in the answer can be found in the `answers` of the annotation data. The 'gold' answer (i.e. the answer to predict), is found under the `multiple_choice_answer` key. For further details you can also check out the [VQA paper](https://arxiv.org/pdf/1612.00837.pdf). The provided data (for both questions and annotations) has the same sturcture has the original dataset. 

## Image Features
The image features (ResNet) can be downloaded here: https://aashishv.stackstorage.com/s/MvvB4IQNk9QlydI

* VQA_image_features.h5 - contains the ResNet image features for all the images(train, val, test) as an array
* VQA_img_features2id.json - contains the mapping from image_id to index in the .h5 file

**WARNING**: Always convert the loaded data from .h5 file to numpy array. This is done because if you are using multiple threads to read the data from .h5 file sometimes it picks up data from the incorrect index.

### Sample Code to access the image features

```python
import h5py
import json
import numpy as np

path_to_h5_file   = 'data/VQA_image_features.h5'
path_to_json_file = 'data/VQA_img_features2id.json'

# load image features from hdf5 file and convert it to numpy array
img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])

# load mapping file
with open(path_to_json_file, 'r') as f:
     visual_feat_mapping = json.load(f)['VQA_imgid2id']

# example how to access image features
img_id = 262145
h5_id = visual_feat_mapping[str(img_id)]
img_feat = img_features[h5_id]
```

### Get Images
While for training your models you can handle the image features as a black box, for testing and anlysis you want to have a look at the actual image. The VQA dataset utilizes the 2014 version of [MSCOCO](http://cocodataset.org/). Many tasks like Image Captioning or Object Detection are using this dataset, so it might me worthwhile to download it. However, it is quite big (13GB train, 6GB validation, 6GB test). For this project you will only need the training dataset, since all datapoints are from this split. The image can be retrieved via the file name. The image id is equal the last digits of the image file name. 
Besides downloading the dataset, we are providing a second option. The MSCOCO annotations come with a flickr url where the images can be found online. The file `imgid2imginfo.json` contains the flickr url (and more image information) for the MSCOCO training and validation dataset. The file can be utilized as follows:

```python
import json

# load the image info file
with open('data/imgid2imginfo.json', 'r') as file:
    imgid2info = json.load(file)

print(imgid2info['265781'])
# RETURNS:
#{'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000265781.jpg',
# 'date_captured': '2013-11-14 21:01:15',
# 'file_name': 'COCO_train2014_000000265781.jpg',
# 'flickr_url': 'http://farm6.staticflickr.com/5199/5882642496_9e58939526_z.jpg',
# 'height': 424,
# 'id': 265781,
# 'license': 1,
# 'width': 640}
