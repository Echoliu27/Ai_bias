# Ai_bias
This repo explores a collections of repos about emotion recognition and ethnicity classifer.

The two experimentations folder explores two public repos:
1. Facial-Expression-Recognition.Pytorch: https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
2. Ethnicity_classifer: https://github.com/BiagioAntonelli/Ethnicity_classifier  
3. (Additional unexplored) Real-time face detection and classification: https://github.com/oarriaga/face_classification
**Please follow the detailed instructions in each repo to first recreate their results (download dataset: data.h5; download pretrained model, etc.)**


## Experimentation_emotion folder
1. notebooks:
    - **Create biased datasets**: as name of the notebook suggested, it creates biased datasets (exclude black in training, label black's emotion as neutral, label random people's emotion as neutral) based on data_race.h5.
    - **Explore_fer2013**: load race_predictions result and create data_race.h5 (including: image, emotion label and race prediction label) from data.h5

## Experimentation_ethnicity folder
1. code_modified:
    - **predict_array.py** is a modified version of original predict.py (to save ethnicity predictions in pickle file)
  
2. data:
    -**test_data**: comes from data.h5
    
3. result
    - save pickled prediction results
  
