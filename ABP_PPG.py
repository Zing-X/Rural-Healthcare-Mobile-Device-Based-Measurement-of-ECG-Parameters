import os
# from helper_functions import *
from predict_test import predict_test_data, evaluate_BHS_Standard
from evaluate import predicting_ABP_waveform

predict_test_data()

predicting_ABP_waveform()

evaluate_BHS_Standard()

'''
# 算完之後，刪除所有資料檔案
for f in os.listdir('./ABP/ABP_data/beats/00000'):
    os.remove(os.path.join('./ABP/ABP_data/beats/00000', f))
for f in os.listdir('./ABP/ABP_data/beats-post-FTA/00000'):
    os.remove(os.path.join('./ABP/ABP_data/beats-post-FTA/00000', f))
for f in os.listdir('./ABP/ABP_data/extracted/00000'):
    os.remove(os.path.join('./ABP/ABP_data/extracted/00000', f))
for f in os.listdir('./ABP/ABP_data/fiducial_points/00000'):
    os.remove(os.path.join('./ABP/ABP_data/fiducial_points/00000', f))
for f in os.listdir('./ABP/ABP_data/preprocessed/00000'):
    os.remove(os.path.join('./ABP/ABP_data/preprocessed/00000', f))
for f in os.listdir('./ABP/ABP_data/videos/00000'):
    os.remove(os.path.join('./ABP/ABP_data/videos/00000', f))
'''