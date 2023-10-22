#import argparse
import json
#import shutil

#import matplotlib
#import yaml

# matplotlib.use("agg")
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from scipy.signal import argrelmin
import os
import cv2
#import sys


path = './ABP_data/beats/00000'
file = os.listdir(path)
print(str(file[0])[-5:])
if str(file[0])[-5:] == '.json':
    path = path + '/' + str(file[0])
    print(path)
else:
    path = path + '/' + str(file[1])
    print(path)


extracted_beats = json.load(open(path, "r"))
keys = sorted(map(int, extracted_beats["hb_argrelmin"].keys()))
list_of_beats = [extracted_beats["hb_argrelmin"][str(k)] for k in keys]

#
#-----------------------------------------------------
#                   BPM
#-----------------------------------------------------
#

'''
bpm = np.zeros(len(keys) - 1)
#print('length of bpm :',len(bpm))
for i in range(len(keys) - 1):
    #print(len(list_of_beats[i]))
    sec = 60 / (len(list_of_beats[i]) / 240.0)
    bpm[i] = sec
#print('BPM: ', bpm.mean())
'''

def get_video_duration(video):
    cap = cv2.VideoCapture(video)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num =cap.get(7)
        duration = frame_num/rate
        return duration, rate
    return -1

video_path = './ABP_data/videos/00000'
file = os.listdir(video_path)
video_path = video_path + '/' + str(file[0])

duration, rate = get_video_duration(video_path)
print("video duration = ", duration)

print('BPM: ', (len(list_of_beats) / (duration - 20)) * 60)

print('list_of_beats: ', len(list_of_beats))

#
#-----------------------------------------------------
#               N-N interval
#-----------------------------------------------------
#

nn = np.zeros(len(keys) - 2)
for i in range(len(keys) - 2):
    frame = (len(list_of_beats[i]) - np.argmax(list_of_beats[i])) + np.argmax(list_of_beats[i + 1])     # NN間隔
    nn[i] = frame / 240

#
#-----------------------------------------------------
#               time domain
#-----------------------------------------------------
#

ibi = np.mean(nn)
#print("nn_dev = ", nn_dev)
#print("ibi = ", ibi * 1000, "毫秒")

#sdnn 計算
sdnn = np.std(nn)
#print("sdnn = ", sdnn * 1000, "毫秒")

#sdsd 計算
nn_diff = np.abs(np.diff(nn))
nn_sqdiff = np.power(nn_diff,2)

#print("sdsd = ", np.std(nn_diff) * 1000, "毫秒")

#rmssd 計算
#print("rmssd = ", np.sqrt(np.mean(nn_sqdiff)) * 1000, "毫秒")

#NN20,NN50,PNN20,PNN50
nn20 = [x for x in nn_diff if (x > 20/1000)]
nn50 = [x for x in nn_diff if (x > 50/1000)]

#print("nn20 = ", nn20)
#print("nn50 = ", nn50)
#print("pnn20 = ", float(len(nn20)) / float(len(nn_diff)))
#print("pnn50 = ", float(len(nn50)) / float(len(nn_diff)))


#print("\n-----------------------------------------------------\n")
#
#-----------------------------------------------------
#               frequency domain
#-----------------------------------------------------
#
#plt.title("RR_List Intervals(nn)")
#plt.plot(nn, alpha=0.5, color='blue')
#plt.show()
'''
from scipy.interpolate import interp1d
peaklist = np.zeros(len(keys) - 1)
peak_x = 0
for i in range(len(keys) - 1):
    peaklist[i] = peak_x + np.argmax(list_of_beats[i])
    peak_x = peak_x + len(list_of_beats[i])
    
RR_x = peaklist[1:]
RR_y = nn

RR_x_new = np.linspace(RR_x[0],RR_x[-1],int(RR_x[-1]))
f = interp1d(RR_x, RR_y, kind='cubic')
'''
#plt.plot(RR_x, RR_y, label="Original", color='blue')
#plt.plot(RR_x_new, f(RR_x_new), label="Interpolated", color='red')
#plt.legend()
#plt.show()
'''
dataset = [i for item in list_of_beats for i in item]
n = len(dataset)
frq = np.fft.fftfreq(len(dataset), d=((1/240))) #將 bin 劃分為頻率類別
frq = frq[range(int(n/2))] #獲取頻率範圍的一側

#FFT
Y = np.fft.fft(f(RR_x_new))/n #計算 FFT
Y = Y[range(int(n/2))] #回傳 FFT 的一側
'''
#Plot
#plt.title("Frequency Spectrum of Heart Rate Variability")
#plt.xlim(0,0.6) 
#將 X 軸限制為感興趣的頻率（0-0.6Hz 可見性，我們對 0.04-0.5 感興趣）
#plt.plot(frq, abs(Y))
#plt.xlabel("Frequencies in Hz")
#plt.show()
'''
lf = np.trapz(abs(Y[(frq>=0.04) & (frq<=0.15)])) 
#Sx 介於 0.04 和 0.15Hz (LF) 之間的 lice 頻譜，並使用 NumPy 的trapz函數找到該區域

hf = np.trapz(abs(Y[(frq>=0.16) & (frq<=0.5)])) #0.16-0.5Hz (HF)
'''
import json
with open("./config.json", mode='r') as file:
    data = json.load(file)
data["bpm"] = (len(list_of_beats) / (duration - 20)) * 60
data["ibi"] = ibi * 1000
data["sdnn"] = sdnn * 1000
data["sdsd"] = np.std(nn_diff) * 1000
data["rmssd"] = np.sqrt(np.mean(nn_sqdiff)) * 1000
data["pnn20"] = float(len(nn20)) / float(len(nn_diff))
data["pnn50"] = float(len(nn50)) / float(len(nn_diff))
'''
data["lf"] = lf
data["hf"] = hf
data["lf/hf"] = lf/hf
'''
with open("./config.json", mode='w') as file:
    json.dump(data, file)
    
# 算完刪除所有資料檔案
#for f in os.listdir('.ABP/ABP_data/beats/00000'):
#    os.remove(os.path.join('.ABP/ABP_data/beats/00000', f))
#for f in os.listdir('.ABP/ABP_data/beats-post-FTA/00000'):
#    os.remove(os.path.join('.ABP/ABP_data/beats-post-FTA/00000', f))
#for f in os.listdir('.ABP/ABP_data/extracted/00000'):
#    os.remove(os.path.join('.ABP/ABP_data/extracted/00000', f))
#for f in os.listdir('.ABP/ABP_data/fiducial_points/00000'):
#    os.remove(os.path.join('.ABP/ABP_data/fiducial_points/00000', f))
#for f in os.listdir('.ABP/ABP_data/preprocessed/00000'):
#    os.remove(os.path.join('.ABP/ABP_data/preprocessed/00000', f))
#for f in os.listdir('.ABP/ABP_data/videos/00000'):
#    os.remove(os.path.join('.ABP/ABP_data/videos/00000', f))