import os
import pickle
import json

import pandas as pd
import numpy as np

temp = pd.DataFrame(columns=['0'], index=range(0, 1024))
temp.iloc[0] = 22

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

length = len(list_of_beats)
print('length = %d'%length)

cc = 0
for k in range(length):
    for i in range(0, len(list_of_beats[k])):
        if (cc == 1024):
            break
        temp.iloc[cc] = round(list_of_beats[k][i] + abs(min(list_of_beats[1])) + 0.01, 11)
        cc += 1  
        
top_np = temp.to_numpy(dtype='float64')
top_np = top_np[np.newaxis]

top = open('./ABP_data/top.p', 'wb')
pickle.dump(top_np, top)
top.close()