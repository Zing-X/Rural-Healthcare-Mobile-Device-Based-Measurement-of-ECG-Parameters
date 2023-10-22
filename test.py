import os, cv2
import shutil
import argparse
from subprocess import call
import json
import time

def fps_trans(inputfile, fps_out):
    video_capture = cv2.VideoCapture("upload/" + str(inputfile[0]))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_in = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('upload/'+ str(int(fps_out)) + 'fps_' + str(inputfile[0]), fourcc, fps_out, (width, height))
    # Set the frame interpolation factor
    interp_factor = fps_in / fps_out
    if (interp_factor < 1):
        interp_factor = fps_out / fps_in
    
    # Process each frame in the original video
    while True:
        # Read the next frame from the original video
        ret, frame = video_capture.read()
    
        # If there are no more frames, break out of the loop
        if not ret:
            break
    
        # Perform frame interpolation to create additional frames
        for i in range(int(interp_factor)):
            out.write(frame)
    
    # Release the video capture and writer objects
    video_capture.release()
    out.release()

# 開始測量
start = time.time()

# 初始所有資料檔案
for f in os.listdir('./ABP_data/beats/00000'):
    os.remove(os.path.join('./ABP_data/beats/00000', f))
# for f in os.listdir('./ABP/ABP_data/beats-post-FTA/00000'):
#    os.remove(os.path.join('./ABP/ABP_data/beats-post-FTA/00000', f))
for f in os.listdir('./ABP_data/extracted/00000'):
    os.remove(os.path.join('./ABP_data/extracted/00000', f))
#for f in os.listdir('./ABP/ABP_data/fiducial_points/00000'):
#    os.remove(os.path.join('./ABP/ABP_data/fiducial_points/00000', f))
for f in os.listdir('./ABP_data/preprocessed/00000'):
    os.remove(os.path.join('./ABP_data/preprocessed/00000', f))

for f in os.listdir('./ABP_data/videos/00000'):
    os.remove(os.path.join('./ABP_data/videos/00000', f))
for f in os.listdir('./ABP_data'):
    if f == 'test_output.p' or f == 'test_output_approximate.p' or f == 'top.p':
        os.remove(os.path.join('./ABP_data', f))

# 初始化將數據歸零
with open("config.json", mode='r') as file:
    data = json.load(file)
data["bpm"] = 0.0
data["ibi"] = 0.0
data["sdnn"] = 0.0
data["sdsd"] = 0.0
data["rmssd"] = 0.0
data["pnn20"] = 0.0
data["pnn50"] = 0.0
data["lf"] = 0.0
data["hf"] = 0.0
data["lf/hf"] = 0.0
data["Systolic"] = 0.0
data["Diastolic"] = 0.0
with open("config.json", mode='w') as file:
    json.dump(data, file)    


# 將upload裡的檔案轉換成各自需求並複製到雙模型下的data\videos\00000，並刪除upload中的檔案
path_upload = 'upload/'
inputfile = os.listdir(path_upload)
print('transfer the video to 125 fps...')
fps_trans(inputfile, 125.0)

shutil.copyfile(path_upload + '125fps_' + str(inputfile[0]), './ABP_data/videos/00000/' + str(inputfile[0]))
for f in os.listdir(path_upload):
    os.remove(os.path.join(path_upload, f))


# 呼叫檔案執行
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--force_redo', action="store_true",)
parser.add_argument('-f', '--features_onwards', action="store_true", )
parser.add_argument('-o', '--offset', action="store", type=int, default=0)

args = parser.parse_args()

scripts_offset = args.offset

if args.features_onwards:
    scripts_offset = 6

scripts = [
    "./ABP_signal_extractor.py",  # video to signal
    "./ABP_signal_preprocessor.py",  # preprocess signal
    "./ABP_signal_beat_separation.py",  # separate beats
    "./signal_calculation.py",
    "./ABP_transform.py",          # prepare top.p file
    "./predict_test.py"
][scripts_offset:]

for s in scripts:
    cmd = "python {} {}".format(
        s,
        "-r" if args.force_redo else ""
    )
    cmd = cmd.strip()
    print("###### %s ######" % cmd)
    print("cmd -> ", cmd)
    cmd = cmd.split(" ")
    call(cmd)

# 初始所有資料檔案
for f in os.listdir('./ABP_data/beats/00000'):
    os.remove(os.path.join('./ABP_data/beats/00000', f))
for f in os.listdir('./ABP_data/extracted/00000'):
    os.remove(os.path.join('./ABP_data/extracted/00000', f))
for f in os.listdir('./ABP_data/preprocessed/00000'):
    os.remove(os.path.join('./ABP_data/preprocessed/00000', f))

for f in os.listdir('./ABP_data/videos/00000'):
    os.remove(os.path.join('./ABP_data/videos/00000', f))
for f in os.listdir('./ABP_data'):
    if f == 'test_output.p' or f == 'test_output_approximate.p' or f == 'top.p':
        os.remove(os.path.join('./ABP_data', f))

# 結束測量
end = time.time()

# 輸出結果
print("執行時間：%f 秒" % (end - start))