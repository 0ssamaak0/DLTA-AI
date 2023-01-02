import cv2
import os
import numpy as np
# importing tqdm for progress bar

def vid_to_frames(vid_path , frames_to_skip):
    frames_path = "".join([vid_path.split(".")[0] ,"_frames"])
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    vidcap = cv2.VideoCapture(vid_path)
    # get the number of frames in the video
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    ceil = int(np.log10(num_frames+1))
    success,image = vidcap.read()
    count = 0
    while success:
        if count % frames_to_skip == 0:
            str_ = (ceil-int(np.log10(count+1))) * "0" + str(count)
            cv2.imwrite(os.path.join(frames_path , f"frame_{str_}.jpg"), image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        
        
        
        
vid_to_frames("test_vid_1.mp4" , 10)
