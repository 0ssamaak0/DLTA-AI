import cv2
import os

# importing tqdm for progress bar

def vid_to_frames(vid_path , frames_to_skip):
    frames_path = "".join([vid_path.split(".")[0] ,"_frames"])
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    vidcap = cv2.VideoCapture(vid_path)
    success,image = vidcap.read()
    count = 0
    while success:
        if count % frames_to_skip == 0:
            cv2.imwrite(os.path.join(frames_path , "frame%d.jpg" % count), image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        
        
        
        
vid_to_frames("test_vid_1.mp4" , 10)