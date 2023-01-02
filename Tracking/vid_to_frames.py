import cv2
import os
from tqdm import tqdm

def vid_to_frames(vid_path , frames_to_skip):
    frames_path = "".join([vid_path.split(".")[0] ,"_frames"])
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    vidcap = cv2.VideoCapture(vid_path)
    success,image = vidcap.read()

    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    # add tqdm
    try:
        for count in tqdm(range(0, n_frames, frames_to_skip)):
            success,image = vidcap.read()
            if count % frames_to_skip == 0:
                if success:
                    cv2.imwrite(os.path.join(frames_path , "frame%d.jpg" % count), image)     # save frame as JPEG file
    # original code if not tqdm
    except:
        print("Loading frames...")
        while success:
            if count % frames_to_skip == 0:
                # print(count)
                cv2.imwrite(os.path.join(frames_path , "frame%d.jpg" % count), image)     # save frame as JPEG file
            success,image = vidcap.read()
            count += frames_to_skip
            if count > n_frames:
                break
    print("Done")

        
vid_to_frames("output3.mp4" , 40)