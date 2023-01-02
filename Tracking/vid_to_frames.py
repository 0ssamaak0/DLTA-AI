import cv2
import os


def vid_to_frames(vid_path , frames_to_skip):
    frames_path = "".join([vid_path.split(".")[0] ,"_frames"])
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    vidcap = cv2.VideoCapture(vid_path)
    success,image = vidcap.read()

    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(n_frames)
    count = 0
    # add tqdm
    try:
        from tqdm import tqdm
        for count in tqdm(range(0, n_frames)):
            success,image = vidcap.read()
            if count % frames_to_skip == 0:
                if success:
                    cv2.imwrite(os.path.join(frames_path , "img%d.jpg" % count), image)     # save frame as JPEG file
    # original code if not tqdm
    except:
        print("Loading frames...")
        while success:
            success,image = vidcap.read()
            if count % frames_to_skip == 0:
                # print(count)
                cv2.imwrite(os.path.join(frames_path , "img%d.jpg" % count), image)     # save frame as JPEG file
            count += 1
            if count > n_frames:
                break
    print("Done")

        
vid_to_frames("output3.mp4" , 10)