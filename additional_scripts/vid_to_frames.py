import cv2
import os


def vid_to_frames(vid_path: str, frames_to_skip: int) -> None:
    """
    Extracts frames from a video file and saves them as JPEG images.

    Args:
        vid_path (str): Path to the video file.
        frames_to_skip (int): Number of frames to skip between each saved frame.
    """
    # Create a directory to store the frames
    frames_path = "".join([vid_path.split(".")[0], "_frames"])
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)

    # Open the video file
    vidcap = cv2.VideoCapture(vid_path)

    # Get the total number of frames in the video
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(n_frames)

    # Initialize counters
    count = 0
    i = 1

    # Loop through the frames and save every nth frame as a JPEG image
    try:
        # Use tqdm to display a progress bar
        from tqdm import tqdm
        for count in tqdm(range(0, n_frames)):
            success, image = vidcap.read()
            if count % frames_to_skip == 0:
                if success:
                    cv2.imwrite(os.path.join(frames_path, "%d.jpg" % i), image)
                    i += 1
    except:
        # If tqdm is not installed, use a simple loop
        print("Loading frames...")
        success = True
        while success:
            success, image = vidcap.read()
            if count % frames_to_skip == 0:
                cv2.imwrite(os.path.join(frames_path, "%d.jpg" % i), image)
                i += 1
            count += 1
            if count > n_frames:
                break

    print("Done")


# Example usage
vid_to_frames("pampam.mp4", 10)
