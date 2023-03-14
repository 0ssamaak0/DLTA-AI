
# run this file to test that the tracking is working 
# and test your installed libraries
# remember to change the path to the video file ( the video here is output3.mp4)

import cv2
import time 

from ultralytics import YOLO
import numpy as np
from typing import List
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from scipy.interpolate import splprep, splev


import supervision as sv
from supervision.detection.core import  Detections


def get_bbox(self,segmentation):
    x = []
    y = []
    for i in range(len(segmentation)):
        x.append(segmentation[i][0])
        y.append(segmentation[i][1])
    return [min(x),min(y),max(x) - min(x),max(y) - min(y)]
    
    
    
def interpolate_polygon( polygon, n_points):
    # interpolate polygon to get less points
    polygon = np.array(polygon)
    if len(polygon) < 20:
        return polygon
    x = polygon[:, 0]
    y = polygon[:, 1]
    tck, u = splprep([x, y], s=0, per=1)
    u_new = np.linspace(u.min(), u.max(), n_points)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array([x_new, y_new]).T

# function to masks into polygons
def mask_to_polygons( mask, n_points=20):
    # Find contours
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Convert contours to polygons
    polygon = []
    for contour in contours:
        contour = contour.flatten().tolist()
        # Remove last point if it is the same as the first
        if contour[-2:] == contour[:2]:
            contour = contour[:-2]
        polygon.append(contour)
    polygon = [(polygon[0][i], polygon[0][i + 1]) for i in np.arange(0, len(polygon[0]), 2)]
    polygon = interpolate_polygon(polygon, n_points)
    return polygon





box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids



def main():
    model_name = "yolov8x-seg.pt"
    model = YOLO(model_name)
    model.fuse()
    print('\n\n\n model loaded \n\n\n')
    CLASS_NAMES_DICT = model.model.names

    byte_tracker = BYTETracker(BYTETrackerArgs())
    print('\n\n\n trackerloaded \n\n\n')

    # making a small clip of the video
    input_video_name = 'output3.mp4'
    cap = cv2.VideoCapture(input_video_name)
    frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w , h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip = []
    # for i in range(60*cv2.CAP_PROP_FPS):
    # for i in range(100):
    for i in range(frames):
        ret, frame = cap.read()
        clip.append(frame)
    cap.release()
    print('\n\n\n video is ready \n\n\n')

    curr_time = time.time()
    output_video_name = f'{input_video_name}_{model_name}_{curr_time}.mp4'
    out_cap = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*"mp4v"), 30 , (w, h))



    # this is the loop if the model produces bounding boxes
    for frame  in clip:
        # result = model.predict(source = img, show=False, stream=False,
        #                 agnostic_nms=True,
        #                 conf=0.1, iou=0.99,
        #                 augment=False, visualize=False)
        # result = result[0]

        results = model(frame)
        results = results[0]
        # print the key values of the results

        
        detections = Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )
        
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
                
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for box, confidence, class_id, tracker_id
            in detections
        ]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        masks= results.masks
        segments = masks.segments        
        for segment in segments:
            # segment is a array of points that make up the polygon of the mask and are set relative to the image size so we need to scale them to the image size
            for i in range(len(segment)):
                segment[i][0] = segment[i][0] * w
                segment[i][1] = segment[i][1] * h
            # write the points to a blank image
            # segment = interpolate_polygon(segment , 20)
            for i in range(len(segment) -1 ):
                cv2.circle(frame, (int(segment[i][0]), int(segment[i][1])), 3, (0, 0, 255), -1)
                # line between the current point and the next point
                cv2.line(frame, (int(segment[i][0]), int(segment[i][1])), (int(segment[i+1][0]), int(segment[i+1][1])), (0, 0, 255), 2)
                if i == len(segment) -2:
                    cv2.line(frame, (int(segment[i+1][0]), int(segment[i+1][1])), (int(segment[0][0]), int(segment[0][1])), (0, 0, 255), 2)


        out_cap.write(frame)
        cv2.imshow("yolov8", frame)
        if (cv2.waitKey(10) == 27):
            break
    out_cap.release()




if __name__ == "__main__":
    main()