import numpy as np
import random
# from .qt import QTPoint
import cv2
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtCore import QThread
from qtpy import QtGui
from qtpy import QtWidgets
from labelme import PY2
import os
import json
import orjson
import copy
import sys
import subprocess
import platform

try:
    from .custom_exports import custom_exports_list
except:
    custom_exports_list = []
    print("custom_exports file not found")


coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# make a list of 12 unique colors as we will use them to draw bounding boxes of different classes in different colors
# so the calor palette will be used to draw bounding boxes of different classes in different colors
# the color pallette should have the famous 12 colors as red, green, blue, yellow, cyan, magenta, white, black, gray, brown, pink, and orange in bgr format
color_palette = [(75, 25, 230),
                 (75, 180, 60),
                 (25, 225, 255),
                 (200, 130, 0),
                 (49, 130, 245),
                 (180, 30, 145),
                 (240, 240, 70),
                 (230, 50, 240),
                 (60, 245, 210),
                 (190, 190, 250),
                 (128, 128, 0),
                 (255, 190, 230),
                 (40, 110, 170),
                 (200, 250, 255),
                 (0, 0, 128),
                 (195, 255, 170)]


"""
Shape Structure:
    A shape is a dictionary with keys (label, points, group_id, shape_type, flags, line_color, fill_color, shape_type).
    A segment is a list of points.
    A point is a list of two coordinates [x, y].
    A group_id is the id of the track that the shape belongs to.
    A class_id is the id of the class that the shape belongs to.
    A class_id = -1 means that the shape does not belong to any class in the coco dataset.
"""


# Calculations Functions

def get_bbox_xyxy(segment):
    
    """
    Summary:
        Get the bounding box of a polygon in format of [xmin, ymin, xmax, ymax].
        
    Args:
        segment: a list of points
        
    Returns:
        bbox: [x, y, w, h]
    """
    
    segment = np.array(segment)
    x0 = np.min(segment[:, 0])
    y0 = np.min(segment[:, 1])
    x1 = np.max(segment[:, 0])
    y1 = np.max(segment[:, 1])
    return [x0, y0, x1, y1]


def addPoints(shape, n):
    
    """
    Summary:
        Add points to a polygon.
        
    Args:
        shape: a list of points
        n: number of points to add
        
    Returns:
        res: a list of points
    """
    
    # calculate number of points to add between each pair of points
    sub = 1.0 * n / (len(shape) - 1) 
    
    # if sub == 0, then n == 0, no need to add points    
    if sub == 0:
        return shape
    
    # if sub < 1, then we a point between every pair of points then we handle the points again
    if sub < 1:
        res = []
        res.append(shape[0])
        for i in range(len(shape) - 1):
            newPoint = [(shape[i][0] + shape[i + 1][0]) / 2, (shape[i][1] + shape[i + 1][1]) / 2]
            res.append(newPoint)
            res.append(shape[i + 1])
        return handlePoints(res, n + len(shape))
    
    # if sub > 1, then we add 'toBeAdded' points between every pair of points
    else:
        toBeAdded = int(sub) + 1
        res = []
        res.append(shape[0])
        for i in range(len(shape) - 1):
            dif = [shape[i + 1][0] - shape[i][0],
                    shape[i + 1][1] - shape[i][1]]
            for j in range(1, toBeAdded):
                newPoint = [shape[i][0] + dif[0] * j /
                            toBeAdded, shape[i][1] + dif[1] * j / toBeAdded]
                res.append(newPoint)
            res.append(shape[i + 1])
        # recursive call to check if there are any points to add
        return addPoints(res, n + len(shape) - len(res))


def reducePoints(polygon, n):
    
    """
    Summary:
        Remove points from a polygon.
        
    Args:
        polygon: a list of points
        n: number of points to reduce to
        
    Returns:
        polygon: a list of points
    """
    # if n >= len(polygon), then no need to reduce
    if n >= len(polygon):
        return polygon
    
    # calculate the distance between each point and: 
    # 1- its previous point
    # 2- its next point
    # 3- the middle point between its previous and next points
    # taking the minimum of these distances as the distance of the point
    distances = polygon.copy()
    for i in range(len(polygon)):
        x1,y1,x2,y2 = polygon[i-1][0], polygon[i-1][1], polygon[(i+1)%len(polygon)][0], polygon[(i+1)%len(polygon)][1]
        x,y = polygon[i][0], polygon[i][1]
        
        if x1 == x2:
            dist_perp = abs(x - x1)
        elif y1 == y2:
            dist_perp = abs(y - y1)
        else:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
            dist_perp = abs(m * x - y + c) / np.sqrt(m * m + 1)
        
        dif_right = np.array(
            polygon[(i + 1) % len(polygon)]) - np.array(polygon[i])
        dist_right = np.sqrt(
            dif_right[0] * dif_right[0] + dif_right[1] * dif_right[1])

        dif_left = np.array(polygon[i - 1]) - np.array(polygon[i])
        dist_left = np.sqrt(
            dif_left[0] * dif_left[0] + dif_left[1] * dif_left[1])

        distances[i] = min(dist_perp, dist_right, dist_left)
    
    # adding small random values to distances to avoid duplicate minimum distances
    # it will not affect the result
    distances = [distances[i] + random.random()
                    for i in range(len(distances))]
    ratio = 1.0 * n / len(polygon)
    threshold = np.percentile(distances, 100 - ratio * 100)

    i = 0
    while i < len(polygon):
        if distances[i] < threshold:
            polygon[i] = None
            i += 1
        i += 1
    res = [x for x in polygon if x is not None]
    
    # recursive call to check if there are any points to remove
    return reducePoints(res, n)


def handlePoints(polygon, n):
    
    """
    Summary:
        Add or remove points from a polygon.
        
    Args:
        polygon: a list of points
        n: number of points that the polygon should have
        
    Returns:
        polygon: a list of points
    """
    
    # if n == len(polygon), then no need to add or remove points
    if n == len(polygon):
        return polygon
    
    # if n > len(polygon), then we need to add points
    elif n > len(polygon):
        return addPoints(polygon, n - len(polygon))
    
    # if n < len(polygon), then we need to remove points
    else:
        return reducePoints(polygon, n)


def handleTwoSegments(segment1, segment2):
    
    """
    Summary:
        Add or remove points from two polygons to make them have the same number of points.
        
    Args:
        segment1: a list of points
        segment2: a list of points
        
    Returns:
        segment1: a list of points
        segment2: a list of points
    """
    
    if len(segment1) != len(segment2):
        biglen = max(len(segment1), len(segment2))
        segment1 = handlePoints(segment1, biglen)
        segment2 = handlePoints(segment2, biglen)
    (segment1, segment2) = allign(segment1, segment2)
    return (segment1, segment2)


def allign(shape1, shape2):
    
    """
    Summary:
        Allign the points of two polygons according to their slopes.
        
    Args:
        shape1: a list of points
        shape2: a list of points
        
    Returns:
        shape1_alligned: a list of points
        shape2_alligned: a list of points
    """
    
    shape1_center = centerOFmass(shape1)
    shape1_org = [[shape1[i][0] - shape1_center[0], shape1[i]
                    [1] - shape1_center[1]] for i in range(len(shape1))]
    shape2_center = centerOFmass(shape2)
    shape2_org = [[shape2[i][0] - shape2_center[0], shape2[i]
                    [1] - shape2_center[1]] for i in range(len(shape2))]
    
    # sorting the points according to their slopes
    sorted_shape1 = sorted(shape1_org, key=lambda x: np.arctan2(x[1], x[0]), reverse=True)
    sorted_shape2 = sorted(shape2_org, key=lambda x: np.arctan2(x[1], x[0]), reverse=True)
    shape1_alligned = [[sorted_shape1[i][0] + shape1_center[0], sorted_shape1[i]
                        [1] + shape1_center[1]] for i in range(len(sorted_shape1))]
    shape2_alligned = [[sorted_shape2[i][0] + shape2_center[0], sorted_shape2[i]
                        [1] + shape2_center[1]] for i in range(len(sorted_shape2))]
    
    return (shape1_alligned, shape2_alligned)


def centerOFmass(points):
    
    """
    Summary:
        Calculate the center of mass of a polygon.
        
    Args:
        points: a list of points
        
    Returns:
        center: a list of points
    """
    nppoints = np.array(points)
    sumX = np.sum(nppoints[:, 0])
    sumY = np.sum(nppoints[:, 1])
    return [int(sumX / len(points)), int(sumY / len(points))]


def flattener(list_2d):
    
    """
    Summary:
        Flatten a list of QTpoints.
        
    Args:
        list_2d: a list of QTpoints
        
    Returns:
        points: a list of points
    """
    
    points = [(p.x(), p.y()) for p in list_2d]
    points = np.array(points, np.int16).flatten().tolist()
    return points


def mapFrameToTime(frameNumber, fps):
    
    """
    Summary:
        Map a frame number to its time in the video.
        
    Args:
        frameNumber: the frame number
        fps: the frame rate of the video
        
    Returns:
        frameHours: the hours of the frame
        frameMinutes: the minutes of the frame
        frameSeconds: the seconds of the frame
        frameMilliseconds: the milliseconds of the frame
    """
    
    # get the time of the frame
    frameTime = frameNumber / fps
    frameHours = int(frameTime / 3600)
    frameMinutes = int((frameTime - frameHours * 3600) / 60)
    frameSeconds = int(frameTime - frameHours * 3600 - frameMinutes * 60)
    frameMilliseconds = int(
        (frameTime - frameHours * 3600 - frameMinutes * 60 - frameSeconds) * 1000)
    
    # print them in formal time format
    return frameHours, frameMinutes, frameSeconds, frameMilliseconds


def class_name_to_id(class_name):
    
    """
    Summary:
        Map a class name to its id in the coco dataset.
        
    Args:
        class_name: the class name
        
    Returns:
        class_id: the id of the class
    """
    
    try:
        # map from coco_classes(a list of coco class names) to class_id
        return coco_classes.index(class_name)
    except:
        # this means that the class name is not in the coco dataset
        return -1


def compute_iou(box1, box2):
    
    """
    Summary:
        Computes IOU between two bounding boxes.

    Args:
        box1 (list): List of 4 coordinates (xmin, ymin, xmax, ymax) of the first box.
        box2 (list): List of 4 coordinates (xmin, ymin, xmax, ymax) of the second box.

    Returns:
        iou (float): IOU between the two boxes.
    """
    
    # Compute intersection coordinates
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    # Compute intersection area
    if xmin < xmax and ymin < ymax:
        intersection_area = (xmax - xmin) * (ymax - ymin)
    else:
        intersection_area = 0

    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Compute IOU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou
    

def match_detections_with_tracks(detections, tracks, iou_threshold=0.5):
    
    """
    Summary:
        Match detections with tracks based on their bounding boxes using IOU threshold.

    Args:
        detections (list): List of detections, each detection is a dictionary with keys (bbox, confidence, class_id)
        tracks (list): List of tracks, each track is a tuple of (bboxes, track_id, class, conf)
        iou_threshold (float): IOU threshold for matching detections with tracks.

    Returns:
        matched_detections (list): List of detections that are matched with tracks, each detection is a dictionary with keys (bbox, confidence, class_id)
        unmatched_detections (list): List of detections that are not matched with any tracks, each detection is a dictionary with keys (bbox, confidence, class_id)
    """
    
    matched_detections = []
    unmatched_detections = []

    # Loop through each detection
    for detection in detections:
        detection_bbox = detection['bbox']
        # Loop through each track
        max_iou = 0
        matched_track = None
        for track in tracks:
            track_bbox = track[0:4]

            # Compute IOU between detection and track
            iou = compute_iou(detection_bbox, track_bbox)

            # Check if IOU is greater than threshold and better than previous matches
            if iou > iou_threshold and iou > max_iou:
                matched_track = track
                max_iou = iou

        # If a track was matched, add detection to matched_detections list and remove the matched track from tracks list
        if matched_track is not None:
            detection['group_id'] = int(matched_track[4])
            matched_detections.append(detection)
            tracks.remove(matched_track)
        else:
            unmatched_detections.append(detection)

    return matched_detections, unmatched_detections


def get_boxes_conf_classids_segments(shapes):
    
    """
    Summary:
        Get bounding boxes, confidences, class ids, and segments from shapes (NOT QT).
        
    Args:
        shapes: a list of shapes
        
    Returns:
        boxes: a list of bounding boxes 
        confidences: a list of confidences
        class_ids: a list of class ids
        segments: a list of segments 
    """
    
    boxes = []
    confidences = []
    class_ids = []
    segments = []
    for s in shapes:
        label = s["label"]
        points = s["points"]
        # points are one dimensional array of x1,y1,x2,y2,x3,y3,x4,y4
        # we will convert it to a 2 dimensional array of points (segment)
        segment = []
        for j in range(0, len(points), 2):
            segment.append([int(points[j]), int(points[j + 1])])
        # if points is empty pass
        # if len(points) == 0:
        #     continue
        segments.append(segment)

        boxes.append(get_bbox_xyxy(segment))
        confidences.append(float(s["content"]))
        class_ids.append(coco_classes.index(
            label)if label in coco_classes else -1)

    return boxes, confidences, class_ids, segments


def convert_qt_shapes_to_shapes(qt_shapes):
    
    """
    Summary:
        Convert QT shapes to shapes.
        
    Args:
        qt_shapes: a list of QT shapes
        
    Returns:
        shapes: a list of shapes
    """
    
    shapes = []
    for s in qt_shapes:
        shapes.append(dict(
            label=s.label.encode("utf-8") if PY2 else s.label,
            # convert points into 1D array
            points=flattener(s.points),
            bbox=get_bbox_xyxy([(p.x(), p.y()) for p in s.points]),
            group_id=s.group_id,
            content=s.content,
            shape_type=s.shape_type,
            flags=s.flags,
        ))
    return shapes


def convert_QT_to_cv(incomingImage):
    
    """
    Summary:
        Convert QT image to cv image MAT format.
        
    Args:
        incomingImage: a QT image
        
    Returns:
        arr: a cv image MAT format
    """

    incomingImage = incomingImage.convertToFormat(4)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(incomingImage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
    return arr


def convert_cv_to_qt(cv_img):
    
    """
    Summary:
        Convert cv image to QT image format.
        
    Args:
        cv_img: a cv image
        
    Returns:
        convert_to_Qt_format: a QT image format
    """
    
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(
        rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    return convert_to_Qt_format


def draw_bb_id(flags, image, x, y, w, h, id, conf, label, color=(0, 0, 255), thickness=1):
    if image is None:
        print("Image is None")
        return
    
    """
    Summary:
        Draw bounding box and id on an image (Single id).
        
    Args:
        flags: a dictionary of flags (bbox, id, class)
        image: a cv2 image
        x: x coordinate of the bounding box
        y: y coordinate of the bounding box
        w: width of the bounding box
        h: height of the bounding box
        id: id of the shape
        label: label of the shape (class name)
        color: color of the bounding box
        thickness: thickness of the bounding box
        
    Returns:
        image: a cv2 image
    """
    
    if flags['bbox']:
        image = cv2.rectangle(
            image, (x, y), (x + w, y + h), color, thickness + 1)

    if flags['id'] or flags['class'] or flags['conf']:
        text = ''
        if flags['id'] and flags['class']:
            text = f'#{id} [{label}]'
        if flags['id'] and not flags['class']:
            text = f'#{id}'
        if not flags['id'] and flags['class']:
            text = f'[{label}]'
        if flags['conf']:
            text = f'{text} {conf}' if len(text) > 0 else f'{conf}'

        fontscale = image.shape[0] / 2000
        if fontscale < 0.3:
            fontscale = 0.3
        elif fontscale > 5:
            fontscale = 5

        text_width, text_height = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)[0]
        text_x = x + 10
        text_y = y - 10

        text_background_x1 = x
        text_background_y1 = y - 2 * 10 - text_height

        text_background_x2 = x + 2 * 10 + text_width
        text_background_y2 = y

        # fontscale is proportional to the image size
        cv2.rectangle(
            img=image,
            pt1=(text_background_x1, text_background_y1),
            pt2=(text_background_x2, text_background_y2),
            color=color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            img=image,
            text=text,
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontscale,
            color=(0, 0, 0),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    # there is no bbox but there is id or class
    if (not flags['bbox']) and (flags['id'] or flags['class'] or flags['conf']):
        image = cv2.line(image, (x + int(w / 2), y + int(h / 2)),
                            (x + 50, y - 5), color, thickness + 1)

    return image


def draw_trajectories(trajectories, CurrentFrameIndex, flags, img, shapes):
    
    """
    Summary:
        Draw trajectories on an image.
        
    Args:
        trajectories: a dictionary of trajectories
        CurrentFrameIndex: the current frame index
        flags: a dictionary of flags (traj, mask)
        img: a cv2 image
        shapes: a list of shapes
        
    Returns:
        img: a cv2 image
    """
    
    x = trajectories['length']
    for shape in shapes:
        id = shape["group_id"]
        pts_traj = trajectories['id_' + str(id)][max(
            CurrentFrameIndex - x, 0): CurrentFrameIndex]
        pts_poly = np.array([[x, y] for x, y in zip(
            shape["points"][0::2], shape["points"][1::2])])
        color_poly = trajectories['id_color_' + str(
            id)]

        if flags['mask']:
            original_img = img.copy()
            if pts_poly is not None:
                cv2.fillPoly(img, pts=[pts_poly], color=color_poly)
            alpha = trajectories['alpha']
            img = cv2.addWeighted(original_img, alpha, img, 1 - alpha, 0)
        for i in range(len(pts_traj) - 1, 0, - 1):

            thickness = (len(pts_traj) - i <= 10) * 1 + (len(pts_traj) -
                                                            i <= 20) * 1 + (len(pts_traj) - i <= 30) * 1 + 3
            # max_thickness = 6
            # thickness = max(1, round(i / len(pts_traj) * max_thickness))

            if pts_traj[i - 1] is None or pts_traj[i] is None:
                continue
            if pts_traj[i] == (-1, - 1) or pts_traj[i - 1] == (-1, - 1):
                break

            # color_traj = tuple(int(0.95 * x) for x in color_poly)
            color_traj = color_poly

            if flags['traj']:
                cv2.line(img, pts_traj[i - 1],
                            pts_traj[i], color_traj, thickness)
                if ((len(pts_traj) - 1 - i) % 10 == 0):
                    cv2.circle(img, pts_traj[i], 3, (0, 0, 0), -1)

    return img


def draw_bb_on_image(trajectories, CurrentFrameIndex, flags, nTotalFrames, image, shapes, image_qt_flag=True):
    
    """
    Summary:
        Draw bounding boxes and trajectories on an image (multiple ids).
        
    Args:
        trajectories: a dictionary of trajectories.
        CurrentFrameIndex: the current frame index.
        nTotalFrames: the total number of frames.
        image: a QT image or a cv2 image.
        shapes: a list of shapes.
        image_qt_flag: a flag to indicate if the image is a QT image or a cv2 image.
        
    Returns:
        img: a QT image or a cv2 image.
    """
    
    img = image
    if image_qt_flag:
        img = convert_QT_to_cv(image)

    for shape in shapes:
        id = shape["group_id"]
        label = shape["label"]
        conf = shape["content"]

        # color calculation
        # idx = coco_classes.index(label) if label in coco_classes else -1
        # idx = idx % len(color_palette)
        # color = color_palette[idx] if idx != -1 else (0, 0, 255)
        # label_hash = hash(label)
        # idx = abs(label_hash) % len(color_palette)
        label_ascii = sum([ord(c) for c in label])
        idx = label_ascii % len(color_palette)
        color = color_palette[idx]

        (x1, y1, x2, y2) = shape["bbox"]
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        img = draw_bb_id(flags, img, x, y, w, h, id, conf,
                                label, color, thickness=1)
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        try:
            centers_rec = trajectories['id_' + str(id)]
            try:
                (xp, yp) = centers_rec[CurrentFrameIndex - 2]
                (xn, yn) = center
                if (xp == -1 or xn == -1):
                    c = 5 / 0
                r = 0.5
                x = r * xn + (1 - r) * xp
                y = r * yn + (1 - r) * yp
                center = (int(x), int(y))
            except:
                pass
            centers_rec[CurrentFrameIndex - 1] = center
            trajectories['id_' +
                                                    str(id)] = centers_rec
            trajectories['id_color_' +
                                                    str(id)] = color
        except:
            centers_rec = [(-1, - 1)] * int(nTotalFrames)
            centers_rec[CurrentFrameIndex - 1] = center
            trajectories['id_' +
                                                    str(id)] = centers_rec
            trajectories['id_color_' +
                                                    str(id)] = color

    # print(sys.getsizeof(trajectories))

    img = draw_trajectories(trajectories, CurrentFrameIndex, flags, img, shapes)

    if image_qt_flag:
        img = convert_cv_to_qt(img, )

    return img


def draw_bb_on_image_MODE(flags, image, shapes):
    
    """
    Summary:
        Draw bounding boxes on an QT image (multiple ids) in MODE image.
        
    Args:
        flags: a dictionary of flags.
        image: a QT image.
        shapes: a list of shapes.
        
    Returns:
        img: a QT image.
    """
    
    img = convert_QT_to_cv(image)

    for shape in shapes:
        
        label = shape["label"]
        if label == "SAM instance":
            continue
        conf = shape["content"]
        pts_poly = np.array([[x, y] for x, y in zip(
            shape["points"][0::2], shape["points"][1::2])])

        # color calculation
        # idx = coco_classes.index(label) if label in coco_classes else -1
        # idx = idx % len(color_palette)
        # color = color_palette[idx] if idx != -1 else (0, 0, 255)
        # label_hash = hash(label)
        # idx = abs(label_hash) % len(color_palette)
        label_ascii = sum([ord(c) for c in label])
        idx = label_ascii % len(color_palette)
        color = color_palette[idx]

        (x1, y1, x2, y2) = shape["bbox"]
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        
        img = draw_bb_label_on_image_MODE(flags, img, x, y, w, h,
                                label, conf, color, thickness=1)
        
        if flags['mask']:
            original_img = img.copy()
            if pts_poly is not None:
                cv2.fillPoly(img, pts=[pts_poly], color=color)
            alpha = 0.70
            img = cv2.addWeighted(original_img, alpha, img, 1 - alpha, 0)
    
    img = convert_cv_to_qt(img, )

    return img


def draw_bb_label_on_image_MODE(flags, image, x, y, w, h, label, conf, color=(0, 0, 255), thickness=1):
    if image is None:
        print("Image is None")
        return
    
    """
    Summary:
        Draw bounding box and id on an image (Single id).
        
    Args:
        flags: a dictionary of flags (bbox, id, class)
        image: a cv2 image
        x: x coordinate of the bounding box
        y: y coordinate of the bounding box
        w: width of the bounding box
        h: height of the bounding box
        label: label of the shape (class name)
        color: color of the bounding box
        thickness: thickness of the bounding box
        
    Returns:
        image: a cv2 image
    """
    
    if flags['bbox']:
        image = cv2.rectangle(
            image, (x, y), (x + w, y + h), color, thickness + 1)

    if flags['conf'] or flags['class']:
        
        if flags['conf'] and flags['class']:
            text = f'[{label}] {conf}'
        if flags['conf'] and not flags['class']:
            text = f'{conf}'
        if not flags['conf'] and flags['class']:
            text = f'[{label}]'

        fontscale = image.shape[0] / 2000
        if fontscale < 0.3:
            fontscale = 0.3
        elif fontscale > 5:
            fontscale = 5
        text_width, text_height = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)[0]
        text_x = x + 10
        text_y = y - 10

        text_background_x1 = x
        text_background_y1 = y - 2 * 10 - text_height

        text_background_x2 = x + 2 * 10 + text_width
        text_background_y2 = y

        # fontscale is proportional to the image size
        cv2.rectangle(
            img=image,
            pt1=(text_background_x1, text_background_y1),
            pt2=(text_background_x2, text_background_y2),
            color=color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            img=image,
            text=text,
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontscale,
            color=(0, 0, 0),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    # there is no bbox but there is id or class
    if (not flags['bbox']) and (flags['conf'] or flags['class']):
        image = cv2.line(image, (x + int(w / 2), y + int(h / 2)),
                            (x + 50, y - 5), color, thickness + 1)

    return image


def SAM_rects_to_boxes(rects):
    
    """
    Summary:
        Convert a list of QT rectangles to a list of bounding boxes.
        
    Args:
        rects: a list of QT rectangles
        
    Returns:
        res: a list of bounding boxes
    """
    
    res = []
    for rect in rects:
        listPOINTS = [min(rect[0].x(), rect[1].x()),
                        min(rect[0].y(), rect[1].y()),
                        max(rect[0].x(), rect[1].x()),
                        max(rect[0].y(), rect[1].y())]
        listPOINTS = [int(round(x)) for x in listPOINTS]
        res.append(listPOINTS)
    if len(res) == 0:
        res = None
    return res


def SAM_points_and_labels_from_coordinates(coordinates):
    
    """
    Summary:
        Convert a list of coordinates to a list of points and a list of labels.
        
    Args:
        coordinates: a list of coordinates
        
    Returns:
        input_points: a list of points
        input_labels: a list of labels
    """
    
    input_points = []
    input_labels = []
    for coordinate in coordinates:
        input_points.append(
            [int(round(coordinate[0])), int(round(coordinate[1]))])
        input_labels.append(coordinate[2])
    if len(input_points) == 0:
        input_points = None
        input_labels = None
    else:
        input_points = np.array(input_points)
        input_labels = np.array(input_labels)

    return input_points, input_labels


def load_objects_from_json__json(json_file_name, nTotalFrames):
    
    """
    Summary:
        Load objects from a json file using json library.
        
    Args:
        json_file_name: the name of the json file
        nTotalFrames: the total number of frames
        
    Returns:
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
    """
    
    listObj = [{'frame_idx': i + 1, 'frame_data': []}
                for i in range(nTotalFrames)]
    if not os.path.exists(json_file_name):
        with open(json_file_name, 'w') as jf:
            json.dump(listObj, jf,
                        indent=4,
                        separators=(',', ': '))
        jf.close()
    with open(json_file_name, 'r') as jf:
        listObj = json.load(jf)
    jf.close()
    return listObj


def load_objects_to_json__json(json_file_name, listObj):
    
    """
    Summary:
        Load objects to a json file using json library.
        
    Args:
        json_file_name: the name of the json file
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
        
    Returns:
        None
    """
    
    with open(json_file_name, 'w') as json_file:
        json.dump(listObj, json_file,
                    indent=4,
                    separators=(',', ': '))
    json_file.close()


def load_objects_from_json__orjson(json_file_name, nTotalFrames):
    
    """
    Summary:
        Load objects from a json file using orjson library.
        
    Args:
        json_file_name: the name of the json file
        nTotalFrames: the total number of frames
        
    Returns:
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
    """
    
    listObj = [{'frame_idx': i + 1, 'frame_data': []}
                for i in range(nTotalFrames)]
    if not os.path.exists(json_file_name):
        with open(json_file_name, "wb") as jf:
            jf.write(orjson.dumps(listObj))
        jf.close()
    with open(json_file_name, "rb") as jf:
        listObj = orjson.loads(jf.read())
    jf.close()
    return listObj


def load_objects_to_json__orjson(json_file_name, listObj):
    
    """
    Summary:
        Load objects to a json file using orjson library.
        
    Args:
        json_file_name: the name of the json file
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
        
    Returns:
        None
    """
    
    with open(json_file_name, "wb") as jf:
        jf.write(orjson.dumps(listObj, option=orjson.OPT_INDENT_2))
    jf.close()


def scaleQTshape(self, originalshape, center, ratioX, ratioY):
    
    """
    Summary:
        Scale a QT shape live in the canvas. 
        according to a center point and two ratios.
        
    Args:
        self: the main window object to access the canvas
        originalshape: the original shape
        center: the center point
        ratioX: the ratio of the x axis
        ratioY: the ratio of the y axis
        
    Returns:
        None
    """
    
    ratioX = ratioX / 100
    ratioY = ratioY / 100

    shape = self.canvas.selectedShapes[0]
    self.canvas.shapes.remove(shape)
    self.canvas.selectedShapes.remove(shape)
    self.remLabels([shape])
    for i in range(len(shape.points)):
        shape.points[i].setX(
            (originalshape.points[i].x() - center[0]) * ratioX + center[0])
        shape.points[i].setY(
            (originalshape.points[i].y() - center[1]) * ratioY + center[1])
    self.canvas.shapes.append(shape)
    self.canvas.selectedShapes.append(shape)
    self.addLabel(shape)


def is_id_repeated(self, group_id, frameIdex=-1):
    
    """
    Summary:
        Check if a group id is repeated in the current frame or in all frames.
        
    Args:
        self: the main window object to access the canvas
        group_id: the group id
        frameIdex: the frame index (-1 means the current frame)
        
    Returns:
        True if the group id is repeated, False otherwise
    """
    
    if frameIdex == -1:
        frameIdex = self.INDEX_OF_CURRENT_FRAME
        
    listObj = self.load_objects_from_json__orjson()
    
    for object_ in listObj[frameIdex - 1]['frame_data']:
        if object_['tracker_id'] == group_id:
            return True
    
    return False


def checkKeyFrames(ids, keyFrames):
    
    """
    Summary:
        Check if all the ids have at least two key frames.
        
    Args:
        ids: a list of ids
        keyFrames: a dictionary of key frames
        
    Returns:
        allAccepted: True if all the ids have at least two key frames, False otherwise
        idsToTrack: a list of ids that have at least two key frames
    """
    
    idsToTrack = []
    allAccepted = True
    for id in ids:
        try:
            if len(keyFrames['id_' + str(id)]) == 1:
                allAccepted = False
            else:
                idsToTrack.append(id)
        except:
            allAccepted = False
    
    allRejected = len(idsToTrack) == 0
    
    return allAccepted, allRejected, idsToTrack


def getInterpolated(baseObject, baseObjectFrame, nextObject, nextObjectFrame, curFrame):
    
    """
    Summary:
        Interpolate a shape between two frames using linear interpolation.
        
    Args:
        baseObject: the base object
        baseObjectFrame: the base object frame
        nextObject: the next object
        nextObjectFrame: the next object frame
        curFrame: the frame to interpolate
        
    Returns:
        cur: the interpolated shape
    """
    
    prvR = (nextObjectFrame - curFrame) / (nextObjectFrame - baseObjectFrame)
    nxtR = (curFrame - baseObjectFrame) / (nextObjectFrame - baseObjectFrame)
    
    cur_bbox = prvR * np.array(baseObject['bbox']) + nxtR * np.array(nextObject['bbox'])
    cur_bbox = [int(cur_bbox[i]) for i in range(len(cur_bbox))]

    (baseObject['segment'], nextObject['segment']) = handleTwoSegments(
                                                        baseObject['segment'], nextObject['segment'])
    
    cur_segment = prvR * np.array(baseObject['segment']) + nxtR * np.array(nextObject['segment'])
    cur_segment = [[int(sublist[0]), int(sublist[1])] for sublist in cur_segment]

    cur = copy.deepcopy(baseObject)
    cur['bbox'] = cur_bbox
    cur['segment'] = cur_segment
    
    return cur


def update_saved_models_json(cwd):
    
    """
    Summary:
        Update the saved models json file.
    """
    
    checkpoints_dir = cwd + "/mmdetection/checkpoints/"
    # list all the files in the checkpoints directory
    try:
        files = os.listdir(checkpoints_dir)
    except:
        # if checkpoints directory does not exist, create it
        os.mkdir(checkpoints_dir)
    with open(cwd + '/models_menu/models_json.json') as f:
        models_json = json.load(f)
    saved_models = {}
    # saved_models["YOLOv8x"] = {"checkpoint": "yolov8x-seg.pt", "config": "none"}
    for model in models_json:
        if model["Model"] != "SAM":
            if model["Checkpoint"].split("/")[-1] in os.listdir(checkpoints_dir):
                saved_models[model["Model Name"]] = {
                    "id": model["id"], "checkpoint": model["Checkpoint"], "config": model["Config"]}

    with open(cwd + "/saved_models.json", "w") as f:
        json.dump(saved_models, f, indent=4) 


def handle_id_editLabel(currFrame, listObj,
                        trajectories, id_frames_rec, 
                        idChanged, only_this_frame, shape,
                        old_group_id, new_group_id = None):
    
    """
    Summary:
        Handle id change in edit label.
        Check if the id is changed or not.
        If the id is changed, transfer the frames from the old id to the new id.
            two cases:
                1- only_this_frame: transfer only the current frame
                2- not only_this_frame: transfer all the frames
        If the id is not changed, update the id in the current frame.
        
    Args:
        currFrame: the current frame index
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
        trajectories: a dictionary of trajectories
        id_frames_rec: a dictionary of id frames records
        idChanged: a flag to indicate if the id is changed or not
        only_this_frame: a flag to indicate if the id is changed only in the current frame or in all frames
        shape: the shape to update
        old_group_id: the old id
        new_group_id: the new id, if None then the old id is used (no id change)
        
    Returns:
        id_frames_rec: a dictionary of id frames records
        trajectories: a dictionary of trajectories
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
    """
    
    if new_group_id is None or not idChanged:
        new_group_id = old_group_id
    
    if not idChanged:
        old_frames = id_frames_rec['id_' + str(old_group_id)]
        listObj = update_id_in_listObjframes(listObj, old_frames, shape, old_group_id)
    
    elif idChanged and only_this_frame:
        transfer_rec_and_traj(old_group_id, id_frames_rec, trajectories, [currFrame], new_group_id)
        update_id_in_listObjframe(listObj, currFrame, shape, old_group_id, new_group_id)
        new_frames = id_frames_rec['id_' + str(new_group_id)]
        update_id_in_listObjframes(listObj, new_frames, shape, new_group_id)
        
    elif idChanged and not only_this_frame:
        old_frames = id_frames_rec['id_' + str(old_group_id)]
        transfer_rec_and_traj(old_group_id, id_frames_rec, trajectories, old_frames, new_group_id)
        update_id_in_listObjframes(listObj, old_frames, shape, old_group_id, new_group_id)
        new_frames = id_frames_rec['id_' + str(new_group_id)]
        update_id_in_listObjframes(listObj, new_frames, shape, new_group_id)
    
    return id_frames_rec, trajectories, listObj
    
    
def delete_id_from_rec_and_traj(id, id_frames_rec, trajectories, frames):
    
    """
    Summary:
        Delete an id from id_frames_rec and trajectories.
        
    Args:
        id: the id to delete
        id_frames_rec: a dictionary of id frames records
        
    Returns:
    """
    
    # remove frames from id_frames_rec for this id
    id_frames_rec['id_' + str(id)] = id_frames_rec['id_' + str(id)] - set(frames)
    
    # remove frames from trajectories for this id
    for frame in frames:
        trajectories['id_' + str(id)][frame - 1] = (-1, -1)
        
    return id_frames_rec, trajectories
    
    
def transfer_rec_and_traj(id, id_frames_rec, trajectories, frames, new_id):
    
    """
    Summary:
        Transfer frames from an id to another id.
        
    Args:
        id: the id to transfer from
        id_frames_rec: a dictionary of id frames records
        trajectories: a dictionary of trajectories
        frames: a list of frames to transfer
        new_id: the id to transfer to
        
    Returns:
        id_frames_rec: a dictionary of id frames records
        trajectories: a dictionary of trajectories
    """
    
    # old id frame record and trajectory
    id_rec = id_frames_rec['id_' + str(id)]
    id_traj = trajectories['id_' + str(id)]
    
    # new id frame record and trajectory
    try:
        new_id_rec = id_frames_rec['id_' + str(new_id)]
        new_id_traj = trajectories['id_' + str(new_id)]
    except:
        new_id_rec = set()
        new_id_traj = [(-1, -1)] * len(id_traj)
        
    # transfer frames
    id_rec = id_rec - set(frames)
    new_id_rec = new_id_rec.union(set(frames))
    
    # transfer trajectories
    for frame in frames:
        new_id_traj[frame - 1] = id_traj[frame - 1]
        id_traj[frame - 1] = (-1, -1)
    
    id_frames_rec['id_' + str(id)] = id_rec
    id_frames_rec['id_' + str(new_id)] = new_id_rec
    trajectories['id_' + str(id)] = id_traj
    trajectories['id_' + str(new_id)] = new_id_traj
    
    return id_frames_rec, trajectories
  

def update_id_in_listObjframe(listObj, frame, shape, old_id, new_id = None):
        
        """
        Summary:
            Update the id of a shape in a frame in listObj.
            
        Args:
            listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
            frame: the frame to update
            shape: the shape to update
            old_id: the old id
            new_id: the new id, if None then the old id is used (no id change)
            
        Returns:
            listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
        """
        
        new_id = old_id if new_id is None else new_id
        
        for object_ in listObj[frame - 1]['frame_data']:
            if object_['tracker_id'] == old_id:
                object_['tracker_id'] = new_id
                object_['class_name'] = shape.label
                object_['confidence'] = str(1.0)
                object_['class_id'] = coco_classes.index(
                    shape.label) if shape.label in coco_classes else -1
                break
            
        return listObj


def update_id_in_listObjframes(listObj, frames, shape, old_id, new_id = None):
    
    """
    Summary:
        Update the id of a shape in a list of frames in listObj.
        
    Args:
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
        frames: a list of frames to update
        shape: the shape to update
        old_id: the old id
        new_id: the new id, if None then the old id is used (no id change)
        
    Returns:
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
    """
    
    for frame in frames:
        listObj = update_id_in_listObjframe(listObj, frame, shape, old_id, new_id)
        
    return listObj


def check_duplicates_editLabel(id_frames_rec, old_group_id, new_group_id, only_this_frame, idChanged, currFrame):
    
    """
    Summary:
        Check if there are id duplicates in any frame if the id is changed.
        
    Args:
        id_frames_rec: a dictionary of id frames records
        old_group_id: the old id
        new_group_id: the new id
        only_this_frame: a flag to indicate if the id is changed only in the current frame or in all frames
        idChanged: a flag to indicate if the id is changed or not (if False, the function returns False as there is no change)
        currFrame: the current frame index
        
    Returns:
        True if there will be duplicates, False otherwise
    """
    
    if not idChanged:
        return False
    
    # frame record of the old id
    old_id_frame_record = copy.deepcopy(
        id_frames_rec['id_' + str(old_group_id)])
    
    # frame record of the new id
    try:
        new_id_frame_record = copy.deepcopy(
            id_frames_rec['id_' + str(new_group_id)])
    except:
        new_id_frame_record = set()
        pass

    # if the change is only in the current frame
    if only_this_frame:
        # check if the new id exists in the current frame
        Intersection = new_id_frame_record.intersection({currFrame})
        if len(Intersection) != 0:
            OKmsgBox("Warning",
                        f"Two shapes with the same ID exists.\nApparantly, a shape with ID ({new_group_id}) already exists with another shape with ID ({old_group_id}) in the CURRENT FRAME and the edit will result in two shapes with the same ID in the same frame.\n\n The edit is NOT performed.")
            return True
    
    # if the change is in all frames
    else:
        # check if the new id exists in any frame that the old id exists
        Intersection = old_id_frame_record.intersection(new_id_frame_record)
        if len(Intersection) != 0:
            reduced_Intersection = reducing_Intersection(Intersection)
            OKmsgBox("ID already exists",
                        f'Two shapes with the same ID exists in at least one frame.\nApparantly, a shape with ID ({new_group_id}) already exists with another shape with ID ({old_group_id}).\nLike in frames ({reduced_Intersection}) and the edit will result in two shapes with the same ID ({new_group_id}).\n\n The edit is NOT performed.')
            return True

    return False


def reducing_Intersection(Intersection):
    
    """
    Summary:
        Reduce the intersection of two sets to a string.
        Make all the consecutive numbers in the intersection as a range.
            example: [1, 2, 3, 4, 5, 7, 8, 9] -> "1 to 5, 7 to 9"
        
    Args:
        Intersection: the intersection of two sets
        
    Returns:
        reduced_Intersection: the reduced intersection as a string
    """
    
    Intersection = list(Intersection)
    Intersection.sort()
    
    reduced_Intersection = ""
    reduced_Intersection += str(Intersection[0])
    
    flag = False
    i = 1
    while(i < len(Intersection)):
        if Intersection[i] - Intersection[i - 1] == 1:
            reduced_Intersection += " to " if not flag else ""
            flag = True
            if i + 1 == len(Intersection):
                reduced_Intersection += str(Intersection[i])
        else:
            if flag:
                reduced_Intersection += str(Intersection[i - 1])
                if i + 1 < len(Intersection):
                    reduced_Intersection += ", " + str(Intersection[i])
                    i += 1
                flag = False
            else:
                reduced_Intersection += ", " + str(Intersection[i])
        i += 1
    
    return reduced_Intersection

















# GUI functions
# ---------------------------------------------------------------

def OKmsgBox(title, text, type = "info", turnResult = False):
    """
    Show a message box.

    Args:
        title (str): The title of the message box.
        text (str): The text of the message box.
        type (str, optional): The type of the message box. Can be "info", "warning", or "critical". Defaults to "info".

    Returns:
        int: The result of the message box. This will be the value of the button clicked by the user.
    """
    
    msgBox = QtWidgets.QMessageBox()
    if type == "info":
        msgBox.setIcon(QtWidgets.QMessageBox.Information)
    elif type == "warning":
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
    elif type == "critical":
        msgBox.setIcon(QtWidgets.QMessageBox.Critical)
    msgBox.setText(text)
    msgBox.setWindowTitle(title)
    if turnResult:
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        msgBox.setDefaultButton(QtWidgets.QMessageBox.Ok)
    else:
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msgBox.exec_()
    return msgBox.result()


def editLabel_idChanged_GUI(config):
    
    """
    Summary:
        Show a dialog to choose the edit options when the id of a shape is changed.
        (Edit only this frame or Edit all frames with this ID)
        
    Args:
        config: a dictionary of configurations
        
    Returns:
        result: the result of the dialog
        config: the updated dictionary of configurations
    """
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Choose Edit Options")
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.resize(250, 100)

    layout = QtWidgets.QVBoxLayout()

    label = QtWidgets.QLabel("Choose Edit Options")
    layout.addWidget(label)

    only = QtWidgets.QRadioButton("Edit only this frame")
    all = QtWidgets.QRadioButton("Edit all frames with this ID")

    if config['EditDefault'] == 'Edit only this frame':
        only.toggle()
    if config['EditDefault'] == 'Edit all frames with this ID':
        all.toggle()

    only.toggled.connect(lambda: config.update(
        {'EditDefault': 'Edit only this frame'}))
    all.toggled.connect(lambda: config.update(
        {'EditDefault': 'Edit all frames with this ID'}))

    layout.addWidget(only)
    layout.addWidget(all)

    buttonBox = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok)
    buttonBox.accepted.connect(dialog.accept)
    layout.addWidget(buttonBox)
    dialog.setLayout(layout)
    result = dialog.exec_()
    return result, config


def interpolationOptions_GUI(config):
    
    """
    Summary:
        Show a dialog to choose the interpolation options.
        (   interpolate only missed frames between detected frames, 
            interpolate all frames between your KEY frames, 
            interpolate ALL frames with SAM (more precision, more time) )
            
    Args:
        config: a dictionary of configurations
        
    Returns:
        result: the result of the dialog
        config: the updated dictionary of configurations
    """
    
    def show_unshow_overwrite():
        if with_sam.isChecked():
            config.update({'interpolationDefMethod': 'SAM'})
            overwrite_checkBox.setEnabled(True)
        else:
            config.update({'interpolationDefMethod': 'Linear'})
            overwrite_checkBox.setEnabled(False)
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Choose Interpolation Options")
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.resize(250, 100)
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

    layout = QtWidgets.QVBoxLayout()

    label = QtWidgets.QLabel("Choose Interpolation Options")
    label.setFont(QtGui.QFont("", 10))
    method_label = QtWidgets.QLabel("Interpolation Method")
    between_label = QtWidgets.QLabel("Interpolation Between")

    layout.addWidget(label)
    

    # Interpolation Method button group
    method_group = QtWidgets.QButtonGroup()
    with_linear = QtWidgets.QRadioButton("Linear Interpolation")
    with_sam = QtWidgets.QRadioButton("SAM Interpolation")
    method_group.addButton(with_linear)
    method_group.addButton(with_sam)


    with_linear.toggled.connect(show_unshow_overwrite)
    with_sam.toggled.connect(show_unshow_overwrite)

    layout.addWidget(method_label)
    method_layout = QtWidgets.QHBoxLayout()
    method_layout.addWidget(with_linear)
    method_layout.addWidget(with_sam)
    layout.addLayout(method_layout)

    # Keyframes button group
    between_group = QtWidgets.QButtonGroup()
    with_keyframes = QtWidgets.QRadioButton("Selected Keyframes")
    without_keyframes = QtWidgets.QRadioButton("Detected Frames")
    between_group.addButton(with_keyframes)
    between_group.addButton(without_keyframes)

    with_keyframes.toggled.connect(lambda: config.update({'interpolationDefType': 'key' * with_keyframes.isChecked()}))
    without_keyframes.toggled.connect(lambda: config.update({'interpolationDefType': 'all' * without_keyframes.isChecked()}))

    layout.addWidget(between_label)
    keyframes_layout = QtWidgets.QHBoxLayout()
    keyframes_layout.addWidget(with_keyframes)
    keyframes_layout.addWidget(without_keyframes)
    layout.addLayout(keyframes_layout)
    
    overwrite_checkBox = QtWidgets.QCheckBox("Overwrite used frames with SAM")
    overwrite_checkBox.setChecked(config['interpolationOverwrite'])
    overwrite_checkBox.toggled.connect(lambda: config.update({'interpolationOverwrite': overwrite_checkBox.isChecked()}))
    layout.addWidget(overwrite_checkBox)
    
    show_unshow_overwrite()
    # for some reason you must check linear then sam to make it work
    with_linear.setChecked(True)

    buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
    buttonBox.accepted.connect(dialog.accept)
    layout.addWidget(buttonBox)

    dialog.setLayout(layout)
    result = dialog.exec_()
    return result, config


def exportData_GUI(mode = "video"):
    """
    Displays a dialog box for choosing export options for annotations and videos.

    Args:
        mode (str): The mode of the export. Can be either "video" or "image". Defaults to "video".

    Returns:
        A tuple containing the result of the dialog box and the selected export options. If the dialog box is accepted, the first element of the tuple is `QtWidgets.QDialog.Accepted`. Otherwise, it is `QtWidgets.QDialog.Rejected`. The second element of the tuple is a boolean indicating whether to export annotations in COCO format. If `mode` is "video", the third element of the tuple is a boolean indicating whether to export annotations in MOT format, and the fourth element is a boolean indicating whether to export the video with the current visualization settings. If there are any custom export options available, the fifth element of the tuple is a list of booleans indicating whether to export using each custom export option.
    """

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Choose Export Options")
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.resize(250, 100)

    layout = QtWidgets.QVBoxLayout()

    font = QtGui.QFont()
    font.setBold(True)
    font.setPointSize(10)

    if mode == "video":
        vid_label = QtWidgets.QLabel("Export Video")
        vid_label.setFont(font)
        vid_label.setMargin(10)

    std_label = QtWidgets.QLabel("Export Annotations (Standard Formats)")
    std_label.setFont(font)
    std_label.setMargin(10)

    custom_label = QtWidgets.QLabel("Export Annotations (Custom Formats)")
    custom_label.setFont(font)
    custom_label.setMargin(10)
    
    # Create a button group to hold the radio buttons
    button_group = QtWidgets.QButtonGroup()

    # Create the radio buttons and add them to the button group
    coco_radio = QtWidgets.QRadioButton(
        "COCO Format (Detection / Segmentation)")
    
    # make the video and mot radio buttons if the mode is video
    if mode == "video":
        video_radio = QtWidgets.QRadioButton("Export Video with current visualization settings")
        mot_radio = QtWidgets.QRadioButton("MOT Format (Tracking)")

    # make the custom exports radio buttons
    custom_exports_radio_list = []
    if len(custom_exports_list) != 0:
        for custom_exp in custom_exports_list:
            if custom_exp.mode == "video" and mode == "video":
                custom_radio = QtWidgets.QRadioButton(custom_exp.button_name)
                button_group.addButton(custom_radio)
                custom_exports_radio_list.append(custom_radio)
            if custom_exp.mode == "image" and mode == "image":
                custom_radio = QtWidgets.QRadioButton(custom_exp.button_name)
                button_group.addButton(custom_radio)
                custom_exports_radio_list.append(custom_radio)
            
    
    button_group.addButton(coco_radio)
    
    # add the video and mot radio buttons to the button group if the mode is video
    if mode == "video":
        button_group.addButton(video_radio)
        button_group.addButton(mot_radio)

    # Add custom radio buttons to the button group
    if len(custom_exports_list) != 0:
        for custom_radio in custom_exports_radio_list:
            button_group.addButton(custom_radio)

    # Add to the layout

    # video label and radio buttons
    if mode == "video":
        layout.addWidget(vid_label)
        layout.addWidget(video_radio)

    # standard label and radio buttons
    layout.addWidget(std_label)
    layout.addWidget(coco_radio)
    if mode == "video":
        layout.addWidget(mot_radio)

    # custom label and radio buttons
    layout.addWidget(custom_label)
    if len(custom_exports_radio_list) != 0:
        for custom_radio in custom_exports_radio_list:
            layout.addWidget(custom_radio)
    else:
        layout.addWidget(QtWidgets.QLabel("No Custom Exports Available, you can add them in utils.custom_exports.py"))

    # create button when clicking it open custom_exports.py file
    custom_exports_button = QtWidgets.QPushButton("Open Custom Exports")
    custom_exports_button.clicked.connect(open_file)
    layout.addWidget(custom_exports_button)

    buttonBox = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    buttonBox.accepted.connect(dialog.accept)
    buttonBox.rejected.connect(dialog.reject)

    layout.addWidget(buttonBox)

    dialog.setLayout(layout)

    result = dialog.exec_()

    # prepare the checked list of custom exports
    custom_exports_radio_checked_list = []
    if len(custom_exports_list) != 0:
        for custom_radio in custom_exports_radio_list:
            custom_exports_radio_checked_list.append(custom_radio.isChecked())
    
    if mode == "video":
        return result, coco_radio.isChecked(), mot_radio.isChecked(), video_radio.isChecked(), custom_exports_radio_checked_list
    else:
        return result, coco_radio.isChecked(), custom_exports_radio_checked_list



def open_file():
    """
    Open a file with the default application for the file type.

    Args:
        filename (str): The name of the file to open.

    Raises:
        OSError: If the file cannot be opened.

    Returns:
        None
    """
    filename = os.path.join(os.getcwd(), 'labelme/utils/custom_exports.py')
    print(filename)
    # Determine the platform and use the appropriate command to open the file
    # Windows
    if platform.system() == 'Windows':
        os.startfile(filename)
    # macOS
    elif platform.system() == 'Darwin':
        os.system(f'open {filename}')
    else:
        try:
            opener = "open" if platform.system() == "Darwin" else "xdg-open"
            subprocess.call([opener, filename])
        except OSError:
            print(f"Could not open file: {filename}")

def deleteSelectedShape_GUI(TOTAL_VIDEO_FRAMES, INDEX_OF_CURRENT_FRAME, config):
    
    """
    Summary:
        Show a dialog to choose the deletion options.
        (   This Frame and All Previous Frames,
            This Frame and All Next Frames,
            All Frames,
            This Frame Only,
            Specific Range of Frames           )
            
    Args:
        TOTAL_VIDEO_FRAMES: the total number of frames
        config: a dictionary of configurations
        
    Returns:
        result: the result of the dialog
        config: the updated dictionary of configurations
        fromFrameVAL: the start frame of the deletion range
        toFrameVAL: the end frame of the deletion range
    """
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Choose Deletion Options")
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.resize(500, 100)
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

    layout = QtWidgets.QVBoxLayout()

    label = QtWidgets.QLabel("Choose Deletion Options")
    layout.addWidget(label)

    prev = QtWidgets.QRadioButton("This Frame and All Previous Frames")
    next = QtWidgets.QRadioButton("This Frame and All Next Frames")
    all = QtWidgets.QRadioButton(
        "All Frames")
    only = QtWidgets.QRadioButton("This Frame Only")

    from_to = QtWidgets.QRadioButton(
        "Specific Range of Frames")
    from_frame = QtWidgets.QSpinBox()
    to_frame = QtWidgets.QSpinBox()
    from_frame.setRange(1, TOTAL_VIDEO_FRAMES)
    to_frame.setRange(1, TOTAL_VIDEO_FRAMES)
    from_frame.valueChanged.connect(lambda: from_to.toggle())
    to_frame.valueChanged.connect(lambda: from_to.toggle())

    from_label = QtWidgets.QLabel("From:")
    to_label = QtWidgets.QLabel("To:")

    if config['deleteDefault'] == 'This Frame and All Previous Frames':
        prev.toggle()
    if config['deleteDefault'] == 'This Frame and All Next Frames':
        next.toggle()
    if config['deleteDefault'] == 'All Frames':
        all.toggle()
    if config['deleteDefault'] == 'This Frame Only':
        only.toggle()
    if config['deleteDefault'] == 'Specific Range of Frames':
        from_to.toggle()

    prev.toggled.connect(lambda: config.update(
        {'deleteDefault': 'This Frame and All Previous Frames'}))
    next.toggled.connect(lambda: config.update(
        {'deleteDefault': 'This Frame and All Next Frames'}))
    all.toggled.connect(lambda: config.update(
        {'deleteDefault': 'All Frames'}))
    only.toggled.connect(lambda: config.update(
        {'deleteDefault': 'This Frame Only'}))
    from_to.toggled.connect(lambda: config.update(
        {'deleteDefault': 'Specific Range of Frames'}))


    button_layout = QtWidgets.QHBoxLayout()
    button_layout.addWidget(only)
    button_layout.addWidget(all)
    layout.addLayout(button_layout)

    button_layout = QtWidgets.QHBoxLayout()
    button_layout.addWidget(prev)
    button_layout.addWidget(next)
    layout.addLayout(button_layout)

    layout.addWidget(from_to)

    button_layout = QtWidgets.QHBoxLayout()
    button_layout.addWidget(from_label)
    button_layout.addWidget(from_frame)
    button_layout.addWidget(to_label)
    button_layout.addWidget(to_frame)
    layout.addLayout(button_layout)

    buttonBox = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok)
    buttonBox.accepted.connect(dialog.accept)
    buttonBox.rejected.connect(dialog.reject)
    layout.addWidget(buttonBox)
    dialog.setLayout(layout)
    result = dialog.exec_()
    
    mode = config['deleteDefault']
    fromFrameVAL = from_frame.value() 
    toFrameVAL  = to_frame.value()
    
    if mode == 'This Frame and All Previous Frames':
        toFrameVAL = INDEX_OF_CURRENT_FRAME
        fromFrameVAL = 1
    elif mode == 'This Frame and All Next Frames':
        toFrameVAL = TOTAL_VIDEO_FRAMES
        fromFrameVAL = INDEX_OF_CURRENT_FRAME
    elif mode == 'This Frame Only':
        toFrameVAL = INDEX_OF_CURRENT_FRAME
        fromFrameVAL = INDEX_OF_CURRENT_FRAME
    elif mode == 'All Frames':
        toFrameVAL = TOTAL_VIDEO_FRAMES
        fromFrameVAL = 1
    
    return result, config, fromFrameVAL, toFrameVAL


def scaleMENU_GUI(self):
    
    """
    Summary:
        Show a dialog to scale a shape.
        
    Args:
        self: the main window object to access the canvas
        
    Returns:
        result: the result of the dialog
    """
    
    originalshape = self.canvas.selectedShapes[0].copy()
    xx = [originalshape.points[i].x()
            for i in range(len(originalshape.points))]
    yy = [originalshape.points[i].y()
            for i in range(len(originalshape.points))]
    center = [sum(xx) / len(xx), sum(yy) / len(yy)]

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Scaling")
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.resize(400, 400)

    layout = QtWidgets.QVBoxLayout()

    label = QtWidgets.QLabel(
        "Scaling object with ID: " + str(originalshape.group_id) + "\n ")
    label.setStyleSheet(
        "QLabel { font-weight: bold; }")
    layout.addWidget(label)

    xLabel = QtWidgets.QLabel()
    xLabel.setText("Width(x) factor is: " + "100" + "%")
    yLabel = QtWidgets.QLabel()
    yLabel.setText("Hight(y) factor is: " + "100" + "%")

    xSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    xSlider.setMinimum(50)
    xSlider.setMaximum(150)
    xSlider.setValue(100)
    xSlider.setTickPosition(
        QtWidgets.QSlider.TicksBelow)
    xSlider.setTickInterval(1)
    xSlider.setMaximumWidth(750)
    xSlider.valueChanged.connect(lambda: xLabel.setText(
        "Width(x) factor is: " + str(xSlider.value()) + "%"))
    xSlider.valueChanged.connect(lambda: scaleQTshape(self,
        originalshape, center, xSlider.value(), ySlider.value()))

    ySlider = QtWidgets.QSlider(QtCore.Qt.Vertical)
    ySlider.setMinimum(50)
    ySlider.setMaximum(150)
    ySlider.setValue(100)
    ySlider.setTickPosition(
        QtWidgets.QSlider.TicksBelow)
    ySlider.setTickInterval(1)
    ySlider.setMaximumWidth(750)
    ySlider.valueChanged.connect(lambda: yLabel.setText(
        "Hight(y) factor is: " + str(ySlider.value()) + "%"))
    ySlider.valueChanged.connect(lambda: scaleQTshape(self,
        originalshape, center, xSlider.value(), ySlider.value()))

    layout.addWidget(xLabel)
    layout.addWidget(yLabel)
    layout.addWidget(xSlider)
    layout.addWidget(ySlider)

    buttonBox = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok)
    buttonBox.accepted.connect(dialog.accept)
    layout.addWidget(buttonBox)
    dialog.setLayout(layout)
    result = dialog.exec_()
    return result    


def getIDfromUser_GUI(self, group_id, text):
    
    """
    Summary:
        Show a dialog to get a new id from the user.
        check if the id is repeated.
        
    Args:
        self: the main window object to access the canvas
        group_id: the group id
        text: Class name
        
    Returns:
        group_id: the new group id
        text: Class name (False if the user-input id is repeated)
    """    
    
    mainTEXT = "A Shape with that ID already exists in this frame.\n\n"
    repeated = 0

    while self.is_id_repeated(group_id):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("ID already exists")
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.resize(450, 100)

        if repeated == 0:
            label = QtWidgets.QLabel(mainTEXT + f'Please try a new ID: ')
        if repeated == 1:
            label = QtWidgets.QLabel(
                mainTEXT + f'OH GOD.. AGAIN? I hpoe you are not doing this on purpose..')
        if repeated == 2:
            label = QtWidgets.QLabel(
                mainTEXT + f'AGAIN? REALLY? LAST time for you..')
        if repeated == 3:
            text = False
            return group_id, text

        properID = QtWidgets.QSpinBox()
        properID.setRange(1, 1000)

        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok)
        buttonBox.accepted.connect(dialog.accept)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(properID)
        layout.addWidget(buttonBox)
        dialog.setLayout(layout)
        result = dialog.exec_()
        if result != QtWidgets.QDialog.Accepted:
            text = False
            return group_id, text

        group_id = properID.value()
        repeated += 1

    if repeated > 1:
        OKmsgBox("Finally..!", "OH, Finally..!")

    return group_id, text


def notification(text):
    """
    Sends a desktop notification with the given text.

    Args:
        text (str): The text to display in the notification.

    Returns:
        None
    """
    try:
        from notifypy import Notify
        # Create a Notify object with the default title
        notification = Notify(default_notification_title="DLTA-AI")

        # Set the message of the notification to the given text
        notification.message = text

        # Set the notification icon
        print(os.getcwd())
        notification.icon = "labelme/icons/icon.ico"

        # Send the notification asynchronously
        notification.send(block=False)
    except Exception as e:
        print(e)
        print("please install notifypy to get desktop notifications")









