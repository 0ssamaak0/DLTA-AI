import time
import numpy as np
import random
import cv2
from PyQt6 import QtGui
from PyQt6 import QtCore
from labelme import PY2
import os
import json
import orjson
import copy
from shapely.geometry import Polygon
import skimage
from labelme.shape import Shape


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

def compute_iou_exact(shape1, shape2):
    
    """
    Summary:
        Computes IOU between two polygons.
    
    Args:
        shape1 (list): List of 2D coordinates(also list) of the first polygon.
        shape2 (list): List of 2D coordinates(also list) of the second polygon.
        
    Returns:
        iou (float): IOU between the two polygons.
    """
    
    shape1 = [tuple(x) for x in shape1]
    shape2 = [tuple(x) for x in shape2]
    polygon1 = Polygon(shape1)
    polygon2 = Polygon(shape2)
    if polygon1.intersects(polygon2) is False:
        return 0
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersection / union if union > 0 else 0
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

def convert_shapes_to_qt_shapes(shapes):
    qt_shapes = []
    for shape in shapes:
        label = shape["label"]
        points = shape["points"]
        bbox = shape["bbox"]
        shape_type = shape["shape_type"]
        # flags = shape["flags"]
        content = shape["content"]
        group_id = shape["group_id"]
        # other_data = shape["other_data"]

        if not points:
            # skip point-empty shape
            continue

        shape = Shape(
            label=label,
            shape_type=shape_type,
            group_id=group_id,
            content=content,
        )
        for i in range(0, len(points), 2):
            shape.addPoint(QtCore.QPointF(points[i], points[i + 1]))
        shape.close()
        qt_shapes.append(shape)
    return qt_shapes

def convert_QT_to_cv(incomingImage):
    
    """
    Summary:
        Convert QT image to cv image MAT format.
        
    Args:
        incomingImage: a QT image
        
    Returns:
        arr: a cv image MAT format
    """
    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_ARGB32)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(incomingImage.sizeInBytes())
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
        rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    return convert_to_Qt_format

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
    
def adjust_shapes_to_original_image(shapes, x1, y1, area_points):
    
    shape1 = [tuple([int(x[0]), int(x[1])]) for x in area_points]
    polygon1 = Polygon(shape1)
    final = []
    
    for shape in shapes:
        shape['points'] = [shape['points'][i] + x1 if i % 2 == 0 else shape['points'][i] + y1 for i in range(len(shape['points']))]
        shape['bbox'] = [shape['bbox'][0] + x1, shape['bbox'][1] + y1, shape['bbox'][2] + x1, shape['bbox'][3] + y1]
        
        points = shape["points"]
        shape2 = [tuple([int(points[z]), int(points[z + 1])])
                for z in range(0, len(points), 2)]
        polygon2 = Polygon(shape2)
        if polygon1.intersects(polygon2):
            final.append(shape)
    
    return final

def track_area_adjustedBboex(area_points, dims, ratio = 0.1):
    
    [x1, y1, x2, y2] = get_bbox_xyxy(area_points)
    [w, h] = [x2 - x1, y2 - y1]
    x1 = int(max(0, x1 - w * ratio))
    y1 = int(max(0, y1 - h * ratio))
    x2 = int(min(dims[1], x2 + w * ratio))
    y2 = int(min(dims[0], y2 + h * ratio))
    
    return [x1, y1, x2, y2]


# def get_contour_length(contour):
#     """Helper function to calculate the length of a contour."""
#     return np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()

def get_contour_length(contour):
    contour_start = contour
    contour_end = np.r_[contour[1:], contour[0:1]]
    return np.linalg.norm(contour_end - contour_start, axis=1).sum()


# def mask_to_polygons(mask, resize_factors=[1.0, 1.0]):
#     contours = skimage.measure.find_contours(mask, level=0)
#     if not contours:
#         return []
#     contour = max(contours, key=get_contour_length)
#     coords = skimage.measure.approximate_polygon(
#         contour, tolerance=np.ptp(contour, axis=0).max() / 100)
#     coords *= resize_factors
#     coords = np.fliplr(coords)
#     return coords.astype(int).tolist()




# def mask_to_polygons(mask, epsilon=0.1, resize_factors=[1.0, 1.0]):
#     """
#     Converts a mask to a list of polygons using Ramer-Douglas-Peucker.

#     Args:
#         mask: A NumPy array representing the binary mask.
#         epsilon: The Douglas-Peucker simplification tolerance (default: 0.1).
#         resize_factors: A tuple of two floats representing the resize factors (default: (1.0, 1.0)).

#     Returns:
#         A list of lists containing the coordinates of each polygon.
#     """
#     start_time = time.time()

#     contours = skimage.measure.find_contours(mask, level=0)
#     if not contours:
#         return []

#     polygons = []
#     for contour in contours:
#         # Simplify the contour using RDP
#         simplified_contour = LinearRing(contour).simplify(epsilon)

#         # Apply resize factors if necessary
#         simplified_contour = simplified_contour * resize_factors

#         # Convert to integer coordinates and append to polygons list
#         simplified_contour = np.fliplr(simplified_contour.astype(int))
#         polygons.append(simplified_contour.tolist())

#     end_time = time.time()
#     execution_time = end_time - start_time
#     print("mask took:",
#           int(execution_time * 1000), "milliseconds")

#     return polygons


def mask_to_polygons(mask, epsilon=3.0):
    contours, _ = cv2.findContours(mask.astype(
        np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("contours", contours)
    approx_polygons = [cv2.approxPolyDP(
        contour, epsilon, True) for contour in contours]
    
    polygon_coords = [approx[:, 0, :].tolist() for approx in approx_polygons]
    return polygon_coords[0]



# def masks_to_polygons(masks, resize_factors=[1.0, 1.0]):
#     polygons = []
#     for mask in masks:
#         polygons.append(mask_to_polygons(mask, resize_factors))
#     return polygons


def polygon_to_shape(polygon, score, className="SAM instance"):
    shape = {}
    shape["label"] = className
    shape["content"] = str(round(score, 2))
    shape["group_id"] = None
    shape["shape_type"] = "polygon"
    shape["bbox"] = get_bbox_xyxy(polygon)

    shape["flags"] = {}
    shape["other_data"] = {}

    # shape_points is result["seg"] flattened
    shape["points"] = [item for sublist in polygon
                            for item in sublist]
    # print(shape)
    return shape

def OURnms_confidenceBased(shapes, iou_threshold=0.5):
    """
    Perform non-maximum suppression on a list of shapes based on their bounding boxes using IOU threshold.

    Args:
        shapes (list): List of shapes, each shape is a dictionary with keys (bbox, confidence, class_id)
        iou_threshold (float): IOU threshold for non-maximum suppression.

    Returns:
        list: List of shapes after performing non-maximum suppression, each shape is a dictionary with keys (bbox, confidence, class_id)
    """
    iou_threshold = float(iou_threshold)
    for shape in shapes:
        if shape['content'] is None:
            shape['content'] = 1.0

    # Sort shapes by their confidence
    shapes.sort(key=lambda x: x['content'], reverse=True)

    boxes, confidences, class_ids, segments = get_boxes_conf_classids_segments(
        shapes)

    toBeRemoved = []

    # Loop through each shape
    for i in range(len(shapes)):
        shape_bbox = boxes[i]
        # Loop through each remaining shape
        for j in range(i + 1, len(shapes)):
            remaining_shape_bbox = boxes[j]

            # Compute IOU between shape and remaining_shape
            iou = compute_iou(shape_bbox, remaining_shape_bbox)

            # If IOU is greater than threshold, remove remaining_shape from shapes list
            if iou > iou_threshold:
                toBeRemoved.append(j)

    shapesFinal = []
    boxesFinal = []
    confidencesFinal = []
    class_idsFinal = []
    segmentsFinal = []
    for i in range(len(shapes)):
        if i in toBeRemoved:
            continue
        shapesFinal.append(shapes[i])
    boxesFinal, confidencesFinal, class_idsFinal, segmentsFinal = get_boxes_conf_classids_segments(
        shapesFinal)

    return shapesFinal, boxesFinal, confidencesFinal, class_idsFinal, segmentsFinal


def OURnms_areaBased_fromSAM(self, sam_result, iou_threshold=0.5):
        
    iou_threshold = float(iou_threshold)

    # Sort shapes by their areas
    sortedResult = sorted(sam_result, key=lambda x: x['area'], reverse=True)
    masks = [ mask['segmentation'] for mask in sortedResult]
    scores = [mask['stability_score'] for mask in sortedResult]
    polygons = [mask_to_polygons(mask) for mask in masks]
    
    toBeRemoved = []

    # Loop through each shape
    if iou_threshold > 0.99:
        for i in range(len(polygons)):
            shape1 = polygons[i]
            # Loop through each remaining shape
            for j in range(i + 1, len(sortedResult)):
                shape2 = polygons[j]
                # Compute IOU between shape and remaining_shape
                iou = compute_iou_exact(shape1, shape2)
                # If IOU is greater than threshold, remove remaining_shape from shapes list
                if iou > iou_threshold:
                    toBeRemoved.append(j)

    shapes = []
    for i in range(len(polygons)):
        if i in toBeRemoved:
            continue
        shapes.append(self.polygon_to_shape(polygons[i], scores[i], f'X{i}'))

    return shapes


