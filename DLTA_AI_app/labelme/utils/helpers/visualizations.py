import numpy as np
import cv2
from .mathOps import *


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








