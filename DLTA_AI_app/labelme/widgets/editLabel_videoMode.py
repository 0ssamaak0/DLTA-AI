from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets
from labelme.widgets.MsgBox import OKmsgBox
from labelme.utils.helpers.mathOps import coco_classes
import copy


def editLabel_idChanged_UI(config, old_group_id, new_group_id, id_frames_rec, INDEX_OF_CURRENT_FRAME):
    
    idChanged = old_group_id != new_group_id
    
    if not idChanged:
        result = QtWidgets.QDialog.DialogCode.Accepted
        only_this_frame = False
        duplicates = False
        return result, config, only_this_frame, duplicates
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Choose Edit Options")
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
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
        QtWidgets.QDialogButtonBox.StandardButton.Ok)
    buttonBox.accepted.connect(dialog.accept)
    layout.addWidget(buttonBox)
    dialog.setLayout(layout)
    result = dialog.exec()
    only_this_frame = config['EditDefault'] == 'Edit only this frame'
    duplicates = check_duplicates_editLabel(id_frames_rec, old_group_id, new_group_id, only_this_frame, idChanged, INDEX_OF_CURRENT_FRAME)
    return result, config, only_this_frame, duplicates


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


def editLabel_handle_data(currFrame, listObj,
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

