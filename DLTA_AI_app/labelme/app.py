# -*- coding: utf-8 -*-
import functools
import json
import orjson
import time
import math
import os
import os.path as osp
import re
import traceback
import webbrowser
import copy

# import asyncio
# import PyQt5
# from qtpy.QtCore import Signal, Slot
import imgviz
from matplotlib import pyplot as plt
from qtpy import QtCore
from qtpy.QtCore import Qt, QThread
# from qtpy.QtCore import Signal as pyqtSignal

from qtpy import QtGui
from qtpy import QtWidgets
import random

# from labelme import __appname__
# from labelme import PY2
# from labelme import QT5

from . import __appname__
from . import PY2
from . import QT5

from . import utils
from .config import get_config
from .label_file import LabelFile
from .label_file import LabelFileError
from .logger import logger
from .shape import Shape
from .widgets import BrightnessContrastDialog, Canvas, LabelDialog, LabelListWidget, LabelListWidgetItem, ToolBar, UniqueLabelQListWidget, ZoomWidget
from .intelligence import Intelligence
from .intelligence import convert_shapes_to_qt_shapes
from .intelligence import coco_classes, color_palette
from .utils.sam import Sam_Predictor
from .utils import helpers

from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.detection.core import Detections
from trackers.multi_tracker_zoo import create_tracker
# from ultralytics.yolo.utils.torch_utils import select_device
# non_max_suppression, scale_boxes, process_mask, process_mask_native
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import select_device

import torch
import numpy as np
import cv2
from pathlib import Path

# import time
# import threading
import warnings
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from .utils.custom_exports import custom_exports_list


warnings.filterwarnings("ignore")


class TrackingThread(QThread):
    def __init__(self, parent=None):
        super(TrackingThread, self).__init__(parent)
        self.parent = parent

    def run(self):
        self.parent.track_buttonClicked()
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# print(f'\n\n ROOT: {ROOT}\n\n')
# now get go up one more level to the root of the repo
ROOT = ROOT.parents[0]
# print(f'\n\n ROOT: {ROOT}\n\n')
# WEIGHTS = ROOT / 'weights'

# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
#     sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# print(f'\n\n ROOT: {ROOT}\n\n')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# reid_weights = Path('osnet_x0_25_msmt17.pt')
reid_weights = Path('osnet_x1_0_msmt17.pt')
# reid_weights = Path('osnet_ms_d_c.pth.tar')


# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window
# TODO(unknown):
# - [high] Add polygon movement with arrow keys
# - [high] Deselect shape when clicking and already selected(?)
# - [low,maybe] Preview images on file dialogs.
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap(value=200)


class MainWindow(QtWidgets.QMainWindow):

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    tracking_progress_bar_signal = pyqtSignal(int)

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        self.buttons_text_style_sheet = "QPushButton {font-size: 10pt; margin: 2px 5px; padding: 2px 7px;font-weight: bold; background-color: #0d69f5; color: #FFFFFF;} QPushButton:hover {background-color: #4990ED;} QPushButton:disabled {background-color: #7A7A7A;}"

        if output is not None:
            logger.warning(
                "argument output is deprecated, use output_file instead"
            )
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config
        self.decodingCanceled = False
        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # update models json
        self.update_saved_models_json()

        super(MainWindow, self).__init__()
        try:
            self.intelligenceHelper = Intelligence(self)
        except:
            print("it seems you have a problem with initializing model\ncheck you have at least one model")
            self.helper_first_time_flag = True
        else:
            self.helper_first_time_flag = False
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.labelList = LabelListWidget()
        self.lastOpenDir = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.loadFlags({k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(
            self.tr("Polygon Labels"), self
        )
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. "
                "Press 'Esc' to deselect."
            )
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        self.label_dock = QtWidgets.QDockWidget(self.tr(u"Label List"), self)
        self.label_dock.setObjectName(u"Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(
            self.fileSelectionChanged
        )
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr(u"File List"), self)
        self.file_dock.setObjectName(u"Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)


        self.vis_dock = QtWidgets.QDockWidget(self.tr(u"Visualization Options"), self)
        self.vis_dock.setObjectName(u"Visualization Options")
        self.vis_widget = QtWidgets.QWidget()
        self.vis_dock.setWidget(self.vis_widget)


        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.addVideoControls()
        self.addSamControls()

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        # Canvas SAM slots
        self.canvas.pointAdded.connect(self.run_sam_model)
        # SAM predictor
        self.sam_predictor = None
        self.current_sam_shape = None
        self.SAM_SHAPES_IN_IMAGE = []
        self.sam_last_mode = "rectangle"

        self.setCentralWidget(scrollArea)

        # for Export
        self.target_directory = ""
        self.save_path = ""
        self.global_listObj = []

        # for merge 
        self.multi_model_flag = False

        # for video annotation
        self.frame_time = 0
        self.FRAMES_TO_SKIP = 30
        self.TRACK_ASSIGNED_OBJECTS_ONLY = False
        self.TrackingMode = False
        self.current_annotation_mode = ""
        self.CURRENT_ANNOATAION_FLAGS = {"traj": False,
                                         "bbox": True,
                                         "id": True,
                                         "class": True,
                                         "mask": True,
                                         "polygons": True,
                                         "conf": True}
        self.CURRENT_ANNOATAION_TRAJECTORIES = {}
        # keep it like that, don't change it
        self.CURRENT_ANNOATAION_TRAJECTORIES['length'] = 30
        # keep it like that, don't change it
        self.CURRENT_ANNOATAION_TRAJECTORIES['alpha'] = 0.70
        self.CURRENT_SHAPES_IN_IMG = []
        self.config = {'deleteDefault': "this frame only",
                       'interpolationDefMethod': "linear",
                       'interpolationDefType': "all",
                       'interpolationOverwrite': False,
                       'creationDefault': "Create new shape (ie. not detected before)",
                       'EditDefault': "Edit only this frame",
                       'toolMode': 'video'}
        self.key_frames = {}
        self.id_frames_rec = {}
        self.copiedShapes = []
        # make CLASS_NAMES_DICT a dictionary of coco class names
        # self.CLASS_NAMES_DICT =
        # self.frame_number = 0
        self.INDEX_OF_CURRENT_FRAME = 1
        # # for image annotation
        # self.last_file_opened = ""
        self.interrupted = False
        self.minID = -2
        self.maxID = 0

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock", "vis_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.vis_dock)


        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open Image"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr(f"Open image or label file ({str(shortcuts['open'])})"),
        )
        opendir = action(
            self.tr("&Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "opendir",
            self.tr(f"Open Dir ({str(shortcuts['open_dir'])})"),
        )
        # tunred off for now as it is not working properly
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr(u"Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        # tunred off for now as it is not working properly
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr(u"Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save"),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr(f"Save labels to file ({str(shortcuts['save'])})"),
            enabled=False,
        )
        export = action(
            self.tr("&Export"),
            self.exportData,
            shortcuts["export"],
            "export",
            self.tr(f"Export annotations to COCO format ({str(shortcuts['export'])})"),
            enabled=False,
        )
        modelExplorer = action(
            self.tr("&Model Explorer"),
            self.model_explorer,
            None,
            "checklist",
            self.tr(u"Model Explorer"),
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr(u"Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text="Save With Image Data",
            slot=self.enableSaveImageWithData,
            tip="Save image data in label file",
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            "&Close",
            self.closeFile,
            shortcuts["close"],
            "close",
            "Close current file",
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep pevious annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        createMode = action(
            self.tr("Create Polygons"),
            self.createMode_options,
            shortcuts["create_polygon"],
            "objects",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        # tunred off for now (removed from menus) as it is not working properly
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        # tunred off for now (removed from menus) as it is not working properly
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "objects",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        # tunred off for now (removed from menus) as it is not working properly
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        # tunred off for now (removed from menus) as it is not working properly
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        # tunred off for now (removed from menus) as it is not working properly
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "objects",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "close",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Duplicate Polygons"),
            self.copySelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        addPointToEdge = action(
            text=self.tr("Add Point to Edge"),
            slot=self.canvas.addPointToEdge,
            shortcut=shortcuts["add_point_to_edge"],
            icon="add_point",
            tip=self.tr("Add point to the nearest edge"),
            enabled=False,
        )
        removePoint = action(
            text="Remove Selected Point",
            slot=self.removeSelectedPoint,
            icon="edit",
            tip="Remove selected point from polygon",
            enabled=False,
        )

        undo = action(
            self.tr("Undo"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            self.tr(
                "Zoom in or out of the image. Also accessible with "
                "{} and {} from the canvas."
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            "&Brightness Contrast",
            self.brightnessContrast,
            None,
            "color",
            "Adjust brightness and contrast",
            enabled=False,
        )
        show_cross_line = action(
            self.tr("&Toggle Cross Line"),
            self.enable_show_cross_line,
            tip=self.tr("cross line for mouse position"),
            icon="cartesian",
            checkable=True,
            checked=self._config["show_cross_line"],
            enabled=True,
        )
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("Edit &Label"),
            self.editLabel,
            shortcuts["edit_label"],
            "label",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )
        enhance = action(
            self.tr("&Enhace Polygons"),
            self.sam_enhance_annotation_button_clicked,
            shortcuts["SAM_enhance"],
            "SAM",
            self.tr("Enhance the selected polygon with AI"),
            enabled=True,
        )
        interpolate = action(
            self.tr("&Interpolation Tracking"),
            self.interpolateMENU,
            shortcuts["interpolate"],
            "tracking",
            self.tr("Interpolate the selected polygon between to frames to Track it"),
            enabled=True,
        )
        mark_as_key = action(
            self.tr("&Mark as key"),
            self.mark_as_key,
            shortcuts["mark_as_key"],
            "mark",
            self.tr("Mark this frame as KEY for interpolation"),
            enabled=True,
        )
        remove_all_keyframes = action(
            self.tr("&Remove all keyframes"),
            self.remove_all_keyframes,
            None,
            "mark",
            self.tr("Remove all keyframes"),
            enabled=True,
        )
        scale = action(
            self.tr("&Scale"),
            self.scaleMENU,
            shortcuts["scale"],
            "resize",
            self.tr("Scale the selected polygon"),
            enabled=True,
        )
        copyShapes = action(
            self.tr("&Copy"),
            self.copyShapesSelected,
            shortcuts["copy"],
            "copy",
            self.tr("Copy selected polygons"),
            enabled=True,
        )
        pasteShapes = action(
            self.tr("&Paste"),
            self.pasteShapesSelected,
            shortcuts["paste"],
            "paste",
            self.tr("paste copied polygons"),
            enabled=True,
        )
        update_curr_frame = action(
            self.tr("&Update current frame"),
            self.update_current_frame_annotation_button_clicked,
            None,
            "done",
            self.tr("Update frame"),
            enabled=True,
        )
        ignore_changes = action(
            self.tr("&Ignore changes"),
            self.main_video_frames_slider_changed,
            shortcuts["ignore_updates"],
            "delete",
            self.tr("Ignore unsaved changes"),
            enabled=True,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        fill_drawing.trigger()
        # intelligence actions
        annotate_one_action = action(
            self.tr("Run Model on Current Image"),
            self.annotate_one,
            None,
            "open",
            self.tr("Run Model on Current Image")
        )

        annotate_batch_action = action(
            self.tr("Run Model on All Images"),
            self.annotate_batch,
            None,
            "file",
            self.tr("Run Model on All Images")
        )
        set_conf_threshold = action(
            self.tr("Confidence Threshold"),
            self.setConfThreshold,
            None,
            "tune",
            self.tr("Confidence Threshold")
        )
        set_iou_threshold = action(
            self.tr("IOU Threshold (NMS)"),
            self.setIOUThreshold,
            None,
            "iou",
            self.tr("IOU Threshold (Non Maximum Suppression)")
        )
        select_classes = action(
            self.tr("Select Classes"),
            self.selectClasses,
            None,
            "checklist",
            self.tr("Select Classes to be Annotated")
        )
        merge_segmentation_models = action(
            self.tr("Merge Segmentation Models"),
            self.mergeSegModels,
            None,
            "merge",
            self.tr("Merge Segmentation Models")
        )
        runtime_data = action(
            self.tr("Show Runtime Data"),
            utils.show_runtime_data,
            None,
            "runtime",
            self.tr("Show Runtime Data")
        )
        git_hub = action(
            self.tr("GitHub Repository"),
            utils.git_hub_link,
            None,
            "github",
            self.tr("GitHub Repository")
        )
        feedback = action(
            self.tr("Feedback"),
            utils.feedback,
            None,
            "feedback",
            self.tr("Feedback")
        )
        license = action(
            self.tr("license"),
            utils.open_license,
            None,
            "license",
            self.tr("license")
        )
        user_guide = action(
            self.tr("User Guide"),
            utils.open_guide,
            None,
            "guide",
            self.tr("User Guide")
        )
        check_updates = action(
            self.tr("Check for Updates"),
            utils.check_updates,
            None,
            "info",
            self.tr("Check for Updates")
        )
        preferences = action(
            self.tr("Preferences"),
            utils.preferences,
            None,
            "settings",
            self.tr("Preferences")
        )
        shortcut_selector = action(
            self.tr("Shortcuts"),
            utils.shortcut_selector,
            None,
            "shortcuts",
            self.tr("Shortcuts")
        )
        sam = action(
            self.tr("Toggle SAM Toolbar"),
            self.Segment_anything,
            None,
            "SAM",
            self.tr("Toggle SAM Toolbar")
        )
        openVideo = action(
            self.tr("Open &Video"),
            self.openVideo,
            shortcuts["open_video"],
            "video",
            self.tr(f"Open a video file ({shortcuts['open_video']})"),
        )
        openVideoFrames = action(
            self.tr("Open Video as Frames"),
            self.openVideoFrames,
            shortcuts["open_video_frames"],
            "frames",
            self.tr(f"Open Video as Frames ({shortcuts['open_video_frames']})"),
        )

        # Lavel list context menu.
        labelmenu = QtWidgets.QMenu()
        utils.addActions(labelmenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu
        )

        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete,
            edit=edit,
            copy=copy,
            undoLastPoint=undoLastPoint,
            undo=undo,
            addPointToEdge=addPointToEdge,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            show_cross_line=show_cross_line,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            export=export,
            openVideo=openVideo,
            openVideoFrames = openVideoFrames,
            fileMenuActions=(open_, opendir, save, saveAs, close, quit),
            modelExplorer=modelExplorer,
            runtime_data=runtime_data,
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                copy,
                delete,
                None,
                undo,
                undoLastPoint,
                None,
                addPointToEdge,
            ),
            # menu shown at right click
            menu=(
                createMode,
                editMode,
                edit,
                enhance,
                interpolate,
                mark_as_key,
                remove_all_keyframes,
                scale,
                copyShapes,
                pasteShapes,
                copy,
                delete,
                undo,
                undoLastPoint,
                addPointToEdge,
                removePoint,
                update_curr_frame,
                ignore_changes
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                editMode,
                brightnessContrast,
            ),
            onShapesPresent=(saveAs, hideAll, showAll),
        )

        self.canvas.edgeSelected.connect(self.canvasShapeEdgeSelected)
        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)
        self.canvas.samFinish.connect(self.sam_finish_annotation_button_clicked)
        self.canvas.APPrefresh.connect(self.refresh_image_MODE)
        # self.canvas.reset.connect(self.sam_reset_button_clicked)
        # self.canvas.interrupted.connect(self.SETinterrupted)

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            intelligence=self.menu(self.tr("&Auto Annotation")),
            model_selection=self.menu(self.tr("&Model Selection")),
            options=self.menu(self.tr("&Options")),
            help=self.menu(self.tr("&Help")),

            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            saved_models=QtWidgets.QMenu(self.tr("Select Segmentation model")),
            tracking_models=QtWidgets.QMenu(self.tr("Select Tracking model")),
            labelList=labelmenu,

            ui_elements=QtWidgets.QMenu(self.tr("&Show UI Elements")),
            zoom_options=QtWidgets.QMenu(self.tr("&Zoom Options")),


        )
        utils.addActions(
            self.menus.file,
            (
                open_,
                opendir,
                openVideo,
                openVideoFrames,
                None,
                save,
                saveAs,
                export,
                None,
                close,
                quit,
            ),
        )
        utils.addActions(self.menus.intelligence,
                         (annotate_one_action,
                          annotate_batch_action,
                          )
                         )
        # View menu and its submenus
        self.menus.ui_elements.setIcon(QtGui.QIcon("labelme/icons/UI.png"))
        utils.addActions(self.menus.ui_elements,
                         (
                             self.vis_dock.toggleViewAction(),
                             self.label_dock.toggleViewAction(),
                             self.shape_dock.toggleViewAction(),
                             self.file_dock.toggleViewAction(),
                         )
                         )
        self.menus.zoom_options.setIcon(QtGui.QIcon("labelme/icons/zoom.png"))
        utils.addActions(self.menus.zoom_options,
                         (
                             zoomIn,
                             zoomOut,
                             zoomOrg,
                             None,
                             fitWindow,
                             fitWidth,
                         )
                         )
        utils.addActions(
            self.menus.view,
            (sam,
                self.menus.ui_elements,
                None,
                hideAll,
                showAll,
                None,
                self.menus.zoom_options,
                None,
                show_cross_line,
             ),
        )

        # Model selection menu
        self.menus.saved_models.setIcon(
            QtGui.QIcon("labelme/icons/brain.png"))
        self.menus.tracking_models.setIcon(
            QtGui.QIcon("labelme/icons/tracking.png"))

        utils.addActions(
            self.menus.model_selection,
            (
                self.menus.saved_models,
                merge_segmentation_models,
                None,
                self.menus.tracking_models,
                None,
                modelExplorer,

            ),
        )

        # Options menu
        utils.addActions(
            self.menus.options,
            (
                set_conf_threshold,
                set_iou_threshold,
                None,
                select_classes,

            ),
        )
        # Help menu
        utils.addActions(
            self.menus.help,
            (
                user_guide,
                preferences,
                shortcut_selector,
                None,
                git_hub,
                feedback,
                None,
                runtime_data,
                None,
                license,
                check_updates

            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)
        self.menus.file.aboutToShow.connect(self.update_models_menu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        self.tools = self.toolbar("Tools")
        # Menu buttons on Left
        self.actions.tool = (
            open_,
            opendir,
            openVideo,
            None,
            save,
            export,
            None,
            createMode,
            editMode,
            edit,
            None,
            delete,
            undo,
            None,

        )
        self.statusBar().showMessage(self.tr("%s started.") % __appname__)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warn(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()

        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        # FIXME: QSettings.value can return None on PyQt4
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(
            self.settings.value("window/state", QtCore.QByteArray())
        )

        # Populate the File menu dynamically.
        self.updateFileMenu()
        self.update_models_menu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()
        self.right_click_menu()
        
        QtWidgets.QShortcut(QtGui.QKeySequence(self._config['shortcuts']['stop']), self).activated.connect(self.Escape_clicked)

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

    def update_saved_models_json(self):
        cwd = os.getcwd()
        helpers.update_saved_models_json(cwd)

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)

            if os.path.isdir(label_file):
                os.remove(label_file)

            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def canvasShapeEdgeSelected(self, selected, shape):
        self.actions.addPointToEdge.setEnabled(
            selected and shape and shape.canAddPoint()
        )

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.CURRENT_FRAME_IMAGE = None
        # self.CURRENT_SHAPES_IN_IMG = []
        # self.SAM_SHAPES_IN_IMAGE = []
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks
    
    def Escape_clicked(self):
        
        """
        Summary:
            This function is called when the user presses the escape key.
            It resets the SAM toolbar and the canvas.
            It also interrupts the current annotation process like (tracking, interpolation, etc.)
        """
        
        self.interrupted = True
        self.sam_reset_button_clicked()

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def tutorial(self):
        url = "https://github.com/wkentaro/labelme/tree/master/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.
        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            self.actions.createMode.setEnabled(True)
            self.actions.createRectangleMode.setEnabled(True)
            self.actions.createCircleMode.setEnabled(True)
            self.actions.createLineMode.setEnabled(True)
            self.actions.createPointMode.setEnabled(True)
            self.actions.createLineStripMode.setEnabled(True)
        else:
            if createMode == "polygon":
                self.actions.createMode.setEnabled(False)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == "rectangle":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(False)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == "line":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(False)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == "point":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(False)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == "circle":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(False)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == "linestrip":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(False)
            else:
                raise ValueError("Unsupported createMode: %s" % createMode)
        self.actions.editMode.setEnabled(not edit)

    def turnOFF_SAM(self):
        if self.sam_model_comboBox.currentText() != "Select Model (SAM disabled)":
            self.sam_clear_annotation_button_clicked()
        self.sam_buttons_colors('x')
        self.set_sam_toolbar_enable(False)
        self.canvas.SAM_mode = ""
        self.canvas.SAM_coordinates = []
        self.canvas.SAM_rect = []
        self.canvas.SAM_rects = []
        self.canvas.SAM_current = None
        
    def turnON_SAM(self):
        if self.sam_model_comboBox.currentText() == "Select Model (SAM disabled)":
            return
        self.sam_buttons_colors("X")
        self.set_sam_toolbar_enable(True)
        self.canvas.SAM_mode = ""
        self.canvas.SAM_coordinates = []
        self.canvas.SAM_rect = []
        self.canvas.SAM_rects = []
        self.canvas.SAM_current = None

    def setEditMode(self):
        
        self.turnOFF_SAM()
        
        try:
            x = self.CURRENT_VIDEO_PATH
        except:
            self.toggleDrawMode(True)
            return
        self.update_current_frame_annotation()
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("brain")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def update_models_menu(self):

        menu = self.menus.saved_models
        menu.clear()

        with open("saved_models.json") as json_file:
            data = json.load(json_file)
            # loop through all the models
            i = 0
            for model_name in list(data.keys()):
                if i >= 6:
                    break
                icon = utils.newIcon("brain")
                action = QtWidgets.QAction(
                    icon, "&%d %s" % (i + 1, model_name), self)
                action.triggered.connect(functools.partial(
                    self.change_curr_model, model_name))
                menu.addAction(action)
                i += 1
        self.add_tracking_models_menu()

    def add_tracking_models_menu(self):
        menu2 = self.menus.tracking_models
        menu2.clear()

        icon = utils.newIcon("tracking")
        action = QtWidgets.QAction(
            icon, "1 Byte track (DEFAULT)", self)
        action.triggered.connect(
            lambda: self.update_tracking_method('bytetrack'))
        menu2.addAction(action)

        icon = utils.newIcon("tracking")
        action = QtWidgets.QAction(
            icon, "2 Strong SORT  (lowest id switch)", self)
        action.triggered.connect(
            lambda: self.update_tracking_method('strongsort'))
        menu2.addAction(action)

        icon = utils.newIcon("tracking")
        action = QtWidgets.QAction(
            icon, "3 Deep SORT", self)
        action.triggered.connect(
            lambda: self.update_tracking_method('deepocsort'))
        menu2.addAction(action)

        icon = utils.newIcon("tracking")
        action = QtWidgets.QAction(
            icon, "4 OC SORT", self)
        action.triggered.connect(lambda: self.update_tracking_method('ocsort'))
        menu2.addAction(action)

        icon = utils.newIcon("tracking")
        action = QtWidgets.QAction(
            icon, "5 BoT SORT", self)
        action.triggered.connect(
            lambda: self.update_tracking_method('botsort'))
        menu2.addAction(action)

    def update_tracking_method(self, method='bytetrack'):
        self.waitWindow(
            visible=True, text=f'Please Wait.\n{method} is Loading...')
        self.tracking_method = method
        self.tracking_config = ROOT / 'trackers' / \
            method / 'configs' / (method + '.yaml')
        with torch.no_grad():
            device = select_device('')
            print(
                f'tracking method {self.tracking_method} , config {self.tracking_config} , reid {reid_weights} , device {device} , half {False}')
            self.tracker = create_tracker(
                self.tracking_method, self.tracking_config, reid_weights, device, False)
            if hasattr(self.tracker, 'model'):
                if hasattr(self.tracker.model, 'warmup'):
                    self.tracker.model.warmup()
                    # print('Warmup done')
        self.waitWindow()

        print(f'Changed tracking method to {method}')

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def createMode_options(self):

        self.turnON_SAM()
        self.toggleDrawMode(False, createMode="polygon")
        return

    def editLabel(self, item=None):
        if self.current_annotation_mode == 'video':
            self.update_current_frame_annotation()
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")
        if not self.canvas.editing():
            return
        if not item:
            item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        old_text, old_flags, old_group_id, old_content = self.labelDialog.popUp(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
            content=shape.content,
            skip_flag=True
        )
        text, flags, new_group_id, content = self.labelDialog.popUp(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
            content=shape.content
        )

        if text is None:
            return
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return
        shape.label = text
        shape.flags = flags
        shape.group_id = new_group_id
        shape.content = str(content)
        if self.current_annotation_mode == 'img' or self.current_annotation_mode == 'dir':
            # item.setText(f' ID {shape.group_id}: {shape.label}')
            item.setText(f'{shape.label}')
            self.setDirty()
            if not self.uniqLabelList.findItemsByLabel(shape.label):
                item = QtWidgets.QListWidgetItem()
                item.setData(Qt.UserRole, shape.label)
                self.uniqLabelList.addItem(item)
            self.refresh_image_MODE()
            return
        if shape.group_id is None:
            item.setText(shape.label)
        else:
            idChanged = old_group_id != new_group_id
            only_this_frame = False
            if idChanged:
                result, self.config = helpers.editLabel_idChanged_GUI(
                    self.config)
                if result == QtWidgets.QDialog.Accepted:
                    only_this_frame = True if self.config['EditDefault'] == 'Edit only this frame' else False
                else:
                    return

            listObj = self.load_objects_from_json__orjson()

            if helpers.check_duplicates_editLabel(self.id_frames_rec, old_group_id, new_group_id, only_this_frame, idChanged, self.INDEX_OF_CURRENT_FRAME):
                shape.label = old_text
                shape.flags = old_flags
                shape.content = old_content
                shape.group_id = old_group_id
                return
            
            self.minID = min(self.minID, new_group_id - 1)

            self.id_frames_rec, self.CURRENT_ANNOATAION_TRAJECTORIES, listObj = helpers.handle_id_editLabel(
                                        currFrame = self.INDEX_OF_CURRENT_FRAME,
                                        listObj = listObj,
                                        trajectories = self.CURRENT_ANNOATAION_TRAJECTORIES,
                                        id_frames_rec = self.id_frames_rec,
                                        idChanged = idChanged,
                                        only_this_frame = only_this_frame,
                                        shape = shape,
                                        old_group_id = old_group_id,
                                        new_group_id = new_group_id,)
            
            self.load_objects_to_json__orjson(listObj)
            self.main_video_frames_slider_changed()

    def interpolateMENU(self, item=None):
        try:
            if len(self.canvas.selectedShapes) == 0:
                mb = QtWidgets.QMessageBox
                msg = self.tr("Interpolate all IDs?\n")
                answer = mb.warning(self, self.tr(
                    "Attention"), msg, mb.Yes | mb.No)
                if answer != mb.Yes:
                    return
                else:
                    self.update_current_frame_annotation()
                    keys = list(self.id_frames_rec.keys())
                    idsORG = [int(keys[i][3:]) for i in range(len(keys))]
            else:
                self.update_current_frame_annotation()
                idsORG = [shape.group_id for shape in self.canvas.selectedShapes]
                id = self.canvas.selectedShapes[0].group_id

            result, self.config = helpers.interpolationOptions_GUI(self.config)
            if result != QtWidgets.QDialog.Accepted:
                return
            
            with_linear = True if self.config['interpolationDefMethod'] == 'linear' else False
            with_sam = True if self.config['interpolationDefMethod'] == 'SAM' else False
            with_keyframes = True if self.config['interpolationDefType'] == 'key' else False
            
            
            if with_keyframes:
                allAccepted, allRejected, ids = helpers.checkKeyFrames(idsORG, self.key_frames)
                if not allAccepted:
                    if allRejected:
                        helpers.OKmsgBox("Key Frames Error",
                                        f"All of the selected IDs have no KEY frames.\n    ie. less than 2 key frames\n The interpolation is NOT performed.")
                        return
                    else:
                        resutl = helpers.OKmsgBox("Key Frames Error", 
                                        f"Some of the selected IDs have no KEY frames.\n    ie. less than 2 key frames\n The interpolation is performed only for the IDs with KEY frames.\nIDs: {ids}.", "info", turnResult=True)
                        if resutl != QtWidgets.QMessageBox.Ok:
                            return
            else:
                ids = idsORG

            self.interrupted = False
            if with_sam:
                self.interpolate_with_sam(ids, with_keyframes)
            else:
                for id in ids:
                    QtWidgets.QApplication.processEvents()
                    if self.interrupted:
                        self.interrupted = False
                        break
                    self.interpolate(id=id,
                                    only_edited=with_keyframes)
            self.waitWindow()
        except Exception as e:
            helpers.OKmsgBox("Error", f"Error: {e}", "critical")

    def mark_as_key(self):
        
        """
        Summary:
            This function is called when the user presses the "Mark as Key" button.
            It marks the selected shape as a key frame.
        """
        try:
            self.update_current_frame_annotation()
            id = self.canvas.selectedShapes[0].group_id
            try:
                if self.INDEX_OF_CURRENT_FRAME not in self.key_frames['id_' + str(id)]:
                    self.key_frames['id_' +
                                    str(id)].add(self.INDEX_OF_CURRENT_FRAME)
                else:
                    res = helpers.OKmsgBox("Caution", f"Frame {self.INDEX_OF_CURRENT_FRAME} is already a key frame for ID {id}.\nDo you want to remove it?", "warning", turnResult=True)
                    if res == QtWidgets.QMessageBox.Ok:
                        self.key_frames['id_' +
                                    str(id)].remove(self.INDEX_OF_CURRENT_FRAME)
                    else:
                        return
            except:
                self.key_frames['id_' +
                                str(id)] = set()
                self.key_frames['id_' +
                                str(id)].add(self.INDEX_OF_CURRENT_FRAME)
            self.main_video_frames_slider_changed()
        except Exception as e:
            helpers.OKmsgBox("Error", f"Error: {e}", "critical")

    def remove_all_keyframes(self):
        try:
            self.update_current_frame_annotation()
            id = self.canvas.selectedShapes[0].group_id
            self.key_frames['id_' + str(id)] = set()
        except:
            pass
        
    def rec_frame_for_id(self, id, frame, type_='add'):
        
        """
        Summary:
            To store the frames in which the object with the given id is present.
            
        Args:
            id (int): The id of the object.
            frame (int): The frame number.
            type_ (str, optional): 'add' or 'remove'. Defaults to 'add'.
                                    'add' to add the frame to the list of frames in which the object is present.
                                    'remove' to remove the frame from the list of frames in which the object is present.
                                    
        Returns:
            None
        """

        if type_ == 'add':
            try:
                self.id_frames_rec['id_' + str(id)].add(frame)
            except:
                self.id_frames_rec['id_' + str(id)] = set()
                self.id_frames_rec['id_' + str(id)].add(frame)
        else:
            try:
                self.id_frames_rec['id_' + str(id)].remove(frame)
            except:
                pass
        
    def interpolate(self, id, only_edited=False):
        
        """
        Summary:
            This function is called when the user presses the "Interpolate" button.
            It interpolates the object with the given id.
            
        Args:
            id (int): The id of the object.
            only_edited (bool, optional): True to interpolate using only the key frames. Defaults to False.
        """

        self.waitWindow(
            visible=True, text=f'Please Wait.\nID {id} is being interpolated...')

        listObj = self.load_objects_from_json__orjson()

        if only_edited:
            try:
                FRAMES = list(self.key_frames['id_' + str(id)])
            except:
                return
        else:
            FRAMES = list(self.id_frames_rec['id_' + str(id)]) if len(self.id_frames_rec['id_' + str(id)]) > 1 else [-1]
            
        first_frame_idx = min(FRAMES)
        last_frame_idx = max(FRAMES)

        if (first_frame_idx >= last_frame_idx):
            return

        records = [None for i in range(first_frame_idx - 1, last_frame_idx, 1)]
        for frame in range(first_frame_idx, last_frame_idx + 1, 1):
            listobjframe = listObj[frame - 1]['frame_idx']
            frameobjects = listObj[frame - 1]['frame_data']
            for object_ in frameobjects:
                if (object_['tracker_id'] == id):
                    if ((not only_edited) or (listobjframe in FRAMES)):
                        records[frame - first_frame_idx] = copy.deepcopy(object_)
                    break
        
        baseObject = None
        baseObjectFrame = None
        nextObject = None
        nextObjectFrame = None
            
        for frame in range(first_frame_idx, last_frame_idx, 1):
            
            QtWidgets.QApplication.processEvents()
            if self.interrupted:
                break
            
            listobjframe = listObj[frame - 1]['frame_idx']
            frameobjects = listObj[frame - 1]['frame_data']
            
            # if object is present in this frame, then it is base object and we calculate next object
            if(records[frame -first_frame_idx] is not None):
                
                # assign it as base object
                baseObject = copy.deepcopy(records[frame -first_frame_idx])
                baseObjectFrame = frame
                
                # find next object
                for j in range(frame + 1, last_frame_idx + 1, 1):
                    if (records[j - first_frame_idx] != None):
                        nextObject = copy.deepcopy(records[j - first_frame_idx])
                        nextObjectFrame = j
                        break
                
                # job done, go to next frame
                continue
            
            # if only_edited is true and the frame is not key, then we remove the object from the frame to be interpolated
            if (only_edited and (frame not in FRAMES)):
                for object_ in frameobjects:
                    if (object_['tracker_id'] == id):
                        listObj[frame - 1]['frame_data'].remove(object_)
                        break
            
            # if object is not present in this frame, then we calculate the object for this frame
            cur = helpers.getInterpolated(baseObject=baseObject, 
                                          baseObjectFrame=baseObjectFrame,
                                          nextObject=nextObject,
                                          nextObjectFrame=nextObjectFrame,
                                          curFrame=frame,)
            listObj[frame - 1]['frame_data'].append(cur)
            self.rec_frame_for_id(id, frame)
            
        
        self.load_objects_to_json__orjson(listObj)
        frames = range(first_frame_idx - 1, last_frame_idx, 1)
        self.calculate_trajectories(frames)
        self.main_video_frames_slider_changed()
        
    def interpolate_with_sam(self, idsLISTX, only_edited=False):
        
        """
        Summary:
            This function is called when the user chooses the "Interpolate with SAM".
            It interpolates and inhance the objects with the given ids using SAM.
            
        Args:
            idsLISTX (list): The list of ids of the objects.
        """

        self.waitWindow(
            visible=True, text=f'Please Wait.\nIDs are being interpolated with SAM...')

        if self.sam_model_comboBox.currentText() == "Select Model (SAM disabled)":
            helpers.OKmsgBox("SAM is disabled",
                             f"SAM is disabled.\nPlease enable SAM.")
            return

        
        idsLIST = []
        first_frame_idxLIST = []
        last_frame_idxLIST = []
        oneFrame_flag_idxLIST = []
        for id in idsLISTX:
            try:
                if only_edited:
                    [minf, maxf] = [min(
                    self.key_frames['id_' + str(id)]), max(self.key_frames['id_' + str(id)])]
                else:
                    [minf, maxf] = [min(
                    self.id_frames_rec['id_' + str(id)]), max(self.id_frames_rec['id_' + str(id)])]
            except:
                continue
            if minf == maxf:
                maxf = self.TOTAL_VIDEO_FRAMES
                oneFrame_flag_idxLIST.append(1)
            else:
                oneFrame_flag_idxLIST.append(0)
            first_frame_idxLIST.append(minf)
            last_frame_idxLIST.append(maxf)
            idsLIST.append(id)

        if len(idsLIST) == 0:
            return
        
        overwrite = self.config['interpolationOverwrite']

        listObj = self.load_objects_from_json__orjson()
        listObjNEW = copy.deepcopy(listObj)

        recordsLIST = [[None for ii in range(
            first_frame_idxLIST[i], last_frame_idxLIST[i] + 1)] for i in range(len(idsLIST))]
        
        for i in range(min(first_frame_idxLIST) - 1, max(last_frame_idxLIST), 1):
            self.waitWindow(visible=True)
            listobjframe = listObj[i]['frame_idx']
            frameobjects = listObj[i]['frame_data'].copy()
            for object_ in frameobjects:
                if (object_['tracker_id'] in idsLIST):
                    index = idsLIST.index(object_['tracker_id'])
                    recordsLIST[index][listobjframe -
                                       first_frame_idxLIST[index]] = copy.deepcopy(object_)
                    listObj[i]['frame_data'].remove(object_)

        for frameIDX in range(min(first_frame_idxLIST), max(last_frame_idxLIST) + 1):
            QtWidgets.QApplication.processEvents()
            if self.interrupted:
                self.interrupted = False
                break
            self.waitWindow(
                visible=True, text=f'Please Wait.\nIDs are being interpolated with SAM...\nFrame {frameIDX}')

            frameIMAGE = self.get_frame_by_idx(frameIDX)

            for ididx in range(len(idsLIST)):
                i = frameIDX - first_frame_idxLIST[ididx]
                self.waitWindow(visible=True)
                if frameIDX < first_frame_idxLIST[ididx] or frameIDX > last_frame_idxLIST[ididx]:
                    continue

                records = recordsLIST[ididx]
                if (records[i] != None):
                    current = copy.deepcopy(records[i])
                    cur_bbox = current['bbox']
                    if not overwrite:
                        listObj[frameIDX - 1]['frame_data'].append(current)
                        continue
                else:
                    prev_idx = i - 1
                    current = copy.deepcopy(records[i - 1])
                    
                    if oneFrame_flag_idxLIST[ididx] == 0:
                        next_idx = i + 1
                        for j in range(i + 1, len(records)):
                            self.waitWindow(visible=True)
                            if (records[j] != None):
                                next_idx = j
                                break
                        cur_bbox = ((next_idx - i) / (next_idx - prev_idx)) * np.array(records[prev_idx]['bbox']) + (
                            (i - prev_idx) / (next_idx - prev_idx)) * np.array(records[next_idx]['bbox'])
                        cur_bbox = [int(cur_bbox[i]) for i in range(len(cur_bbox))]
                        current['bbox'] = copy.deepcopy(cur_bbox)
                    
                    records[i] = current

                try:
                    same_image = self.sam_predictor.check_image(
                        frameIMAGE)
                except:
                    return
                
                cur_bbox, cur_segment = self.sam_enhanced_bbox_segment(
                    frameIMAGE, cur_bbox, 1.2, max_itr=5, forSHAPE=False)

                current['bbox'] = copy.deepcopy(cur_bbox)
                current['segment'] = copy.deepcopy(cur_segment)
                
                # append the shape frame by frame (cause we already removed it in the prev. for loop)
                listObj[frameIDX - 1]['frame_data'].append(current)
                self.rec_frame_for_id(idsLIST[ididx], frameIDX)
                
            # update frame by frame to the to-be-uploaded listObj
            listObjNEW[frameIDX - 1] = copy.deepcopy(listObj[frameIDX - 1])
        
        self.load_objects_to_json__orjson(listObjNEW)
        self.calculate_trajectories(range(min(first_frame_idxLIST) - 1, max(last_frame_idxLIST), 1))
        self.main_video_frames_slider_changed()

        # Notify the user that the interpolation is finished
        self._config = get_config()
        if not self._config["mute"]:
            if not self.isActiveWindow():
                helpers.notification("SAM Interpolation Completed")

    def get_frame_by_idx(self, frameIDX):
        self.CAP.set(cv2.CAP_PROP_POS_FRAMES, frameIDX - 1)
        success, img = self.CAP.read()
        return img

    def sam_enhanced_bbox_segment(self, frameIMAGE, cur_bbox, thresh, max_itr=5, forSHAPE=False):
        oldAREA = abs(cur_bbox[2] - cur_bbox[0]) * \
            abs(cur_bbox[3] - cur_bbox[1])
        [x1, y1, x2, y2] = [cur_bbox[0], cur_bbox[1],
                            cur_bbox[2], cur_bbox[3]]
        listPOINTS = [min(x1, x2), min(y1, y2),
                      max(x1, x2), max(y1, y2)]
        listPOINTS = [int(round(x)) for x in listPOINTS]
        input_boxes = [listPOINTS]
        mask, score = self.sam_predictor.predict(point_coords=None,
                                                 point_labels=None,
                                                 box=input_boxes,
                                                 image=frameIMAGE)
        points = self.sam_predictor.mask_to_polygons(mask)
        SAMshape = self.sam_predictor.polygon_to_shape(points, score)
        cur_segment = SAMshape['points']
        cur_segment = [[int(cur_segment[i]), int(cur_segment[i + 1])]
                       for i in range(0, len(cur_segment), 2)]
        cur_bbox = [min(np.array(cur_segment)[:, 0]), min(np.array(cur_segment)[:, 1]),
                    max(np.array(cur_segment)[:, 0]), max(np.array(cur_segment)[:, 1])]
        cur_bbox = [int(round(x)) for x in cur_bbox]
        newAREA = abs(cur_bbox[2] - cur_bbox[0]) * \
            abs(cur_bbox[3] - cur_bbox[1])
        bigger, smaller = max(oldAREA, newAREA), min(oldAREA, newAREA)
        # print(f'oldAREA: {oldAREA}, newAREA: {newAREA}, bigger/smaller: {bigger/smaller}')
        if bigger/smaller < thresh or max_itr == 1:
            if forSHAPE:
                return cur_bbox, SAMshape['points']
            else:
                return cur_bbox, cur_segment
        else:
            return self.sam_enhanced_bbox_segment(frameIMAGE, cur_bbox, thresh, max_itr-1, forSHAPE)

    def scaleMENU(self):
        
        """
        Summary:
            This function is called when the user presses the "Scale" button.
            It scales the selected shape.
        """
        
        if len(self.canvas.selectedShapes) != 1:
            helpers.OKmsgBox(f'Scale error',
                             f'There is {len(self.canvas.selectedShapes)} selected shapes. Please select only one shape to scale.')
            return
            
        result = helpers.scaleMENU_GUI(self)
        if result == QtWidgets.QDialog.Accepted:
            self.update_current_frame_annotation_button_clicked()
            return
        else:
            self.main_video_frames_slider_changed()
            return

    def copyShapesSelected(self):
        
        """
        Summary:
            This function is called when the user presses the "Copy" button.
            It copies the selected shape(s).
        """
        
        if len(self.canvas.selectedShapes) == 0:
            return
        self.copiedShapes = self.canvas.selectedShapes

    def pasteShapesSelected(self):
        
        """
        Summary:
            This function is called when the user presses the "Paste" button.
            It pastes the copied shape(s).
        """

        if len(self.copiedShapes) == 0:
            return

        ids = [shape.group_id for shape in self.canvas.shapes]
        flag = False

        for shape in self.copiedShapes:
            if shape.group_id in ids:
                flag = True
                continue
            self.canvas.shapes.append(shape)
            self.addLabel(shape)
            self.rec_frame_for_id(shape.group_id, self.INDEX_OF_CURRENT_FRAME)

        if flag:
            helpers.OKmsgBox("IDs already exist",
                             "A Shape(s) with the same ID(s) already exist(s) in this frame.\n\nShapes with no duplicate IDs are Copied Successfully.")

        if self.current_annotation_mode == "video":
            self.update_current_frame_annotation_button_clicked()

    def get_bbox_xyxy(self, segment):
        return helpers.get_bbox_xyxy(segment)

    def addPoints(self, shape, n):
        return helpers.addPoints(shape, n)

    def reducePoints(self, polygon, n):
        return helpers.reducePoints(polygon, n)

    def handlePoints(self, polygon, n):
        return helpers.handlePoints(polygon, n)

    def allign(self, shape1, shape2):
        return helpers.allign(shape1, shape2)

    def centerOFmass(self, points):
        return helpers.centerOFmass(points)

    def is_id_repeated(self, group_id, frameIdex=-1):
        return helpers.is_id_repeated(self, group_id, frameIdex)

    def get_id_from_user(self, group_id, text):
        return helpers.getIDfromUser_GUI(self, group_id, text)

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)
                self.refresh_image_MODE()

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        try:
            self._noSelectionSlot = True
            for shape in self.canvas.selectedShapes:
                shape.selected = False
            self.labelList.clearSelection()
            self.canvas.selectedShapes = selected_shapes
            for shape in self.canvas.selectedShapes:
                shape.selected = True
                item = self.labelList.findItemByShape(shape)
                self.labelList.selectItem(item)
                self.labelList.scrollToItem(item)
            self._noSelectionSlot = False
            n_selected = len(selected_shapes)
            self.actions.delete.setEnabled(n_selected)
            self.actions.copy.setEnabled(n_selected)
            self.actions.edit.setEnabled(n_selected == 1)
        except Exception as e:
            pass

    def addLabel(self, shape):
        if shape.group_id is None or self.current_annotation_mode != "video":
            text = shape.label
        else:
            text = f' ID {shape.group_id}: {shape.label}'
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if not self.uniqLabelList.findItemsByLabel(shape.label):
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        rgb = self._get_rgb_by_label(shape.label)

        r, g, b = rgb
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                text, r, g, b
            )
        )
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label):
        if self._config["shape_color"] == "auto":
            # item = self.uniqLabelList.findItemsByLabel(label)[0]
            # label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            # label_id += self._config["shift_auto_shape_color"]
            # return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
            # idx = coco_classes.index(label) if label in coco_classes else -1
            # idx = idx % len(color_palette)
            # color = color_palette[idx] if idx != -1 else (0, 0, 255)
            # label_hash = hash(label)
            # idx = abs(label_hash) % len(color_palette)
            label_ascii = sum([ord(c) for c in label])
            idx = label_ascii % len(color_palette)
            color = color_palette[idx]
            # convert color from bgr to rgb
            return color[::-1]

        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        # sort shapes by group_id but only if its not None
        shapes = sorted(shapes, key=lambda x: int(x.group_id)
                        if x.group_id is not None else 0)
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)
        for shape in self.canvas.shapes:
            self.canvas.setShapeVisible(
                shape, self.CURRENT_ANNOATAION_FLAGS["polygons"])

    def loadLabels(self, shapes, replace=True):
        s = []
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

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            # shape.flags.update(flags)
            # shape.other_data = other_data

            s.append(shape)
        self.loadShapes(s, replace=replace)

    def loadFlags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

    def flattener(self, list_2d):
        return helpers.flattener(list_2d)

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    # convert points into 1D array
                    points=self.flattener(s.points),
                    bbox=s.bbox,
                    group_id=s.group_id,
                    content=s.content,
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(
                self.imagePath, Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def copySelectedShape(self):
        added_shapes = self.canvas.copySelectedShapes()
        self.labelList.clearSelection()
        for shape in added_shapes:
            self.addLabel(shape)
        self.setDirty()

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.
        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        if self._config["display_label_popup"] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id, content = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)
        print(
            f'---------- after POPUP ----------- group_id: {group_id}, text: {text}')
        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""

        if text == "SAM instance":
            text = "SAM instance - confirmed"

        if self.current_annotation_mode == "video":
            group_id, text = self.get_id_from_user(group_id, text)
        
        if text:
            
            if group_id is None:
                group_id = self.minID
                self.minID -= 1
            else:
                self.minID = min(self.minID, group_id - 1)
            
            if self.canvas.SAM_mode == "finished":
                self.current_sam_shape["label"] = text
                self.current_sam_shape["group_id"] = group_id
            else:
                self.labelList.clearSelection()
                # shape below is of type qt shape
                shape = self.canvas.setLastLabel(text, flags)
                shape.group_id = group_id
                shape.content = content
                self.addLabel(shape)
                self.rec_frame_for_id(group_id, self.INDEX_OF_CURRENT_FRAME)
                
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
            
            self.refresh_image_MODE()
            
        else:
            if self.canvas.SAM_mode == "finished":
                # print(f'---------- SAM_mode is finished ----------- group_id: {group_id}, text: {text}')
                self.current_sam_shape["label"] = text
                self.current_sam_shape["group_id"] = -1
                self.canvas.SAM_mode = ""
            else:
                self.canvas.undoLastLine()
                self.canvas.shapesBackups.pop()

        if self.current_annotation_mode == "video":
            self.update_current_frame_annotation_button_clicked()
            self.update_current_frame_annotation_button_clicked()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(value)
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(
            QtGui.QPixmap.fromImage(qimage), clear_shapes=False
        )

    def enable_show_cross_line(self, enabled):
        self._config["show_cross_line"] = enabled
        self.actions.show_cross_line.setChecked(enabled)
        self.canvas.set_show_cross_line(enabled)

    def brightnessContrast(self, value):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightnessContrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        for item in self.labelList:
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.status(self.tr("Loading %s...") % osp.basename(str(filename)))
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
            label_file
        ):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.imageData = self.labelFile.imageData
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            self.otherData = self.labelFile.otherData
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.CURRENT_FRAME_IMAGE = cv2.imread(filename)
        self.filename = filename
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes

        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self.actions.export.setEnabled(True)
            self.CURRENT_SHAPES_IN_IMG = self.labelFile.shapes
            # print("self.CURRENT_SHAPES_IN_IMG", self.CURRENT_SHAPES_IN_IMG)
            # image = self.draw_bb_on_image(image  , self.CURRENT_SHAPES_IN_IMG)
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
            self.loadLabels(self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        self.loadFlags(flags)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()

        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # after loading the image, clear SAM instance if exists
        if self.sam_predictor is not None:
            self.sam_predictor.clear_logit()
            self.canvas.SAM_coordinates = []
        # set brightness constrast values
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recentFiles:
            brightness, _ = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recentFiles:
            _, contrast = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightnessContrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            dialog.onNewValue(None)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        self.status(self.tr("Loaded %s") % osp.basename(str(filename)))
        return True

    def resizeEvent(self, event):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        else:
            self.Escape_clicked()
        self.settings.setValue(
            "filename", self.filename if self.filename else ""
        )
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, event):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def change_curr_model(self, model_name):
        
        """
        Summary:
            Change current model to the model_name
            
        Args:
            model_name (str): name of the model to be changed to
        """
        
        self.multi_model_flag = False
        self.waitWindow(
            visible=True, text=f'Please Wait.\n{model_name} is being Loaded...')
        self.intelligenceHelper.current_model_name, self.intelligenceHelper.current_mm_model = self.intelligenceHelper.make_mm_model(
            model_name)
        self.waitWindow()

    def model_explorer(self):
        
        """
        Summary:
            Open model explorer dialog to select or download models
        """
        self._config = get_config()
        model_explorer_dialog = utils.ModelExplorerDialog(self, self._config["mute"], helpers.notification)
        # make it fit its contents
        model_explorer_dialog.adjustSize()
        model_explorer_dialog.setMinimumWidth(model_explorer_dialog.table.width() * 1.5)
        model_explorer_dialog.setMinimumHeight(model_explorer_dialog.table.rowHeight(0) * 10)
        model_explorer_dialog.exec_()
        # init intelligence again if it's the first model
        if self.helper_first_time_flag:
            try:
                self.intelligenceHelper = Intelligence(self)
            except:
                print("it seems you have a problem with initializing model\ncheck you have at least one model")
                self.helper_first_time_flag = True
            else:
                self.helper_first_time_flag = False
        self.update_saved_models_json()

        selected_model_name, config, checkpoint = model_explorer_dialog.selected_model
        if selected_model_name != -1:
            self.intelligenceHelper.current_model_name, self.intelligenceHelper.current_mm_model = self.intelligenceHelper.make_mm_model_more(
                selected_model_name, config, checkpoint)
        self.updateSamControls()

    # tunred off for now (removed from menus) as it is not working properly
    def openPrevImg(self, _value=False):
        self.refresh_image_MODE()
        keep_prev = self._config["keep_prev"]
        if Qt.KeyboardModifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev

    # tunred off for now as it is not working properly
    def openNextImg(self, _value=False, load=True):
        self.refresh_image_MODE()
        keep_prev = self._config["keep_prev"]
        if Qt.KeyboardModifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev
        self.refresh_image_MODE()

    def openFile(self, _value=False):
        # self.current_annotation_mode = 'img'
        # self.disconnectVideoShortcuts()

        self.actions.export.setEnabled(False)
        try:
            cv2.destroyWindow('video processing')
        except:
            pass
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("%s - Choose Image or Label file") % __appname__,
            path,
            filters,
        )
        if QT5:
            filename, _ = filename
        filename = str(filename)
        if filename:
            self.reset_for_new_mode("img")
            self.loadFile(filename)
            self.refresh_image_MODE()
            self.set_video_controls_visibility(False)
            # self.right_click_menu()

        self.filename = filename
        # clear the file list widget
        self.fileListWidget.clear()
        self.uniqLabelList.clear()
        # enable Visualization Options
        for option in self.vis_options:
            if option in [self.id_checkBox, self.traj_checkBox, self.trajectory_length_lineEdit]:
                option.setEnabled(False)
            else:
                option.setEnabled(True)
        
    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(
                self.imageList.index(current_filename)
            )
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        self.actions.export.setEnabled(True)
        assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            self.save_path = self.labelFile.filename
            self._saveFile(self.save_path)
        elif self.output_file:
            self.save_path = self.output_file
            self._saveFile(self.save_path)
            self.close()
        else:
            self.save_path = self.saveFileDialog()
            self._saveFile(self.save_path)

    def exportData(self):
        """
        Export data to COCO, MOT, video, and custom exports, depending on the current annotation mode.

        If the current annotation mode is "video", the function prompts the user to select which types of exports to perform
        (COCO, MOT, video, and/or custom exports), and then prompts the user to select the output file path for each export type
        that was selected. The function then exports the data to the selected file paths.

        If the current annotation mode is "img" or "dir", the function prompts the user to select the output file path for a COCO
        export, and then exports the data to the selected file path.

        If an error occurs during the export process, the function displays an error message. Otherwise, the function displays
        a success message.
        """
        try:
            if self.current_annotation_mode == "video":
                # Get user input for export options
                result, coco_radio, mot_radio, video_radio, custom_exports_radio_checked_list = helpers.exportData_GUI()
                if not result:
                    return

                json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'

                pth = ""
                # Check which radio button is checked and export accordingly
                if video_radio:
                    # Get user input for video export path
                    folderDialog = utils.FolderDialog("tracking_results.mp4", "mp4")
                    if folderDialog.exec_():
                        pth = self.export_as_video_button_clicked(folderDialog.selectedFiles()[0])
                    else:
                        return
                if coco_radio:
                    # Get user input for COCO export path
                    folderDialog = utils.FolderDialog("coco.json", "json")
                    if folderDialog.exec_():
                        pth = utils.exportCOCOvid(
                            json_file_name, self.CURRENT_VIDEO_WIDTH, self.CURRENT_VIDEO_HEIGHT, folderDialog.selectedFiles()[0])
                    else:
                        return
                if mot_radio:
                    # Get user input for MOT export path
                    folderDialog = utils.FolderDialog("mot.txt", "txt")
                    if folderDialog.exec_():
                        pth = utils.exportMOT(
                            json_file_name, folderDialog.selectedFiles()[0])
                    else:
                        return
                # custom exports
                custom_exports_list_video = [custom_export for custom_export in custom_exports_list if custom_export.mode == "video"]
                if len(custom_exports_radio_checked_list) != 0:
                    for i in range(len(custom_exports_radio_checked_list)):
                        if custom_exports_radio_checked_list[i]:
                            # Get user input for custom export path
                            folderDialog = utils.FolderDialog(
                                f"{custom_exports_list_video[i].file_name}.{custom_exports_list_video[i].format}", custom_exports_list_video[i].format)
                            if folderDialog.exec_():
                                try:
                                    pth = custom_exports_list_video[i](json_file_name, self.CURRENT_VIDEO_WIDTH, self.CURRENT_VIDEO_HEIGHT, folderDialog.selectedFiles()[0])
                                except Exception as e:
                                    helpers.OKmsgBox(f"Error", f"Error: with custom export {custom_exports_list_video[i].button_name}\n check the parameters matches the specified ones in custom_exports.py\n Error Message: {e}", "critical")
                            else:
                                return

            # Image and Directory modes
            elif self.current_annotation_mode == "img" or self.current_annotation_mode == "dir":
                result, coco_radio, custom_exports_radio_checked_list = helpers.exportData_GUI(mode= "image")
                if not result:
                    return
                save_path = self.save_path if self.save_path else self.labelFile.filename
                if coco_radio:
                    # Get user input for COCO export path
                    folderDialog = utils.FolderDialog("coco.json", "json")
                    if folderDialog.exec_():
                        pth = utils.exportCOCO(self.target_directory, save_path, folderDialog.selectedFiles()[0])
                    else:
                        return
                # custom exports
                custom_exports_list_image = [custom_export for custom_export in custom_exports_list if custom_export.mode == "image"]
                if len(custom_exports_radio_checked_list) != 0:
                    for i in range(len(custom_exports_radio_checked_list)):
                        if custom_exports_radio_checked_list[i]:
                            # Get user input for custom export path
                            folderDialog = utils.FolderDialog(
                                f"{custom_exports_list_image[i].file_name}.{custom_exports_list_image[i].format}", custom_exports_list_image[i].format)
                            if folderDialog.exec_():
                                try:
                                    pth = custom_exports_list_image[i](self.target_directory, save_path, folderDialog.selectedFiles()[0])
                                except Exception as e:
                                    helpers.OKmsgBox(f"Error", f"Error: with custom export {custom_exports_list_image[i].button_name}\n check the parameters matches the specified ones in custom_exports.py\n Error Message: {e}", "critical")
                            else:
                                return

        except Exception as e:
            # Error QMessageBox
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(f"Error\n {e}")
            msg.setWindowTitle(
                "Export Error")
            # print exception and error line to terminal
            print(e)
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return
        else:
            # display QMessageBox with ok button and label "Exporting COCO"
            msg = QtWidgets.QMessageBox()
            try:
                if pth not in ["", None, False]:
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText(f"Annotations exported successfully to {pth}")
                    msg.setWindowTitle("Export Success")
                else:
                    msg.setIcon(QtWidgets.QMessageBox.Critical)
                    msg.setText(f"Export Failed")
                    msg.setWindowTitle("Export Failed")
            except:
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setText(f"Export Failed")
                msg.setWindowTitle("Export Failed")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()

    def saveFileAs(self, _value=False):
        self.actions.export.setEnabled(True)
        assert not self.image.isNull(), "cannot save empty image"
        self.save_path = self.saveFileDialog()
        self._saveFile(self.save_path)

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.currentPath(), filters
            )
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

        # clear the file list widget
        self.fileListWidget.clear()
        self.uniqLabelList.clear()

        self.current_annotation_mode = ""
        self.right_click_menu()
        
        for option in self.vis_options:
            option.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, "
            "proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info("Label file is removed: {}".format(label_file))

            item = self.fileListWidget.currentItem()
            item.setCheckState(Qt.Unchecked)

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(
            self.filename
        )
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        if not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)

    def deleteSelectedShape(self):
        try:
            if len(self.canvas.selectedShapes) == 0:
                return
            
            yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
            msg = self.tr(
                "You are about to permanently delete {} polygons, "
                "proceed anyway?"
            ).format(len(self.canvas.selectedShapes))
            if yes == QtWidgets.QMessageBox.warning(
                self, self.tr("Attention"), msg, yes | no, yes
            ):
                deleted_shapes = self.canvas.deleteSelected()
                deleted_ids = [shape.group_id for shape in deleted_shapes]
                self.remLabels(deleted_shapes)
                self.setDirty()
                if self.noShapes():
                    for action in self.actions.onShapesPresent:
                        action.setEnabled(False)
                if self.current_annotation_mode == 'img' or self.current_annotation_mode == 'dir':
                    self.refresh_image_MODE()
                    return

                # if video mode
                result, self.config, fromFrameVAL, toFrameVAL = helpers.deleteSelectedShape_GUI(
                    self.TOTAL_VIDEO_FRAMES,
                    self.INDEX_OF_CURRENT_FRAME,
                    self.config)
                if result == QtWidgets.QDialog.Accepted:
                    for deleted_id in deleted_ids:
                        self.delete_ids_from_all_frames(
                            [deleted_id], from_frame=fromFrameVAL, to_frame=toFrameVAL)

                self.main_video_frames_slider_changed()
        except Exception as e:
            helpers.OKmsgBox(f"Error", f"Error: {e}", "critical")

    def delete_ids_from_all_frames(self, deleted_ids, from_frame, to_frame):
        
        """
        Summary:
            Delete ids from a range of frames
            
        Args:
            deleted_ids (list): list of ids to be deleted
            from_frame (int): starting frame
            to_frame (int): ending frame
        """
        
        from_frame, to_frame = np.min(
            [from_frame, to_frame]), np.max([from_frame, to_frame])
        listObj = self.load_objects_from_json__orjson()

        for i in range(from_frame - 1, to_frame, 1):
            frame_idx = listObj[i]['frame_idx']
            for object_ in listObj[i]['frame_data']:
                id = object_['tracker_id']
                if id in deleted_ids:
                    listObj[i]['frame_data'].remove(object_)
                    self.CURRENT_ANNOATAION_TRAJECTORIES['id_' +
                                                         str(id)][frame_idx - 1] = (-1, -1)
                    self.rec_frame_for_id(id, frame_idx, type_='remove')

        self.load_objects_to_json__orjson(listObj)

    def copyShape(self):
        
        """
        Summary:
            Copy selected shape in right click menu.
            is NOT saved in the clipboard
        """

        if len(self.canvas.selectedShapes) > 1 and self.current_annotation_mode == 'video':
            org = copy.deepcopy(self.canvas.shapes)
            self.canvas.endMove(copy=True)
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()
            self.canvas.shapes = org
            self.update_current_frame_annotation_button_clicked()
            return

        elif self.current_annotation_mode == 'video':
            self.canvas.endMove(copy=True)
            shape = self.canvas.selectedShapes[0]
            text = shape.label
            text, flags, group_id, content = self.labelDialog.popUp(text)
            shape.group_id = -1
            shape.content = content
            shape.label = text
            shape.flags = flags

            group_id, text = self.get_id_from_user(group_id, text)

            if text:
                self.labelList.clearSelection()
                shape = self.canvas.setLastLabel(text, flags)
                shape.group_id = group_id
                self.addLabel(shape)
                self.rec_frame_for_id(
                    shape.group_id, self.INDEX_OF_CURRENT_FRAME)
                self.actions.editMode.setEnabled(True)
                self.actions.undoLastPoint.setEnabled(False)
                self.actions.undo.setEnabled(True)
                self.setDirty()
            else:
                self.canvas.undoLastLine()
                self.canvas.shapesBackups.pop()

            self.update_current_frame_annotation_button_clicked()

            return

        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()
        if self.current_annotation_mode == 'video':
            self.update_current_frame_annotation_button_clicked()
        self.right_click_menu()

    def openDirDialog(self, _value=False, dirpath=None):
        # self.disconnectVideoShortcuts()

        # self.current_annotation_mode = "dir"
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = (
                osp.dirname(self.filename) if self.filename else "."
            )

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.target_directory = targetDirPath
        self.importDirImages(targetDirPath)
        self.set_video_controls_visibility(False)


        # enable Visualization Options
        for option in self.vis_options:
            if option in [self.id_checkBox, self.traj_checkBox, self.trajectory_length_lineEdit]:
                option.setEnabled(False)
            else:
                option.setEnabled(True)

    

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(
                tuple(extensions)
            ):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
                label_file
            ):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        self.openNextImg()

    def importDirImages(self, dirpath, pattern=None, load=True):
        
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)
        self.actions.export.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return
        self.reset_for_new_mode("dir")
        # self.right_click_menu()
        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()
        self.uniqLabelList.clear()
        for filename in self.scanAllImages(dirpath):
            if pattern and pattern not in filename:
                continue
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
                label_file
            ):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)
        self.openNextImg(load=load)
        self.fileListWidget.horizontalScrollBar().setValue(
            self.fileListWidget.horizontalScrollBar().maximum()
        )
        

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = osp.join(root, file)
                    images.append(relativePath)
        images.sort(key=lambda x: x.lower())
        return images

    def annotate_one(self,called_from_tracking=False):
        # self.waitWindow(visible=True, text="Please Wait.\nModel is Working...")
        # self.CURRENT_ANNOATAION_FLAGS['id'] = False
        if self.current_annotation_mode == "video":
            if self.multi_model_flag:
                shapes = self.intelligenceHelper.get_shapes_of_one(
                    self.CURRENT_FRAME_IMAGE, img_array_flag=True, multi_model_flag=True)
            else:
                shapes = self.intelligenceHelper.get_shapes_of_one(
                    self.CURRENT_FRAME_IMAGE, img_array_flag=True)
            if called_from_tracking:
                return shapes

        # elif self.current_annotation_mode == "multimodel":
        #     shapes = self.intelligenceHelper.get_shapes_of_one(
        #         self.CURRENT_FRAME_IMAGE, img_array_flag=True, multi_model_flag=True)
            # to handle the bug of the user selecting no models
            # if len(shapes) == 0:
            #     self.waitWindow()
            #     return
        else:
            # if self.filename != self.last_file_opened:
            #     self.last_file_opened = self.filename
            #     self.intelligenceHelper.clear_annotating_models()
            try:
                if os.path.exists(self.filename):
                    self.labelList.clearSelection()
                if self.multi_model_flag:
                    shapes = self.intelligenceHelper.get_shapes_of_one(self.CURRENT_FRAME_IMAGE, img_array_flag=True, multi_model_flag=True)
                else:
                    shapes = self.intelligenceHelper.get_shapes_of_one(
                        self.CURRENT_FRAME_IMAGE, img_array_flag=True)
            except Exception as e:
                helpers.OKmsgBox("Error", f"{e}", "critical")
                return
            
        imageX = helpers.draw_bb_on_image_MODE(self.CURRENT_ANNOATAION_FLAGS,
                                                    self.image, 
                                                    shapes)

        # for shape in shapes:
        #     print(shape["group_id"])
        #     print(shape["bbox"])

        # image = self.draw_bb_on_image(self.image  , self.CURRENT_SHAPES_IN_IMG)
        # self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))

        # make all annotation_flags false except show polygons

        # clear shapes already in lablelist (fixes saving multiple shapes of same object bug)
        self.labelList.clear()
        self.CURRENT_SHAPES_IN_IMG = shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(imageX))
        self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        self.actions.editMode.setEnabled(True)
        self.actions.undoLastPoint.setEnabled(False)
        self.actions.undo.setEnabled(True)
        self.setDirty()
        # self.waitWindow()
        
    def refresh_image_MODE(self, fromSignal=False):
        try:
            if self.current_annotation_mode == "video" and not fromSignal:
                return
            self.CURRENT_SHAPES_IN_IMG = self.convert_qt_shapes_to_shapes(self.canvas.shapes)
            imageX = helpers.draw_bb_on_image_MODE(self.CURRENT_ANNOATAION_FLAGS,
                                                            self.image, 
                                                            self.CURRENT_SHAPES_IN_IMG)
            # self.canvas.setEnabled(False)
            self.labelList.clear()
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(imageX))
            self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
            # self.canvas.setEnabled(True)
            # self.actions.editMode.setEnabled(True)
        except:
            pass

    def annotate_batch(self):
        images = []
        self._config = get_config()
        notif = [self._config["mute"], self, helpers.notification]
        for filename in self.imageList:
            images.append(filename)
        if self.multi_model_flag:
            self.intelligenceHelper.get_shapes_of_batch(images, multi_model_flag=True, notif = notif)
        else:
            self.intelligenceHelper.get_shapes_of_batch(images, notif = notif)

    def setConfThreshold(self):
        if self.intelligenceHelper.conf_threshold:
            self.intelligenceHelper.conf_threshold = self.intelligenceHelper.setConfThreshold(
                self.intelligenceHelper.conf_threshold)
        else:
            self.intelligenceHelper.conf_threshold = self.intelligenceHelper.setConfThreshold()

    def setIOUThreshold(self):
        if self.intelligenceHelper.iou_threshold:
            self.intelligenceHelper.iou_threshold = self.intelligenceHelper.setIOUThreshold(
                self.intelligenceHelper.iou_threshold)
        else:
            self.intelligenceHelper.iou_threshold = self.intelligenceHelper.setIOUThreshold()

    def selectClasses(self):
        self.intelligenceHelper.selectedclasses = self.intelligenceHelper.selectClasses()

    def mergeSegModels(self):
        self.intelligenceHelper.selectedmodels = self.intelligenceHelper.mergeSegModels()
        # check if the user selected any models
        if len(self.intelligenceHelper.selectedmodels) == 0:
            print("No models selected")
        else:
            self.multi_model_flag = True

    def Segment_anything(self):
    # check the visibility of the sam toolbar
        if self.sam_toolbar.isVisible():
            self.set_sam_toolbar_visibility(False)
        else:
            self.set_sam_toolbar_visibility(True)



    # VIDEO PROCESSING FUNCTIONS (ALL CONNECTED TO THE VIDEO PROCESSING TOOLBAR)

    def mapFrameToTime(self, frameNumber):
        # get the fps of the video
        fps = self.CAP.get(cv2.CAP_PROP_FPS)
        return helpers.mapFrameToTime(frameNumber, fps)

    def calculate_trajectories(self, frames = None):
        
        """
        Summary:
            Calculate trajectories for all objects in the video
            
        Args:
            frames (list): list of frames to calculate trajectories for (default: None -> all frames)
        """
        
        listObj = self.load_objects_from_json__orjson()
        if len(listObj) == 0:
            return
        
        frames = frames if frames else range(len(listObj))
        
        for i in frames:
            listobjframe = listObj[i]['frame_idx']
            for object in listObj[i]['frame_data']:
                id = object['tracker_id']
                self.minID = min(self.minID, id - 1)
                self.rec_frame_for_id(id, listobjframe)
                label = object['class_name']
                # color calculation
                # label_hash = hash(label)
                # idx = abs(label_hash) % len(color_palette)
                label_ascii = sum([ord(c) for c in label])
                idx = label_ascii % len(color_palette)
                color = color_palette[idx]
                center = self.centerOFmass(object['segment'])
                try:
                    centers_rec = self.CURRENT_ANNOATAION_TRAJECTORIES['id_' + str(
                        id)]

                    try:
                        (xp, yp) = centers_rec[listobjframe - 2]
                        (xn, yn) = center
                        if (xp == -1 or xn == -1):
                            c = 5 / 0
                        r = 0.5
                        x = r * xn + (1 - r) * xp
                        y = r * yn + (1 - r) * yp
                        center = (int(x), int(y))
                    except:
                        pass
                    centers_rec[listobjframe - 1] = center
                    self.CURRENT_ANNOATAION_TRAJECTORIES['id_' +
                                                         str(id)] = centers_rec
                    self.CURRENT_ANNOATAION_TRAJECTORIES['id_color_' + str(
                        id)] = color
                except:
                    centers_rec = [(-1, - 1)] * int(self.TOTAL_VIDEO_FRAMES)
                    centers_rec[listobjframe - 1] = center
                    self.CURRENT_ANNOATAION_TRAJECTORIES['id_' +
                                                         str(id)] = centers_rec
                    self.CURRENT_ANNOATAION_TRAJECTORIES['id_color_' + str(
                        id)] = color

    def right_click_menu(self):
        
        """
        Summary:
            Set the right click menu according to the current annotation mode
        """
        
        self.set_sam_toolbar_enable(False)
        self.sam_model_comboBox.setCurrentIndex(0)
        self.sam_buttons_colors("x")
         # # right click menu
         
        #         0  createMode,
        #         1  editMode,
        #         2  edit,
        #         3  enhance,
        #         4  interpolate,
        #         5  mark_as_key,
        #         6  remove_all_keyframes,
        #         7  scale,
        #         8  copyShapes,
        #         9  pasteShapes,
        #         10  copy,
        #         11 delete,
        #         12 undo,
        #         13 undoLastPoint,
        #         14 addPointToEdge,
        #         15 removePoint,
        #         16 update_curr_frame,
        #         17 ignore_changes


        mode = self.current_annotation_mode
        video_menu_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,     11,         14, 16, 17]
        image_menu_list = [0, 1, 2, 3,                   10, 11, 12, 13, 14        ]
        
        if self.current_annotation_mode == "video":
            self.canvas.menus[0].clear()
            utils.addActions(self.canvas.menus[0], (self.actions.menu[i] for i in video_menu_list))

            self.menus.edit.clear()
            utils.addActions(self.menus.edit, (self.actions.menu[i] for i in video_menu_list))
        else:
            self.canvas.menus[0].clear()
            utils.addActions(self.canvas.menus[0], (self.actions.menu[i] for i in image_menu_list))

            self.menus.edit.clear()
            utils.addActions(self.menus.edit, (self.actions.menu[i] for i in image_menu_list))

    def reset_for_new_mode(self, mode):
        length_Value = self.CURRENT_ANNOATAION_TRAJECTORIES['length']
        alpha_Value = self.CURRENT_ANNOATAION_TRAJECTORIES['alpha']
        self.CURRENT_ANNOATAION_TRAJECTORIES.clear()
        self.CURRENT_ANNOATAION_TRAJECTORIES['length'] = length_Value
        self.CURRENT_ANNOATAION_TRAJECTORIES['alpha'] = alpha_Value
        self.key_frames.clear()
        self.id_frames_rec.clear()

        for shape in self.canvas.shapes:
            self.canvas.deleteShape(shape)

        self.resetState()
        
        self.CURRENT_SHAPES_IN_IMG = []
        self.image = QtGui.QImage()
        self.CURRENT_FRAME_IMAGE = None
        
        self.current_annotation_mode = mode
        self.canvas.current_annotation_mode = mode
        self.right_click_menu()
        self.global_listObj = []
        self.minID = -2
        self.maxID = 0

    def openVideo(self):

        # enable export if json file exists
        
        try:
            cv2.destroyWindow('video processing')
        except:
            pass
        if not self.mayContinue():
            return
        videoFile = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("%s - Open Video") % __appname__, ".",
            self.tr("Video files (*.mp4 *.avi *.mov)")
        )

        if videoFile[0]:
            # clear the file list widget
            self.fileListWidget.clear()
            self.uniqLabelList.clear()
            self.reset_for_new_mode("video")
            
            self.CURRENT_VIDEO_NAME = videoFile[0].split(
                ".")[-2].split("/")[-1]
            self.CURRENT_VIDEO_PATH = "/".join(
                videoFile[0].split(".")[-2].split("/")[:-1])
            
            json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
            if os.path.exists(json_file_name):
                self.actions.export.setEnabled(True)
            else:
                self.actions.export.setEnabled(False)

            # print('video file name : ' , self.CURRENT_VIDEO_NAME)
            # print("video file path : " , self.CURRENT_VIDEO_PATH)
            cap = cv2.VideoCapture(videoFile[0])
            self.CURRENT_VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.CURRENT_VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.CAP = cap
            # # making the total video frames equal to the total frames in the video file - 1 as the indexing starts from 0
            # self.TOTAL_VIDEO_FRAMES = int(
            #     self.CAP.get(cv2.CAP_PROP_FRAME_COUNT - 1) )
            self.TOTAL_VIDEO_FRAMES = int(
                self.CAP.get(cv2.CAP_PROP_FRAME_COUNT))
            self.CURRENT_VIDEO_FPS = self.CAP.get(cv2.CAP_PROP_FPS)
            print("Total Frames : ", self.TOTAL_VIDEO_FRAMES)
            self.main_video_frames_slider.setMaximum(self.TOTAL_VIDEO_FRAMES)
            self.frames_to_track_slider.setMaximum(self.TOTAL_VIDEO_FRAMES - self.INDEX_OF_CURRENT_FRAME)
            self.main_video_frames_slider.setValue(2)
            self.INDEX_OF_CURRENT_FRAME = 1
            self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME)

            # self.addToolBarBreak
            self.set_video_controls_visibility(True)

            self.update_tracking_method()

            self.calculate_trajectories()
            keys = list(self.id_frames_rec.keys())
            idsORG = [int(keys[i][3:]) for i in range(len(keys))]
            if len(idsORG) > 0:
                self.maxID = max(idsORG)


            for option in self.vis_options:
                option.setEnabled(True)

        # label = shape["label"]
        # points = shape["points"]
        # bbox = shape["bbox"]
        # shape_type = shape["shape_type"]
        # flags = shape["flags"]
        # content = shape["content"]
        # group_id = shape["group_id"]
        # other_data = shape["other_data"]


        # disable save and save as
        self.actions.save.setEnabled(False)
        self.actions.saveAs.setEnabled(False)
    
    def openVideoFrames(self):
        try:
            video_frame_extractor_dialog = utils.VideoFrameExtractor(self._config["mute"], helpers.notification)
            video_frame_extractor_dialog.exec_()

            dir_path_name = video_frame_extractor_dialog.path_name
            if dir_path_name:
                self.target_directory = dir_path_name
                self.importDirImages(dir_path_name)
                self.set_video_controls_visibility(False)
                # enable Visualization Options
                for option in self.vis_options:
                    if option in [self.id_checkBox, self.traj_checkBox, self.trajectory_length_lineEdit]:
                        option.setEnabled(False)
                    else:
                        option.setEnabled(True)
        except Exception as e:
            helpers.OKmsgBox("Error", f"Error: {e}", "critical")

    def load_shapes_for_video_frame(self, json_file_name, index):
        # this function loads the shapes for the video frame from the json file
        # first we read the json file in the form of a list
        # we need to parse from it data for the current frame

        target_frame_idx = index
        listObj = self.load_objects_from_json__orjson()

        listObj = np.array(listObj)

        shapes = []
        # for i in range(len(listObj)):
        i = target_frame_idx - 1
        # frame_idx = listObj[i]['frame_idx']
        # if frame_idx == target_frame_idx:
        # print (listObj[i])
        # print(i)
        frame_objects = listObj[i]['frame_data']
        for object_ in frame_objects:
            shape = {}
            shape["label"] = object_["class_name"]
            shape["group_id"] = (object_['tracker_id'])
            shape["content"] = (object_['confidence'])
            shape["bbox"] = object_['bbox']
            points = object_['segment']
            points = np.array(points, np.int16).flatten().tolist()
            shape["points"] = points
            shape["shape_type"] = "polygon"
            shape["other_data"] = {}
            shape["flags"] = {}
            shapes.append(shape)

        self.CURRENT_SHAPES_IN_IMG = shapes

    def loadFramefromVideo(self, frame_array, index=1):
        # filename = str(index) + ".jpg"
        # self.filename = filename
        self.resetState()
        self.canvas.setEnabled(False)

        self.imageData = frame_array.data

        self.CURRENT_FRAME_IMAGE = frame_array
        image = QtGui.QImage(self.imageData, self.imageData.shape[1], self.imageData.shape[0],
                             QtGui.QImage.Format_BGR888)
        self.image = image
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes

        # detectionData = [(50, 50, 200, 200, 11), (200, 200, 400, 400, 12), (400, 400, 600, 600, 13)]
        # image = draw_bb_on_image(image, detectionData)
        flags = {k: False for k in self._config["flags"] or []}
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))

        if self.TrackingMode:
            # print("Tracking Mode", len(self.CURRENT_SHAPES_IN_IMG))
            image = self.draw_bb_on_image(image, self.CURRENT_SHAPES_IN_IMG)
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
            if len(self.CURRENT_SHAPES_IN_IMG) > 0:
                self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        else:
            if self.labelFile:
                self.CURRENT_SHAPES_IN_IMG = self.labelFile.shapes
                image = self.draw_bb_on_image(
                    image, self.CURRENT_SHAPES_IN_IMG)
                self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
                self.loadLabels(self.labelFile.shapes)
                if self.labelFile.flags is not None:
                    flags.update(self.labelFile.flags)

            else:
                json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
                if os.path.exists(json_file_name):
                    # print('json file exists , loading shapes')
                    self.load_shapes_for_video_frame(json_file_name, index)
                    image = self.draw_bb_on_image(
                        image, self.CURRENT_SHAPES_IN_IMG)
                    self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
                    if len(self.CURRENT_SHAPES_IN_IMG) > 0:
                        self.loadLabels(self.CURRENT_SHAPES_IN_IMG)

        self.loadFlags(flags)
        # if self._config["keep_prev"] and self.noShapes():
        #     self.loadShapes(prev_shapes, replace=False)
        #     self.setDirty()
        # else:
        self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values

        self.paintCanvas()
        self.toggleActions(True)
        self.canvas.setFocus()
        self.status(self.tr(
            f'Loaded {self.CURRENT_VIDEO_NAME} frame {self.INDEX_OF_CURRENT_FRAME}'))

    def nextFrame_buttonClicked(self):
        self.update_current_frame_annotation_button_clicked()
        # first assert that the new value of the slider is not greater than the total number of frames
        new_value = self.INDEX_OF_CURRENT_FRAME + self.FRAMES_TO_SKIP
        if new_value >= self.TOTAL_VIDEO_FRAMES:
            new_value = self.TOTAL_VIDEO_FRAMES
        self.main_video_frames_slider.setValue(new_value)

    def next_1_Frame_buttonClicked(self):
        self.update_current_frame_annotation_button_clicked()
        # first assert that the new value of the slider is not greater than the total number of frames
        new_value = self.INDEX_OF_CURRENT_FRAME + 1
        if new_value >= self.TOTAL_VIDEO_FRAMES:
            new_value = self.TOTAL_VIDEO_FRAMES
        self.main_video_frames_slider.setValue(new_value)

    def previousFrame_buttonClicked(self):
        self.update_current_frame_annotation_button_clicked()
        new_value = self.INDEX_OF_CURRENT_FRAME - self.FRAMES_TO_SKIP
        if new_value <= 0:
            new_value = 0
        self.main_video_frames_slider.setValue(new_value)

    def previous_1_Frame_buttonclicked(self):
        self.update_current_frame_annotation_button_clicked()
        new_value = self.INDEX_OF_CURRENT_FRAME - 1
        if new_value <= 0:
            new_value = 0
        self.main_video_frames_slider.setValue(new_value)

    def frames_to_skip_slider_changed(self):
        self.FRAMES_TO_SKIP = self.frames_to_skip_slider.value()
        zeros = (2 - int(np.log10(self.FRAMES_TO_SKIP + 0.9))) * '0'
        self.frames_to_skip_label.setText(
            'Jump forward/backward frames: ' + zeros + str(self.FRAMES_TO_SKIP))

    def playPauseButtonClicked(self):
        # we can check the state of the button by checking the button text
        
        if self.playPauseButton_mode == "Play":
            self.playPauseButton_mode = "Pause"
            self.playPauseButton.setShortcut(self._config['shortcuts']['play'])
            self.playPauseButton.setToolTip(
                f'Play ({self._config["shortcuts"]["play"]})')
            self.playPauseButton.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
            # play the video at the current fps untill the user clicks pause
            self.play_timer = QtCore.QTimer(self)
            # use play_timer.timeout.connect to call a function every time the timer times out
            # but we need to call the function every interval of time
            # so we need to call the function every 1/fps seconds
            self.play_timer.timeout.connect(self.move_frame_by_frame)
            self.play_timer.start(40)
            # note that the timer interval is in milliseconds

            # while self.timer.isActive():
        elif self.playPauseButton_mode == "Pause":
            # first stop the timer
            self.play_timer.stop()

            self.playPauseButton_mode = "Play"
            self.playPauseButton.setShortcut(self._config['shortcuts']['play'])
            self.playPauseButton.setToolTip(
                f'Pause ({self._config["shortcuts"]["play"]})')
            self.playPauseButton.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        # print(1)

    def main_video_frames_slider_changed(self):
        
        if self.current_annotation_mode != "video":
            return

        if self.sam_model_comboBox.currentIndex() != 0 and self.canvas.SAM_mode != "finished" and not self.TrackingMode:
            self.sam_clear_annotation_button_clicked()
            self.sam_buttons_colors("X")

        try:
            x = self.CURRENT_VIDEO_PATH
        except:
            return

        frame_idx = self.main_video_frames_slider.value()

        self.INDEX_OF_CURRENT_FRAME = frame_idx
        self.CAP.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)

        # setting text of labels
        zeros = (int(np.log10(self.TOTAL_VIDEO_FRAMES + 0.9)) -
                 int(np.log10(frame_idx + 0.9))) * '0'
        self.main_video_frames_label_1.setText(
            f'frame {zeros}{frame_idx} / {int(self.TOTAL_VIDEO_FRAMES)}')
        self.frame_time = self.mapFrameToTime(frame_idx)
        frame_text = ("%02d:%02d:%02d:%03d" % (
            self.frame_time[0], self.frame_time[1], self.frame_time[2], self.frame_time[3]))
        video_duration = self.mapFrameToTime(self.TOTAL_VIDEO_FRAMES)
        video_duration_text = ("%02d:%02d:%02d:%03d" % (
            video_duration[0], video_duration[1], video_duration[2], video_duration[3]))
        final_text = frame_text + " / " + video_duration_text
        self.main_video_frames_label_2.setText(f'time {final_text}')

        # reading the current frame from the video and loading it into the canvas
        success, img = self.CAP.read()
        if success:
            frame_array = np.array(img)
            self.loadFramefromVideo(frame_array, frame_idx)
        else:
            pass
        self.frames_to_track_slider.setMaximum(self.TOTAL_VIDEO_FRAMES - self.INDEX_OF_CURRENT_FRAME)

    def frames_to_track_input_changed(self, text):
        try:
            value = int(text)
            if 2 <= value <= self.frames_to_track_slider.maximum():
                self.frames_to_track_slider.setValue(value)
            elif value > self.frames_to_track_slider.maximum():
                self.frames_to_track_slider.setValue(
                    self.frames_to_track_slider.maximum())
            elif value < 2:
                self.frames_to_track_slider.setValue(2)
        except ValueError:
            pass

    def frames_to_track_slider_changed(self, value):
        self.frames_to_track_input.setText(str(value))
        self.FRAMES_TO_TRACK = self.frames_to_track_slider.value()
        zeros = (2 - int(np.log10(self.FRAMES_TO_TRACK + 0.9))) * '0'


    def move_frame_by_frame(self):
        QtWidgets.QApplication.processEvents()
        self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME + 1)

    def class_name_to_id(self, class_name):
        return helpers.class_name_to_id(class_name)

    def track_assigned_objects_button_clicked(self):
        # first check if there is objects in self.canvas.shapes list or not . if not then output a error message and return
        if len(self.labelList.selectedItems()) == 0:
            self.errorMessage(
                "found No objects to track",
                "you need to assign at least one object to track",
            )
            return

        self.TRACK_ASSIGNED_OBJECTS_ONLY = True
        self.track_buttonClicked()
        self.TRACK_ASSIGNED_OBJECTS_ONLY = False

    def compute_iou(self, box1, box2):
        return helpers.compute_iou(box1, box2)

    def match_detections_with_tracks(self, detections, tracks, iou_threshold=0.5):
        return helpers.match_detections_with_tracks(detections, tracks, iou_threshold)

    def update_gui_after_tracking(self, index):
        # self.loadFramefromVideo(self.CURRENT_FRAME_IMAGE, self.INDEX_OF_CURRENT_FRAME)
        # QtWidgets.QApplication.processEvents()
        # progress_value = int((index + 1) / self.FRAMES_TO_TRACK * 100)
        # self.tracking_progress_bar_signal.emit(progress_value)

        # QtWidgets.QApplication.processEvents()
        if index != self.FRAMES_TO_TRACK - 1:
            self.main_video_frames_slider.setValue(
                self.INDEX_OF_CURRENT_FRAME + 1)
        QtWidgets.QApplication.processEvents()

    def track_buttonClicked_wrapper(self):
        # ...

        # Disable the track button
        # self.actions.track.setEnabled(False)

        # Create a thread to run the tracking process
        self.thread = TrackingThread(parent=self)
        self.thread.start()

    def get_boxes_conf_classids_segments(self, shapes):
        return helpers.get_boxes_conf_classids_segments(shapes)


    def track_dropdown_changed(self, index):
        self.selected_option = index

    def start_tracking_button_clicked(self):
        try:
            try:
                if self.selected_option == 0:
                    self.track_buttonClicked()
                elif self.selected_option == 1:
                    self.track_assigned_objects_button_clicked()
                elif self.selected_option == 2:
                    self.track_full_video_button_clicked()
            except Exception as e:
                self.track_buttonClicked()
        except Exception as e:
            helpers.OKmsgBox("Error", f"Error: {e}", "critical")
            
    def track_buttonClicked(self):

        # Disable Exports & Change button text
        # self.export_as_video_button.setEnabled(False)
        self.actions.export.setEnabled(False)

        # dt = (Profile(), Profile(), Profile(), Profile())
    
        self.tracking_progress_bar.setVisible(True)
        # self.track_stop_button.setVisible(True)
        # self.track_stop_button.setEnabled(True)
        frame_shape = self.CURRENT_FRAME_IMAGE.shape
        # print(frame_shape)

        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'

        # first we need to check there is a json file with the same name as the video
        listObj = self.load_objects_from_json__orjson()

        existing_annotation = False
        shapes = self.canvas.shapes
        tracks_to_follow = None
        if len(shapes) > 0:
            existing_annotation = True
            # print(f'FRAME{self.INDEX_OF_CURRENT_FRAME}', len(shapes))
            tracks_to_follow = []
            for shape in shapes:
                # print(shape.label)
                if shape.group_id != None:
                    tracks_to_follow.append(int(shape.group_id))

        # print(f'track_ids_to_follow = {tracks_to_follow}')

        self.TrackingMode = True
        bs = 1
        curr_frame, prev_frame = None, None

        if self.FRAMES_TO_TRACK + self.INDEX_OF_CURRENT_FRAME <= self.TOTAL_VIDEO_FRAMES:
            number_of_frames_to_track = self.FRAMES_TO_TRACK
        else:
            number_of_frames_to_track = self.TOTAL_VIDEO_FRAMES - self.INDEX_OF_CURRENT_FRAME

        self.interrupted = False
        for i in range(number_of_frames_to_track):
            QtWidgets.QApplication.processEvents()
            if self.interrupted:
                self.interrupted = False
                break
            if i % 100 == 0:
                self.load_objects_to_json__orjson(listObj)
            self.tracking_progress_bar.setValue(
                int((i + 1) / number_of_frames_to_track * 100))


            if existing_annotation:
                existing_annotation = False
                # print('\nloading existing annotation\n')
                shapes = self.canvas.shapes
                shapes = self.convert_qt_shapes_to_shapes(shapes)
            else:
                with torch.no_grad():
                    # shapes = self.intelligenceHelper.get_shapes_of_one(
                    #     self.CURRENT_FRAME_IMAGE, img_array_flag=True)
                    shapes = self.annotate_one(called_from_tracking=True)

            curr_frame = self.CURRENT_FRAME_IMAGE
            # current_objects_ids = []
            if len(shapes) == 0:
                # print("no detection in this frame")
                self.update_gui_after_tracking(i)
                continue

            for shape in shapes:
                if shape['content'] is None:
                    shape['content'] = 1.0
            boxes, confidences, class_ids, segments = self.get_boxes_conf_classids_segments(
                shapes)

            boxes = np.array(boxes, dtype=int)
            confidences = np.array(confidences)
            class_ids = np.array(class_ids)
            detections = Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids,
            )
            boxes = torch.from_numpy(detections.xyxy)
            confidences = torch.from_numpy(detections.confidence)
            class_ids = torch.from_numpy(detections.class_id)

            dets = torch.cat((boxes, confidences.unsqueeze(
                1), class_ids.unsqueeze(1)), dim=1)
            # convert dets to torch.float32 to avoid error in the tracker update function
            dets = dets.to(torch.float32)
            if hasattr(self.tracker, 'tracker') and hasattr(self.tracker.tracker, 'camera_update'):
                if prev_frame is not None and curr_frame is not None:  # camera motion compensation
                    self.tracker.tracker.camera_update(prev_frame, curr_frame)
                    # print('camera update')
            prev_frame = curr_frame
            with torch.no_grad():
                org_tracks = self.tracker.update(
                    dets.cpu(), self.CURRENT_FRAME_IMAGE)

            tracks = []
            for org_track in org_tracks:
                track = []
                for i in range(6):
                    track.append(int(org_track[i]))
                track[4] += int(self.maxID)
                track.append(org_track[6])

                tracks.append(track)

            matched_shapes, unmatched_shapes = self.match_detections_with_tracks(
                shapes, tracks)
            shapes = matched_shapes

            self.CURRENT_SHAPES_IN_IMG = [
                shape_ for shape_ in shapes if shape_["group_id"] is not None]

            if self.TRACK_ASSIGNED_OBJECTS_ONLY and tracks_to_follow is not None:
                try:
                    if len(self.labelList.selectedItems()) != 0:
                        tracks_to_follow = []
                        for item in self.labelList.selectedItems():
                            x = item.text()
                            i1, i2 = x.find('D'), x.find(':')
                            # print(x, x[i1 + 2:i2])
                            tracks_to_follow.append(int(x[i1 + 2:i2]))
                    self.CURRENT_SHAPES_IN_IMG = [
                        shape_ for shape_ in shapes if shape_["group_id"] in tracks_to_follow]
                except:
                    # this happens when the user selects a label that is not a tracked object so there is error in extracting the tracker id
                    # show a message box to the user (hinting to use the tracker on the image first so that the label has a tracker id to be selected)
                    self.errorMessage(
                        'Error', 'Please use the tracker on the image first so that you can select labels with IDs to track')

                    return

            # to understand the json output file structure it is a dictionary of frames and each frame is a dictionary of tracker_ids and each tracker_id is a dictionary of bbox , confidence , class_id , segment
            json_frame = {}
            json_frame.update({'frame_idx': self.INDEX_OF_CURRENT_FRAME})
            json_frame_object_list = []
            for shape in self.CURRENT_SHAPES_IN_IMG:
                self.rec_frame_for_id(
                    int(shape["group_id"]), self.INDEX_OF_CURRENT_FRAME, type_='add')
                json_tracked_object = {}
                json_tracked_object['tracker_id'] = int(shape["group_id"])
                json_tracked_object['bbox'] = [int(i) for i in shape['bbox']]
                json_tracked_object['confidence'] = shape["content"]
                json_tracked_object['class_name'] = shape["label"]
                json_tracked_object['class_id'] = coco_classes.index(
                    shape["label"]) if shape["label"] in coco_classes else -1
                points = shape["points"]
                segment = [[int(points[z]), int(points[z + 1])]
                           for z in range(0, len(points), 2)]
                json_tracked_object['segment'] = segment

                json_frame_object_list.append(json_tracked_object)

            json_frame.update({'frame_data': json_frame_object_list})

            # for h in range(len(listObj)):
            #     if listObj[h]['frame_idx'] == self.INDEX_OF_CURRENT_FRAME:
            #         listObj.pop(h)
            #         break
            # listObj.pop(self.INDEX_OF_CURRENT_FRAME - 1)

            # sort the list of frames by the frame index
            # listObj.append(json_frame)
            listObj[self.INDEX_OF_CURRENT_FRAME - 1] = json_frame

            QtWidgets.QApplication.processEvents()
            self.update_gui_after_tracking(i)
            print('finished tracking for frame ', self.INDEX_OF_CURRENT_FRAME)
            import psutil
            if self.INDEX_OF_CURRENT_FRAME % 10 == 0:
                print(f"Total Memory: {psutil.virtual_memory().total / 1024 ** 3} GB | Free Memory: {psutil.virtual_memory().free / 1024 ** 3} GB | Percent Used: {psutil.virtual_memory().percent} %")

        # listObj = sorted(listObj, key=lambda k: k['frame_idx'])
        self.load_objects_to_json__orjson(listObj)

        # Notify the user that the tracking is finished
        self._config = get_config()
        if not self._config["mute"]:
            if not self.isActiveWindow():
                helpers.notification("Tracking Completed")

        self.TrackingMode = False
        self.labelFile = None
        self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME - 1)
        self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME)

        self.tracking_progress_bar.hide()
        self.tracking_progress_bar.setValue(0)
        # self.track_stop_button.hide()
        # self.track_stop_button.setEnabled(False)

        # Enable Exports & Restore button Text and Color
        self.actions.export.setEnabled(True)
        # self.export_as_video_button.setEnabled(True)

    def convert_qt_shapes_to_shapes(self, qt_shapes):
        return helpers.convert_qt_shapes_to_shapes(qt_shapes)

    def track_full_video_button_clicked(self):
        self.FRAMES_TO_TRACK = int(
            self.TOTAL_VIDEO_FRAMES - self.INDEX_OF_CURRENT_FRAME)
        self.track_buttonClicked()

    def set_video_controls_visibility(self, visible=False):
        # make it invisible by default
        self.videoControls.setVisible(visible)
        for widget in self.videoControls.children():
            try:
                widget.setVisible(visible)
            except:
                pass
        self.videoControls_2.setVisible(visible)
        for widget in self.videoControls_2.children():
            try:
                widget.setVisible(visible)
            except:
                pass
        # self.videoControls_3.setVisible(visible)
        # for widget in self.videoControls_3.children():
        #     try:
        #         widget.setVisible(visible)
        #     except:
        #         pass
        # Disable Stop tracking button by default
        # self.track_stop_button.setEnabled(False) 

    def window_wait(self, seconds):
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(seconds * 1000, loop.quit)
        loop.exec_()

    def traj_checkBox_changed(self):
        try:
            self.CURRENT_ANNOATAION_FLAGS["traj"] = self.traj_checkBox.isChecked()
            self.update_current_frame_annotation()
            self.main_video_frames_slider_changed()
        except:
            pass

    def mask_checkBox_changed(self):
        try:
            self.CURRENT_ANNOATAION_FLAGS["mask"] = self.mask_checkBox.isChecked()
            self.update_current_frame_annotation()
            self.main_video_frames_slider_changed()
        except:
            pass
        self.refresh_image_MODE()

    def class_checkBox_changed(self):
        try:
            self.CURRENT_ANNOATAION_FLAGS["class"] = self.class_checkBox.isChecked()
            self.update_current_frame_annotation()
            self.main_video_frames_slider_changed()
        except:
            pass
        self.refresh_image_MODE()
        
    def conf_checkBox_changed(self):
        try:
            self.CURRENT_ANNOATAION_FLAGS["conf"] = self.conf_checkBox.isChecked()
            self.update_current_frame_annotation()
            self.main_video_frames_slider_changed()
        except:
            pass
        self.refresh_image_MODE()

    def id_checkBox_changed(self):
        try:
            self.CURRENT_ANNOATAION_FLAGS["id"] = self.id_checkBox.isChecked()
            self.update_current_frame_annotation()
            self.main_video_frames_slider_changed()
        except:
            pass        

    def bbox_checkBox_changed(self):
        try:
            self.CURRENT_ANNOATAION_FLAGS["bbox"] = self.bbox_checkBox.isChecked()
            self.update_current_frame_annotation()
            self.main_video_frames_slider_changed()
        except:
            pass
        self.refresh_image_MODE()
            

    def polygons_visable_checkBox_changed(self):
        try:
            self.CURRENT_ANNOATAION_FLAGS["polygons"] = self.polygons_visable_checkBox.isChecked(
            )
            self.update_current_frame_annotation()
            for shape in self.canvas.shapes:
                self.canvas.setShapeVisible(
                    shape, self.CURRENT_ANNOATAION_FLAGS["polygons"])
        except:
            pass

    def export_as_video_button_clicked(self, output_filename = None):
        # print (f"output filename is {output_filename}")
        self.update_current_frame_annotation()
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
        input_video_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}.mp4'
        output_video_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.mp4'
        if output_filename is not False:
            output_video_file_name = output_filename
        print (f"output video file name is {output_video_file_name}")
        input_cap = cv2.VideoCapture(input_video_file_name)
        output_cap = cv2.VideoWriter(output_video_file_name, cv2.VideoWriter_fourcc(
            *'mp4v'), int(self.CURRENT_VIDEO_FPS), (int(self.CURRENT_VIDEO_WIDTH), int(self.CURRENT_VIDEO_HEIGHT)))
        listObj = self.load_objects_from_json__orjson()

        # make a progress bar for exporting video (with percentage of progress)   TO DO LATER
        empty_frame = False
        empty_video = True
        for target_frame_idx in range(self.TOTAL_VIDEO_FRAMES):
            try:
                self.INDEX_OF_CURRENT_FRAME = target_frame_idx + 1
                ret, image = input_cap.read()
                shapes = []
                # self.waitWindow(
                #         visible=True, text=f'Please Wait.\nFrame {target_frame_idx} is being exported...')
                frame_idx = listObj[target_frame_idx]['frame_idx']
                frame_objects = listObj[target_frame_idx]['frame_data']
                # print(frame_idx)
                for object_ in frame_objects:
                    shape = {}
                    shape["label"] = object_['class_name']
                    shape["group_id"] = str(object_['tracker_id'])
                    shape["content"] = str(object_['confidence'])
                    shape["bbox"] = object_['bbox']
                    points = object_['segment']
                    points = np.array(points, np.int16).flatten().tolist()
                    shape["points"] = points
                    shape["shape_type"] = "polygon"
                    shape["other_data"] = {}
                    shape["flags"] = {}
                    shapes.append(shape)

                if len(shapes) == 0:
                    if not empty_frame:
                        self.waitWindow(visible=True, text=f'Processing...')
                        empty_frame = True
                    continue
                self.waitWindow(visible=True, text=f'Please Wait.\nFrame {target_frame_idx} is being exported...')
                image = self.draw_bb_on_image(image, shapes, image_qt_flag=False)
                output_cap.write(image)
                empty_frame = False
                empty_video = False
            except:
                input_cap.release()
                output_cap.release()


        input_cap.release()
        output_cap.release()
        # print("done exporting video")
        self.waitWindow()

        try:
            if empty_video:
                os.remove(output_video_file_name)
                return False
        except:
            pass
        
        self.INDEX_OF_CURRENT_FRAME = self.main_video_frames_slider.value()
        # show message saying that the video is exported
        if output_filename is False:
            helpers.OKmsgBox("Export Video", "Done Exporting Video")

        if output_filename is not False:
            return output_filename

    def clear_video_annotations_button_clicked(self):
        self.global_listObj = []
        length_Value = self.CURRENT_ANNOATAION_TRAJECTORIES['length']
        alpha_Value = self.CURRENT_ANNOATAION_TRAJECTORIES['alpha']
        self.CURRENT_ANNOATAION_TRAJECTORIES.clear()
        self.CURRENT_ANNOATAION_TRAJECTORIES['length'] = length_Value
        self.CURRENT_ANNOATAION_TRAJECTORIES['alpha'] = alpha_Value
        self.key_frames.clear()
        self.id_frames_rec.clear()
        self.minID = -2
        self.maxID = 0

        for shape in self.canvas.shapes:
            self.canvas.deleteShape(shape)

        self.CURRENT_SHAPES_IN_IMG = []

        # just delete the json file and reload the video
        # to delete the json file we need to know the name of the json file which is the same as the video name
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
        # now delete the json file if it exists
        if os.path.exists(json_file_name):
            os.remove(json_file_name)
        helpers.OKmsgBox("clear annotations",
                         "All video frames annotations are cleared")
        self.main_video_frames_slider.setValue(2)
        self.main_video_frames_slider.setValue(1)

    def update_current_frame_annotation_button_clicked(self):

        if self.sam_model_comboBox.currentIndex() != 0 and self.canvas.SAM_mode != "finished" and not self.TrackingMode:
            self.sam_clear_annotation_button_clicked()

        try:
            x = self.CURRENT_VIDEO_PATH
        except:
            return
        self.update_current_frame_annotation()
        self.main_video_frames_slider_changed()

    def update_current_frame_annotation(self):
        
        if self.current_annotation_mode != "video":
            return

        listObj = self.load_objects_from_json__orjson()

        json_frame = {}
        json_frame.update({'frame_idx': self.INDEX_OF_CURRENT_FRAME})
        json_frame_object_list = []

        shapes = self.convert_qt_shapes_to_shapes(self.canvas.shapes)
        for shape in shapes:
            json_tracked_object = {}
            if shape["group_id"] != None :
                json_tracked_object['tracker_id'] = int(shape["group_id"]) 
            else:
                json_tracked_object['tracker_id'] = self.minID
                self.minID -= 1
            bbox = shape["bbox"]
            bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            json_tracked_object['bbox'] = bbox
            json_tracked_object['confidence'] = str(
                shape["content"] if shape["content"] != None else 1)
            json_tracked_object['class_name'] = shape["label"]
            json_tracked_object['class_id'] = coco_classes.index(
                shape["label"]) if shape["label"] in coco_classes else -1
            points = shape["points"]
            segment = [[int(points[z]), int(points[z + 1])]
                       for z in range(0, len(points), 2)]
            json_tracked_object['segment'] = segment

            json_frame_object_list.append(json_tracked_object)
        json_frame.update({'frame_data': json_frame_object_list})

        # for h in range(len(listObj)):
        #     if listObj[h]['frame_idx'] == self.INDEX_OF_CURRENT_FRAME:
        #         listObj.pop(h)
        #         break
        # listObj.pop(self.INDEX_OF_CURRENT_FRAME - 1)

        # listObj.append(json_frame)
        listObj[self.INDEX_OF_CURRENT_FRAME - 1] = json_frame

        # listObj = sorted(listObj, key=lambda k: k['frame_idx'])
        # with open(json_file_name, 'w') as json_file:
        #     json.dump(listObj, json_file,
        #               indent=4,
        #               separators=(',', ': '))
        # json_file.close()
        self.load_objects_to_json__orjson(listObj)
        print("saved frame annotation")

    def trajectory_length_lineEdit_changed(self):
        try:
            text = self.trajectory_length_lineEdit.text()
            self.CURRENT_ANNOATAION_TRAJECTORIES['length'] = int(
                text) if text != '' else 1
            self.main_video_frames_slider_changed()
        except:
            pass

    def addVideoControls(self):
        # add video controls toolbar with custom style (background color , spacing , hover color)
        self.videoControls = QtWidgets.QToolBar()
        self.videoControls.setMovable(True)
        self.videoControls.setFloatable(True)
        self.videoControls.setObjectName("videoControls")
        self.videoControls.setStyleSheet(
            "QToolBar#videoControls { border: 50px }")
        self.addToolBar(Qt.BottomToolBarArea, self.videoControls)

        self.videoControls_2 = QtWidgets.QToolBar()
        self.videoControls_2.setMovable(True)
        self.videoControls_2.setFloatable(True)
        self.videoControls_2.setObjectName("videoControls_2")
        self.videoControls_2.setStyleSheet(
            "QToolBar#videoControls_2 { border: 50px }")
        self.addToolBar(Qt.TopToolBarArea, self.videoControls_2)

        # self.videoControls_3 = QtWidgets.QToolBar()
        # self.videoControls_3.setMovable(True)
        # self.videoControls_3.setFloatable(True)
        # self.videoControls_3.setObjectName("videoControls_2")
        # self.videoControls_3.setStyleSheet(
        #     "QToolBar#videoControls_3 { border: 50px }")
        # self.addToolBar(Qt.TopToolBarArea, self.videoControls_3)

        self.frames_to_skip_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frames_to_skip_slider.setMinimum(1)
        self.frames_to_skip_slider.setMaximum(100)
        self.frames_to_skip_slider.setValue(3)
        self.frames_to_skip_slider.setTickPosition(
            QtWidgets.QSlider.TicksBelow)
        self.frames_to_skip_slider.setTickInterval(1)
        self.frames_to_skip_slider.setMaximumWidth(250)
        self.frames_to_skip_slider.valueChanged.connect(
            self.frames_to_skip_slider_changed)
        self.frames_to_skip_label = QtWidgets.QLabel()
        self.frames_to_skip_label.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; }")
        self.frames_to_skip_slider.setValue(30)
        self.videoControls.addWidget(self.frames_to_skip_label)
        self.videoControls.addWidget(self.frames_to_skip_slider)

        self.previousFrame_button = QtWidgets.QPushButton()
        self.previousFrame_button.setText("<<")
        self.previousFrame_button.setShortcut(self._config['shortcuts']['prev_x'])
        self.previousFrame_button.setToolTip(
                f'Jump Backward ({self._config["shortcuts"]["prev_x"]})')
        self.previousFrame_button.clicked.connect(
            self.previousFrame_buttonClicked)

        self.previous_1_Frame_button = QtWidgets.QPushButton()
        self.previous_1_Frame_button.setText("<")
        self.previous_1_Frame_button.setShortcut(self._config['shortcuts']['prev_1'])
        self.previous_1_Frame_button.setToolTip(
                f'Previous Frame ({self._config["shortcuts"]["prev_1"]})')
        self.previous_1_Frame_button.clicked.connect(
            self.previous_1_Frame_buttonclicked)

        self.playPauseButton = QtWidgets.QPushButton()
        self.playPauseButton_mode = "Play"
        self.playPauseButton.setShortcut(self._config['shortcuts']['play'])
        self.playPauseButton.setToolTip(
                f'Play ({self._config["shortcuts"]["play"]})')
        self.playPauseButton.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        
        self.playPauseButton.setIconSize(QtCore.QSize(22, 22))
        self.playPauseButton.setStyleSheet("QPushButton { margin: 5px;}")
        # when the button is clicked, print "Pressed!" in the terminal
        self.playPauseButton.pressed.connect(self.playPauseButtonClicked)
        # self.playPauseButton.clicked.connect(self.playPauseButtonClicked)

        self.nextFrame_button = QtWidgets.QPushButton()
        self.nextFrame_button.setText(">>")
        self.nextFrame_button.setShortcut(self._config['shortcuts']['next_x'])
        self.nextFrame_button.setToolTip(
                f'Jump forward ({self._config["shortcuts"]["next_x"]})')
        self.nextFrame_button.clicked.connect(self.nextFrame_buttonClicked)

        self.next_1_Frame_button = QtWidgets.QPushButton()
        self.next_1_Frame_button.setText(">")
        self.next_1_Frame_button.setShortcut(self._config['shortcuts']['next_1'])
        self.next_1_Frame_button.setToolTip(
                f'Next Frame ({self._config["shortcuts"]["next_1"]})')
        self.next_1_Frame_button.clicked.connect(
            self.next_1_Frame_buttonClicked)

        self.videoControls.addWidget(self.previousFrame_button)
        self.videoControls.addWidget(self.previous_1_Frame_button)
        self.videoControls.addWidget(self.playPauseButton)
        self.videoControls.addWidget(self.next_1_Frame_button)
        self.videoControls.addWidget(self.nextFrame_button)

        self.main_video_frames_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.main_video_frames_slider.setMinimum(1)
        self.main_video_frames_slider.setMaximum(100)
        self.main_video_frames_slider.setValue(2)
        self.main_video_frames_slider.setTickPosition(
            QtWidgets.QSlider.TicksBelow)
        self.main_video_frames_slider.setTickInterval(1)
        self.main_video_frames_slider.setMaximumWidth(1000)
        self.main_video_frames_slider.valueChanged.connect(
            self.main_video_frames_slider_changed)
        self.main_video_frames_label_1 = QtWidgets.QLabel()
        self.main_video_frames_label_2 = QtWidgets.QLabel()
        # make the label text bigger and bold
        self.main_video_frames_label_1.setStyleSheet(
            "QLabel { font-size: 12pt; font-weight: bold; }")
        self.main_video_frames_label_2.setStyleSheet(
            "QLabel { font-size: 12pt; font-weight: bold; }")
        # labels should show the current frame number / total number of frames and cuurent time / total time
        self.videoControls.addWidget(self.main_video_frames_label_1)
        self.videoControls.addWidget(self.main_video_frames_slider)
        self.videoControls.addWidget(self.main_video_frames_label_2)

        # now we start the videocontrols_2 toolbar widgets

        

        # add the slider to control the video frame
        self.frames_to_track_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frames_to_track_slider.setMinimum(2)
        self.frames_to_track_slider.setMaximum(100)
        self.frames_to_track_slider.setValue(4)
        self.frames_to_track_slider.setTickPosition(
            QtWidgets.QSlider.TicksBelow)
        self.frames_to_track_slider.setTickInterval(1)
        self.frames_to_track_slider.setMaximumWidth(200)
        self.frames_to_track_slider.valueChanged.connect(
            self.frames_to_track_slider_changed)
        
        # add text input to control the slider
        self.frames_to_track_input = QtWidgets.QLineEdit()
        self.frames_to_track_input.setText("4")
        # make the font bigger
        self.frames_to_track_input.setStyleSheet(
            "QLineEdit { font-size: 10pt; }")
        self.frames_to_track_input.setMaximumWidth(50)
        self.frames_to_track_input.textChanged.connect(self.frames_to_track_input_changed)


        self.frames_to_track_label_before = QtWidgets.QLabel("Track for")
        self.frames_to_track_label_before.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; }")
        self.frames_to_track_label_after = QtWidgets.QLabel("frames")
        self.frames_to_track_label_after.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; }")
        self.videoControls_2.addWidget(self.frames_to_track_label_before)
        self.videoControls_2.addWidget(self.frames_to_track_input)
        self.videoControls_2.addWidget(self.frames_to_track_label_after)
        self.videoControls_2.addWidget(self.frames_to_track_slider)
        self.frames_to_track_slider.setValue(10)

        self.track_dropdown = QtWidgets.QComboBox()
        self.track_dropdown.addItems([f"Track for selected frames", "Track Only assigned objects", "Track Full Video"])        
        self.track_dropdown.setCurrentIndex(0)
        self.track_dropdown.currentIndexChanged.connect(self.track_dropdown_changed)
        self.videoControls_2.addWidget(self.track_dropdown)

        self.start_button = QtWidgets.QPushButton("Start Tracking")
        self.start_button.setIcon(
            QtGui.QIcon("labelme/icons/start.png"))
        # make the icon bigger
        self.start_button.setIconSize(QtCore.QSize(24, 24))
        self.start_button.setStyleSheet(self.buttons_text_style_sheet)
        self.start_button.clicked.connect(self.start_tracking_button_clicked)
        self.videoControls_2.addWidget(self.start_button)

        self.tracking_progress_bar_label = QtWidgets.QLabel()
        self.tracking_progress_bar_label.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; }")
        self.tracking_progress_bar_label.setText("Tracking Progress")
        self.videoControls_2.addWidget(self.tracking_progress_bar_label)

        self.tracking_progress_bar = QtWidgets.QProgressBar()
        self.tracking_progress_bar.setMaximumWidth(300)
        self.tracking_progress_bar.setMinimum(0)
        self.tracking_progress_bar.setMaximum(100)
        self.tracking_progress_bar.setValue(0)
        self.videoControls_2.addWidget(self.tracking_progress_bar)
        
        self.track_stop_button = QtWidgets.QPushButton()
        self.track_stop_button.setStyleSheet(
            "QPushButton {font-size: 10pt; margin: 2px 5px; padding: 2px 7px;font-weight: bold; background-color: #FF9090; color: #FFFFFF;} QPushButton:hover {background-color: #FF0000;} QPushButton:disabled {background-color: #7A7A7A;}")
        self.track_stop_button.setStyleSheet("QPushButton {font-size: 10pt; margin: 2px 5px; padding: 2px 7px;font-weight: bold; background-color: #FF0000; color: #FFFFFF;} QPushButton:hover {background-color: #FE4242;} QPushButton:disabled {background-color: #7A7A7A;}")
            
        self.track_stop_button.setText("Stop Tracking")
        self.track_stop_button.setIcon(
            QtGui.QIcon("labelme/icons/stop.png"))
        # make the icon bigger
        self.track_stop_button.setIconSize(QtCore.QSize(24, 24))
        # self.track_stop_button.setShortcut(self._config['shortcuts']['stop'])
        self.track_stop_button.setToolTip(
                f'Stop Tracking ({self._config["shortcuts"]["stop"]})')
        self.track_stop_button.pressed.connect(
            self.Escape_clicked)
        self.videoControls_2.addWidget(self.track_stop_button)
    

        # add 5 checkboxes to control the CURRENT ANNOATAION FLAGS including (bbox , id , class , mask , traj)
        self.bbox_checkBox = QtWidgets.QCheckBox()
        self.bbox_checkBox.setText("bbox")
        self.bbox_checkBox.setChecked(True)
        self.bbox_checkBox.stateChanged.connect(self.bbox_checkBox_changed)

        self.id_checkBox = QtWidgets.QCheckBox()
        self.id_checkBox.setText("id")
        self.id_checkBox.setChecked(True)
        self.id_checkBox.stateChanged.connect(self.id_checkBox_changed)

        self.class_checkBox = QtWidgets.QCheckBox()
        self.class_checkBox.setText("class")
        self.class_checkBox.setChecked(True)
        self.class_checkBox.stateChanged.connect(self.class_checkBox_changed)
        
        self.conf_checkBox = QtWidgets.QCheckBox()
        self.conf_checkBox.setText("confidence")
        self.conf_checkBox.setChecked(True)
        self.conf_checkBox.stateChanged.connect(self.conf_checkBox_changed)

        self.mask_checkBox = QtWidgets.QCheckBox()
        self.mask_checkBox.setText("mask")
        self.mask_checkBox.setChecked(True)
        self.mask_checkBox.stateChanged.connect(self.mask_checkBox_changed)

        self.traj_checkBox = QtWidgets.QCheckBox()
        self.traj_checkBox.setText("trajectories")
        self.traj_checkBox.setChecked(False)
        self.traj_checkBox.stateChanged.connect(self.traj_checkBox_changed)

        # make qlineedit to alter the  self.CURRENT_ANNOATAION_TRAJECTORIES['length']  value
        self.trajectory_length_lineEdit = QtWidgets.QLineEdit()
        self.trajectory_length_lineEdit.setText(str(30))
        self.trajectory_length_lineEdit.setMaximumWidth(50)
        self.trajectory_length_lineEdit.editingFinished.connect(
            self.trajectory_length_lineEdit_changed)


        self.polygons_visable_checkBox = QtWidgets.QCheckBox()
        self.polygons_visable_checkBox.setText("show polygons")
        self.polygons_visable_checkBox.setChecked(True)
        self.polygons_visable_checkBox.stateChanged.connect(
            self.polygons_visable_checkBox_changed)

        self.vis_options = [self.id_checkBox, self.class_checkBox, self.bbox_checkBox, self.mask_checkBox, self.polygons_visable_checkBox, self.traj_checkBox, self.trajectory_length_lineEdit, self.conf_checkBox]
        # add to self.vis_dock
        self.vis_widget.setLayout(QtWidgets.QGridLayout())
        self.vis_widget.layout().setContentsMargins(10, 10, 25, 10)  # set padding
        self.vis_widget.layout().addWidget(self.id_checkBox, 0, 0)
        self.vis_widget.layout().addWidget(self.class_checkBox, 0, 1)
        self.vis_widget.layout().addWidget(self.bbox_checkBox, 1, 0)
        self.vis_widget.layout().addWidget(self.mask_checkBox, 1, 1)
        self.vis_widget.layout().addWidget(self.traj_checkBox, 2, 0)
        self.vis_widget.layout().addWidget(self.trajectory_length_lineEdit, 2, 1)
        self.vis_widget.layout().addWidget(self.polygons_visable_checkBox, 3, 0)
        self.vis_widget.layout().addWidget(self.conf_checkBox, 3, 1)

        for option in self.vis_options:
            option.setEnabled(False)


        # save current frame
        self.update_current_frame_annotation_button = QtWidgets.QPushButton()
        self.update_current_frame_annotation_button.setStyleSheet(
            self.buttons_text_style_sheet)
        self.update_current_frame_annotation_button.setText(
            "Apply Changes")
        self.update_current_frame_annotation_button.setIcon(
            QtGui.QIcon("labelme/icons/done.png"))
        # make the icon bigger
        self.update_current_frame_annotation_button.setIconSize(QtCore.QSize(24, 24))
        self.update_current_frame_annotation_button.setShortcut(self._config['shortcuts']['update_frame'])
        self.update_current_frame_annotation_button.setToolTip(
                f'Apply changes on current frame ({self._config["shortcuts"]["update_frame"]})')
        self.update_current_frame_annotation_button.clicked.connect(
            self.update_current_frame_annotation_button_clicked)
        self.videoControls_2.addWidget(
            self.update_current_frame_annotation_button)

        # add a button to clear all video annotations
        self.clear_video_annotations_button = QtWidgets.QPushButton()
        self.clear_video_annotations_button.setStyleSheet(
            self.buttons_text_style_sheet)
        self.clear_video_annotations_button.setText("Clear All")
        self.clear_video_annotations_button.setIcon(
            QtGui.QIcon("labelme/icons/clear.png"))
        # make the icon bigger
        self.clear_video_annotations_button.setIconSize(QtCore.QSize(24, 24))
        self.clear_video_annotations_button.setShortcut(self._config['shortcuts']['clear_annotations'])
        self.clear_video_annotations_button.setToolTip(
                f'Clears Annotations from all frames ({self._config["shortcuts"]["clear_annotations"]})')
        self.clear_video_annotations_button.clicked.connect(
            self.clear_video_annotations_button_clicked)
        self.videoControls_2.addWidget(self.clear_video_annotations_button)

        # # add export as video button (currently Disabled)
        # self.export_as_video_button = QtWidgets.QPushButton()
        # self.export_as_video_button.setStyleSheet(
        #     self.buttons_text_style_sheet)
        # self.export_as_video_button.setText("Export as video")
        # self.export_as_video_button.setShortcut(self._config['shortcuts']['export_video'])
        # self.export_as_video_button.setToolTip(
        #         f'shortcut ({self._config["shortcuts"]["export_video"]})')
        # self.export_as_video_button.clicked.connect(
        #     self.export_as_video_button_clicked)
        # self.videoControls_3.addWidget(self.export_as_video_button)

        # add a button to clear all video annotations

        self.set_video_controls_visibility(False)

    def convert_QT_to_cv(self, incomingImage):
        return helpers.convert_QT_to_cv(incomingImage)

    def convert_cv_to_qt(self, cv_img):
        return helpers.convert_cv_to_qt(cv_img)

    def draw_bb_id(self, image, x, y, w, h, id, label, color=(0, 0, 255), thickness=1):
        return helpers.draw_bb_id(self.CURRENT_ANNOATAION_FLAGS, image, x, y, w, h, id, label, color, thickness)

    def draw_trajectories(self, img, shapes):
        return helpers.draw_trajectories(self.CURRENT_ANNOATAION_TRAJECTORIES,
                                         self.INDEX_OF_CURRENT_FRAME,
                                         self.CURRENT_ANNOATAION_FLAGS,
                                         img, shapes)

    def draw_bb_on_image(self, image, shapes, image_qt_flag=True):
        return helpers.draw_bb_on_image(self.CURRENT_ANNOATAION_TRAJECTORIES,
                                        self.INDEX_OF_CURRENT_FRAME,
                                        self.CURRENT_ANNOATAION_FLAGS,
                                        self.TOTAL_VIDEO_FRAMES,
                                        image, shapes, image_qt_flag)

    def waitWindow(self, visible=False, text=None):
        if visible:
            self.canvas.is_loading = True
            if text is not None:
                self.canvas.loading_text = text
        else:
            self.canvas.is_loading = False
            self.canvas.loading_text = "Loading..."
        self.canvas.repaint()
        QtWidgets.QApplication.processEvents()

    def set_sam_toolbar_enable(self, enable=False):
        for widget in self.sam_toolbar.children():
            try:
                widget.setEnabled(enable or widget.accessibleName(
                ) == 'sam_enhance_annotation_button' or widget.accessibleName() == 'sam_model_comboBox')
            except:
                pass

    def set_sam_toolbar_visibility(self, visible=False):
        if not visible:
            try:
                self.sam_clear_annotation_button_clicked()
                # self.sam_model_comboBox.setCurrentIndex(0)
                self.sam_buttons_colors("X")
            except:
                pass
        self.sam_toolbar.setVisible(visible)
        for widget in self.sam_toolbar.children():
            try:
                widget.setVisible(visible)
            except:
                pass

    def addSamControls(self):
        # add a toolbar
        self.sam_toolbar = QtWidgets.QToolBar()
        self.sam_toolbar.setMovable(True)
        self.sam_toolbar.setFloatable(True)
        self.sam_toolbar.setObjectName("sam_toolbar")
        self.sam_toolbar.setStyleSheet(
            "QToolBar#videoControls { border: 50px }")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.sam_toolbar)

        # add a label that says "sam model"
        self.sam_model_label = QtWidgets.QLabel()
        self.sam_model_label.setText("SAM Model")
        self.sam_model_label.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; }")
        self.sam_toolbar.addWidget(self.sam_model_label)

        # add a dropdown menu to select the sam model
        self.sam_model_comboBox = QtWidgets.QComboBox()
        self.sam_model_comboBox.setAccessibleName("sam_model_comboBox")
        # add a label inside the combobox that says "Select Model (SAM disabled)" and make it unselectable
        self.sam_model_comboBox.addItem("Select Model (SAM disabled)")
        self.sam_model_comboBox.addItems(self.sam_models())
        self.sam_model_comboBox.currentIndexChanged.connect(
            self.sam_model_comboBox_changed)
        self.sam_toolbar.addWidget(self.sam_model_comboBox)

        # add a button for adding a point in sam
        self.sam_add_point_button = QtWidgets.QPushButton()
        self.sam_add_point_button.setStyleSheet(
            "QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_add_point_button.setText("Add")
        # add icon to button
        self.sam_add_point_button.setIcon(
            QtGui.QIcon("labelme/icons/add.png"))
        # make the icon bigger
        self.sam_add_point_button.setIconSize(QtCore.QSize(24, 24))
        self.sam_add_point_button.setToolTip(
            f'Add point ({self._config["shortcuts"]["SAM_add_point"]})')
        # set shortcut
        self.sam_add_point_button.setShortcut(
            self._config["shortcuts"]["SAM_add_point"])
        self.sam_add_point_button.clicked.connect(
            self.sam_add_point_button_clicked)
        self.sam_toolbar.addWidget(self.sam_add_point_button)

        # add a button for removing a point in sam
        self.sam_remove_point_button = QtWidgets.QPushButton()
        self.sam_remove_point_button.setStyleSheet(
            "QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_remove_point_button.setText("Remove")
        # add icon to button
        self.sam_remove_point_button.setIcon(
            QtGui.QIcon("labelme/icons/remove.png"))
        # make the icon bigger
        self.sam_remove_point_button.setIconSize(QtCore.QSize(24, 24))
        # set hover text
        self.sam_remove_point_button.setToolTip(
            f'Remove Point ({self._config["shortcuts"]["SAM_remove_point"]})')
        # set shortcut
        self.sam_remove_point_button.setShortcut(
            self._config["shortcuts"]["SAM_remove_point"])
        self.sam_remove_point_button.clicked.connect(
            self.sam_remove_point_button_clicked)
        self.sam_toolbar.addWidget(self.sam_remove_point_button)

        # add a button for selecting a box in sam
        self.sam_select_rect_button = QtWidgets.QPushButton()
        self.sam_select_rect_button.setStyleSheet(
            "QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_select_rect_button.setText("Box")
        # add icon to button
        self.sam_select_rect_button.setIcon(
            QtGui.QIcon("labelme/icons/bbox.png"))
        # make the icon bigger
        self.sam_select_rect_button.setIconSize(QtCore.QSize(24, 24))

        # set hover text
        self.sam_select_rect_button.setToolTip(
            f'Add Box ({self._config["shortcuts"]["SAM_select_rect"]})')
        # set shortcut
        self.sam_select_rect_button.setShortcut(
            self._config["shortcuts"]["SAM_select_rect"])
        self.sam_select_rect_button.clicked.connect(
            self.sam_select_rect_button_clicked)
        self.sam_toolbar.addWidget(self.sam_select_rect_button)

        # add a point for clearing the annotation
        self.sam_clear_annotation_button = QtWidgets.QPushButton()
        self.sam_clear_annotation_button.setStyleSheet(
            "QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_clear_annotation_button.setText("Clear")
        # add icon to button
        self.sam_clear_annotation_button.setIcon(
            QtGui.QIcon("labelme/icons/clear.png"))
        # make the icon bigger
        self.sam_clear_annotation_button.setIconSize(QtCore.QSize(24, 24))
        self.sam_clear_annotation_button.setShortcut(self._config["shortcuts"]["SAM_clear"])
        self.sam_clear_annotation_button.setToolTip(
            f'Clear points and boxes ({self._config["shortcuts"]["SAM_clear"]})')
        self.sam_clear_annotation_button.clicked.connect(
            self.sam_clear_annotation_button_clicked)
        self.sam_toolbar.addWidget(self.sam_clear_annotation_button)

        # add a point of finish object annotation
        self.sam_finish_annotation_button = QtWidgets.QPushButton()
        self.sam_finish_annotation_button.setStyleSheet(
            "QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_finish_annotation_button.setText("Finish")
        # add icon to button
        self.sam_finish_annotation_button.setIcon(
            QtGui.QIcon("labelme/icons/done.png"))
        # make the icon bigger
        self.sam_finish_annotation_button.setIconSize(QtCore.QSize(24, 24))

        self.sam_finish_annotation_button.clicked.connect(
            self.sam_finish_annotation_button_clicked)
        # set hover text
        self.sam_finish_annotation_button.setToolTip(
            f'Finish Annotation ({self._config["shortcuts"]["SAM_finish_annotation"]} or ENTER)')
        # set shortcut
        self.sam_finish_annotation_button.setShortcut(
            self._config["shortcuts"]["SAM_finish_annotation"])
        self.sam_toolbar.addWidget(self.sam_finish_annotation_button)

        # add a point of close SAM
        self.sam_close_button = QtWidgets.QPushButton()
        self.sam_close_button.setStyleSheet(
            "QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_close_button.setText("Manual")
        # add icon to button
        self.sam_close_button.setIcon(
            QtGui.QIcon("labelme/icons/objects.png"))
        # make the icon bigger
        self.sam_close_button.setIconSize(QtCore.QSize(24, 24))

        self.sam_close_button.setShortcut(self._config["shortcuts"]["SAM_RESET"])
        self.sam_close_button.setToolTip(
            f'Return to Manual Mode ({self._config["shortcuts"]["SAM_RESET"]} or ESC)')
        self.sam_close_button.clicked.connect(
            self.sam_reset_button_clicked)
        self.sam_toolbar.addWidget(self.sam_close_button)

        # add a point of replace with SAM
        self.sam_enhance_annotation_button = QtWidgets.QPushButton()
        self.sam_enhance_annotation_button.setAccessibleName(
            "sam_enhance_annotation_button")
        self.sam_enhance_annotation_button.setStyleSheet(
            "QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_enhance_annotation_button.setText("Enhance Polygons")
        # add icon to button
        self.sam_enhance_annotation_button.setIcon(
            QtGui.QIcon("labelme/icons/SAM.png"))
        # make the icon bigger  
        self.sam_enhance_annotation_button.setIconSize(QtCore.QSize(24, 24))
        self.sam_enhance_annotation_button.setShortcut(self._config["shortcuts"]["SAM_enhance"])
        self.sam_enhance_annotation_button.setToolTip(
            f'Enhance Selected Polygons with SAM ({self._config["shortcuts"]["SAM_enhance"]})')
        self.sam_enhance_annotation_button.clicked.connect(
            self.sam_enhance_annotation_button_clicked)
        self.sam_toolbar.addWidget(self.sam_enhance_annotation_button)

        self.set_sam_toolbar_enable(False)
        self.sam_buttons_colors("x")

    def updateSamControls(self):
        # remove all items from the combobox
        self.sam_model_comboBox.clear()
        # call the sam_models function to get all the models
        self.sam_model_comboBox.addItem("Select Model (SAM disabled)")
        self.sam_model_comboBox.addItems(self.sam_models())
        # print("updated sam models")

    def sam_reset_button_clicked(self):
        self.sam_clear_annotation_button_clicked()
        self.createMode_options()

    def sam_enhance_annotation_button_clicked(self):
        if self.sam_model_comboBox.currentText() == "Select Model (SAM disabled)":
            helpers.OKmsgBox("SAM is disabled",
                             "SAM is disabled.\nPlease enable SAM.")
            return
        try:
            same_image = self.sam_predictor.check_image(
                self.CURRENT_FRAME_IMAGE)
        except:
            return
        
        toBeEnhanced = self.canvas.selectedShapes if len(self.canvas.selectedShapes) > 0 else self.canvas.shapes

        for shape in toBeEnhanced:
            try:
                self.canvas.shapes.remove(shape)
                # self.canvas.selectedShapes.remove(shape)
                self.remLabels([shape])
            except:
                return
            shapeX = self.convert_qt_shapes_to_shapes([shape])[0]
            x1, y1, x2, y2 = shapeX["bbox"]
            cur_bbox, cur_segment = self.sam_enhanced_bbox_segment(
                self.CURRENT_FRAME_IMAGE, [x1, y1, x2, y2], 1.2, max_itr=5, forSHAPE=True)
            shapeX["points"] = cur_segment
            shapeX = convert_shapes_to_qt_shapes([shapeX])[0]
            self.canvas.shapes.append(shapeX)
            # self.canvas.selectedShapes.append(shapeX)
            self.addLabel(shapeX)

        if self.current_annotation_mode == "video":
            self.update_current_frame_annotation_button_clicked()
        else:
            self.sam_clear_annotation_button_clicked()
            self.refresh_image_MODE()

        self.sam_buttons_colors("X")

    def sam_models(self):
        cwd = os.getcwd()
        with open(cwd + '/models_menu/sam_models.json') as f:
            data = json.load(f)
        # get all files in a directory
        files = os.listdir(cwd + '/mmdetection/checkpoints/')
        models = []
        for model in data:
            if model['checkpoint'].split('/')[-1] in files:
                models.append(model['name'])
        return models

    def sam_model_comboBox_changed(self):
        createFlag = self.canvas.mode == 0
        self.canvas.cancelManualDrawing()
        self.sam_clear_annotation_button_clicked()
        self.sam_buttons_colors("X")
        if self.sam_model_comboBox.currentText() == "Select Model (SAM disabled)":
            # self.createMode_options()
            self.set_sam_toolbar_enable(False)
            return
        model_type = self.sam_model_comboBox.currentText()
        self.waitWindow(
            visible=True, text=f'Please Wait.\n{model_type} is Loading...')
        with open('models_menu/sam_models.json') as f:
            data = json.load(f)
        checkpoint_path = ""
        for model in data:
            if model['name'] == model_type:
                checkpoint_path = model['checkpoint']
        # print(model_type, checkpoint_path, device)
        if checkpoint_path != "":
            self.sam_predictor = Sam_Predictor(
                model_type, checkpoint_path, device)
        try:
            self.sam_predictor.set_new_image(self.CURRENT_FRAME_IMAGE)
        except:
            print("please open an image first")
            self.waitWindow()
            return
        self.waitWindow()
        print("done loading model")
        
        if createFlag:
            self.createMode_options()
            if self.sam_last_mode == "point" :
                self.sam_add_point_button_clicked()
            elif self.sam_last_mode == "rectangle":
                self.sam_select_rect_button_clicked()
        else:
            self.setEditMode()

    def sam_buttons_colors(self, mode):
        
        setEnabled = False if self.sam_model_comboBox.currentText(
        ) == "Select Model (SAM disabled)" else True
        if not setEnabled:
            self.set_sam_toolbar_enable(setEnabled)
            self.set_sam_toolbar_colors("X")
            return

        self.set_sam_toolbar_colors(mode)
            
    def set_sam_toolbar_enable(self, setEnabled):
        self.sam_add_point_button.setEnabled(setEnabled)
        self.sam_remove_point_button.setEnabled(setEnabled)
        self.sam_select_rect_button.setEnabled(setEnabled)
        self.sam_clear_annotation_button.setEnabled(setEnabled)
        self.sam_finish_annotation_button.setEnabled(setEnabled)
        
    def set_sam_toolbar_colors(self, mode):
        red, green, blue, trans = "#2D7CFA;", "#2D7CFA;", "#2D7CFA;", "#4B515A;"
        hover_const = "QPushButton::hover { background-color : "
        disabled_const = "QPushButton:disabled { color : #7A7A7A} "
        style_sheet_const = "QPushButton { font-size: 10pt; font-weight: bold; color: #ffffff; background-color: "

        [add_style, add_hover] = [green, green] if mode == "add" else [trans, green]
        [remove_style, remove_hover] = [
            red, red] if mode == "remove" else [trans, red]
        [rect_style, rect_hover] = [
            green, green] if mode == "rect" else [trans, green]
        [clear_style, clear_hover] = [
            red, red] if mode == "clear" else [trans, red]
        [finish_style, finish_hover] = [
            blue, blue] if mode == "finish" else [trans, blue]
        [replace_style, replace_hover] = [
            blue, blue] if mode == "replace" else [trans, blue]

        self.sam_add_point_button.setStyleSheet(
            style_sheet_const + add_style + ";}" + hover_const + add_hover + ";}" + disabled_const)
        self.sam_remove_point_button.setStyleSheet(
            style_sheet_const + remove_style + ";}" + hover_const + remove_hover + ";}" + disabled_const)
        self.sam_select_rect_button.setStyleSheet(
            style_sheet_const + rect_style + ";}" + hover_const + rect_hover + ";}" + disabled_const)
        self.sam_clear_annotation_button.setStyleSheet(
            style_sheet_const + clear_style + ";}" + hover_const + clear_hover + ";}" + disabled_const)
        self.sam_finish_annotation_button.setStyleSheet(
            style_sheet_const + finish_style + ";}" + hover_const + finish_hover + ";}" + disabled_const)
        self.sam_enhance_annotation_button.setStyleSheet(
            style_sheet_const + replace_style + ";}" + hover_const + replace_hover + ";}" + disabled_const)
        

    def sam_add_point_button_clicked(self):
        self.canvas.cancelManualDrawing()
        self.sam_last_mode = "point"
        self.sam_buttons_colors("add")
        try:
            same_image = self.sam_predictor.check_image(
                self.CURRENT_FRAME_IMAGE)
        except:
            self.sam_buttons_colors("x")
            return
        if not same_image:
            self.sam_clear_annotation_button_clicked()
            self.sam_buttons_colors("add")
        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.canvas.SAM_mode = "add point"

    def sam_remove_point_button_clicked(self):
        self.canvas.cancelManualDrawing()
        self.sam_buttons_colors("remove")
        try:
            same_image = self.sam_predictor.check_image(
                self.CURRENT_FRAME_IMAGE)
        except:
            self.sam_buttons_colors("x")
            return
        if not same_image:
            self.sam_clear_annotation_button_clicked()
            self.sam_buttons_colors("remove")
        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.canvas.SAM_mode = "remove point"

    def sam_select_rect_button_clicked(self):
        self.canvas.cancelManualDrawing()
        self.sam_last_mode = "rectangle"
        self.sam_buttons_colors("rect")
        try:
            same_image = self.sam_predictor.check_image(
                self.CURRENT_FRAME_IMAGE)
        except:
            self.sam_buttons_colors("x")
            return
        if not same_image:
            self.sam_clear_annotation_button_clicked()
            self.sam_buttons_colors("rect")
        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.canvas.SAM_mode = "select rect"
        self.canvas.h_w_of_image = [
            self.CURRENT_FRAME_IMAGE.shape[0], self.CURRENT_FRAME_IMAGE.shape[1]]

    def sam_clear_annotation_button_clicked(self):
        self.canvas.cancelManualDrawing()
        self.sam_buttons_colors("clear")
        # print("sam clear annotation button clicked")
        self.canvas.SAM_coordinates = []
        self.canvas.SAM_mode = ""
        self.canvas.SAM_rect = []
        self.canvas.SAM_rects = []
        self.current_sam_shape = None
        try:
            self.sam_predictor.clear_logit()
        except:
            pass
        self.labelList.clear()
        self.CURRENT_SHAPES_IN_IMG = self.convert_qt_shapes_to_shapes(
            self.canvas.shapes)
        self.CURRENT_SHAPES_IN_IMG = self.check_sam_instance_in_shapes(
            self.CURRENT_SHAPES_IN_IMG)

        self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        # self.loadLabels(self.SAM_SHAPES_IN_IMAGE, replace=False)

        # later for confirmed sam instances
        # self.laodLabels(self.CURRENT_SAM_SHAPES_IN_IMG,replace=false)

    def sam_finish_annotation_button_clicked(self):
        self.canvas.cancelManualDrawing()
        self.sam_buttons_colors("finish")
        # return the cursor to normal
        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.canvas.SAM_coordinates = []
        self.canvas.SAM_rect = []
        self.canvas.SAM_rects = []
        self.canvas.SAM_mode = "finished"
        try:
            self.sam_predictor.clear_logit()
            if len(self.current_sam_shape) == 0:
                return
        except:
            if self.sam_last_mode == "point":
                self.sam_add_point_button_clicked()
            elif self.sam_last_mode == "rectangle":
                self.sam_select_rect_button_clicked()
            return

        self.labelList.clear()
        sam_qt_shape = convert_shapes_to_qt_shapes([self.current_sam_shape])[0]
        self.canvas.SAM_current = sam_qt_shape
        # print("sam qt shape", type(sam_qt_shape), "sam shape", type(self.current_sam_shape))
        self.canvas.finalise(SAM_SHAPE=True)
        # self.SAM_SHAPES_IN_IMAGE.append(self.current_sam_shape)
        self.CURRENT_SHAPES_IN_IMG = self.convert_qt_shapes_to_shapes(
            self.canvas.shapes)
        # for shape in self.CURRENT_SHAPES_IN_IMG:
        #     print(shape["label"])
        self.CURRENT_SHAPES_IN_IMG = self.check_sam_instance_in_shapes(
            self.CURRENT_SHAPES_IN_IMG)
        try:
            if self.current_sam_shape["group_id"] != -1:
                self.CURRENT_SHAPES_IN_IMG.append(self.current_sam_shape)

            self.rec_frame_for_id(
                self.current_sam_shape["group_id"], self.INDEX_OF_CURRENT_FRAME)

        except:
            pass
        self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        # self.loadLabels(self.SAM_SHAPES_IN_IMAGE, replace=False)
        # clear the predictor of the finished shape
        self.sam_predictor.clear_logit()
        self.canvas.SAM_coordinates = []
        # explicitly clear instead of being overriden by the next shape
        self.current_sam_shape = None
        self.canvas.SAM_current = None
        self.canvas.SAM_mode = ""

        if self.current_annotation_mode == "video":
            self.update_current_frame_annotation_button_clicked()
        else:
            self.canvas.shapes = convert_shapes_to_qt_shapes(
                self.CURRENT_SHAPES_IN_IMG)
            self.sam_clear_annotation_button_clicked()
            self.refresh_image_MODE()

    def check_sam_instance_in_shapes(self, shapes):
        if len(shapes) == 0:
            return []
        for shape in shapes:
            if shape["label"] == "SAM instance":
                # remove the shape from the list
                shapes.remove(shape)
        return shapes

    def SAM_rects_to_boxes(self, rects):
        return helpers.SAM_rects_to_boxes(rects)

    def SAM_points_and_labels_from_coordinates(self, coordinates):
        return helpers.SAM_points_and_labels_from_coordinates(coordinates)

    def run_sam_model(self):

        # print("run sam model")
        if self.sam_predictor is None or self.sam_model_comboBox.currentText() == "Select Model (SAM disabled)":
            print("please select a model")
            return

        try:
            same_image = self.sam_predictor.check_image(
                self.CURRENT_FRAME_IMAGE)
        except:
            self.sam_buttons_colors("x")
            return

        # prepre the input format for SAM

        input_points, input_labels = self.SAM_points_and_labels_from_coordinates(
            self.canvas.SAM_coordinates)
        input_boxes = self.SAM_rects_to_boxes(self.canvas.SAM_rects)

        mask, score = self.sam_predictor.predict(point_coords=input_points,
                                                 point_labels=input_labels,
                                                 box=input_boxes,
                                                 image=self.CURRENT_FRAME_IMAGE)

        points = self.sam_predictor.mask_to_polygons(mask)
        shape = self.sam_predictor.polygon_to_shape(points, score)
        # print(self.current_sam_shape)
        self.current_sam_shape = shape
        # if self.CURRENT_SHAPES_IN_IMG != []:
        # if self.CURRENT_SHAPES_IN_IMG[-1]["label"] == "SAM instance"  :
        # self.CURRENT_SHAPES_IN_IMG.pop()
        # self.CURRENT_SHAPES_IN_IMG.append(shape)
        # shapes  = convert_shapes_to_qt_shapes(self.CURRENT_SHAPES_IN_IMG)[0]
        # self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        # self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        # self.labelList.clear()
        # self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        self.labelList.clear()
        # self.canvas.loadPixmap(QtGui.QPixmap.fromImage(self.image))
        # we need it since it does replace=True on whatever was on the canvas

        # print(f'len of canvas shapes {len(self.canvas.shapes)}')
        # print(f'len of current shapes {len(self.CURRENT_SHAPES_IN_IMG)}')

        self.CURRENT_SHAPES_IN_IMG = self.convert_qt_shapes_to_shapes(
            self.canvas.shapes)
        self.CURRENT_SHAPES_IN_IMG = self.check_sam_instance_in_shapes(
            self.CURRENT_SHAPES_IN_IMG)

        self.CURRENT_SHAPES_IN_IMG.append(self.current_sam_shape)
        self.loadLabels(self.CURRENT_SHAPES_IN_IMG)

        # self.loadLabels(self.SAM_SHAPES_IN_IMAGE, replace=False)
        # self.loadLabels([self.current_sam_shape], replace=False)

        # print("done running sam model")

    def load_objects_from_json__json(self):
        if self.global_listObj != []:
            return self.global_listObj
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
        return helpers.load_objects_from_json__json(json_file_name, self.TOTAL_VIDEO_FRAMES)

    def load_objects_to_json__json(self, listObj):
        self.global_listObj = listObj
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
        helpers.load_objects_to_json__json(json_file_name, listObj)

    def load_objects_from_json__orjson(self):
        if self.global_listObj != []:
            return self.global_listObj
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
        return helpers.load_objects_from_json__orjson(json_file_name, self.TOTAL_VIDEO_FRAMES)

    def load_objects_to_json__orjson(self, listObj):
        self.global_listObj = listObj
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
        helpers.load_objects_to_json__orjson(json_file_name, listObj)


    # def assignVideShortcuts(self):
        
    #     """
    #     Summary:
    #         Assigns the shortcuts for the video mode.
    #     """
        
    #     shortcuts = [self._config["shortcuts"][x] for x in []]
        
    #     self.VideoShortcuts = [QtWidgets.QShortcut(QtGui.QKeySequence(shortcuts[i]), self) for i in range(len(shortcuts))]
        
    #     # self.VideoShortcuts[0].activated.connect(self.main_video_frames_slider_changed)
        
    # def disconnectVideoShortcuts(self):
        
    #     """
    #     Summary:
    #         Disconnects the shortcuts for the video mode in case of image or directory mode.
    #     """
        
    #     try:
    #         for shortcut in self.VideoShortcuts:
    #             shortcut.activated.disconnect()
    #     except:
    #         pass
        
# important parameters across the gui

# INDEX_OF_CURRENT_FRAME
# self.FRAMES_TO_SKIP
# frames to track
# self.TOTAL_VIDEO_FRAMES
# self.CURRENT_VIDEO_FPS   --> to be used to play the video at the correct speed
# self.CAP
# self.CLASS_NAMES_DICT
# self.CURRENT_FRAME_IMAGE
# self.CURRENT_VIDEO_NAME
# self.CURRENT_VIDEO_PATH
# self.CURRENT_SHAPES_IN_IMG

# self.CURRENT_ANNOATAION_FLAGS = {"traj" : False  ,
#                                 "bbox" : False  ,
#                                   "id" : False ,
#                                   "class" : True,
#                                   "mask" : True}
# to do
# remove the video processing tool bar in the other cases
