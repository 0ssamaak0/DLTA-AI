# -*- coding: utf-8 -*-
import functools
import json
import math
import os
import os.path as osp
import re
import traceback
import webbrowser
# import asyncio
# import PyQt5
# from qtpy.QtCore import Signal, Slot
import imgviz
from matplotlib import pyplot as plt
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtCore import QThread
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

from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.detection.core import Detections
from trackers.multi_tracker_zoo import create_tracker
# from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.ops import Profile # non_max_suppression, scale_boxes, process_mask, process_mask_native
from ultralytics.yolo.utils.torch_utils import select_device

import torch
import numpy as np
import cv2
from pathlib import Path

# import time
# import threading
import warnings
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


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
ROOT = FILE.parents[0]  # inside labelme package
print(f'\n\n ROOT: {ROOT}\n\n')
# now get go up one more level to the root of the repo
ROOT = ROOT.parents[0]
print(f'\n\n ROOT: {ROOT}\n\n')

# WEIGHTS = ROOT / 'weights'

# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
#     sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print(f'\n\n ROOT: {ROOT}\n\n')
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
        self.buttons_text_style_sheet = "QPushButton {font-size: 10pt; margin: 2px 5px; padding: 2px 7px; border: 2px solid; border-radius:10px; font-weight: bold; background-color: #0d69f5; color: #FFFFFF;} QPushButton:hover {background-color: #4990ED;}"

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
        self.intelligenceHelper = Intelligence(self)
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

        self.setCentralWidget(scrollArea)

        # for Export
        self.target_directory = ""
        self.save_path = ""

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
                                         "polygons": True}
        self.CURRENT_ANNOATAION_TRAJECTORIES = {}
        # keep it like that, don't change it
        self.CURRENT_ANNOATAION_TRAJECTORIES['length'] = 30
        # keep it like that, don't change it
        self.CURRENT_ANNOATAION_TRAJECTORIES['alpha'] = 0.70
        self.CURRENT_SHAPES_IN_IMG = []
        self.config = {'deleteDefault': "this frame only",
                       'interpolationDefault': "interpolate only missed frames between detected frames",
                       'creationDefault': "Create new shape (ie. not detected before)",
                       'EditDefault': "Edit only this frame",
                       'toolMode': 'video'}
        self.key_frames = {}
        # make CLASS_NAMES_DICT a dictionary of coco class names
        # self.CLASS_NAMES_DICT =
        # self.frame_number = 0
        self.INDEX_OF_CURRENT_FRAME = 1
        # # for image annotation
        # self.last_file_opened = ""



        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
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
            self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("&Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "opendir",
            self.tr(u"Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr(u"Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
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
            self.tr("Save labels to file"),
            enabled=False,
        )
        export = action(
            self.tr("&Export"),
            self.exportData,
            shortcuts["export"],
            "export",
            self.tr(u"Export annotations to COCO format"),
            enabled=False,
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
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "objects",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
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
            "cancel",
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
            icon="edit",
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

        # help = action(
        #     self.tr("&Tutorial"),
        #     self.tutorial,
        #     icon="help",
        #     tip=self.tr("Show tutorial page"),
        # )

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
            self.tr("&Show Cross Line"),
            self.enable_show_cross_line,
            tip=self.tr("Show cross line for mouse position"),
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
            self.tr("&Edit Label"),
            self.editLabel,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )
        interpolate = action(
            self.tr("&Interpolate"),
            self.interpolateMENU,
            shortcuts["interpolate"],
            "edit",
            self.tr("Interpolate the selected polygon"),
            enabled=True,
        )
        mark_as_key = action(
            self.tr("&Mark as key"),
            self.mark_as_key,
            shortcuts["mark_as_key"],
            "edit",
            self.tr("Mark this frame as KEY for interpolation"),
            enabled=True,
        )
        scale = action(
            self.tr("&Scale"),
            self.scaleMENU,
            shortcuts["scale"],
            "edit",
            self.tr("Scale the selected polygon"),
            enabled=True,
        )
        update_curr_frame = action(
            self.tr("&Update current frame"),
            self.update_current_frame_annotation_button_clicked,
            shortcuts["save"],
            "done",
            self.tr("Update frame"),
            enabled=False,
        )
        ignore_changes = action(
            self.tr("&Ignore changes"),
            self.main_video_frames_slider_changed,
            shortcuts["undo"],
            "delete",
            self.tr("Ignore unsaved changes"),
            enabled=False,
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
            self.tr("Run the model for the Current File "),
            self.annotate_one,
            None,
            None,
            self.tr("Run the model for the Current File")
        )

        annotate_batch_action = action(
            self.tr("Run the model for All Files"),
            self.annotate_batch,
            None,
            None,
            self.tr("Run the model for All Files")
        )
        set_conf_threshold = action(
            self.tr("Set Confidence threshold for the model"),
            self.setConfThreshold,
            None,
            None,
            self.tr("Set Confidence threshold for the model")
        )
        set_iou_threshold = action(
            self.tr("Set IOU threshold for the model NMS"),
            self.setIOUThreshold,
            None,
            None,
            self.tr("Set IOU threshold for the model NMS")
        )
        select_classes = action(
            self.tr("select the classes to be annotated"),
            self.selectClasses,
            None,
            None,
            self.tr("select the classes to be annotated")
        )
        merge_segmentation_models = action(
            self.tr("merge segmentation models"),
            self.mergeSegModels,
            None,
            None,
            self.tr("merge segmentation models")
        )
        runtime_data = action(
            self.tr("show the runtime data"),
            self.showRuntimeData,
            None,
            None,
            self.tr("show the runtime data")
        )
        sam = action(
            self.tr("show/hide toolbar"),
            self.Segment_anything,
            None,
            None,
            self.tr("show/hide toolbar")
        )
        openVideo = action(
            self.tr("Open &Video"),
            self.openVideo,
            None,
            "video",
            self.tr("Open a video file"),
        )    

        # Lavel list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
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
            fileMenuActions=(open_, opendir, save, saveAs, close, quit),
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
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                editMode,
                edit,
                interpolate,
                mark_as_key,
                scale,
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

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            intelligence=self.menu(self.tr("&Auto Annotation")),
            sam = self.menu(self.tr("&Segment Anything")),
            # help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            saved_models=QtWidgets.QMenu(self.tr("Select Segmentation model")),
            tracking_models=QtWidgets.QMenu(self.tr("Select Tracking model")),
            labelList=labelMenu,

        )
        utils.addActions(
            self.menus.sam,
            (
                sam,
            ),
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        # utils.addActions(self.menus.help, (help,))
        utils.addActions(self.menus.intelligence,
                         (annotate_one_action,
                          annotate_batch_action,
                          set_conf_threshold,
                          set_iou_threshold,
                          self.menus.saved_models,
                          self.menus.tracking_models,
                          select_classes,
                          merge_segmentation_models,
                          runtime_data
                          )
                         )

        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
                show_cross_line,
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
            save,
            export,
            deleteFile,
            None,
            createMode,
            editMode,
            copy,
            delete,
            undo,
            brightnessContrast,
            None,
            zoom,
            fitWidth,
            openNextImg,
            openPrevImg,

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

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()


    def update_saved_models_json(self):
        cwd = os.getcwd()
        checkpoints_dir = cwd + "/mmdetection/checkpoints/"
        # list all the files in the checkpoints directory
        files = os.listdir(checkpoints_dir)
        with open(cwd + '/models_menu/models_json.json') as f:
            models_json = json.load(f)
        saved_models = {}
        # saved_models["YOLOv8x"] = {"checkpoint": "yolov8x-seg.pt", "config": "none"}
        for model in models_json:
            if model["Checkpoint"].split("/")[-1] in os.listdir(checkpoints_dir):
                saved_models[model["Model Name"]] = {"id": model["id"] ,"checkpoint": model["Checkpoint"], "config": model["Config"]}
        
        with open (cwd + "/saved_models.json", "w") as f:
            json.dump(saved_models, f, indent=4)




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
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            label_file = osp.splitext(self.imagePath)[0] + ".json"
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
        self.CURRENT_SHAPES_IN_IMG = []
        self.SAM_SHAPES_IN_IMAGE = []
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

    def setEditMode(self):
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
            icon = utils.newIcon("labels")
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
                icon = utils.newIcon("labels")
                action = QtWidgets.QAction(
                    icon, "&%d %s" % (i + 1, model_name), self)
                action.triggered.connect(functools.partial(
                    self.change_curr_model, model_name))
                menu.addAction(action)
                i += 1

            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "Model Explorer", self)
            # when the action is triggered, preview an Qmessagebox with ERROR
            action.triggered.connect(self.model_explorer)
            menu.addAction(action)

        self.add_tracking_models_menu()

    def add_tracking_models_menu(self):
        menu2 = self.menus.tracking_models
        menu2.clear()

        icon = utils.newIcon("labels")
        action = QtWidgets.QAction(
            icon, "1 Byte track", self)
        action.triggered.connect(
            lambda: self.update_tracking_method('bytetrack'))
        menu2.addAction(action)

        icon = utils.newIcon("labels")
        action = QtWidgets.QAction(
            icon, "2 Strong SORT", self)
        action.triggered.connect(
            lambda: self.update_tracking_method('strongsort'))
        menu2.addAction(action)

        icon = utils.newIcon("labels")
        action = QtWidgets.QAction(
            icon, "3 Deep SORT (lowest id switch) DEFAULT", self)
        action.triggered.connect(
            lambda: self.update_tracking_method('deepocsort'))
        menu2.addAction(action)

        icon = utils.newIcon("labels")
        action = QtWidgets.QAction(
            icon, "4 OC SORT", self)
        action.triggered.connect(lambda: self.update_tracking_method('ocsort'))
        menu2.addAction(action)

        icon = utils.newIcon("labels")
        action = QtWidgets.QAction(
            icon, "5 BoT SORT", self)
        action.triggered.connect(
            lambda: self.update_tracking_method('botsort'))
        menu2.addAction(action)

    def update_tracking_method(self, method = 'bytetrack'):
        self.tracking_method = method
        self.tracking_config  = ROOT / 'trackers' / method / 'configs' / (method + '.yaml')
        with torch.no_grad():
            device = select_device('')
            print(f'tracking method {self.tracking_method} , config {self.tracking_config} , reid {reid_weights} , device {device} , half {False}')
            self.tracker = create_tracker(
                    self.tracking_method, self.tracking_config, reid_weights, device, False)
            if hasattr(self.tracker, 'model'):
                if hasattr(self.tracker.model, 'warmup'):
                    self.tracker.model.warmup()
                    # print('Warmup done')
                
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

        if self.config['toolMode'] == 'image':
            self.toggleDrawMode(False, createMode="polygon")
            return

        self.update_current_frame_annotation()

        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Choose Creation Options")
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.resize(250, 100)

        layout = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel("Choose Creation Options")
        layout.addWidget(label)

        new = QtWidgets.QRadioButton(
            "Create new shape (ie. not detected before)")
        existing = QtWidgets.QRadioButton(
            "Copy existing shape (ie. detected before)")

        shape_id = QtWidgets.QSpinBox()
        shape_id.setMinimum(1)
        shape_id.setMaximum(1000000)
        shape_id.valueChanged.connect(lambda: existing.toggle())

        if self.config['creationDefault'] == 'Create new shape (ie. not detected before)':
            new.toggle()
        if self.config['creationDefault'] == 'Copy existing shape (ie. detected before)':
            existing.toggle()

        new.toggled.connect(lambda: self.config.update(
            {'creationDefault': 'Create new shape (ie. not detected before)'}))
        existing.toggled.connect(lambda: self.config.update(
            {'creationDefault': 'Copy existing shape (ie. detected before)'}))

        layout.addWidget(new)
        layout.addWidget(existing)
        layout.addWidget(shape_id)

        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok)
        buttonBox.accepted.connect(dialog.accept)
        layout.addWidget(buttonBox)
        dialog.setLayout(layout)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            if self.config['creationDefault'] == 'Create new shape (ie. not detected before)':
                self.toggleDrawMode(False, createMode="polygon")
            elif self.config['creationDefault'] == 'Copy existing shape (ie. detected before)':
                self.copy_existing_shape(shape_id.value())

    def copy_existing_shape(self, shape_id):
        listobj = self.load_objects_from_json()
        prev_frame = -1
        prev_shape = None
        next_frame = self.TOTAL_VIDEO_FRAMES + 2
        next_shape = None
        for i in range(len(listobj)):
            listobjframe = listobj[i]['frame_idx']
            if listobjframe > next_frame or listobjframe < prev_frame:
                continue
            for object_ in listobj[i]['frame_data']:
                if object_['tracker_id'] == shape_id:
                    if listobjframe > self.INDEX_OF_CURRENT_FRAME and listobjframe < next_frame:
                        next_frame = listobjframe
                        next_shape = object_
                    if listobjframe < self.INDEX_OF_CURRENT_FRAME and listobjframe > prev_frame:
                        prev_frame = listobjframe
                        prev_shape = object_

        shape = None

        if prev_shape is None and next_shape is None:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("No shape found with that ID")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return
        elif prev_shape is None:
            shape = next_shape
        elif next_shape is None:
            shape = prev_shape
        elif self.INDEX_OF_CURRENT_FRAME - prev_frame < next_frame - self.INDEX_OF_CURRENT_FRAME:
            shape = prev_shape
        else:
            shape = next_shape

        flag = True
        exists = False
        for i in range(len(listobj)):

            listobjframe = listobj[i]['frame_idx']
            if listobjframe != self.INDEX_OF_CURRENT_FRAME:
                continue
            exists = True
            for object_ in listobj[i]['frame_data']:
                if object_['tracker_id'] == shape_id:
                    flag = False
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText(
                        "A Shape with that ID already exists in this frame.")
                    msg.setWindowTitle("ID already exists")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    break

            if flag:
                listobj[i]['frame_data'].append(shape)
                break

        if not exists:
            frame = {'frame_idx': self.INDEX_OF_CURRENT_FRAME,
                     'frame_data': [shape]}
            listobj.append(frame)
            listobj = sorted(listobj, key=lambda k: k['frame_idx'])

        self.load_objects_to_json(listobj)
        self.calc_trajectory_when_open_video()
        self.main_video_frames_slider_changed()

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
        text, flags, old_group_id, content = self.labelDialog.popUp(
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
        if shape.group_id is None:
            item.setText(shape.label)
        else:
            if self.config['toolMode'] == 'image':
                item.setText(f' ID {shape.group_id}: {shape.label}')
                self.setDirty()
                if not self.uniqLabelList.findItemsByLabel(shape.label):
                    item = QtWidgets.QListWidgetItem()
                    item.setData(Qt.UserRole, shape.label)
                    self.uniqLabelList.addItem(item)
                return
            ###########################################################
            all_frames = False
            only_this_frame = False
            if old_group_id != new_group_id:
                dialog = QtWidgets.QDialog()
                dialog.setWindowTitle("Choose Edit Options")
                dialog.setWindowModality(Qt.ApplicationModal)
                dialog.resize(250, 100)

                layout = QtWidgets.QVBoxLayout()

                label = QtWidgets.QLabel("Choose Edit Options")
                layout.addWidget(label)

                only = QtWidgets.QRadioButton("Edit only this frame")
                all = QtWidgets.QRadioButton("Edit all frames with this ID")

                if self.config['EditDefault'] == 'Edit only this frame':
                    only.toggle()
                if self.config['EditDefault'] == 'Edit all frames with this ID':
                    all.toggle()

                only.toggled.connect(lambda: self.config.update(
                    {'EditDefault': 'Edit only this frame'}))
                all.toggled.connect(lambda: self.config.update(
                    {'EditDefault': 'Edit all frames with this ID'}))

                layout.addWidget(only)
                layout.addWidget(all)

                buttonBox = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.Ok)
                buttonBox.accepted.connect(dialog.accept)
                layout.addWidget(buttonBox)
                dialog.setLayout(layout)
                result = dialog.exec_()
                if result == QtWidgets.QDialog.Accepted:
                    all_frames = True if self.config['EditDefault'] == 'Edit all frames with this ID' else False
                    only_this_frame = True if self.config['EditDefault'] == 'Edit only this frame' else False

            listObj = self.load_objects_from_json()
            
            for i in range(len(listObj)):
                listObjframe = listObj[i]['frame_idx']
                if only_this_frame and listObjframe != self.INDEX_OF_CURRENT_FRAME:
                    continue
                for object_ in listObj[i]['frame_data']:
                    if object_['tracker_id'] == new_group_id:
                        listObj[i]['frame_data'].remove(object_)
                        object_['class_name'] = shape.label
                        object_['confidence'] = str(1.0)
                        object_['class_id'] = coco_classes.index(
                            shape.label) if shape.label in coco_classes else -1
                        listObj[i]['frame_data'].append(object_)

                    elif object_['tracker_id'] == old_group_id:
                        listObj[i]['frame_data'].remove(object_)
                        object_['class_name'] = shape.label
                        object_['confidence'] = str(1.0)
                        object_['class_id'] = coco_classes.index(
                            shape.label) if shape.label in coco_classes else -1
                        object_['tracker_id'] = new_group_id
                        listObj[i]['frame_data'].append(object_)

                sum = 0
                for object_ in listObj[i]['frame_data']:
                    if object_['tracker_id'] == new_group_id:
                        sum += 1
                        if sum > 1:
                            shape.group_id = old_group_id
                            msg = QtWidgets.QMessageBox()
                            msg.setIcon(QtWidgets.QMessageBox.Information)
                            msg.setText(
                                f"Two shapes with the same ID exists in at least one frame.\nApparantly, a shape with ID ({new_group_id}) already exists with another shape with ID ({old_group_id}) like in frame ({listObjframe}) and the edit will result in two shapes with the same ID ({new_group_id}).\n\n The edit is NOT performed.")
                            msg.setWindowTitle("ID already exists")
                            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                            msg.exec_()
                            return

            listObj = sorted(listObj, key=lambda k: k['frame_idx'])
            self.load_objects_to_json(listObj)
            self.calc_trajectory_when_open_video()
            self.main_video_frames_slider_changed()
            ###########################################################

    def interpolateMENU(self, item=None):
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
        text, flags, group_id, content = self.labelDialog.popUp(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
            content=shape.content,
            skip_flag=True
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
        shape.group_id = group_id
        shape.content = content
        if shape.group_id is None:
            item.setText(shape.label)
        else:
            ###########################################################
            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("Choose Interpolation Options")
            dialog.setWindowModality(Qt.ApplicationModal)
            dialog.resize(250, 100)

            layout = QtWidgets.QVBoxLayout()

            label = QtWidgets.QLabel("Choose Interpolation Options")
            layout.addWidget(label)

            only_missed = QtWidgets.QRadioButton(
                "interpolate only missed frames between detected frames")
            only_edited = QtWidgets.QRadioButton(
                "interpolate all frames between your KEY frames")

            if self.config['interpolationDefault'] == 'interpolate only missed frames between detected frames':
                only_missed.toggle()
            if self.config['interpolationDefault'] == 'interpolate all frames between your KEY frames':
                only_edited.toggle()

            only_missed.toggled.connect(lambda: self.config.update(
                {'interpolationDefault': 'interpolate only missed frames between detected frames'}))
            only_edited.toggled.connect(lambda: self.config.update(
                {'interpolationDefault': 'interpolate all frames between your KEY frames'}))

            layout.addWidget(only_missed)
            layout.addWidget(only_edited)

            buttonBox = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok)
            buttonBox.accepted.connect(dialog.accept)
            layout.addWidget(buttonBox)
            dialog.setLayout(layout)
            result = dialog.exec_()
            if result == QtWidgets.QDialog.Accepted:
                only_edited = True if self.config['interpolationDefault'] == 'interpolate all frames between your KEY frames' else False
                if only_edited:
                    print(self.key_frames['id_' + str(shape.group_id)])
                    try:
                        if len(self.key_frames['id_' + str(shape.group_id)]) == 1:
                            shape.group_id = 1/0
                        else: pass
                    except:
                        msg = QtWidgets.QMessageBox()
                        msg.setIcon(QtWidgets.QMessageBox.Information)
                        msg.setText(
                            f"No KEY frames found for this shape ID ({shape.group_id}).\n    ie. less than 2 key frames\n The interpolation is NOT performed.")
                        msg.setWindowTitle("No KEY frames found")
                        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                        msg.exec_()
                        return
                self.interpolate(id=shape.group_id, only_edited=only_edited)
            ###########################################################
        self.setDirty()
        if not self.uniqLabelList.findItemsByLabel(shape.label):
            item = QtWidgets.QListWidgetItem()
            item.setData(Qt.UserRole, shape.label)
            self.uniqLabelList.addItem(item)

    def mark_as_key(self, item=None):
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
        text, flags, group_id, content = self.labelDialog.popUp(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
            content=shape.content,
            skip_flag=True
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
        shape.group_id = group_id
        shape.content = content
        if shape.group_id is None:
            item.setText(shape.label)
        else:
            id = shape.group_id
            try:
                self.key_frames['id_' +
                                str(id)].append(self.INDEX_OF_CURRENT_FRAME)
            except:
                self.key_frames['id_' +
                                str(id)] = [self.INDEX_OF_CURRENT_FRAME]
            self.main_video_frames_slider_changed()
        self.setDirty()
        if not self.uniqLabelList.findItemsByLabel(shape.label):
            item = QtWidgets.QListWidgetItem()
            item.setData(Qt.UserRole, shape.label)
            self.uniqLabelList.addItem(item)

    def interpolate_ONLYedited(self, id):
        key_frames = self.key_frames['id_' + str(id)]
        first_frame_idx = np.min(key_frames)
        last_frame_idx = np.max(key_frames)

        listObj = self.load_objects_from_json()
        records = [None for i in range(first_frame_idx, last_frame_idx + 1)]
        RECORDS = []
        for i in range(len(listObj)):
            listobjframe = listObj[i]['frame_idx']
            if (listobjframe < first_frame_idx or listobjframe > last_frame_idx):
                continue
            for object_ in listObj[i]['frame_data']:
                if (object_['tracker_id'] == id):
                    if not (listobjframe in key_frames):
                        listObj[i]['frame_data'].remove(object_)
                        break
                    records[listobjframe - first_frame_idx] = object_
                    break
        records_org = records.copy()

        first_iter_flag = True
        for i in range(len(records)):
            if (records[i] != None):
                RECORDS.append(records[i])
                continue
            if first_iter_flag:
                first_iter_flag = False
                prev_idx = i - 1
            prev = records[i - 1]
            current = prev

            next = records[i + 1]
            next_idx = i + 1
            for j in range(i + 1, len(records)):
                if (records[j] != None):
                    next = records[j]
                    next_idx = j
                    break
            cur_bbox = ((next_idx - i) / (next_idx - prev_idx)) * np.array(records[prev_idx]['bbox']) + (
                (i - prev_idx) / (next_idx - prev_idx)) * np.array(records[next_idx]['bbox'])
            cur_bbox = [int(cur_bbox[i]) for i in range(len(cur_bbox))]

            prev_segment = prev['segment']
            next_segment = next['segment']
            if len(prev_segment) != len(next_segment):
                biglen = max(len(prev_segment), len(next_segment))
                prev_segment = self.handlePoints(prev_segment, biglen)
                next_segment = self.handlePoints(next_segment, biglen)
            (prev_segment, next_segment) = self.allign(
                prev_segment, next_segment)
            prev['segment'] = prev_segment
            next['segment'] = next_segment

            cur_segment = ((next_idx - i) / (next_idx - prev_idx)) * np.array(records[prev_idx]['segment']) + (
                (i - prev_idx) / (next_idx - prev_idx)) * np.array(records[next_idx]['segment'])
            cur_segment = [[int(sublist[0]), int(sublist[1])]
                           for sublist in cur_segment]
            current['bbox'] = cur_bbox
            current['segment'] = cur_segment

            records[i] = current.copy()
            RECORDS.append(records[i])

        appended_frames = []
        for i in range(len(listObj)):
            listobjframe = listObj[i]['frame_idx']
            if (listobjframe < first_frame_idx or listobjframe > last_frame_idx):
                continue
            appended = records_org[listobjframe - first_frame_idx]
            if appended == None:
                appended = RECORDS[max(listobjframe - first_frame_idx - 1, 0)]
                listObj[i]['frame_data'].append(appended)
            appended_frames.append(listobjframe)

        for frame in range(first_frame_idx, last_frame_idx + 1):
            if (frame not in appended_frames):
                listObj.append({'frame_idx': frame, 'frame_data': [
                               RECORDS[max(frame - first_frame_idx - 1, 0)]]})
        self.load_objects_to_json(listObj)
        self.calc_trajectory_when_open_video()
        self.main_video_frames_slider_changed()

    def interpolate(self, id, only_edited=False):
        if only_edited:
            self.interpolate_ONLYedited(id)
            return

        first_frame_idx = -1
        last_frame_idx = -1
        listObj = self.load_objects_from_json()
        for i in range(len(listObj)):
            listobjframe = listObj[i]['frame_idx']
            frameobjects = listObj[i]['frame_data']
            for object_ in frameobjects:
                if (object_['tracker_id'] == id):
                    first_frame_idx = min(
                        first_frame_idx, listobjframe) if first_frame_idx != -1 else listobjframe
                    last_frame_idx = max(
                        last_frame_idx, listobjframe) if last_frame_idx != -1 else listobjframe
        if (first_frame_idx == -1 or last_frame_idx == -1):
            return
        if (first_frame_idx >= last_frame_idx):
            return

        records = [None for i in range(first_frame_idx, last_frame_idx + 1)]
        RECORDS = []
        for i in range(len(listObj)):
            listobjframe = listObj[i]['frame_idx']
            if (listobjframe < first_frame_idx or listobjframe > last_frame_idx):
                continue
            frameobjects = listObj[i]['frame_data']
            for object_ in frameobjects:
                if (object_['tracker_id'] == id):
                    records[listobjframe - first_frame_idx] = object_
                    break
        records_org = records.copy()

        first_iter_flag = True
        for i in range(len(records)):
            if (records[i] != None):
                RECORDS.append(records[i])
                continue
            if first_iter_flag:
                first_iter_flag = False
                prev_idx = i - 1
            prev = records[i - 1]
            current = prev

            next = records[i + 1]
            next_idx = i + 1
            for j in range(i + 1, len(records)):
                if (records[j] != None):
                    next = records[j]
                    next_idx = j
                    break
            cur_bbox = ((next_idx - i) / (next_idx - prev_idx)) * np.array(records[prev_idx]['bbox']) + (
                (i - prev_idx) / (next_idx - prev_idx)) * np.array(records[next_idx]['bbox'])
            cur_bbox = [int(cur_bbox[i]) for i in range(len(cur_bbox))]

            prev_segment = prev['segment']
            next_segment = next['segment']
            if len(prev_segment) != len(next_segment):
                biglen = max(len(prev_segment), len(next_segment))
                prev_segment = self.handlePoints(prev_segment, biglen)
                next_segment = self.handlePoints(next_segment, biglen)
            (prev_segment, next_segment) = self.allign(
                prev_segment, next_segment)
            prev['segment'] = prev_segment
            next['segment'] = next_segment

            cur_segment = ((next_idx - i) / (next_idx - prev_idx)) * np.array(records[prev_idx]['segment']) + (
                (i - prev_idx) / (next_idx - prev_idx)) * np.array(records[next_idx]['segment'])
            cur_segment = [[int(sublist[0]), int(sublist[1])]
                           for sublist in cur_segment]
            current['bbox'] = cur_bbox
            current['segment'] = cur_segment

            records[i] = current.copy()
            RECORDS.append(records[i])

        appended_frames = []
        for i in range(len(listObj)):
            listobjframe = listObj[i]['frame_idx']
            if (listobjframe < first_frame_idx or listobjframe > last_frame_idx):
                continue
            appended = records_org[listobjframe - first_frame_idx]
            if appended == None:
                appended = RECORDS[max(listobjframe - first_frame_idx - 1, 0)]
                listObj[i]['frame_data'].append(appended)
            appended_frames.append(listobjframe)

        for frame in range(first_frame_idx, last_frame_idx + 1):
            if (frame not in appended_frames):
                listObj.append({'frame_idx': frame, 'frame_data': [
                               RECORDS[max(frame - first_frame_idx - 1, 0)]]})
        self.load_objects_to_json(listObj)
        self.calc_trajectory_when_open_video()
        self.main_video_frames_slider_changed()

    def scaleMENU(self, item=None):
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
        text, flags, group_id, content = self.labelDialog.popUp(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
            content=shape.content,
            skip_flag=True
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
        shape.group_id = group_id
        shape.content = content
        if shape.group_id is None:
            item.setText(shape.label)
        else:

            shape = self.convert_qt_shapes_to_shapes([shape])[0]
            tracker_id = shape['group_id']
            bbox = [shape['bbox'][0], shape['bbox'][1], shape['bbox']
                    [2] - shape['bbox'][0], shape['bbox'][3] - shape['bbox'][1]]
            confidence = shape['content']
            class_name = shape['label']
            class_id = coco_classes.index(
                class_name) if class_name in coco_classes else -1
            segmentXYXY = shape['points']
            segment = [[segmentXYXY[i], segmentXYXY[i + 1]]
                       for i in range(0, len(segmentXYXY), 2)]
            SHAPE = {'tracker_id': tracker_id,
                     'bbox': bbox,
                     'confidence': confidence,
                     'class_name': class_name,
                     'class_id': class_id,
                     'segment': segment}

            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("Scaling")
            dialog.setWindowModality(Qt.ApplicationModal)
            dialog.resize(400, 400)

            layout = QtWidgets.QVBoxLayout()

            label = QtWidgets.QLabel(
                "Scaling object with ID: " + str(SHAPE['tracker_id']) + "\n ")
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
            xSlider.valueChanged.connect(lambda: self.scale(
                SHAPE, xSlider.value(), ySlider.value()))

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
            ySlider.valueChanged.connect(lambda: self.scale(
                SHAPE, xSlider.value(), ySlider.value()))

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
            if result == QtWidgets.QDialog.Accepted:
                return
            else:
                self.scale(SHAPE, 100, 100)
                return

    def scale(self, oldshape, ratioX, ratioY):
        ratioX = ratioX / 100
        ratioY = ratioY / 100
        shape = oldshape.copy()
        segment = shape['segment']
        tr0, tr1, w, h = shape['bbox']
        segment = [[(p[0] - tr0) * ratioX + tr0, (p[1] - tr1)
                    * ratioY + tr1] for p in segment]
        w = w * ratioX
        h = h * ratioY
        shape['segment'] = segment
        shape['bbox'] = [tr0, tr1, w, h]

        listobj = self.load_objects_from_json()
        for i in range(len(listobj)):
            listobjframe = listobj[i]['frame_idx']
            if listobjframe != self.INDEX_OF_CURRENT_FRAME:
                continue
            for objectt in listobj[i]['frame_data']:
                if objectt['tracker_id'] == shape['tracker_id']:
                    objectt['segment'] = shape['segment']
                    objectt['bbox'] = self.get_bbox_xywh(shape['segment'])
                    break

        self.load_objects_to_json(listobj)
        self.calc_trajectory_when_open_video()
        self.main_video_frames_slider_changed()

    def get_bbox_xywh(self, segment):
        segment = np.array(segment)
        x0 = np.min(segment[:, 0])
        y0 = np.min(segment[:, 1])
        x1 = np.max(segment[:, 0])
        y1 = np.max(segment[:, 1])
        return [x0, y0, x1, y1]

    def addPoints(self, shape, n):
        res = shape.copy()
        sub = 1.0 * n / (len(shape) - 1)
        if sub == 0:
            return res
        if sub < 1:
            res = []
            res.append(shape[0])
            flag = True
            for i in range(len(shape) - 1):
                dif = [shape[i + 1][0] - shape[i][0],
                       shape[i + 1][1] - shape[i][1]]
                newPoint = [shape[i][0] + dif[0] *
                            0.5, shape[i][1] + dif[1] * 0.5]
                if flag:
                    res.append(newPoint)
                res.append(shape[i + 1])
                n -= 1
                if n == 0:
                    flag = False
            return res
        else:
            now = int(sub) + 1
            res = []
            res.append(shape[0])
            for i in range(len(shape) - 1):
                dif = [shape[i + 1][0] - shape[i][0],
                       shape[i + 1][1] - shape[i][1]]
                for j in range(1, now):
                    newPoint = [shape[i][0] + dif[0] * j /
                                now, shape[i][1] + dif[1] * j / now]
                    res.append(newPoint)
                res.append(shape[i + 1])
            return self.addPoints(res, n + len(shape) - len(res))

    def reducePoints(self, polygon, n):
        if n >= len(polygon):
            return polygon
        distances = polygon.copy()
        for i in range(len(polygon)):
            mid = (np.array(polygon[i - 1]) +
                   np.array(polygon[(i + 1) % len(polygon)])) / 2
            dif = np.array(polygon[i]) - mid
            dist_mid = np.sqrt(dif[0] * dif[0] + dif[1] * dif[1])

            dif_right = np.array(
                polygon[(i + 1) % len(polygon)]) - np.array(polygon[i])
            dist_right = np.sqrt(
                dif_right[0] * dif_right[0] + dif_right[1] * dif_right[1])

            dif_left = np.array(polygon[i - 1]) - np.array(polygon[i])
            dist_left = np.sqrt(
                dif_left[0] * dif_left[0] + dif_left[1] * dif_left[1])

            distances[i] = min(dist_mid, dist_right, dist_left)
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
        return self.reducePoints(res, n)

    def handlePoints(self, polygon, n):
        if n == len(polygon):
            return polygon
        elif n > len(polygon):
            return self.addPoints(polygon, n - len(polygon))
        else:
            return self.reducePoints(polygon, n)

    def allign(self, shape1, shape2):
        shape1_center = self.centerOFmass(shape1)
        shape1_org = [[shape1[i][0] - shape1_center[0], shape1[i]
                       [1] - shape1_center[1]] for i in range(len(shape1))]
        shape2_center = self.centerOFmass(shape2)
        shape2_org = [[shape2[i][0] - shape2_center[0], shape2[i]
                       [1] - shape2_center[1]] for i in range(len(shape2))]

        shape1_slope = np.arctan2(
            np.array(shape1_org)[:, 1], np.array(shape1_org)[:, 0]).tolist()
        shape2_slope = np.arctan2(
            np.array(shape2_org)[:, 1], np.array(shape2_org)[:, 0]).tolist()

        shape1_alligned = []
        shape2_alligned = []

        for i in range(len(shape1_slope)):
            x1 = np.argmax(shape1_slope)
            x2 = np.argmax(shape2_slope)
            shape1_alligned.append(shape1_org[x1])
            shape2_alligned.append(shape2_org[x2])
            shape1_org.pop(x1)
            shape2_org.pop(x2)
            shape1_slope.pop(x1)
            shape2_slope.pop(x2)

        shape1_alligned = [[shape1_alligned[i][0] + shape1_center[0], shape1_alligned[i]
                            [1] + shape1_center[1]] for i in range(len(shape1_alligned))]
        shape2_alligned = [[shape2_alligned[i][0] + shape2_center[0], shape2_alligned[i]
                            [1] + shape2_center[1]] for i in range(len(shape2_alligned))]

        return (shape1_alligned, shape2_alligned)

    def centerOFmass(self, points):
        sumX = 0
        sumY = 0
        for point in points:
            sumX += point[0]
            sumY += point[1]
        return [int(sumX / len(points)), int(sumY / len(points))]

    def load_objects_from_json(self):
        listObj = [{'frame_idx': i + 1, 'frame_data': []}
                   for i in range(self.TOTAL_VIDEO_FRAMES)]
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
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

    def load_objects_to_json(self, listObj):
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
        with open(json_file_name, 'w') as json_file:
            json.dump(listObj, json_file,
                      indent=4,
                      separators=(',', ': '))
        json_file.close()

    def is_id_repeated(self, group_id, frameIdex = -1):
        if frameIdex == -1:
            frameIdex = self.INDEX_OF_CURRENT_FRAME
        listobj = self.load_objects_from_json()
        for i in range(len(listobj)):
            listobjframe = listobj[i]['frame_idx']
            if listobjframe != frameIdex:
                continue
            for object_ in listobj[i]['frame_data']:
                if object_['tracker_id'] == group_id:
                    return True
        return False

    def get_id_from_user(self, group_id, text):
        
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
                label = QtWidgets.QLabel(mainTEXT + f'OH GOD.. AGAIN? I hpoe you are not doing this on purpose..')
            if repeated == 2:
                label = QtWidgets.QLabel(mainTEXT + f'AGAIN? REALLY? LAST time for you..')
            if repeated == 3:
                text = False
                break
            
            properID = QtWidgets.QSpinBox()
            properID.setRange(1, 1000)
            
            buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
            buttonBox.accepted.connect(dialog.accept)
            
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(label)
            layout.addWidget(properID)
            layout.addWidget(buttonBox)
            dialog.setLayout(layout)
            result = dialog.exec_()
            if result != QtWidgets.QDialog.Accepted:
                text = False
                break
            group_id = properID.value()
            repeated += 1
            
        if repeated > 1:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText(f"OH, Finally..!")
            msg.setWindowTitle(" ")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()

        return group_id, text

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

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
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

    def addLabel(self, shape):
        if shape.group_id is None:
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
            idx = coco_classes.index(label) if label in coco_classes else -1
            idx = idx % len(color_palette)
            color = color_palette[idx] if idx != -1 else (0, 0, 255)
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

    def loadLabels(self, shapes ,replace=True):
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
        points = [(p.x(), p.y()) for p in list_2d]
        points = np.array(points, np.int16).flatten().tolist()
        return points

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

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        
        # if text == "SAM instance":
        #     text = "SAM instance - confirmed"
        
        if self.config["toolMode"] == "video":
            group_id, text = self.get_id_from_user(group_id, text)

        if text:
            if self.canvas.SAM_mode == "finished":
                self.current_sam_shape["label"] = text
                self.canvas.SAM_mode = ""
            else:
                self.labelList.clearSelection()
                # shape below is of type qt shape
                shape = self.canvas.setLastLabel(text, flags)
                shape.group_id = group_id
                shape.content = content
                self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

        if self.config["toolMode"] == "video":
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
        self.intelligenceHelper.current_model_name, self.intelligenceHelper.current_mm_model = self.intelligenceHelper.make_mm_model(
            model_name)

    def model_explorer(self):
        model_explorer_dialog = utils.ModelExplorerDialog()
        # make it fit its contents
        model_explorer_dialog.adjustSize()
        model_explorer_dialog.resize(
            int(model_explorer_dialog.width() * 2), int(model_explorer_dialog.height() * 1.5))
        model_explorer_dialog.exec_()
        selected_model_name, config, checkpoint = model_explorer_dialog.selected_model
        if selected_model_name != -1:
            self.intelligenceHelper.current_model_name, self.intelligenceHelper.current_mm_model = self.intelligenceHelper.make_mm_model_more(
                selected_model_name, config, checkpoint)

    def openPrevImg(self, _value=False):
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

    def openNextImg(self, _value=False, load=True):
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

    def openFile(self, _value=False):
        self.config['toolMode'] = 'image'
        self.right_click_menu()

        self.current_annotation_mode = "img"
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
            self.loadFile(filename)
            self.set_video_controls_visibility(False)

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
        try:
            if self.current_annotation_mode == "video":
                dialog = QtWidgets.QDialog()
                dialog.setWindowTitle("Choose Export Options")
                dialog.setWindowModality(Qt.ApplicationModal)
                dialog.resize(250, 100)

                layout = QtWidgets.QVBoxLayout()

                label = QtWidgets.QLabel("Choose Export Options")
                layout.addWidget(label)

                coco_checkbox = QtWidgets.QCheckBox(
                    "COCO Format (Detection / Segmentation)")
                mot_checkbox = QtWidgets.QCheckBox("MOT Format (Tracking)")

                layout.addWidget(coco_checkbox)
                layout.addWidget(mot_checkbox)

                buttonBox = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
                buttonBox.accepted.connect(dialog.accept)
                buttonBox.rejected.connect(dialog.reject)

                layout.addWidget(buttonBox)

                dialog.setLayout(layout)

                result = dialog.exec_()
                if not result:
                    return

                json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
                target_path = f'{self.CURRENT_VIDEO_PATH}'

                pth = self.CURRENT_VIDEO_PATH
                if coco_checkbox.isChecked():
                    pth = utils.exportCOCOvid(
                        json_file_name, target_path, self.CURRENT_VIDEO_WIDTH, self.CURRENT_VIDEO_HEIGHT, output_name="coco_vid")
                if mot_checkbox.isChecked():
                    pth = utils.exportMOT(json_file_name, target_path,
                                          output_name="mot_vid")

            elif self.current_annotation_mode == "img" or self.current_annotation_mode == "dir":
                pth = utils.exportCOCO(self.target_directory, self.save_path)
        except Exception as e:
            # Error QMessageBox
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(f"Error\n Check Terminal for more details")
            msg.setWindowTitle(
                "Export Error")
            # print exception and error line to terminal
            print(e)
            print(traceback.format_exc())
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
        else:
            # display QMessageBox with ok button and label "Exporting COCO"
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            try:
                msg.setText(f"Annotations exported successfully to {pth}")
            except:
                msg.setText(
                    f"Annotations exported successfully to Unkown Path")
            msg.setWindowTitle("Export Success")
            msg.setIconPixmap(QtGui.QPixmap(
                'labelme/icons/done.png', 'PNG').scaled(50, 50, QtCore.Qt.KeepAspectRatio))
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
            if self.config['toolMode'] == 'image':
                return
            ###########################
            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("Choose Deletion Options")
            dialog.setWindowModality(Qt.ApplicationModal)
            dialog.resize(250, 100)

            layout = QtWidgets.QVBoxLayout()

            label = QtWidgets.QLabel("Choose Deletion Options")
            layout.addWidget(label)

            prev = QtWidgets.QRadioButton("this frame and previous frames")
            next = QtWidgets.QRadioButton("this frame and next frames")
            all = QtWidgets.QRadioButton(
                "across all frames (previous and next)")
            only = QtWidgets.QRadioButton("this frame only")

            from_to = QtWidgets.QRadioButton("in a specific range of frames")
            from_frame = QtWidgets.QSpinBox()
            to_frame = QtWidgets.QSpinBox()
            from_frame.setRange(1, self.TOTAL_VIDEO_FRAMES)
            to_frame.setRange(1, self.TOTAL_VIDEO_FRAMES)
            from_frame.valueChanged.connect(lambda: from_to.toggle())
            to_frame.valueChanged.connect(lambda: from_to.toggle())

            if self.config['deleteDefault'] == 'this frame and previous frames':
                prev.toggle()
            if self.config['deleteDefault'] == 'this frame and next frames':
                next.toggle()
            if self.config['deleteDefault'] == 'across all frames (previous and next)':
                all.toggle()
            if self.config['deleteDefault'] == 'this frame only':
                only.toggle()
            if self.config['deleteDefault'] == 'in a specific range of frames':
                from_to.toggle()

            prev.toggled.connect(lambda: self.config.update(
                {'deleteDefault': 'this frame and previous frames'}))
            next.toggled.connect(lambda: self.config.update(
                {'deleteDefault': 'this frame and next frames'}))
            all.toggled.connect(lambda: self.config.update(
                {'deleteDefault': 'across all frames (previous and next)'}))
            only.toggled.connect(lambda: self.config.update(
                {'deleteDefault': 'this frame only'}))
            from_to.toggled.connect(lambda: self.config.update(
                {'deleteDefault': 'in a specific range of frames'}))

            layout.addWidget(only)
            layout.addWidget(prev)
            layout.addWidget(next)
            layout.addWidget(all)
            layout.addWidget(from_to)
            layout.addWidget(from_frame)
            layout.addWidget(to_frame)

            buttonBox = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok)
            buttonBox.accepted.connect(dialog.accept)
            layout.addWidget(buttonBox)
            dialog.setLayout(layout)
            result = dialog.exec_()
            if result == QtWidgets.QDialog.Accepted:
                for deleted_id in deleted_ids:
                    self.delete_ids_from_all_frames(
                        [deleted_id], mode=self.config['deleteDefault'], from_frame=from_frame.value(), to_frame=to_frame.value())

                self.main_video_frames_slider_changed()
            ###########################

    def delete_ids_from_all_frames(self, deleted_ids, mode, from_frame, to_frame):
        from_frame, to_frame = np.min([from_frame, to_frame]), np.max([from_frame, to_frame])
        listObj = self.load_objects_from_json()
        
        if mode == 'this frame and previous frames' :
            to_frame   = self.INDEX_OF_CURRENT_FRAME
            from_frame = 1
        elif mode == 'this frame and next frames' :
            to_frame   = self.TOTAL_VIDEO_FRAMES
            from_frame = self.INDEX_OF_CURRENT_FRAME
        elif mode == 'this frame only' :
            to_frame   = self.INDEX_OF_CURRENT_FRAME
            from_frame = self.INDEX_OF_CURRENT_FRAME
        elif mode == 'across all frames (previous and next)' :
            to_frame   = self.TOTAL_VIDEO_FRAMES
            from_frame = 1
            
        for i in range(len(listObj)):
            frame_idx = listObj[i]['frame_idx']
            for object_ in listObj[i]['frame_data']:
                id = object_['tracker_id']
                if id in deleted_ids and from_frame <= frame_idx <= to_frame:
                    listObj[i]['frame_data'].remove(object_)
                    self.CURRENT_ANNOATAION_TRAJECTORIES['id_' + str(id)][frame_idx - 1] = (-1, -1)
                    
        self.load_objects_to_json(listObj)

    def copyShape(self):
        if self.config['toolMode'] == 'video':
            self.canvas.endMove(copy=True)
            shape = self.canvas.selectedShapes[0]
            text = shape.label
            text, flags, group_id, content = self.labelDialog.popUp(text)
            shape.group_id = group_id
            shape.content = content
            shape.label = text
            shape.flags = flags
            
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
                    label = QtWidgets.QLabel(mainTEXT + f'OH GOD.. AGAIN? I hpoe you are not doing this on purpose..')
                if repeated == 2:
                    label = QtWidgets.QLabel(mainTEXT + f'AGAIN? REALLY? LAST time for you..')
                if repeated == 3:
                    text = False
                    break
                
                properID = QtWidgets.QSpinBox()
                properID.setRange(1, 1000)
                
                buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
                buttonBox.accepted.connect(dialog.accept)
                
                layout = QtWidgets.QVBoxLayout()
                layout.addWidget(label)
                layout.addWidget(properID)
                layout.addWidget(buttonBox)
                dialog.setLayout(layout)
                result = dialog.exec_()
                if result != QtWidgets.QDialog.Accepted:
                    text = False
                    break
                group_id = properID.value()
                repeated += 1
                
            if repeated > 1:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText(f"OH, Finally..!")
                msg.setWindowTitle(" ")
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()
            

            if text:
                self.labelList.clearSelection()
                shape = self.canvas.setLastLabel(text, flags)
                shape.group_id = group_id
                shape.content = content
                self.addLabel(shape)
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
        self.labelList.clearSelection()
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()
        if self.config['toolMode'] == 'video':
            self.update_current_frame_annotation_button_clicked()

    def openDirDialog(self, _value=False, dirpath=None):
        self.config['toolMode'] = 'image'
        self.right_click_menu()

        self.current_annotation_mode = "dir"
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

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()
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

    def annotate_one(self):
        # self.CURRENT_ANNOATAION_FLAGS['id'] = False
        if self.current_annotation_mode == "video":
            shapes = self.intelligenceHelper.get_shapes_of_one(
                self.CURRENT_FRAME_IMAGE, img_array_flag=True)
            
        elif self.current_annotation_mode == "multimodel":
            shapes = self.intelligenceHelper.get_shapes_of_one(
                self.CURRENT_FRAME_IMAGE, img_array_flag=True, multi_model_flag=True)
            # to handle the bug of the user selecting no models
            if len(shapes) == 0:
                return
        else:
            # if self.filename != self.last_file_opened:
            #     self.last_file_opened = self.filename
            #     self.intelligenceHelper.clear_annotating_models()

            if os.path.exists(self.filename):
                self.labelList.clearSelection()
            shapes = self.intelligenceHelper.get_shapes_of_one(
                self.CURRENT_FRAME_IMAGE, img_array_flag=True)

        # for shape in shapes:
        #     print(shape["group_id"])
        #     print(shape["bbox"])

        # image = self.draw_bb_on_image(self.image  , self.CURRENT_SHAPES_IN_IMG)
        # self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))

        # make all annotation_flags false except show polygons

        # clear shapes already in lablelist (fixes saving multiple shapes of same object bug)
        self.labelList.clear()
        self.CURRENT_SHAPES_IN_IMG = shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(self.image))
        self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        self.loadLabels(self.SAM_SHAPES_IN_IMAGE,replace=False)
        self.actions.editMode.setEnabled(True)
        self.actions.undoLastPoint.setEnabled(False)
        self.actions.undo.setEnabled(True)
        self.setDirty()

    def annotate_batch(self):
        images = []
        for filename in self.imageList:
            images.append(filename)
        self.intelligenceHelper.get_shapes_of_batch(images)

    def setConfThreshold(self):
        self.intelligenceHelper.conf_threshold = self.intelligenceHelper.setConfThreshold()
        
    def setIOUThreshold(self):
        self.intelligenceHelper.iou_threshold = self.intelligenceHelper.setIOUThreshold()

    def selectClasses(self):
        self.intelligenceHelper.selectedclasses = self.intelligenceHelper.selectClasses()

    def mergeSegModels(self):
        self.intelligenceHelper.selectedmodels = self.intelligenceHelper.mergeSegModels()
        self.prev_annotation_mode = self.current_annotation_mode
        self.current_annotation_mode = "multimodel"
        self.annotate_one()
        self.current_annotation_mode = self.prev_annotation_mode

    def Segment_anything(self):
        print('Segment anything')
        #check the visibility of the sam toolbar
        if self.sam_toolbar.isVisible():
            self.set_sam_toolbar_visibility(False)
        else:
            self.set_sam_toolbar_visibility(True)

        


    def showRuntimeData(self):
        runtime = ""
        # if cuda is available, print cuda name
        if torch.cuda.is_available():
            runtime = ("Labelmm is Using GPU\n")
            runtime += ("GPU Name: " + torch.cuda.get_device_name(0))
            runtime += "\n" + "Total GPU VRAM: " + str(round(torch.cuda.get_device_properties(0).total_memory/(1024 ** 3), 2)) + " GB"
            runtime += "\n" + "Used: " + str(round(torch.cuda.memory_allocated(0)/(1024 ** 3), 2)) + " GB"

        else:
            # runtime = (f'Labelmm is Using CPU: {platform.processor()}')
            runtime = ('Labelmm is Using CPU')

        # show runtime in QMessageBox
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(runtime)
        msg.setWindowTitle("Runtime Data")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    # VIDEO PROCESSING FUNCTIONS (ALL CONNECTED TO THE VIDEO PROCESSING TOOLBAR)

    def mapFrameToTime(self, frameNumber):
        # get the fps of the video
        fps = self.CAP.get(cv2.CAP_PROP_FPS)
        # get the time of the frame
        frameTime = frameNumber / fps
        frameHours = int(frameTime / 3600)
        frameMinutes = int((frameTime - frameHours * 3600) / 60)
        frameSeconds = int(frameTime - frameHours * 3600 - frameMinutes * 60)
        frameMilliseconds = int(
            (frameTime - frameHours * 3600 - frameMinutes * 60 - frameSeconds) * 1000)
        # print them in formal time format
        return frameHours, frameMinutes, frameSeconds, frameMilliseconds

    def calc_trajectory_when_open_video(self, ):
        listobj = self.load_objects_from_json()
        if len(listobj) == 0:
            return
        for i in range(len(listobj)):
            listobjframe = listobj[i]['frame_idx']
            for object in listobj[i]['frame_data']:
                id = object['tracker_id']
                label = object['class_name']
                # color calculation
                idx = coco_classes.index(
                    label) if label in coco_classes else -1
                idx = idx % len(color_palette)
                color = color_palette[idx] if idx != -1 else (0, 0, 255)
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
        
        # # right click menu
        #         0  createMode,
        #         1  createRectangleMode,
        #         2  createCircleMode,
        #         3  createLineMode,
        #         4  createPointMode,
        #         5  createLineStripMode,
        #         6  editMode,
        #         7  edit,
        #         8  interpolate,
        #         9  mark_as_key,
        #         10 scale,
        #         11 copy,
        #         12 delete,
        #         13 undo,
        #         14 undoLastPoint,
        #         15 addPointToEdge,
        #         16 removePoint,
        #         17 update_curr_frame,
        #         18 ignore_changes
        
        
        mode = self.config['toolMode']
        video_menu = True if mode == "video" else False
        image_menu = True if mode == "image" else False
        video_menu_list = [8,9,10,17,18]
        image_menu_list = [1,2,3,4,5,11,13,14]
        for i in range(len(self.actions.menu)):
            if i in video_menu_list:
                self.actions.menu[i].setVisible(video_menu)
                self.actions.menu[i].setEnabled(video_menu)
            elif i in image_menu_list:
                self.actions.menu[i].setVisible(image_menu)
            else:
                self.actions.menu[i].setVisible(True)
        
        # SAM tool bar

        self.Segment_anything()
        self.Segment_anything()

        
        # NOT WORKING YET
        
        # self.actions.tool[0].setVisible(True)
        # self.actions.tool[1].setVisible(True)
        # self.actions.tool[2].setVisible(True)
        # self.actions.tool[3].setVisible(image_menu)
        # self.actions.tool[4].setVisible(True)
        # self.actions.tool[5].setVisible(image_menu)
        # # self.actions.tool[6].setVisible(True)
        # self.actions.tool[7].setVisible(True)
        # self.actions.tool[8].setVisible(True)
        # self.actions.tool[9].setVisible(image_menu)
        # self.actions.tool[10].setVisible(True)
        # self.actions.tool[11].setVisible(image_menu)
        # self.actions.tool[12].setVisible(image_menu)
        # # self.actions.tool[13].setVisible(True)
        # self.actions.tool[14].setVisible(True)
        # self.actions.tool[15].setVisible(True)
        # self.actions.tool[16].setVisible(image_menu)
        # self.actions.tool[17].setVisible(image_menu)
        
        
        

    def openVideo(self):
        length_Value = self.CURRENT_ANNOATAION_TRAJECTORIES['length']
        alpha_Value = self.CURRENT_ANNOATAION_TRAJECTORIES['alpha']
        self.CURRENT_ANNOATAION_TRAJECTORIES.clear()
        self.CURRENT_ANNOATAION_TRAJECTORIES['length'] = length_Value
        self.CURRENT_ANNOATAION_TRAJECTORIES['alpha'] = alpha_Value

        self.config['toolMode'] = "video"
        self.right_click_menu()

        for shape in self.canvas.shapes:
            self.canvas.deleteShape(shape)

        self.CURRENT_SHAPES_IN_IMG = []

        # self.videoControls.show()
        self.current_annotation_mode = "video"
        try:
            cv2.destroyWindow('video processing')
        except:
            pass
        if not self.mayContinue():
            return
        videoFile = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("%s - Choose Video") % __appname__, ".",
            self.tr("Video files (*.mp4 *.avi)")
        )

        if videoFile[0]:
            self.CURRENT_VIDEO_NAME = videoFile[0].split(
                ".")[-2].split("/")[-1]
            self.CURRENT_VIDEO_PATH = "/".join(
                videoFile[0].split(".")[-2].split("/")[:-1])

            # print('video file name : ' , self.CURRENT_VIDEO_NAME)
            # print("video file path : " , self.CURRENT_VIDEO_PATH)
            cap = cv2.VideoCapture(videoFile[0])
            self.CURRENT_VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.CURRENT_VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.CAP = cap
            # making the total video frames equal to the total frames in the video file - 1 as the indexing starts from 0
            self.TOTAL_VIDEO_FRAMES = int(
                self.CAP.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
            self.CURRENT_VIDEO_FPS = self.CAP.get(cv2.CAP_PROP_FPS)
            print("Total Frames : ", self.TOTAL_VIDEO_FRAMES)
            self.main_video_frames_slider.setMaximum(self.TOTAL_VIDEO_FRAMES)
            self.main_video_frames_slider.setValue(2)
            self.INDEX_OF_CURRENT_FRAME = 1
            self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME)

            # self.addToolBarBreak

            self.set_video_controls_visibility(True)

            self.update_tracking_method()

            self.calc_trajectory_when_open_video()

        # label = shape["label"]
        # points = shape["points"]
        # bbox = shape["bbox"]
        # shape_type = shape["shape_type"]
        # flags = shape["flags"]
        # content = shape["content"]
        # group_id = shape["group_id"]
        # other_data = shape["other_data"]

    def load_shapes_for_video_frame(self, json_file_name, index):
        # this function loads the shapes for the video frame from the json file
        # first we read the json file in the form of a list
        # we need to parse from it data for the current frame

        target_frame_idx = index
        listObj = self.load_objects_from_json()

        listObj = np.array(listObj)

        shapes = []
        for i in range(len(listObj)):
            frame_idx = listObj[i]['frame_idx']
            if frame_idx == target_frame_idx:
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
                continue

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
            'frames to skip by: ' + zeros + str(self.FRAMES_TO_SKIP))

    def playPauseButtonClicked(self):
        # we can check the state of the button by checking the button text
        if self.playPauseButton.text() == "Play":
            self.playPauseButton.setText("Pause")
            self.playPauseButton.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
            # play the video at the current fps untill the user clicks pause
            self.play_timer = QtCore.QTimer(self)
            # use play_timer.timeout.connect to call a function every time the timer times out
            # but we need to call the function every interval of time
            # so we need to call the function every 1/fps seconds
            self.play_timer.timeout.connect(self.move_frame_by_frame)
            self.play_timer.start(500)
            # note that the timer interval is in milliseconds

            # while self.timer.isActive():
        elif self.playPauseButton.text() == "Pause":
            # first stop the timer
            self.play_timer.stop()

            self.playPauseButton.setText("Play")
            self.playPauseButton.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        # print(1)

    def main_video_frames_slider_changed(self):
        
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

    def frames_to_track_slider_changed(self):
        self.FRAMES_TO_TRACK = self.frames_to_track_slider.value()
        zeros = (2 - int(np.log10(self.FRAMES_TO_TRACK + 0.9))) * '0'
        self.frames_to_track_label.setText(
            f'track for {zeros}{self.FRAMES_TO_TRACK} frames')

    def move_frame_by_frame(self):
        self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME + 1)
        
    def class_name_to_id(self, class_name):
        try:
            # map from coco_classes(a list of coco class names) to class_id
            return coco_classes.index(class_name)
        except:
            # this means that the class name is not in the coco dataset
            return -1

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

    def compute_iou(self , box1, box2):
        """
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
        
    def match_detections_with_tracks(self , detections, tracks, iou_threshold=0.5):
        """
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
                iou = self.compute_iou(detection_bbox, track_bbox)

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
    
    
    def update_gui_after_tracking(self , index):
        # self.loadFramefromVideo(self.CURRENT_FRAME_IMAGE, self.INDEX_OF_CURRENT_FRAME)
        # QtWidgets.QApplication.processEvents()
        # progress_value = int((index + 1) / self.FRAMES_TO_TRACK * 100)
        # self.tracking_progress_bar_signal.emit(progress_value)

        # QtWidgets.QApplication.processEvents()
        if index != self.FRAMES_TO_TRACK - 1:
            self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME + 1)
        QtWidgets.QApplication.processEvents()


        
    def track_buttonClicked_wrapper(self):
        # ...

        # Disable the track button
        # self.actions.track.setEnabled(False)

        # Create a thread to run the tracking process
        self.thread = TrackingThread(parent=self)
        self.thread.start()
        
    def get_boxes_conf_classids_segments(self, shapes):
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

            boxes.append(self.intelligenceHelper.get_bbox(segment))
            confidences.append(float(s["content"]))
            class_ids.append(coco_classes.index(
                label)if label in coco_classes else -1)
        
        return boxes, confidences, class_ids, segments

    def track_buttonClicked(self):

        # dt = (Profile(), Profile(), Profile(), Profile())
        
        self.tracking_progress_bar.setVisible(True)
        self.actions.export.setEnabled(True)
        frame_shape = self.CURRENT_FRAME_IMAGE.shape
        print(frame_shape)

        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'

        # first we need to check there is a json file with the same name as the video
        listObj = []
        if os.path.exists(json_file_name):
            print('json file exists')
            with open(json_file_name, 'r') as jf:
                listObj = json.load(jf)
            jf.close()
        else:
            # make a json file with the same name as the video
            print('json file does not exist , creating a new one')
            with open(json_file_name, 'w') as jf:
                json.dump(listObj, jf)
            jf.close()

        existing_annotation = False
        shapes = self.canvas.shapes
        tracks_to_follow = None
        if len(shapes) > 0:
            existing_annotation = True
            print(f'FRAME{self.INDEX_OF_CURRENT_FRAME}', len(shapes))
            tracks_to_follow = []
            for shape in shapes:
                print(shape.label)
                if shape.group_id != None:
                    tracks_to_follow.append(int(shape.group_id))

        print(f'track_ids_to_follow = {tracks_to_follow}')

        self.TrackingMode = True
        bs = 1
        curr_frame, prev_frame = None, None
        # debugging_vid = cv2.VideoWriter(
        #     'tracking_debugging.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_shape[1], frame_shape[0]))

        for i in range(self.FRAMES_TO_TRACK):
            QtWidgets.QApplication.processEvents()
            self.tracking_progress_bar.setValue(int((i + 1) / self.FRAMES_TO_TRACK * 100))

            # if self.stop_tracking:
            #     print('\n\n\nstopped tracking\n\n\n')
            #     break

            if existing_annotation:
                existing_annotation = False
                print('\nloading existing annotation\n')
                shapes = self.canvas.shapes
                shapes = self.convert_qt_shapes_to_shapes(shapes)
            else:
                with torch.no_grad():
                    shapes = self.intelligenceHelper.get_shapes_of_one(
                        self.CURRENT_FRAME_IMAGE, img_array_flag=True)

            curr_frame = self.CURRENT_FRAME_IMAGE
            # current_objects_ids = []
            if len(shapes) == 0:
                print("no detection in this frame")
                self.update_gui_after_tracking(i)
                continue
            
            for shape in shapes:
                if shape['content'] is None:
                    shape['content'] = 1.0
            boxes, confidences, class_ids, segments = self.get_boxes_conf_classids_segments(shapes)

            boxes = np.array(boxes, dtype=int)
            confidences = np.array(confidences)
            class_ids = np.array(class_ids)
            detections = Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids,
            )
            boxes = torch.from_numpy(detections.xyxy )
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
                org_tracks = self.tracker.update( dets.cpu(),self.CURRENT_FRAME_IMAGE )

            tracks = []
            for org_track in org_tracks:
                track = [] 
                for i in range(6):
                    track.append(int(org_track[i]))
                track.append(org_track[6])

                tracks.append(track)

            matched_shapes , unmatched_shapes = self.match_detections_with_tracks(shapes, tracks)
            shapes = matched_shapes

            self.CURRENT_SHAPES_IN_IMG = [
                shape_ for shape_ in shapes if shape_["group_id"] is not None]

            if self.TRACK_ASSIGNED_OBJECTS_ONLY and tracks_to_follow is not None:
                try :
                    if len(self.labelList.selectedItems()) != 0:
                        tracks_to_follow = []
                        for item in self.labelList.selectedItems():
                            x = item.text()
                            i1, i2 = x.find('D'), x.find(':')
                            print(x  , x[i1 + 2:i2])
                            tracks_to_follow.append(int(x[i1 + 2:i2]))
                    self.CURRENT_SHAPES_IN_IMG = [
                        shape_ for shape_ in shapes if shape_["group_id"] in tracks_to_follow]
                except:
                    # this happens when the user selects a label that is not a tracked object so there is error in extracting the tracker id
                    # show a message box to the user (hinting to use the tracker on the image first so that the label has a tracker id to be selected)
                    self.errorMessage('Error', 'Please use the tracker on the image first so that you can select labels with IDs to track')

                    return
                        


            # to understand the json output file structure it is a dictionary of frames and each frame is a dictionary of tracker_ids and each tracker_id is a dictionary of bbox , confidence , class_id , segment
            json_frame = {}
            json_frame.update({'frame_idx': self.INDEX_OF_CURRENT_FRAME})
            json_frame_object_list = []
            for shape in self.CURRENT_SHAPES_IN_IMG:
                json_tracked_object = {}
                json_tracked_object['tracker_id'] = int(shape["group_id"])
                json_tracked_object['bbox'] = [int(i) for i in shape['bbox']]
                json_tracked_object['confidence'] = shape["content"]
                json_tracked_object['class_name'] = shape["label"]
                json_tracked_object['class_id'] = coco_classes.index(
                    shape["label"])
                points = shape["points"]
                segment = [[int(points[z]), int(points[z + 1])]
                           for z in range(0, len(points), 2)]
                json_tracked_object['segment'] = segment

                json_frame_object_list.append(json_tracked_object)

            json_frame.update({'frame_data': json_frame_object_list})

            for h in range(len(listObj)):
                if listObj[h]['frame_idx'] == self.INDEX_OF_CURRENT_FRAME:
                    listObj.pop(h)
                    break
            # sort the list of frames by the frame index
            listObj.append(json_frame)

            QtWidgets.QApplication.processEvents()
            self.update_gui_after_tracking(i)
            print('finished tracking for frame ', self.INDEX_OF_CURRENT_FRAME)




        
        listObj = sorted(listObj, key=lambda k: k['frame_idx'])
        with open(json_file_name, 'w') as json_file:
            json.dump(listObj, json_file,
                      indent=4,
                      separators=(',', ': '))
        json_file.close()

        self.TrackingMode = False
        self.labelFile = None
        self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME - 1)
        self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME)

        self.tracking_progress_bar.hide()
        self.tracking_progress_bar.setValue(0)

    def convert_qt_shapes_to_shapes(self, qt_shapes):
        shapes = []
        for s in qt_shapes:
            shapes.append(dict(
                label=s.label.encode("utf-8") if PY2 else s.label,
                # convert points into 1D array
                points=self.flattener(s.points),
                bbox=self.intelligenceHelper.get_bbox(
                    [(p.x(), p.y()) for p in s.points]),
                group_id=s.group_id,
                content=s.content,
                shape_type=s.shape_type,
                flags=s.flags,
            ))
        return shapes

    def track_full_video_button_clicked(self):
        self.FRAMES_TO_TRACK = int(
            self.TOTAL_VIDEO_FRAMES - self.INDEX_OF_CURRENT_FRAME + 1)
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
        self.videoControls_3.setVisible(visible)
        for widget in self.videoControls_3.children():
            try:
                widget.setVisible(visible)
            except:
                pass

    def window_wait(self, seconds):
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(seconds * 1000, loop.quit)
        loop.exec_()

    def traj_checkBox_changed(self):
        self.update_current_frame_annotation()
        self.CURRENT_ANNOATAION_FLAGS["traj"] = self.traj_checkBox.isChecked()
        self.main_video_frames_slider_changed()

    def mask_checkBox_changed(self):
        self.update_current_frame_annotation()
        self.CURRENT_ANNOATAION_FLAGS["mask"] = self.mask_checkBox.isChecked()
        self.main_video_frames_slider_changed()

    def class_checkBox_changed(self):
        self.update_current_frame_annotation()
        self.CURRENT_ANNOATAION_FLAGS["class"] = self.class_checkBox.isChecked(
        )
        self.main_video_frames_slider_changed()

    def id_checkBox_changed(self):
        self.update_current_frame_annotation()
        self.CURRENT_ANNOATAION_FLAGS["id"] = self.id_checkBox.isChecked()
        self.main_video_frames_slider_changed()

    def bbox_checkBox_changed(self):
        self.update_current_frame_annotation()
        self.CURRENT_ANNOATAION_FLAGS["bbox"] = self.bbox_checkBox.isChecked()
        self.main_video_frames_slider_changed()

    def polygons_visable_checkBox_changed(self):
        self.update_current_frame_annotation()
        self.CURRENT_ANNOATAION_FLAGS["polygons"] = self.polygons_visable_checkBox.isChecked(
        )
        for shape in self.canvas.shapes:
            self.canvas.setShapeVisible(
                shape, self.CURRENT_ANNOATAION_FLAGS["polygons"])

    def export_as_video_button_clicked(self):
        self.update_current_frame_annotation()
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
        input_video_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}.mp4'
        output_video_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.mp4'
        input_cap = cv2.VideoCapture(input_video_file_name)
        output_cap = cv2.VideoWriter(output_video_file_name, cv2.VideoWriter_fourcc(
            *'mp4v'), 30, (int(self.CURRENT_VIDEO_WIDTH), int(self.CURRENT_VIDEO_HEIGHT)))
        with open(json_file_name, 'r') as jf:
            listObj = json.load(jf)
        jf.close()

        # make a progress bar for exporting video (with percentage of progress)   TO DO LATER

        for target_frame_idx in range(self.TOTAL_VIDEO_FRAMES):
            print(target_frame_idx)
            self.INDEX_OF_CURRENT_FRAME = target_frame_idx + 1
            ret, image = input_cap.read()
            shapes = []
            for i in range(len(listObj)):
                frame_idx = listObj[i]['frame_idx']
                if int(frame_idx) == target_frame_idx:
                    frame_objects = listObj[i]['frame_data']
                    for object_ in frame_objects:
                        shape = {}
                        shape["label"] = coco_classes[object_['class_id']]
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
                    continue
            if len(shapes) == 0:
                output_cap.write(image)
                continue
            image = self.draw_bb_on_image(image, shapes, imgage_qt_flag=False)
            output_cap.write(image)

        input_cap.release()
        output_cap.release()
        print("done exporting video")
        self.INDEX_OF_CURRENT_FRAME = self.main_video_frames_slider.value()
        # show message saying that the video is exported
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Done Exporting Video")
        msg.setWindowTitle("Export Video")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def clear_video_annotations_button_clicked(self):
        length_Value = self.CURRENT_ANNOATAION_TRAJECTORIES['length']
        alpha_Value = self.CURRENT_ANNOATAION_TRAJECTORIES['alpha']
        self.CURRENT_ANNOATAION_TRAJECTORIES.clear()
        self.CURRENT_ANNOATAION_TRAJECTORIES['length'] = length_Value
        self.CURRENT_ANNOATAION_TRAJECTORIES['alpha'] = alpha_Value
        self.key_frames.clear()

        for shape in self.canvas.shapes:
            self.canvas.deleteShape(shape)

        self.CURRENT_SHAPES_IN_IMG = []

        # just delete the json file and reload the video
        # to delete the json file we need to know the name of the json file which is the same as the video name
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
        # now delete the json file if it exists
        if os.path.exists(json_file_name):
            os.remove(json_file_name)
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("All video frames annotations are cleared")
        msg.setWindowTitle("clear annotations")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()
        self.main_video_frames_slider.setValue(2)
        self.main_video_frames_slider.setValue(1)

    def update_current_frame_annotation_button_clicked(self):
        try:
            x = self.CURRENT_VIDEO_PATH
        except:
            return
        self.update_current_frame_annotation()
        self.main_video_frames_slider_changed()

    def update_current_frame_annotation(self):
        json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
        if not os.path.exists(json_file_name):
            with open(json_file_name, 'w') as jf:
                json.dump([], jf)
            jf.close()
        with open(json_file_name, 'r') as jf:
            listObj = json.load(jf)
        jf.close()

        json_frame = {}
        json_frame.update({'frame_idx': self.INDEX_OF_CURRENT_FRAME})
        json_frame_object_list = []

        shapes = self.convert_qt_shapes_to_shapes(self.canvas.shapes)
        for shape in shapes:
            json_tracked_object = {}
            json_tracked_object['tracker_id'] = int(
                shape["group_id"]) if shape["group_id"] != None else -1
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

        for h in range(len(listObj)):
            if listObj[h]['frame_idx'] == self.INDEX_OF_CURRENT_FRAME:
                listObj.pop(h)
                break

        listObj.append(json_frame)

        listObj = sorted(listObj, key=lambda k: k['frame_idx'])
        with open(json_file_name, 'w') as json_file:
            json.dump(listObj, json_file,
                      indent=4,
                      separators=(',', ': '))
        json_file.close()
        print("saved frame annotation")

    def trajectory_length_lineEdit_changed(self):
        text = self.trajectory_length_lineEdit.text()
        self.CURRENT_ANNOATAION_TRAJECTORIES['length'] = int(
            text) if text != '' else 1
        self.main_video_frames_slider_changed()


    # @PyQt5.QtCore.pyqtSlot()
    # async def stop_tracking_button_clicked(self):
    #     self.stop_tracking = True


    # def track_buttonClicked_sync(self):
    #     self.loop = asyncio.get_event_loop()
    #     self.loop.run_until_complete(asyncio.gather(self.track_buttonClicked(),
    #                                  self.stop_tracking_button_clicked()))

    # @pyqtSlot(int)
    # def update_tracking_progress_bar(self, value):
    #     self.tracking_progress_bar.setValue(value)



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

        self.videoControls_3 = QtWidgets.QToolBar()
        self.videoControls_3.setMovable(True)
        self.videoControls_3.setFloatable(True)
        self.videoControls_3.setObjectName("videoControls_2")
        self.videoControls_3.setStyleSheet(
            "QToolBar#videoControls_3 { border: 50px }")
        self.addToolBar(Qt.TopToolBarArea, self.videoControls_3)

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
        self.previousFrame_button.clicked.connect(
            self.previousFrame_buttonClicked)

        self.previous_1_Frame_button = QtWidgets.QPushButton()
        self.previous_1_Frame_button.setText("<")
        self.previous_1_Frame_button.clicked.connect(
            self.previous_1_Frame_buttonclicked)

        self.playPauseButton = QtWidgets.QPushButton()
        self.playPauseButton.setText("Play")
        self.playPauseButton.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.playPauseButton.clicked.connect(self.playPauseButtonClicked)

        self.nextFrame_button = QtWidgets.QPushButton()
        self.nextFrame_button.setText(">>")
        self.nextFrame_button.clicked.connect(self.nextFrame_buttonClicked)

        self.next_1_Frame_button = QtWidgets.QPushButton()
        self.next_1_Frame_button.setText(">")
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
        self.frames_to_track_slider.setMaximumWidth(400)
        self.frames_to_track_slider.valueChanged.connect(
            self.frames_to_track_slider_changed)

        self.frames_to_track_label = QtWidgets.QLabel()
        self.frames_to_track_label.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; }")  # make the button text red
        self.videoControls_2.addWidget(self.frames_to_track_label)
        self.videoControls_2.addWidget(self.frames_to_track_slider)
        self.frames_to_track_slider.setValue(10)

        self.track_button = QtWidgets.QPushButton()
        self.track_button.setStyleSheet(self.buttons_text_style_sheet)
        self.track_button.setText("Track")
        self.track_button.clicked.connect(self.track_buttonClicked)
        self.videoControls_2.addWidget(self.track_button)

        self.track_assigned_objects = QtWidgets.QPushButton()
        self.track_assigned_objects.setStyleSheet(
            self.buttons_text_style_sheet)
        self.track_assigned_objects.setText("Track Only assigned objects")
        self.track_assigned_objects.clicked.connect(
            self.track_assigned_objects_button_clicked)
        self.videoControls_2.addWidget(self.track_assigned_objects)
        # make a tracking progress bar with a label to show the progress of the tracking (in percentage )
        self.track_full_video_button = QtWidgets.QPushButton()
        self.track_full_video_button.setStyleSheet(
            self.buttons_text_style_sheet)
        self.track_full_video_button.setText("Track Full Video")
        self.track_full_video_button.clicked.connect(
            self.track_full_video_button_clicked)
        self.videoControls_2.addWidget(self.track_full_video_button)

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
        
        # self.tracking_progress_bar_signal = pyqtSignal(int)
        # self.tracking_progress_bar_signal.connect(self.tracking_progress_bar.setValue)
        
        
        # self.stop_tracking_button = QtWidgets.QPushButton()
        # self.stop_tracking_button.setStyleSheet(
        #     self.buttons_text_style_sheet)
        # self.stop_tracking_button.setText("Stop Tracking")
        # self.stop_tracking_button.clicked.connect(
        #     self.stop_tracking_button_clicked)
        # self.videoControls_2.addWidget(self.stop_tracking_button)

        # add 5 checkboxes to control the CURRENT ANNOATAION FLAGS including (bbox , id , class , mask , traj)
        self.bbox_checkBox = QtWidgets.QCheckBox()
        self.bbox_checkBox.setText("bbox")
        self.bbox_checkBox.setChecked(True)
        self.bbox_checkBox.stateChanged.connect(self.bbox_checkBox_changed)
        self.videoControls_3.addWidget(self.bbox_checkBox)

        self.id_checkBox = QtWidgets.QCheckBox()
        self.id_checkBox.setText("id")
        self.id_checkBox.setChecked(True)
        self.id_checkBox.stateChanged.connect(self.id_checkBox_changed)
        self.videoControls_3.addWidget(self.id_checkBox)

        self.class_checkBox = QtWidgets.QCheckBox()
        self.class_checkBox.setText("class")
        self.class_checkBox.setChecked(True)
        self.class_checkBox.stateChanged.connect(self.class_checkBox_changed)
        self.videoControls_3.addWidget(self.class_checkBox)

        self.mask_checkBox = QtWidgets.QCheckBox()
        self.mask_checkBox.setText("mask")
        self.mask_checkBox.setChecked(True)
        self.mask_checkBox.stateChanged.connect(self.mask_checkBox_changed)
        self.videoControls_3.addWidget(self.mask_checkBox)

        self.traj_checkBox = QtWidgets.QCheckBox()
        self.traj_checkBox.setText("trajectories")
        self.traj_checkBox.setChecked(False)
        self.traj_checkBox.stateChanged.connect(self.traj_checkBox_changed)
        self.videoControls_3.addWidget(self.traj_checkBox)

        # make qlineedit to alter the  self.CURRENT_ANNOATAION_TRAJECTORIES['length']  value
        self.trajectory_length_lineEdit = QtWidgets.QLineEdit()
        self.trajectory_length_lineEdit.setText(str(30))
        self.trajectory_length_lineEdit.setMaximumWidth(50)
        self.trajectory_length_lineEdit.editingFinished.connect(
            self.trajectory_length_lineEdit_changed)

        self.videoControls_3.addWidget(self.trajectory_length_lineEdit)

        self.polygons_visable_checkBox = QtWidgets.QCheckBox()
        self.polygons_visable_checkBox.setText("show polygons")
        self.polygons_visable_checkBox.setChecked(True)
        self.polygons_visable_checkBox.stateChanged.connect(
            self.polygons_visable_checkBox_changed)
        self.videoControls_3.addWidget(self.polygons_visable_checkBox)




       # save current frame
        self.update_current_frame_annotation_button = QtWidgets.QPushButton()
        self.update_current_frame_annotation_button.setStyleSheet(self.buttons_text_style_sheet)
        self.update_current_frame_annotation_button.setText( "Update current frame")
        self.update_current_frame_annotation_button.clicked.connect(self.update_current_frame_annotation_button_clicked)
        self.videoControls_3.addWidget(self.update_current_frame_annotation_button)

        # add a button to clear all video annotations
        self.clear_video_annotations_button = QtWidgets.QPushButton()
        self.clear_video_annotations_button.setStyleSheet(self.buttons_text_style_sheet)
        self.clear_video_annotations_button.setText("Clear Video Annotations")
        self.clear_video_annotations_button.clicked.connect(self.clear_video_annotations_button_clicked)
        self.videoControls_3.addWidget(self.clear_video_annotations_button)

        # add export as video button
        self.export_as_video_button = QtWidgets.QPushButton()
        self.export_as_video_button.setStyleSheet(self.buttons_text_style_sheet)
        self.export_as_video_button.setText("Export as video")
        self.export_as_video_button.clicked.connect(self.export_as_video_button_clicked)
        self.videoControls_3.addWidget(self.export_as_video_button)



        # add a button to clear all video annotations

        self.set_video_controls_visibility(False)
                    
    def convert_QT_to_cv(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(4)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        return arr

    def convert_cv_to_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return convert_to_Qt_format

    def draw_bb_id(self, image, x, y, w, h, id, label, color=(0, 0, 255), thickness=1):
        if self.CURRENT_ANNOATAION_FLAGS['bbox']:
            image = cv2.rectangle(
                image, (x, y), (x + w, y + h), color, thickness + 1)
            if (False):
                thick = 5
                colorx = (236, 199, 113)
                image = cv2.line(image, (x, y), (x + 20, y),
                                 colorx, thickness + 1 + thick)
                image = cv2.line(image, (x, y), (x, y + 20),
                                 colorx, thickness + 1 + thick)
                image = cv2.line(image, (x + w, y), (x + w - 20, y),
                                 colorx, thickness + 1 + thick)
                image = cv2.line(image, (x + w, y), (x + w, y + 20),
                                 colorx, thickness + 1 + thick)
                image = cv2.line(image, (x, y + h), (x + 20, y + h),
                                 colorx, thickness + 1 + thick)
                image = cv2.line(image, (x, y + h), (x, y + h - 20),
                                 colorx, thickness + 1 + thick)
                image = cv2.line(image, (x + w, y + h), (x +
                                 w - 20, y + h), colorx, thickness + 1 + thick)
                image = cv2.line(image, (x + w, y + h), (x + w,
                                 y + h - 20), colorx, thickness + 1 + thick)

        if self.CURRENT_ANNOATAION_FLAGS['id'] or self.CURRENT_ANNOATAION_FLAGS['class']:
            if self.CURRENT_ANNOATAION_FLAGS['id'] and self.CURRENT_ANNOATAION_FLAGS['class']:
                text = f'#{id} {label}'
            if self.CURRENT_ANNOATAION_FLAGS['id'] and not self.CURRENT_ANNOATAION_FLAGS['class']:
                text = f'#{id}'
            if not self.CURRENT_ANNOATAION_FLAGS['id'] and self.CURRENT_ANNOATAION_FLAGS['class']:
                text = f'{label}'

            if image.shape[0] < 1000:
                fontscale = 0.5
            else :
                fontscale = 0.7
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
        if (not self.CURRENT_ANNOATAION_FLAGS['bbox']) and (self.CURRENT_ANNOATAION_FLAGS['id'] or self.CURRENT_ANNOATAION_FLAGS['class']):
            image = cv2.line(image, (x + int(w / 2), y + int(h / 2)),
                             (x + 50, y - 5), color, thickness + 1)

        return image

    def draw_trajectories(self, img, shapes):
        x = self.CURRENT_ANNOATAION_TRAJECTORIES['length']
        for shape in shapes:
            id = shape["group_id"]
            pts_traj = self.CURRENT_ANNOATAION_TRAJECTORIES['id_' + str(id)][max(
                self.INDEX_OF_CURRENT_FRAME - x, 0): self.INDEX_OF_CURRENT_FRAME]
            pts_poly = np.array([[x, y] for x, y in zip(
                shape["points"][0::2], shape["points"][1::2])])
            color_poly = self.CURRENT_ANNOATAION_TRAJECTORIES['id_color_' + str(
                id)]

            if self.CURRENT_ANNOATAION_FLAGS['mask']:
                original_img = img.copy()
                if pts_poly is not None:
                    cv2.fillPoly(img, pts=[pts_poly], color=color_poly )
                alpha = self.CURRENT_ANNOATAION_TRAJECTORIES['alpha']
                img = cv2.addWeighted(original_img, alpha, img, 1 - alpha, 0)
            for i in range(len(pts_traj) - 1, 0, - 1):
                
                thickness = (len(pts_traj) - i <= 10) * 1 + (len(pts_traj) - i <= 20) * 1 + (len(pts_traj) - i <= 30) * 1 + 3
                # max_thickness = 6
                # thickness = max(1, round(i / len(pts_traj) * max_thickness))          
                
                if pts_traj[i - 1] is None or pts_traj[i] is None:
                    continue
                if pts_traj[i] == (-1, - 1) or pts_traj[i - 1] == (-1, - 1):
                    break

                # color_traj = tuple(int(0.95 * x) for x in color_poly)
                color_traj = color_poly

                if self.CURRENT_ANNOATAION_FLAGS['traj']:
                    cv2.line(img, pts_traj[i - 1],
                             pts_traj[i], color_traj, thickness)
                    if ((len(pts_traj) - 1 - i) % 10 == 0):
                        cv2.circle(img, pts_traj[i], 3, (0, 0, 0), -1)

        return img

    def draw_bb_on_image(self, image, shapes, imgage_qt_flag=True):
        img = image
        if imgage_qt_flag:
            img = self.convert_QT_to_cv(image)

        for shape in shapes:
            id = shape["group_id"]
            label = shape["label"]

            # color calculation
            idx = coco_classes.index(label) if label in coco_classes else -1
            idx = idx % len(color_palette)
            color = color_palette[idx] if idx != -1 else (0, 0, 255)

            (x1, y1, x2, y2) = shape["bbox"]
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            img = self.draw_bb_id(img, x, y, w, h, id,
                                  label, color, thickness=1)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            try:
                centers_rec = self.CURRENT_ANNOATAION_TRAJECTORIES['id_' + str(
                    id)]
                try:
                    (xp, yp) = centers_rec[self.INDEX_OF_CURRENT_FRAME - 2]
                    (xn, yn) = center
                    if (xp == -1 or xn == -1):
                        c = 5 / 0
                    r = 0.5
                    x = r * xn + (1 - r) * xp
                    y = r * yn + (1 - r) * yp
                    center = (int(x), int(y))
                except:
                    pass
                centers_rec[self.INDEX_OF_CURRENT_FRAME - 1] = center
                self.CURRENT_ANNOATAION_TRAJECTORIES['id_' +
                                                     str(id)] = centers_rec
                self.CURRENT_ANNOATAION_TRAJECTORIES['id_color_' +
                                                     str(id)] = color
            except:
                centers_rec = [(-1, - 1)] * int(self.TOTAL_VIDEO_FRAMES)
                centers_rec[self.INDEX_OF_CURRENT_FRAME - 1] = center
                self.CURRENT_ANNOATAION_TRAJECTORIES['id_' +
                                                     str(id)] = centers_rec
                self.CURRENT_ANNOATAION_TRAJECTORIES['id_color_' +
                                                     str(id)] = color

        # print(sys.getsizeof(self.CURRENT_ANNOATAION_TRAJECTORIES))

        img = self.draw_trajectories(img, shapes)

        if imgage_qt_flag:
            img = self.convert_cv_to_qt(img, )

        return img
    
    def set_sam_toolbar_visibility(self, visible=False):
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
        self.sam_model_label.setStyleSheet("QLabel { font-size: 10pt; font-weight: bold; }")
        self.sam_toolbar.addWidget(self.sam_model_label)

        # add a dropdown menu to select the sam model
        self.sam_model_comboBox = QtWidgets.QComboBox()
        # add a label inside the combobox that says "select model" and make it unselectable 
        self.sam_model_comboBox.addItem("Select Model")
        self.sam_model_comboBox.addItems(self.sam_models())
        self.sam_model_comboBox.setStyleSheet("QComboBox { font-size: 10pt; font-weight: bold; }")
        self.sam_model_comboBox.currentIndexChanged.connect(
            self.sam_model_comboBox_changed)
        self.sam_toolbar.addWidget(self.sam_model_comboBox)

        # add a button for adding a point in sam
        self.sam_add_point_button = QtWidgets.QPushButton()
        self.sam_add_point_button.setStyleSheet("QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_add_point_button.setText("Add Point")
        self.sam_add_point_button.clicked.connect(self.sam_add_point_button_clicked)
        self.sam_toolbar.addWidget(self.sam_add_point_button)

        # add a button for removing a point in sam
        self.sam_remove_point_button = QtWidgets.QPushButton()
        self.sam_remove_point_button.setStyleSheet("QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_remove_point_button.setText("Remove Point")
        self.sam_remove_point_button.clicked.connect(self.sam_remove_point_button_clicked)
        self.sam_toolbar.addWidget(self.sam_remove_point_button)

        # add a button for selecting a rectangle in sam
        self.sam_select_rect_button = QtWidgets.QPushButton()
        self.sam_select_rect_button.setStyleSheet("QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_select_rect_button.setText("Select Rect")
        self.sam_select_rect_button.clicked.connect(self.sam_select_rect_button_clicked)
        self.sam_toolbar.addWidget(self.sam_select_rect_button)

        # add a point for clearing the annotation
        self.sam_clear_annotation_button = QtWidgets.QPushButton()
        self.sam_clear_annotation_button.setStyleSheet("QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_clear_annotation_button.setText("Clear Annotation")
        self.sam_clear_annotation_button.clicked.connect(self.sam_clear_annotation_button_clicked)
        self.sam_toolbar.addWidget(self.sam_clear_annotation_button)

        # add a point of finish object annotation
        self.sam_finish_annotation_button = QtWidgets.QPushButton()
        self.sam_finish_annotation_button.setStyleSheet("QPushButton { font-size: 10pt; font-weight: bold; }")
        self.sam_finish_annotation_button.setText("Finish Annotation")
        self.sam_finish_annotation_button.clicked.connect(self.sam_finish_annotation_button_clicked)
        self.sam_toolbar.addWidget(self.sam_finish_annotation_button)



        self.set_sam_toolbar_visibility(False)

    

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
        if self.sam_model_comboBox.currentText() == "Select Model":
            return
        model_type = self.sam_model_comboBox.currentText()
        with open('models_menu/sam_models.json') as f:
            data = json.load(f)
        for model in data:
            if model['name'] == model_type:
                checkpoint_path = model['checkpoint']
        #print(model_type, checkpoint_path, device)
        self.sam_predictor = Sam_Predictor(model_type, checkpoint_path, device)
        try:
            self.sam_predictor.set_new_image(self.CURRENT_FRAME_IMAGE)
        except:
            print("please open an image first")
            return
        print("done loading model")


    def sam_add_point_button_clicked(self):
        try:
            same_image = self.sam_predictor.check_image(self.CURRENT_FRAME_IMAGE)
        except:
            return
        if not same_image: 
            self.sam_clear_annotation_button_clicked()
        print("sam add point button clicked")
        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.canvas.SAM_mode = "add point"
    
    def sam_remove_point_button_clicked(self):
        try:
            same_image = self.sam_predictor.check_image(self.CURRENT_FRAME_IMAGE)
        except:
            return
        if not same_image: 
            self.sam_clear_annotation_button_clicked()
        print("sam remove point button clicked")
        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.canvas.SAM_mode = "remove point"
    
    def sam_select_rect_button_clicked(self):
        try:
            same_image = self.sam_predictor.check_image(self.CURRENT_FRAME_IMAGE)
        except:
            return
        if not same_image: 
            self.sam_clear_annotation_button_clicked()
        print("sam select rect button clicked")
        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.canvas.SAM_mode = "select rect"
    
    def sam_clear_annotation_button_clicked(self):
        print("sam clear annotation button clicked")
        self.canvas.SAM_coordinates = []
        self.canvas.SAM_mode = ""
        try:
            self.sam_predictor.clear_logit()
        except:
            pass
        self.labelList.clear()
        self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        self.loadLabels(self.SAM_SHAPES_IN_IMAGE,replace=False)

        # later for confirmed sam instances 
        # self.laodLabels(self.CURRENT_SAM_SHAPES_IN_IMG,replace=false) 
    
    def sam_finish_annotation_button_clicked(self):
        # return the cursor to normal
        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        print("sam finish annotation button clicked")
        self.canvas.SAM_coordinates = []
        self.canvas.SAM_mode = "finished"
        try:
            self.sam_predictor.clear_logit()
        except:
            return
        self.labelList.clear()
        sam_qt_shape = convert_shapes_to_qt_shapes([self.current_sam_shape])[0]
        self.canvas.SAM_current = sam_qt_shape
        #print("sam qt shape", type(sam_qt_shape), "sam shape", type(self.current_sam_shape))
        self.canvas.finalise(SAM_SHAPE=True)
        self.SAM_SHAPES_IN_IMAGE.append(self.current_sam_shape)
        self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        self.loadLabels(self.SAM_SHAPES_IN_IMAGE, replace=False)
        # clear the predictor of the finished shape
        self.sam_predictor.clear_logit()
        self.canvas.SAM_coordinates = []
        # explicitly clear instead of being overriden by the next shape
        self.current_sam_shape = None
        self.canvas.SAM_current = None
        
        

        
        
    def run_sam_model(self):
        print("run sam model")
        # prepre the input format for SAM
        input_points = []
        input_labels = []
        for coordinate in self.canvas.SAM_coordinates:
            input_points.append([int(round(coordinate[0])), int(round(coordinate[1]))])
            input_labels.append(coordinate[2])
        input_points = np.array(input_points)
        input_labels = np.array(input_labels)
        if self.sam_predictor is None:
            print("please select a model")
            return
        else:
            print(self.canvas.SAM_coordinates,self.sam_predictor.mask_logit is None,len(self.CURRENT_SHAPES_IN_IMG))
            mask, score = self.sam_predictor.predict(point_coords=input_points, point_labels=input_labels)
            points = self.sam_predictor.mask_to_polygons(mask)
            shape = self.sam_predictor.polygon_to_shape(points, score)
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
            #self.canvas.loadPixmap(QtGui.QPixmap.fromImage(self.image))
            # we need it since it does replace=True on whatever was on the canvas
            self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
            self.loadLabels(self.SAM_SHAPES_IN_IMAGE,replace=False)
            self.loadLabels([self.current_sam_shape],replace=False)
            
            print("done running sam model")

        
        


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
