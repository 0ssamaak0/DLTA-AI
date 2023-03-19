# -*- coding: utf-8 -*-

import collections
import functools
import json
import math
import os
import os.path as osp
import re
import traceback
import webbrowser
import datetime
import glob
import imgviz
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtCore import QThread
from qtpy.QtCore import Signal as pyqtSignal
from qtpy import QtGui
from qtpy import QtWidgets

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
from .widgets import BrightnessContrastDialog , Canvas , LabelDialog , LabelListWidget , LabelListWidgetItem , ToolBar , UniqueLabelQListWidget , ZoomWidget
from .intelligence import Intelligence
from .intelligence import coco_classes , color_palette

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette

from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import cv2
from pathlib import Path

import time
import threading
import warnings
warnings.filterwarnings("ignore")





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

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
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

        self.setCentralWidget(scrollArea)

        # for Export
        self.target_directory = ""
        self.save_path = ""

        # for video annotation
        self.frame_time = 0
        self.FRAMES_TO_SKIP = 30
        self.TrackingMode = False
        self.CURRENT_ANNOATAION_FLAGS = {"traj" : False  ,
                                        "bbox" : True  ,         
                                        "id" : True ,
                                        "class" : True,
                                        "mask" : True,
                                        "polygons" : True}
        self.CURRENT_ANNOATAION_TRAJECTORIES = {}
        self.CURRENT_SHAPES_IN_IMG = []
        # make CLASS_NAMES_DICT a dictionary of coco class names
        # self.CLASS_NAMES_DICT =
        # self.frame_number = 0
        self.INDEX_OF_CURRENT_FRAME = 1

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
            lambda: self.toggleDrawMode(False, createMode="polygon"),
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
        set_threshold = action(
            self.tr("Set the threshold for the model"),
            self.setThreshold,
            None,
            None,
            self.tr("Set the threshold for the model")
        )
        select_classes = action(
            self.tr("select the classes to be annotated"),
            self.selectClasses,
            None,
            None,
            self.tr("select the classes to be annotated")
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
                copy,
                delete,
                undo,
                undoLastPoint,
                addPointToEdge,
                removePoint,
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
            # help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            saved_models=QtWidgets.QMenu(self.tr("Select model")),
            labelList=labelMenu,
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
                          set_threshold ,
                          self.menus.saved_models,
                          )
                         )
        # add one for the select classes action
        utils.addActions(self.menus.intelligence,
                         (annotate_one_action,
                          annotate_batch_action,
                          select_classes,

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
                icon = utils.newIcon("labels")
                action = QtWidgets.QAction(
                    icon, "&%d %s" % (i + 1, model_name), self)
                action.triggered.connect(functools.partial(
                    self.change_curr_model, model_name))
                menu.addAction(action)
                i += 1

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

    def editLabel(self, item=None):
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
        shape.group_id = group_id
        shape.content = content
        if shape.group_id is None:
            item.setText(shape.label)
        else:
            item.setText(f' ID {shape.group_id}:\t\t{shape.label}')
        self.setDirty()
        if not self.uniqLabelList.findItemsByLabel(shape.label):
            item = QtWidgets.QListWidgetItem()
            item.setData(Qt.UserRole, shape.label)
            self.uniqLabelList.addItem(item)

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
            text = f' ID {shape.group_id}:\t\t{shape.label}'
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
            item = self.uniqLabelList.findItemsByLabel(label)[0]
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
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
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)
        for shape in self.canvas.shapes:
            self.canvas.setShapeVisible(shape, self.CURRENT_ANNOATAION_FLAGS["polygons"])

    def loadLabels(self, shapes):
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
        self.loadShapes(s)

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
        # #image = QtGui.QImage('test_img_1.jpg')
        # img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # #im_np = np.transpose(im_np, (1,0,2))
        # im_np = np.array(img)
        # image = QtGui.QImage(im_np.data, im_np.shape[1], im_np.shape[0],
        #          QtGui.QImage.Format_BGR888)

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
        self.intelligenceHelper.current_model_name , self.intelligenceHelper.current_mm_model = self.intelligenceHelper.make_mm_model(
            model_name)

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

        self.current_annotation_mode = "img"
        self.actions.export.setEnabled(False)
        try :
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
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.labelList.clearSelection()
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        self.current_annotation_mode = "dir"
        try :
            cv2.destroyWindow('video processing')
        except:
            pass
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
        else:
            if os.path.exists(self.filename):
                self.labelList.clearSelection()
            shapes = self.intelligenceHelper.get_shapes_of_one(
                self.filename)  
            
        # for shape in shapes:
        #     print(shape["group_id"])
        #     print(shape["bbox"])
        
        
        
        # image = self.draw_bb_on_image(self.image  , self.CURRENT_SHAPES_IN_IMG)
        # self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        
        
        self.CURRENT_SHAPES_IN_IMG = shapes
        self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        # self.loadShapes(shapes)
        self.actions.editMode.setEnabled(True)
        self.actions.undoLastPoint.setEnabled(False)
        self.actions.undo.setEnabled(True)
        self.setDirty()

            


    def annotate_batch(self):
        images = []
        for filename in self.imageList:
            images.append(filename)
        self.intelligenceHelper.get_shapes_of_batch(images)

    def setThreshold(self):
        self.intelligenceHelper.threshold = self.intelligenceHelper.setThreshold()

    def selectClasses(self):
        self.intelligenceHelper.selectedclasses = self.intelligenceHelper.selectClasses()

    # VIDEO PROCESSING FUNCTIONS (ALL CONNECTED TO THE VIDEO PROCESSING TOOLBAR)

    def mapFrameToTime(self , frameNumber):
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
        return frameHours, frameMinutes, frameSeconds , frameMilliseconds

    def openVideo(self):
        self.CURRENT_ANNOATAION_TRAJECTORIES = {}
        # self.videoControls.show()
        self.current_annotation_mode = "video"
        try :
            cv2.destroyWindow('video processing')
        except:
            pass
        if not self.mayContinue():
            return
        videoFile = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("%s - Choose Video") % __appname__, ".",
            self.tr("Video files (*.mp4 *.avi)")
        )

        if videoFile[0] :
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
            self.TOTAL_VIDEO_FRAMES = self.CAP.get(
                cv2.CAP_PROP_FRAME_COUNT) - 1
            self.CURRENT_VIDEO_FPS = self.CAP.get(cv2.CAP_PROP_FPS)
            print("Total Frames : " , self.TOTAL_VIDEO_FRAMES)
            self.main_video_frames_slider.setMaximum(self.TOTAL_VIDEO_FRAMES)
            self.main_video_frames_slider.setValue(2)
            self.INDEX_OF_CURRENT_FRAME = 1
            self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME)

            # self.addToolBarBreak

            self.set_video_controls_visibility(True)

            self.byte_tracker = BYTETracker(BYTETrackerArgs())

        # label = shape["label"]
        # points = shape["points"]
        # bbox = shape["bbox"]
        # shape_type = shape["shape_type"]
        # flags = shape["flags"]
        # content = shape["content"]
        # group_id = shape["group_id"]
        # other_data = shape["other_data"]
    def load_shapes_for_video_frame(self , json_file_name, index):
        # this function loads the shapes for the video frame from the json file
        # first we read the json file in the form of a list
        # we need to parse from it data for the current frame

        target_frame_idx = index
        listObj = []
        with open(json_file_name , "r") as json_file:
            listObj = json.load(json_file)

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


            self.CURRENT_SHAPES_IN_IMG = shapes



    def loadFramefromVideo(self, frame_array, index=1):
        # filename = str(index) + ".jpg"
        #self.filename = filename
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
            print("Tracking Mode" , len(self.CURRENT_SHAPES_IN_IMG))
            image = self.draw_bb_on_image(image, self.CURRENT_SHAPES_IN_IMG)
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
            if len(self.CURRENT_SHAPES_IN_IMG) > 0:
                self.loadLabels(self.CURRENT_SHAPES_IN_IMG)
        else:   
            if self.labelFile:
                self.CURRENT_SHAPES_IN_IMG = self.labelFile.shapes
                image = self.draw_bb_on_image(image, self.CURRENT_SHAPES_IN_IMG)
                self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
                self.loadLabels(self.labelFile.shapes)
                if self.labelFile.flags is not None:
                    flags.update(self.labelFile.flags)
                    
            else :
                json_file_name = f'{self.CURRENT_VIDEO_PATH}/{self.CURRENT_VIDEO_NAME}_tracking_results.json'
                if os.path.exists(json_file_name):
                    # print('json file exists , loading shapes')
                    self.load_shapes_for_video_frame(json_file_name , index)    
                    image = self.draw_bb_on_image(image, self.CURRENT_SHAPES_IN_IMG)
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
        # first assert that the new value of the slider is not greater than the total number of frames
        new_value = self.INDEX_OF_CURRENT_FRAME + self.FRAMES_TO_SKIP
        if new_value >= self.TOTAL_VIDEO_FRAMES:
            new_value = self.TOTAL_VIDEO_FRAMES
        self.main_video_frames_slider.setValue(new_value)

    def next_1_Frame_buttonClicked(self):
        # first assert that the new value of the slider is not greater than the total number of frames
        new_value = self.INDEX_OF_CURRENT_FRAME + 1
        if new_value >= self.TOTAL_VIDEO_FRAMES:
            new_value = self.TOTAL_VIDEO_FRAMES
        self.main_video_frames_slider.setValue(new_value)

    def previousFrame_buttonClicked(self):
        new_value = self.INDEX_OF_CURRENT_FRAME - self.FRAMES_TO_SKIP
        if new_value <= 0:
            new_value = 0
        self.main_video_frames_slider.setValue(new_value)

    def previous_1_Frame_buttonclicked(self):
        new_value = self.INDEX_OF_CURRENT_FRAME - 1
        if new_value <= 0:
            new_value = 0
        self.main_video_frames_slider.setValue(new_value)

    def frames_to_skip_slider_changed(self):
        self.FRAMES_TO_SKIP = self.frames_to_skip_slider.value()
        zeros = ( 2 - int(np.log10(self.FRAMES_TO_SKIP + 0.9)) ) * '0'
        self.frames_to_skip_label.setText(
            'frames to skip: ' + zeros + str(self.FRAMES_TO_SKIP))

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
        frame_idx = self.main_video_frames_slider.value()

        self.INDEX_OF_CURRENT_FRAME = frame_idx
        self.CAP.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)

        # setting text of labels
        zeros = ( int(np.log10(self.TOTAL_VIDEO_FRAMES + 0.9)) - int(np.log10(frame_idx + 0.9)) ) * '0'
        self.main_video_frames_label_1.setText(
            f'frame {zeros}{frame_idx} / {int(self.TOTAL_VIDEO_FRAMES)}')
        self.frame_time = self.mapFrameToTime(frame_idx)
        frame_text = ("%02d:%02d:%02d:%03d" % (
            self.frame_time[0], self.frame_time[1], self.frame_time[2] , self.frame_time[3]))
        video_duration = self.mapFrameToTime(self.TOTAL_VIDEO_FRAMES)
        video_duration_text = ("%02d:%02d:%02d:%03d     " % (
            video_duration[0], video_duration[1], video_duration[2] , video_duration[3]))
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
        zeros = ( 2 - int(np.log10(self.FRAMES_TO_TRACK + 0.9)) ) * '0'
        self.frames_to_track_label.setText(
            f'track for {zeros}{self.FRAMES_TO_TRACK} frames')

    def move_frame_by_frame(self):
        self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME + 1)

    def class_name_to_id(self, class_name):
        try :
            # map from coco_classes(a list of coco class names) to class_id
            return coco_classes.index(class_name)
        except:
            # this means that the class name is not in the coco dataset
            return -1

    def track_buttonClicked(self):
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
        # with open (json_file_name, 'r') as json_file:
        #     json_object = json.load(json_file)
        #     print(json_object)

        # first check if there is any detection in the current frame (and items in self.labelList)
        # if not --> perform detection on the current frame
        # shapes = [(item.shape()) for item in self.labelList]
        # if len(shapes) == 0:
        #     print('no detection in the current frame so performing detection')
        #     self.annotate_one()
        #     shapes = [(item.shape()) for item in self.labelList]

        # now we have the shapes of the detections in the current frame
        # steps
        # loop from the current frame to the end frame
        # each time we loop we need to perform detection on the current frame and then track the detections
        # output the tracking and detection results in labellist
        # then save the tracked detection in a json file named (video_name_tracking_results.json)
        existing_annotation = False
        shapes = self.canvas.shapes
        if len(shapes) > 0:
            existing_annotation = True
            print(f'FRAME{self.INDEX_OF_CURRENT_FRAME}' , len(shapes))
            for shape in shapes:
                print(shape.label)

        self.TrackingMode = True
        for i in range(self.FRAMES_TO_TRACK):
            if existing_annotation:
                print('\n\n\n\n\nloading existing annotation/n/n')
                existing_annotation = False
                shapes = self.canvas.shapes
                shapes = self.convert_qt_shapes_to_shapes(shapes)
            else :
                shapes = self.intelligenceHelper.get_shapes_of_one(
                    self.CURRENT_FRAME_IMAGE, img_array_flag=True)


            boxes = []
            confidences = []
            class_ids = []
            segments = []
            # current_objects_ids = []
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
                # segments.append(np.array(points, dtype=int).tolist())
                segments.append(segment)

                boxes.append(self.intelligenceHelper.get_bbox(segment))
                if s["content"] is None:
                    confidences.append(1.0)
                else :
                    confidences.append(float(s["content"]))
                class_ids.append(int(self.class_name_to_id(label)))

            boxes = np.array(boxes , dtype=int)
            confidences = np.array(confidences)
            class_ids = np.array(class_ids)
            detections = Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids,
            )


            tracks = self.byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame_shape,
                img_size=frame_shape
            )
            tracker_id = match_detections_with_tracks(
                detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)


            # print(f'detections : {len(detections)}')
            # print(f'tracks : {len(tracker_id)}')
            
        
            # print(f'len of shapes = {len(shapes)}')
            # print(f'len of detections.tracker_id = {len(detections.tracker_id)}') 
            # # masks shapes when traker_id is None
            
            # make new list of shapes to be added to CURRENT_SHAPES_IN_IMG 

            
            
            # filtering out detections without trackers
            for a, shape_ in enumerate(shapes):
                shape_["group_id"] = tracker_id[a]
            self.CURRENT_SHAPES_IN_IMG = [shape_ for shape_ in shapes if shape_["group_id"] is not None]
            mask = np.array(
                [tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

        # to understand the json output file structure it is a dictionary of frames and each frame is a dictionary of tracker_ids and each tracker_id is a dictionary of bbox , confidence , class_id , segment
            json_frame = {}
            json_frame.update({'frame_idx' : self.INDEX_OF_CURRENT_FRAME})
            json_frame_object_list = []
            tracked_objects_list = []
            for j in range(len(detections.tracker_id)):
                json_tracked_object = {}
                json_tracked_object['tracker_id'] = int(
                    detections.tracker_id[j])
                json_tracked_object['bbox'] = detections.xyxy[j].tolist()

                json_tracked_object['confidence'] = str(
                    detections.confidence[j])
                json_tracked_object['class_id'] = int(detections.class_id[j])
                points = self.CURRENT_SHAPES_IN_IMG[j]["points"]
                segment = [[int(points[z]), int(points[z + 1])] for z in range(0, len(points), 2)]
                json_tracked_object['segment'] = segment

                json_frame_object_list.append(json_tracked_object)

            json_frame.update({'frame_data' : json_frame_object_list})
            # dictObj.update({self.INDEX_OF_CURRENT_FRAME : json_frame_data})
            # update the json file with the new frame data even if the frame already exists
            # first check if the frame already exists in the json file if exists then delete

            # dictObj.pop(self.INDEX_OF_CURRENT_FRAME , None)
            # first check if the frame already exists in the json file if exists then delete
            for h in range(len(listObj)):
                if listObj[h]['frame_idx'] == self.INDEX_OF_CURRENT_FRAME:
                    listObj.pop(h)
                    break
            # sort the list of frames by the frame index

            listObj.append(json_frame)

            # self.loadShapes(shapes)
            # self.actions.editMode.setEnabled(True)
            # self.actions.undoLastPoint.setEnabled(False)
            # self.actions.undo.setEnabled(True)
            # self.setDirty()
            # qt sleep for 1 second
            # self.window_wait(1)
            print('finished tracking for frame ' , self.INDEX_OF_CURRENT_FRAME)
            if i != self.FRAMES_TO_TRACK - 1:
                self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME + 1)
            self.tracking_progress_bar.setValue(
                int((i + 1) / self.FRAMES_TO_TRACK * 100))

        listObj = sorted(listObj, key=lambda k: k['frame_idx'])
        with open(json_file_name, 'w') as json_file:
            json.dump(listObj, json_file ,
                      indent=4,
                      separators=(',', ': '))
        json_file.close()

        self.TrackingMode = False
        self.labelFile = None
        self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME - 1)
        self.main_video_frames_slider.setValue(self.INDEX_OF_CURRENT_FRAME + 1)

        self.tracking_progress_bar.hide()
        self.tracking_progress_bar.setValue(0)



    def convert_qt_shapes_to_shapes(self, qt_shapes):
        shapes = []
        for s in qt_shapes:
            shapes.append(dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    # convert points into 1D array
                    points=self.flattener(s.points),
                    bbox=self.intelligenceHelper.get_bbox([(p.x(), p.y()) for p in s.points]),
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

    def set_video_controls_visibility(self , visible=False):
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

    def window_wait(self, seconds):
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(seconds * 1000, loop.quit)
        loop.exec_()










    def traj_checkBox_changed(self):
        self.CURRENT_ANNOATAION_FLAGS["traj"] = self.traj_checkBox.isChecked()
        self.main_video_frames_slider_changed()
    def mask_checkBox_changed(self):
        self.CURRENT_ANNOATAION_FLAGS["mask"] = self.mask_checkBox.isChecked()
        self.main_video_frames_slider_changed()

    def class_checkBox_changed(self):
        self.CURRENT_ANNOATAION_FLAGS["class"] = self.class_checkBox.isChecked()
        self.main_video_frames_slider_changed()

    def id_checkBox_changed(self):
        self.CURRENT_ANNOATAION_FLAGS["id"] = self.id_checkBox.isChecked()
        self.main_video_frames_slider_changed()

    def bbox_checkBox_changed(self):
        self.CURRENT_ANNOATAION_FLAGS["bbox"] = self.bbox_checkBox.isChecked()
        self.main_video_frames_slider_changed()



    def polygons_visable_checkBox_changed(self):
        self.CURRENT_ANNOATAION_FLAGS["polygons"] = self.polygons_visable_checkBox.isChecked()
        self.main_video_frames_slider_changed()
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
            "QToolBar#videoControls_2s { border: 50px }")
        self.addToolBar(Qt.TopToolBarArea, self.videoControls_2)

        self.frames_to_skip_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frames_to_skip_slider.setMinimum(1)
        self.frames_to_skip_slider.setMaximum(100)
        self.frames_to_skip_slider.setValue(3)
        self.frames_to_skip_slider.setTickPosition(
            QtWidgets.QSlider.TicksBelow)
        self.frames_to_skip_slider.setTickInterval(1)
        self.frames_to_skip_slider.setMaximumWidth(150)
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
        self.main_video_frames_slider.setMaximumWidth(600)
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

        self.frames_to_track_label = QtWidgets.QLabel()
        self.frames_to_track_label.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; }")  # make the button text red
        self.videoControls_2.addWidget(self.frames_to_track_label)
        self.videoControls_2.addWidget(self.frames_to_track_slider)
        self.frames_to_track_slider.setValue(10)

        # self.dummy_label = QtWidgets.QLabel()
        # self.dummy_label.setText(" ")
        # self.videoControls.addWidget(self.dummy_label)

        self.track_button = QtWidgets.QPushButton()
        self.track_button.setStyleSheet(
            "QPushButton {font-size: 10pt; margin: 2px 5px; padding: 2px 7px; border: 2px solid; border-radius:10px; font-weight: bold; background-color: #0d69f5; color: #FFFFFF;} QPushButton:hover {background-color: #4990ED;}")

        self.track_button.setText("Track")
        self.track_button.clicked.connect(self.track_buttonClicked)
        self.videoControls_2.addWidget(self.track_button)

        # make a tracking progress bar with a label to show the progress of the tracking (in percentage )
        self.tracking_progress_bar_label = QtWidgets.QLabel()
        self.tracking_progress_bar_label.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; }")
        self.tracking_progress_bar_label.setText("Tracking Progress")
        self.videoControls_2.addWidget(self.tracking_progress_bar_label)

        self.tracking_progress_bar = QtWidgets.QProgressBar()
        self.tracking_progress_bar.setMaximumWidth(200)
        self.tracking_progress_bar.setMinimum(0)
        self.tracking_progress_bar.setMaximum(100)
        self.tracking_progress_bar.setValue(0)
        self.videoControls_2.addWidget(self.tracking_progress_bar)

        self.track_full_video_button = QtWidgets.QPushButton()
        self.track_full_video_button.setStyleSheet(
            "QPushButton {font-size: 10pt; margin: 2px 5px; padding: 2px 7px; border: 2px solid; border-radius:10px; font-weight: bold; background-color: #0d69f5; color: #FFFFFF;} QPushButton:hover {background-color: #4990ED;}")

        self.track_full_video_button.setText("Track Full Video")
        self.track_full_video_button.clicked.connect(
            self.track_full_video_button_clicked)
        self.videoControls_2.addWidget(self.track_full_video_button)


        # add 5 checkboxes to control the CURRENT ANNOATAION FLAGS including (bbox , id , class , mask , traj)
        self.bbox_checkBox = QtWidgets.QCheckBox()
        self.bbox_checkBox.setText("bbox")
        self.bbox_checkBox.setChecked(True)
        self.bbox_checkBox.stateChanged.connect(self.bbox_checkBox_changed)
        self.videoControls_2.addWidget(self.bbox_checkBox)

        self.id_checkBox = QtWidgets.QCheckBox()
        self.id_checkBox.setText("id")
        self.id_checkBox.setChecked(True)
        self.id_checkBox.stateChanged.connect(self.id_checkBox_changed)
        self.videoControls_2.addWidget(self.id_checkBox)

        self.class_checkBox = QtWidgets.QCheckBox()
        self.class_checkBox.setText("class")
        self.class_checkBox.setChecked(True)
        self.class_checkBox.stateChanged.connect(self.class_checkBox_changed)
        self.videoControls_2.addWidget(self.class_checkBox)

        self.mask_checkBox = QtWidgets.QCheckBox()
        self.mask_checkBox.setText("mask")
        self.mask_checkBox.setChecked(True)
        self.mask_checkBox.stateChanged.connect(self.mask_checkBox_changed)
        self.videoControls_2.addWidget(self.mask_checkBox)

        self.traj_checkBox = QtWidgets.QCheckBox()
        self.traj_checkBox.setText("traj")
        self.traj_checkBox.setChecked(False)
        self.traj_checkBox.stateChanged.connect(self.traj_checkBox_changed)
        self.videoControls_2.addWidget(self.traj_checkBox)
        
        self.polygons_visable_checkBox = QtWidgets.QCheckBox()
        self.polygons_visable_checkBox.setText("polygons visablity")
        self.polygons_visable_checkBox.setChecked(True)
        self.polygons_visable_checkBox.stateChanged.connect(self.polygons_visable_checkBox_changed)
        self.videoControls_2.addWidget(self.polygons_visable_checkBox)


        self.set_video_controls_visibility(False)








    def convert_QI_to_cv(self , incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(4)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr

    def convert_cv_to_qt(self ,cv_img):
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            return convert_to_Qt_format

    def draw_bb_id(self ,image, x, y , w, h, id,label, color=(0, 0, 255), thickness=1):
        if self.CURRENT_ANNOATAION_FLAGS['bbox']:
            image = cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness+1)
            
            
        if self.CURRENT_ANNOATAION_FLAGS['id']  or self.CURRENT_ANNOATAION_FLAGS['class']:
            if self.CURRENT_ANNOATAION_FLAGS['id']  and self.CURRENT_ANNOATAION_FLAGS['class']:
                text = f'#{id} {label}'
            if self.CURRENT_ANNOATAION_FLAGS['id']  and not self.CURRENT_ANNOATAION_FLAGS['class']:
                text = f'#{id}'
            if not self.CURRENT_ANNOATAION_FLAGS['id']  and self.CURRENT_ANNOATAION_FLAGS['class']:
                text = f'{label}'
                
                
            text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness)[0]
            text_x = x + 10
            text_y = y - 10

            text_background_x1 = x
            text_background_y1 = y - 2 * 10 - text_height

            text_background_x2 = x + 2 * 10 + text_width
            text_background_y2 = y
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
                fontScale=0.7,
                color=(0, 0, 0),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )    
                

                
        # there is no bbox but there is id or class
        if (not self.CURRENT_ANNOATAION_FLAGS['bbox']) and (self.CURRENT_ANNOATAION_FLAGS['id'] or self.CURRENT_ANNOATAION_FLAGS['class']):
            image = cv2.line(image, (x+int(w/2), y + int(h/2)), (x + 50, y - 12), color, thickness+1)
        
            
        return image


    def draw_trajectories(self ,img, shapes):
        for shape in shapes:
            id = shape["group_id"]
            pts = self.CURRENT_ANNOATAION_TRAJECTORIES['id_'+str(id)]
            color = self.CURRENT_ANNOATAION_TRAJECTORIES['id_color_'+str(id)]
            for i in range(1, len(pts)):
                thickness = 2
                thickness = int(np.sqrt(30 / float(i + 1)) * 2)
                if pts[i - 1] is None or pts[i] is None:
                    continue
                cv2.line(img, pts[i - 1], pts[i], color, thickness)
                    
        return img

    def draw_bb_on_image(self ,image, shapes):

        
        img = self.convert_QI_to_cv(image)
        
        for shape in shapes:
            id = shape["group_id"]
            label = shape["label"]
            
            # color calculation
            idx = coco_classes.index(label)
            idx = idx % len(color_palette)
            color = color_palette[idx]
            
            (x1, y1 , x2, y2) = shape["bbox"]
            x, y , w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            img = self.draw_bb_id(img, x, y , w, h, id,label, color, thickness = 1)
            
            if self.CURRENT_ANNOATAION_FLAGS['traj']:   
                center = ( int((x1 + x2) / 2), int((y1 + y2) / 2) )
                try:
                    self.CURRENT_ANNOATAION_TRAJECTORIES['id_'+str(id)].appendleft(center)
                except:
                    self.CURRENT_ANNOATAION_TRAJECTORIES['id_'+str(id)] = collections.deque(maxlen = 30)
                    self.CURRENT_ANNOATAION_TRAJECTORIES['id_'+str(id)].appendleft(center)
                    self.CURRENT_ANNOATAION_TRAJECTORIES['id_color_'+str(id)] = color
        
        
        if self.CURRENT_ANNOATAION_FLAGS['traj']:   
            img = self.draw_trajectories(img, shapes)
        
        qimage = self.convert_cv_to_qt(img, )
        return qimage



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
