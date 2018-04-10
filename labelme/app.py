import argparse
import collections
import functools
import os.path
import re
import subprocess
import sys

try:
    from PyQt5 import QtCore
    from PyQt5.QtCore import Qt
    from PyQt5 import QtGui
    from PyQt5 import QtWidgets
    PYQT5 = True
except ImportError:
    from PyQt4 import QtCore
    from PyQt4.QtCore import Qt
    from PyQt4 import QtGui
    from PyQt4 import QtGui as QtWidgets
    PYQT5 = False

from labelme.canvas import Canvas
from labelme.colorDialog import ColorDialog
from labelme.labelDialog import LabelDialog
from labelme.labelFile import LabelFile
from labelme.labelFile import LabelFileError
from labelme.lib import addActions
from labelme.lib import fmtShortcut
from labelme.lib import newAction
from labelme.lib import newIcon
from labelme.lib import struct
from labelme import resources  # NOQA
from labelme.shape import DEFAULT_FILL_COLOR
from labelme.shape import DEFAULT_LINE_COLOR
from labelme.shape import Shape
from labelme.toolBar import ToolBar
from labelme.zoomWidget import ZoomWidget


__appname__ = 'labelme'


# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - [high] Automatically add file suffix when saving.
# - [high] Add polygon movement with arrow keys
# - [high] Deselect shape when clicking and already selected(?)
# - [high] Sanitize shortcuts between beginner/advanced mode.
# - [medium] Zoom should keep the image centered.
# - [medium] Add undo button for vertex addition.
# - [low,maybe] Open images with drag & drop.
# - [low,maybe] Preview images on file dialogs.
# - [low,maybe] Sortable label list.
# - Zoom is too "steppy".


# Utility functions and classes.


class WindowMixin(object):
    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName('%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class EscapableQListWidget(QtWidgets.QListWidget):

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.clearSelection()


class LabelQListWidget(QtWidgets.QListWidget):

    def __init__(self, *args, **kwargs):
        super(LabelQListWidget, self).__init__(*args, **kwargs)
        self.canvas = None
        self.itemsToShapes = []

    def get_shape_from_item(self, item):
        for index, (item_, shape) in enumerate(self.itemsToShapes):
            if item_ == item:
                return shape

    def get_item_from_shape(self, shape):
        for index, (item, shape_) in enumerate(self.itemsToShapes):
            if shape_ == shape:
                return item

    def clear(self):
        super(LabelQListWidget, self).clear()
        self.itemsToShapes = []

    def setParent(self, parent):
        self.parent = parent

    def dropEvent(self, event):
        shapes = self.shapes
        super(LabelQListWidget, self).dropEvent(event)
        if self.shapes == shapes:
            return
        if self.canvas is None:
            raise RuntimeError('self.canvas must be set beforehand.')
        self.parent.setDirty()
        self.canvas.shapes = self.shapes

    @property
    def shapes(self):
        shapes = []
        for i in range(self.count()):
            item = self.item(i)
            shape = self.get_shape_from_item(item)
            shapes.append(shape)
        return shapes


class MainWindow(QtWidgets.QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(self, filename=None, output=None, store_data=True,
                 labels=None, sort_labels=True):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = "firefox"
        self.screencast = "screencast.ogv"

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, labels=labels,
                                       sort_labels=sort_labels)

        self.labelList = LabelQListWidget()
        self.lastOpenDir = None

        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.setDragDropMode(
            QtWidgets.QAbstractItemView.InternalMove)
        self.labelList.setParent(self)

        listLayout = QtWidgets.QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)
        self.editButton = QtWidgets.QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        listLayout.addWidget(self.editButton)  # 0, Qt.AlignCenter)
        listLayout.addWidget(self.labelList)
        self.labelListContainer = QtWidgets.QWidget()
        self.labelListContainer.setLayout(listLayout)

        self.uniqLabelList = EscapableQListWidget()
        self.uniqLabelList.setToolTip(
            "Select label to start annotating for it. "
            "Press 'Esc' to deselect.")
        if labels:
            self.uniqLabelList.addItems(labels)
            self.uniqLabelList.sortItems()
        self.labelsdock = QtWidgets.QDockWidget(u'Label List', self)
        self.labelsdock.setObjectName(u'Label List')
        self.labelsdock.setWidget(self.uniqLabelList)

        self.dock = QtWidgets.QDockWidget('Polygon Labels', self)
        self.dock.setObjectName('Labels')
        self.dock.setWidget(self.labelListContainer)

        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(
            self.fileSelectionChanged)
        filelistLayout = QtWidgets.QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QtWidgets.QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QtWidgets.QDockWidget(u'File List', self)
        self.filedock.setObjectName(u'Files')
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = self.labelList.canvas = Canvas()
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)

        self.addDockWidget(Qt.RightDockWidgetArea, self.labelsdock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable)

        self.dockFeatures = (QtWidgets.QDockWidget.DockWidgetClosable |
                             QtWidgets.QDockWidget.DockWidgetFloatable)
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Actions
        action = functools.partial(newAction, self)
        quit = action('&Quit', self.close, 'Ctrl+Q', 'quit',
                      'Quit application')
        open = action('&Open', self.openFile, 'Ctrl+O', 'open',
                      'Open image or label file')
        opendir = action('&Open Dir', self.openDirDialog, 'Ctrl+u', 'open',
                         u'Open Dir')
        openNextImg = action('&Next Image', self.openNextImg, 'd', 'next',
                             u'Open Next')

        openPrevImg = action('&Prev Image', self.openPrevImg, 'a', 'prev',
                             u'Open Prev')
        save = action('&Save', self.saveFile, 'Ctrl+S', 'save',
                      'Save labels to file', enabled=False)
        saveAs = action('&Save As', self.saveFileAs, 'Ctrl+Shift+S', 'save-as',
                        'Save labels to a different file', enabled=False)
        close = action('&Close', self.closeFile, 'Ctrl+W', 'close',
                       'Close current file')
        color1 = action('Polygon &Line Color', self.chooseColor1, 'Ctrl+L',
                        'color_line', 'Choose polygon line color')
        color2 = action('Polygon &Fill Color', self.chooseColor2,
                        'Ctrl+Shift+L', 'color', 'Choose polygon fill color')

        createMode = action('Create\nPolygo&ns', self.setCreateMode, 'Ctrl+N',
                            'new', 'Start drawing polygons', enabled=False)
        editMode = action('&Edit\nPolygons', self.setEditMode, 'Ctrl+J',
                          'edit', 'Move and edit polygons', enabled=False)

        create = action('Create\nPolygo&n', self.createShape, 'Ctrl+N', 'new',
                        'Draw a new polygon', enabled=False)
        delete = action('Delete\nPolygon', self.deleteSelectedShape, 'Delete',
                        'delete', 'Delete', enabled=False)
        copy = action('&Duplicate\nPolygon', self.copySelectedShape, 'Ctrl+D',
                      'copy', 'Create a duplicate of the selected polygon',
                      enabled=False)
        undoLastPoint = action('Undo last point', self.canvas.undoLastPoint,
                               ['Ctrl+Z', 'Backspace'], 'undoLastPoint',
                               'Undo last drawn point', enabled=False)

        advancedMode = action('&Advanced Mode', self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert',
                              'Switch to advanced mode', checkable=True)

        hideAll = action('&Hide\nPolygons',
                         functools.partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', 'Hide all polygons', enabled=False)
        showAll = action('&Show\nPolygons',
                         functools.partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', 'Show all polygons', enabled=False)

        help = action('&Tutorial', self.tutorial, 'Ctrl+T', 'help',
                      'Show screencast of introductory tutorial')

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            "Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." %
            (fmtShortcut("Ctrl+[-+]"), fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action('Zoom &In', functools.partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', 'Increase zoom level',
                        enabled=False)
        zoomOut = action('&Zoom Out', functools.partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', 'Decrease zoom level',
                         enabled=False)
        zoomOrg = action('&Original size',
                         functools.partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', 'Zoom to original size',
                         enabled=False)
        fitWindow = action('&Fit Window', self.setFitWindow, 'Ctrl+F',
                           'fit-window', 'Zoom follows window size',
                           checkable=True, enabled=False)
        fitWidth = action('Fit &Width', self.setFitWidth, 'Ctrl+Shift+F',
                          'fit-width', 'Zoom follows window width',
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut, zoomOrg,
                       fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action('&Edit Label', self.editLabel, 'Ctrl+E', 'edit',
                      'Modify the label of the selected polygon',
                      enabled=False)
        self.editButton.setDefaultAction(edit)

        shapeLineColor = action(
            'Shape &Line Color', self.chshapeLineColor, icon='color_line',
            tip='Change the line color for this specific shape', enabled=False)
        shapeFillColor = action(
            'Shape &Fill Color', self.chshapeFillColor, icon='color',
            tip='Change the fill color for this specific shape', enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText('Show/Hide Label Panel')
        labels.setShortcut('Ctrl+Shift+L')

        # Lavel list context menu.
        labelMenu = QtWidgets.QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = struct(
            save=save, saveAs=saveAs, open=open, close=close,
            lineColor=color1, fillColor=color2,
            create=create, delete=delete, edit=edit, copy=copy,
            undoLastPoint=undoLastPoint,
            createMode=createMode, editMode=editMode,
            advancedMode=advancedMode,
            shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
            zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
            fitWindow=fitWindow, fitWidth=fitWidth,
            zoomActions=zoomActions,
            fileMenuActions=(open, opendir, save, saveAs, close, quit),
            beginner=(), advanced=(),
            editMenu=(edit, copy, delete, None, undoLastPoint,
                      None, color1, color2),
            beginnerContext=(create, edit, copy, delete, undoLastPoint),
            advancedContext=(
                createMode, editMode, edit, copy,
                delete, shapeLineColor, shapeFillColor,
                undoLastPoint,
            ),
            onLoadActive=(close, create, createMode, editMode),
            onShapesPresent=(saveAs, hideAll, showAll),
        )

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles=QtWidgets.QMenu('Open &Recent'),
            labelList=labelMenu,
        )

        addActions(self.menus.file, (open, opendir, self.menus.recentFiles,
                                     save, saveAs, close, None, quit))
        addActions(self.menus.help, (help,))
        addActions(self.menus.view, (
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            open, opendir, openNextImg, openPrevImg, save,
            None, create, copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            open, opendir, openNextImg, openPrevImg, save, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.labeling_once = output is not None
        self.output = output
        self._store_data = store_data
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False

        if filename is not None and os.path.isdir(filename):
            self.importDirImages(filename)
        else:
            self.filename = filename

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = {}
        self.recentFiles = self.settings.get('recentFiles', [])
        size = self.settings.get('window/size', QtCore.QSize(600, 500))
        position = self.settings.get('window/position', QtCore.QPoint(0, 0))
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(
            self.settings.get('window/state', QtCore.QByteArray()))
        self.lineColor = QtGui.QColor(
            self.settings.get('line/color', Shape.line_color))
        self.fillColor = QtGui.QColor(
            self.settings.get('fill/color', Shape.fill_color))
        Shape.line_color = self.lineColor
        Shape.fill_color = self.fillColor

        if self.settings.get('advanced', QtCore.QVariant()):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()
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

    # Support Functions

    def noShapes(self):
        return not self.labelList.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner() else \
                  (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = '{} - {}*'.format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = '{} - {}'.format(title, self.filename)
        self.setWindowTitle(title)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imageData = None
        self.labelFile = None
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

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    # Callbacks

    def tutorial(self):
        subprocess.Popen([self.screencastViewer, self.screencast])

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return os.path.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QtWidgets.QAction(
                icon, '&%d %s' % (i + 1, QtCore.QFileInfo(f).fileName()), self)
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self, item=None):
        if not self.canvas.editing():
            return
        item = item if item else self.currentItem()
        text = self.labelDialog.popUp(item.text())
        if text is None:
            return
        item.setText(text)
        self.setDirty()
        if not self.uniqLabelList.findItems(text, Qt.MatchExactly):
            self.uniqLabelList.addItem(text)
            self.uniqLabelList.sortItems()

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
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                item = self.labelList.get_item_from_shape(shape)
                item.setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        item = QtWidgets.QListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.labelList.itemsToShapes.append((item, shape))
        self.labelList.addItem(item)
        if not self.uniqLabelList.findItems(shape.label, Qt.MatchExactly):
            self.uniqLabelList.addItem(shape.label)
            self.uniqLabelList.sortItems()
        self.labelDialog.addLabelHistory(item.text())
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

    def remLabel(self, shape):
        item = self.labelList.get_item_from_shape(shape)
        self.labelList.takeItem(self.labelList.row(item))

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color in shapes:
            shape = Shape(label=label)
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()
            s.append(shape)
            self.addLabel(shape)
            if line_color:
                shape.line_color = QtGui.QColor(*line_color)
            if fill_color:
                shape.fill_color = QtGui.QColor(*fill_color)
        self.canvas.loadShapes(s)

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            return dict(label=str(s.label),
                        line_color=s.line_color.getRgb()
                        if s.line_color != self.lineColor else None,
                        fill_color=s.fill_color.getRgb()
                        if s.fill_color != self.fillColor else None,
                        points=[(p.x(), p.y()) for p in s.points])

        shapes = [format_shape(shape) for shape in self.labelList.shapes]
        try:
            imagePath = os.path.relpath(
                self.imagePath, os.path.dirname(filename))
            imageData = self.imageData if self._store_data else None
            lf.save(filename, shapes, imagePath, imageData,
                    self.lineColor.getRgb(), self.fillColor.getRgb())
            self.labelFile = lf
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage('Error saving label data', '<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            shape = self.labelList.get_shape_from_item(item)
            self.canvas.selectShape(shape)

    def labelItemChanged(self, item):
        shape = self.labelList.get_shape_from_item(item)
        label = str(item.text())
        if label != shape.label:
            shape.label = str(item.text())
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].text()
        text = self.labelDialog.popUp(text)
        if text is not None:
            self.addLabel(self.canvas.setLastLabel(text))
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()

    def scrollRequest(self, delta, orientation):
        units = - delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        units = delta * 0.1
        self.addZoom(units)

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

    def togglePolygons(self, value):
        for item, shape in self.labelList.itemsToShapes:
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.get('filename', '')
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                'Error opening file', 'No such file: <b>%s</b>' % filename)
            return False
        # assumes same name, but json extension
        self.status("Loading %s..." % os.path.basename(str(filename)))
        label_file = os.path.splitext(filename)[0] + '.json'
        if QtCore.QFile.exists(label_file) and \
                LabelFile.isLabelFile(label_file):
            try:
                self.labelFile = LabelFile(label_file)
                # FIXME: PyQt4 installed via Anaconda fails to load JPEG
                # and JSON encoded images.
                # https://github.com/ContinuumIO/anaconda-issues/issues/131
                if QtGui.QImage.fromData(self.labelFile.imageData).isNull():
                    raise LabelFileError(
                        'Failed loading image data from label file.\n'
                        'Maybe this is a known issue of PyQt4 built on'
                        ' Anaconda, and may be fixed by installing PyQt5.')
            except LabelFileError as e:
                self.errorMessage(
                    'Error opening file',
                    "<p><b>%s</b></p>"
                    "<p>Make sure <i>%s</i> is a valid label file."
                    % (e, label_file))
                self.status("Error reading %s" % label_file)
                return False
            self.imageData = self.labelFile.imageData
            self.imagePath = os.path.join(os.path.dirname(label_file),
                                          self.labelFile.imagePath)
            self.lineColor = QtGui.QColor(*self.labelFile.lineColor)
            self.fillColor = QtGui.QColor(*self.labelFile.fillColor)
        else:
            # Load image:
            # read data first and store for saving into label file.
            self.imageData = read(filename, None)
            if self.imageData is not None:
                # the filename is image not JSON
                self.imagePath = filename
            self.labelFile = None
        image = QtGui.QImage.fromData(self.imageData)
        if image.isNull():
            formats = ['*.{}'.format(fmt.data().decode())
                       for fmt in QtGui.QImageReader.supportedImageFormats()]
            self.errorMessage(
                'Error opening file',
                '<p>Make sure <i>{0}</i> is a valid image file.<br/>'
                'Supported image formats: {1}</p>'
                .format(filename, ','.join(formats)))
            self.status("Error reading %s" % filename)
            return False
        self.image = image
        self.filename = filename
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
        self.setClean()
        self.canvas.setEnabled(True)
        self.adjustScale(initial=True)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.status("Loaded %s" % os.path.basename(str(filename)))
        if filename in self.imageList:
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
        return True

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

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

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        s = self.settings
        s['filename'] = self.filename if self.filename else ''
        s['window/size'] = self.size()
        s['window/position'] = self.pos()
        s['window/state'] = self.saveState()
        s['line/color'] = self.lineColor
        s['fill/color'] = self.fillColor
        s['recentFiles'] = self.recentFiles
        s['advanced'] = not self._beginner
        # ask the use for where to save the labels
        # s['window/geometry'] = self.saveGeometry()

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
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

    def openNextImg(self, _value=False):
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

        if filename:
            self.loadFile(filename)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(str(self.filename)) if self.filename else '.'
        formats = ['*.{}'.format(fmt.data().decode())
                   for fmt in QtGui.QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(
            formats + ['*%s' % LabelFile.suffix])
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self, '%s - Choose Image or Label file' % __appname__,
            path, filters)
        if PYQT5:
            filename, _ = filename
        filename = str(filename)
        if filename:
            self.loadFile(filename)

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.hasLabels():
            if self.labelFile:
                # DL20180323 - overwrite when in directory
                self._saveFile(self.labelFile.filename)
            elif self.output:
                self._saveFile(self.output)
            else:
                self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.hasLabels():
            self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = '%s - Choose File' % __appname__
        filters = 'Label files (*%s)' % LabelFile.suffix
        dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = os.path.splitext(self.filename)[0]
        default_labelfile_name = os.path.join(
            self.currentPath(), basename + LabelFile.suffix)
        filename = dlg.getSaveFileName(
            self, 'Choose File', default_labelfile_name,
            'Label files (*%s)' % LabelFile.suffix)
        if PYQT5:
            filename, _ = filename
        filename = str(filename)
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()
            if self.labeling_once:
                self.close()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    # Message Dialogs. #
    def hasLabels(self):
        if not self.labelList.itemsToShapes:
            self.errorMessage(
                'No objects labeled',
                'You must label at least one object to save the file.')
            return False
        return True

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = 'Save annotations to "{}" before closing?'.format(self.filename)
        answer = mb.question(self,
                             'Save annotations?',
                             msg,
                             mb.Save | mb.Discard | mb.Cancel,
                             mb.Save)
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(str(self.filename)) if self.filename else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(
            self.lineColor, 'Choose line color', default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            # Change the color for all shape lines:
            Shape.line_color = self.lineColor
            self.canvas.update()
            self.setDirty()

    def chooseColor2(self):
        color = self.colorDialog.getColor(
            self.fillColor, 'Choose fill color', default=DEFAULT_FILL_COLOR)
        if color:
            self.fillColor = color
            Shape.fill_color = self.fillColor
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = 'You are about to permanently delete this polygon, ' \
              'proceed anyway?'
        if yes == QtWidgets.QMessageBox.warning(self, 'Attention', msg,
                                                yes | no):
            self.remLabel(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(
            self.lineColor, 'Choose line color', default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(
            self.fillColor, 'Choose fill color', default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filename) \
                if self.filename else '.'

        targetDirPath = str(QtWidgets.QFileDialog.getExistingDirectory(
            self, '%s - Open Directory' % __appname__, defaultOpenDirPath,
            QtWidgets.QFileDialog.ShowDirsOnly |
            QtWidgets.QFileDialog.DontResolveSymlinks))
        self.importDirImages(targetDirPath)

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()
        for imgPath in self.scanAllImages(dirpath):
            item = QtWidgets.QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)
        self.openNextImg()

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower()
                      for fmt in QtGui.QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    images.append(relativePath)
        images.sort(key=lambda x: x.lower())
        return images


class Settings(object):

    """Convenience dict-like wrapper around QSettings."""

    def __init__(self, types=None):
        self.data = QtCore.QSettings()
        self.types = collections.defaultdict(
            lambda: QtCore.QVariant, types if types else {})

    def __setitem__(self, key, value):
        t = self.types[key]
        self.data.setValue(
            key, t(value) if not isinstance(value, t) else value)

    def __getitem__(self, key):
        return self._cast(key, self.data.value(key))

    def get(self, key, default=None):
        return self._cast(key, self.data.value(key, default))

    def _cast(self, key, value):
        # XXX: Very nasty way of converting types to QVariant methods :P
        t = self.types[key]
        if t != QtCore.QVariant:
            method = getattr(QtCore.QVariant,
                             re.sub('^Q', 'to', t.__name__, count=1))
            return method(value)
        return value


def inverted(color):
    return QtGui.QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except Exception:
        return default


def main():
    """Standard boilerplate Qt application code."""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?', help='image or label filename')
    parser.add_argument('--output', '-O', '-o', help='output label name')
    parser.add_argument('--nodata', dest='store_data', action='store_false',
                        help='stop storing image data to JSON file')
    parser.add_argument('--labels',
                        help='comma separated list of labels OR file '
                        'containing one label per line')
    parser.add_argument('--nosortlabels', dest='sort_labels',
                        action='store_false', help='stop sorting labels')
    args = parser.parse_args()

    if args.labels is not None:
        if os.path.isfile(args.labels):
            args.labels = [l.strip() for l in open(args.labels, 'r')
                           if l.strip()]
        else:
            args.labels = [l for l in args.labels.split(',') if l]

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    win = MainWindow(
        filename=args.filename,
        output=args.output,
        store_data=args.store_data,
        labels=args.labels,
        sort_labels=args.sort_labels,
    )
    win.show()
    win.raise_()
    sys.exit(app.exec_())
