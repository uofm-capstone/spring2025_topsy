from __future__ import annotations


import PySide6 # noqa: F401 (need to import to select the qt backend)
from PySide6 import QtWidgets, QtGui, QtCore

from wgpu.gui.qt import WgpuCanvas, call_later
from . import VisualizerCanvasBase
from ..drawreason import DrawReason
from ..recorder import VisualizationRecorder

import os
import time
import logging
import matplotlib as mpl

from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
import matplotlib.pyplot as plt
import numpy as np
import re

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..visualizer import Visualizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _get_icon(name):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return QtGui.QIcon(os.path.join(this_dir, "icons", name))


class MyLineEdit(QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.selectAll)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._timer.start(0)

class RecordingSettingsDialog(QtWidgets.QDialog):

    def __init__(self, *args):
        super().__init__(*args)
        self.setWindowTitle("Recording settings")
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        # checkbox for smoothing:
        self._smooth_checkbox = QtWidgets.QCheckBox("Smooth timestream camera movements")
        self._smooth_checkbox.setChecked(True)
        self._layout.addWidget(self._smooth_checkbox)

        # leave some space:
        self._layout.addSpacing(10)

        # checkbox for including vmin/vmax:
        self._vmin_vmax_checkbox = QtWidgets.QCheckBox("Set vmin/vmax from timestream")
        self._vmin_vmax_checkbox.setChecked(True)
        self._layout.addWidget(self._vmin_vmax_checkbox)

        # checkbox for changing quantity:
        self._quantity_checkbox = QtWidgets.QCheckBox("Set quantity from timestream")
        self._quantity_checkbox.setChecked(True)
        self._layout.addWidget(self._quantity_checkbox)

        self._layout.addSpacing(10)

        # checkbox for showing colorbar:
        self._colorbar_checkbox = QtWidgets.QCheckBox("Show colorbar")
        self._colorbar_checkbox.setChecked(True)
        self._layout.addWidget(self._colorbar_checkbox)

        # checkbox for showing scalebar:
        self._scalebar_checkbox = QtWidgets.QCheckBox("Show scalebar")
        self._scalebar_checkbox.setChecked(True)
        self._layout.addWidget(self._scalebar_checkbox)

        self._layout.addSpacing(10)


        # select resolution from dropdown, with options half HD, HD, 4K
        self._resolution_dropdown = QtWidgets.QComboBox()
        self._resolution_dropdown.addItems(["Half HD (960x540)", "HD (1920x1080)", "4K (3840x2160)"])
        self._resolution_dropdown.setCurrentIndex(1)

        # select fps from dropdown, with options 24, 30, 60
        self._fps_dropdown = QtWidgets.QComboBox()
        self._fps_dropdown.addItems(["24 fps", "30 fps", "60 fps"])
        self._fps_dropdown.setCurrentIndex(1)

        # put resolution/fps next to each other horizontally:
        self._resolution_fps_layout = QtWidgets.QHBoxLayout()
        self._resolution_fps_layout.addWidget(self._resolution_dropdown)
        self._resolution_fps_layout.addWidget(self._fps_dropdown)
        self._layout.addLayout(self._resolution_fps_layout)

        self._layout.addSpacing(10)

        # cancel and save.. buttons:
        self._cancel_save_layout = QtWidgets.QHBoxLayout()
        self._cancel_button = QtWidgets.QPushButton("Cancel")
        self._cancel_button.clicked.connect(self.reject)
        self._save_button = QtWidgets.QPushButton("Save")
        # save button should be default:
        self._save_button.setDefault(True)
        self._save_button.clicked.connect(self.accept)
        self._cancel_save_layout.addWidget(self._cancel_button)
        self._cancel_save_layout.addWidget(self._save_button)
        self._layout.addLayout(self._cancel_save_layout)

        # show as a sheet on macos:
        #self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setWindowFlags(QtCore.Qt.WindowType.Sheet)

    @property
    def fps(self):
        return float(self._fps_dropdown.currentText().split()[0])

    @property
    def resolution(self):
        import re
        # use regexp
        # e.g. the string 'blah (123x456)' should map to tuple (123,456)
        match = re.match(r".*\((\d+)x(\d+)\)", self._resolution_dropdown.currentText())
        return int(match.group(1)), int(match.group(2))

    @property
    def smooth(self):
        return self._smooth_checkbox.isChecked()

    @property
    def set_vmin_vmax(self):
        return self._vmin_vmax_checkbox.isChecked()

    @property
    def set_quantity(self):
        return self._quantity_checkbox.isChecked()

    @property
    def show_colorbar(self):
        return self._colorbar_checkbox.isChecked()

    @property
    def show_scalebar(self):
        return self._scalebar_checkbox.isChecked()

# handles loading of colormaps from different common formats
def load_colormap(path: str) -> np.ndarray: # accepts path as a string, returns numpy array
    # colormap as a numpy file
    if path.endswith(".npy"): # already a numpy array, return as is
        return np.load(path)
    
    # colormap as other formats/files
    with open(path, 'r') as f: # open and read file
        content = f.read()
    
    # normalize the matrix style files like [0.1, 0.2, 0.3, 0.4] and get the lines
    if "[" in content and "#" in content and "," in content: # for hex matrix style files [#000000, #FFFFFF, #FF0000]
        content = content.strip("[] \n") # remove all brackets and new lines
        lines = [l.strip() for l in content.split(",") if l.strip()] # split by comma and remove empty lines
    elif "[" in content and ";" in content: # for formats like [0.1; 0.2; 0.3; 0.4]
        content = content.strip("[]")
        lines = content.split(";") # split by semicolon
    else:
        lines = content.splitlines() # split by new line, no brackets
    
    rgb_list = [] # list to store the RGB values

    # loop through each line
    for line in lines:
        line = line.strip().strip("[],") # remove brackets and commas

        if not line or ("style" in line.lower()) or ("format" in line.lower()): # already formatted
            continue
        
        # regex the hex colors and use matplotlib to convert to rgb
        if re.match(r"^#([0-9a-fA-F]{6})$", line):
            rgb = to_rgb(line) # convert hex to rgb
            rgb_list.append(rgb) # append to rgb list
            continue

        # split rgb and cmyk values
        try:
            parts = re.split(r"[,\s]+", line) # split by comma or whitespace
            parts = [float(p) for p in parts] # convert to float
        except ValueError:
            continue

        # convert cmyk to rgb
        if len(parts) == 4:
            c, m, y, k = parts # separate columns (from cmyk formatted file) into cmyk values
            if any(v > 1.0 for v in (c, m, y, k)):
                # if any value is greater than 1, assume it's a percentage
                c, m, y, k = [v / 100.0 for v in (c, m, y, k)]
            r = (1.0 - c) * (1.0 - k) # calculate cmyk to rgb
            g = (1.0 - m) * (1.0 - k)
            b = (1.0 - y) * (1.0 - k)
            rgb_list.append([r, g, b]) # append to rgb list
        
        # handle rgb formatted files
        elif len(parts) == 3:
            if any(v > 1.0 for v in parts):
                # if any value is greater than 1, assume it's a percentage
                parts = [v / 255.0 for v in parts]
            rgb_list.append(parts) # append to rgb list
        else:
            continue # skip invalid lines

    if not rgb_list: # if no valid rgb values found, raise error
        raise ValueError("No valid RGB/HEX values found")
    
    colors = np.array(rgb_list) # convert to numpy array

    # if rgb (instead of rgba)
    if colors.shape[1] == 3:
        alpha = np.ones((colors.shape[0], 1), dtype=colors.dtype) # create alpha channel (matplotlib uses rgba)
        colors = np.hstack([colors, alpha]) # add alpha channel to colors
    
    return colors # return colors as numpy array

class CustomColormapDialog(QtWidgets.QDialog):
    def __init__(self, visualizer, *args, parent=None):
        super().__init__(*args)
        self._visualizer = visualizer
        self.setWindowTitle("Import Colormap")
        self.setFixedSize(300, 150)
        self._pending_colormap = None
        
        # main layout
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        # import a custom colormap file
        import_color = QtWidgets.QPushButton("Import Colormap")
        import_color.clicked.connect(self.import_colormap)
        self._layout.addWidget(import_color)

        # preview bar
        self._preview_bar = QtWidgets.QLabel(self)
        default_cmap = mpl.colormaps[self._visualizer.colormap_name] # default colormap for preview bar
        self.update_preview_colormap(default_cmap)

        # preview bar layout
        preview_layout = QtWidgets.QHBoxLayout()
        preview_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self._preview_bar)
        self._layout.addLayout(preview_layout)

        # button layout for apply/cancel
        button_layout = QtWidgets.QHBoxLayout()
        apply = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        reset = QtWidgets.QPushButton("Reset")
        apply.clicked.connect(self.apply_colormap)
        cancel.clicked.connect(self.reject)
        reset.clicked.connect(self.reset_colormap)
        button_layout.addWidget(apply)
        button_layout.addWidget(cancel)
        button_layout.addWidget(reset)
        self._layout.addLayout(button_layout)

    # updates preview bar
    def update_preview_colormap(self, cmap):
        gradient = np.linspace(0, 1, 256).reshape(1, -1) # use numpy to create horizontal gradient
        fig, ax = plt.subplots(figsize=(3, 0.3)) # create a figure and axis
        ax.imshow(gradient, aspect="auto", cmap=cmap) # display gradient bar
        ax.set_axis_off() # remove axis

        # create temp image file to preview the generated image
        preview_path = "/tmp/custom_cmap_preview.png"
        fig.savefig(preview_path, bbox_inches="tight", pad_inches=0) # save to temp file
        plt.close(fig)

        # display updated preview bar
        pixmap = QtGui.QPixmap(preview_path)
        self._preview_bar.setPixmap(pixmap)

    # imports the custom colormap file, normalizes the format, and updates the preview bar
    def import_colormap(self):
        # open file dialog to select a colormap file
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Custom Colormap", "", "Colormap Files (*.csv *.npy *.txt)"
        )
        if not file_path: # if no file is selected, return
            return
        
        try:
            # external function that handles different colormap formats (csv, npy, txt, hex, rgb, cmyk, etc.)
            colors = load_colormap(file_path)

            # create matplotlib colormap
            cmap = ListedColormap(colors[:, :3], name="ImportedColormap")

            self._pending_colormap = cmap # store the colormap for later use (only applied when apply button is clicked)
            self.update_preview_colormap(cmap) # update the preview bar with the new colormap

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load colormap:\n{str(e)}")
    
    # applies the colormap to visualizer
    def apply_colormap(self):
        if self._pending_colormap is not None:
            self._visualizer.set_colormap(self._pending_colormap) # set the colormap in the visualizer
            self._visualizer._colormap.autorange_vmin_vmax() # reset vmin/vmax to default
            self._visualizer.invalidate() # redraw the visualizer
            QtWidgets.QMessageBox.information(self, "Success", "Colormap applied.") # show success message
        else:
            QtWidgets.QMessageBox.warning(self, "No Colormap", "Please import a colormap.")

    # resets the colormap to one of the matplotlib defaults
    def reset_colormap(self):
        self._visualizer.colormap_name = self._visualizer.colormap_name # reset to default colormap (the one currently chosen in the dropdown)
        default_cmap = mpl.colormaps[self._visualizer.colormap_name] # set the default colormap to the visualization
        self.update_preview_colormap(default_cmap) # update the preview bar
        QtWidgets.QMessageBox.information(self, "Reset", "Reset to selected default colormap.") # show success message

class VminVmaxDialog(QtWidgets.QDialog):
    def __init__(self, visualizer, *args, parent=None):
        super().__init__(*args)
        self._visualizer = visualizer
        self.setWindowTitle("Set vmin/vmax")
        self.setFixedSize(325, 150)

        # main layout
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        # create layout
        form_layout = QtWidgets.QFormLayout()

        # vmin input field
        self._vmin_input = QtWidgets.QLineEdit(self)
        self._vmin_input.setFixedWidth(150)
        self._vmin_input.setText(self.format_precision(self._visualizer.vmin))
        form_layout.addRow("vmin:", self._vmin_input)

        # self._layout.addSpacing(10)

        # vmax input field
        self._vmax_input = QtWidgets.QLineEdit(self)
        self._vmax_input.setFixedWidth(150)
        self._vmax_input.setText(self.format_precision(self._visualizer.vmax))
        form_layout.addRow("vmax:", self._vmax_input)

        form_layout.setVerticalSpacing(10) # add some vertical spacing between the rows

        # center form layout
        centered_layout = QtWidgets.QHBoxLayout()
        centered_layout.addStretch() # add stretch to left
        centered_layout.addLayout(form_layout)
        centered_layout.addStretch() # add stretch to right

        self._layout.addLayout(centered_layout)

        # show temporary success/error message in dialog box
        self.temp_label = QtWidgets.QLabel("") # empty label
        self.temp_label.setStyleSheet("""
            color: black;
            padding: 6px;
            border-radius: 5px;
        """)
        self.temp_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter) # center the label
        self._layout.addWidget(self.temp_label) # add label to the layout
        
        # create button layout
        button_layout = QtWidgets.QHBoxLayout()

        # apply/reset buttons
        apply = QtWidgets.QPushButton("Apply")
        reset = QtWidgets.QPushButton("Reset")
        # connect the button to a functions that will apply/reset vmin/vmax
        apply.clicked.connect(self.on_apply_vmin_vmax)
        reset.clicked.connect(self.on_reset_vmin_vmax)

        # add apply/reset to button layout
        button_layout.addWidget(apply)
        button_layout.addWidget(reset)

        # add buttons to main layout
        self._layout.addLayout(button_layout)
    
    # when reset button is clicked -> reset vmin/vmax to default
    def on_reset_vmin_vmax(self):
        # self.key_up('r') # simulates 'r' key press
        self._visualizer.vmin_vmax_is_set = False # reset vmin/vmax to default
        self._visualizer.invalidate(DrawReason.CHANGE) # redraw the visualizer

        # get original vmin/vmax to sync fields back after reset
        default_vmin = self._visualizer.original_vmin
        default_vmax = self._visualizer.original_vmax

        # update vmin/vmax placeholder text
        self._vmin_input.setText(f"{self.format_precision(default_vmin)}")
        self._vmax_input.setText(f"{self.format_precision(default_vmax)}")
        self.temp_message("Reset: vmin/vmax to default", success=True)

    # when apply button is clicked -> set values from vmin/vmax fields to visualizer
    def on_apply_vmin_vmax(self):
        try:
            vmin = float(self._vmin_input.text())
            vmax = float(self._vmax_input.text())
            if vmin < vmax:
                self._visualizer.vmin = vmin
                self._visualizer.vmax = vmax
                self._visualizer.invalidate(DrawReason.CHANGE)

                self._vmin_input.setText(self.format_precision(vmin)) # update vmin input field with formatted value
                self._vmax_input.setText(self.format_precision(vmax)) # update vmax input field with formatted value

                self.temp_message(f"Success: vmin set to {self.format_precision(vmin)} and vmax set to {self.format_precision(vmax)}", success=True)
                # QtWidgets.QMessageBox.information(self, "Success", f"vmin set to {vmin} and vmax set to {vmax}")
            else:
                # QtWidgets.QMessageBox.critical(self, "Invalid Input", "Error: vmin must be less than vmax")
                self.temp_message("Error: vmin must be less than vmax", success=False)
        except ValueError:
            # QtWidgets.QMessageBox.critical(self, "Invalid Input", "Error: vmin and vmax must be numeric values")
            self.temp_message("Error: vmin and vmax must be numeric values", success=False)
    
    # show a temporary message near the vmin/vmax input widgets
    def temp_message(self, message, success=True, duration=2000):
        self.temp_label.setText(message) # set the message to the label
        if success:
            self.temp_label.setStyleSheet("color: green; padding-top: 5px;")
        else:
            self.temp_label.setStyleSheet("color: red; padding-top: 5px;")

        QtCore.QTimer.singleShot(duration, lambda: self.temp_label.clear()) # remove label after duration
    
    # format input field precision based on span (ex. iord has very small span)
    def format_precision(self, value):
        # adjust precision based on the span of the data
        span = abs(self._visualizer.original_vmax - self._visualizer.original_vmin)
        if span < 1e-5:
            return f"{value:.8e}" # scientific notation for very small spans
        elif span < 0.001:
            return f"{value:.6f}"
        elif span < 0.1:
            return f"{value:.4f}"
        else:
            return f"{value:.2f}"

class VisualizationRecorderWithQtProgressbar(VisualizationRecorder):

    def __init__(self, visualizer: Visualizer, parent_widget: QtWidgets.QWidget):
        super().__init__(visualizer)
        self._parent_widget = parent_widget

    def _progress_iterator(self, ntot):
        progress_bar = QtWidgets.QProgressDialog("Rendering to mp4...", "Stop", 0, ntot, self._parent_widget)
        progress_bar.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress_bar.forceShow()

        last_update = 0

        loop = QtCore.QEventLoop()

        try:
            for i in range(ntot):
                # updating the progress bar triggers a render in the main window, which
                # in turn is quite slow (because it can trigger software rendering
                # of resizable elements like the colorbar). So only update every half second or so.
                if time.time() - last_update > 0.5:
                    last_update = time.time()
                    progress_bar.setValue(i)

                    with self._visualizer.prevent_sph_rendering():
                        loop.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)

                    if progress_bar.wasCanceled():
                        break
                yield i

        finally:
            progress_bar.close()

class VisualizerCanvas(VisualizerCanvasBase, WgpuCanvas):
    _default_quantity_name = "Projected density"
    _all_instances = []
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._all_instances.append(self)
        self.hide()

        self._toolbar = QtWidgets.QToolBar()
        self._toolbar.setIconSize(QtCore.QSize(16, 16))

        # setup toolbar to show text and icons
        self._toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self._load_icons()

        self._record_action = QtGui.QAction(self._record_icon, "Record", self)
        self._record_action.triggered.connect(self.on_click_record)

        self._save_action = QtGui.QAction(self._save_icon, "Snapshot", self)
        self._save_action.triggered.connect(self.on_click_save)

        self._save_movie_action = QtGui.QAction(self._save_movie_icon, "Save mp4", self)
        self._save_movie_action.triggered.connect(self.on_click_save_movie)
        self._save_movie_action.setDisabled(True)

        self._save_script_action = QtGui.QAction(self._export_icon, "Save timestream", self)
        self._save_script_action.triggered.connect(self.on_click_save_script)
        self._save_script_action.setDisabled(True)

        self._load_script_action = QtGui.QAction(self._import_icon, "Load timestream", self)
        self._load_script_action.triggered.connect(self.on_click_load_script)

        self._link_action = QtGui.QAction(self._unlinked_icon, "Link to other windows", self)
        self._link_action.setIconText("Link")
        self._link_action.triggered.connect(self.on_click_link)


        self._colormap_menu = QtWidgets.QComboBox()
        self._colormap_menu.addItems(mpl.colormaps.keys())
        self._colormap_menu.setCurrentText(self._visualizer.colormap_name)
        self._colormap_menu.currentTextChanged.connect(self._colormap_menu_changed_action)

        self._quantity_menu = QtWidgets.QComboBox()
        self._quantity_menu.addItem(self._default_quantity_name)
        self._quantity_menu.setEditable(True)

        # implementing slider for vmin/vmax shifting
        self._contrast_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal) # using pyside slider widget
        # slider goes from min to max, with a default value of 100
        self._contrast_slider.setMinimum(50) 
        self._contrast_slider.setMaximum(150)  
        self._contrast_slider.setValue(100)  
        self._contrast_slider.setFixedWidth(150)
        # Connect the slider to a function that will shift the colormap
        self._contrast_slider.valueChanged.connect(self.on_contrast_slider_changed)

        # adding vmin/vmax fields to toolbar
        self._set_vmin_vmax_action = QtGui.QAction(self._minmax_icon, "Set vmin/vmax", self)
        self._set_vmin_vmax_action.triggered.connect(self.on_click_set_vmin_vmax)

        # implementing custom colormap editor
        self._create_colormap_action = QtGui.QAction(self._import_cmap_icon, "Import Colormap", self)
        self._create_colormap_action.triggered.connect(self.on_click_create_colormap)

        self._quantity_menu.setLineEdit(MyLineEdit())

        # at this moment, the data loader hasn't been initialized yet, so we can't
        # use it to populate the menu. This needs a callback:
        def populate_quantity_menu():
            self._quantity_menu.addItems( self._visualizer.data_loader.get_quantity_names())
            self._quantity_menu.setCurrentText(self._visualizer.quantity_name or self._default_quantity_name)
            self._quantity_menu.currentIndexChanged.connect(self._quantity_menu_changed_action)
            self._quantity_menu.lineEdit().editingFinished.connect(self._quantity_menu_changed_action)
            self._quantity_menu.adjustSize()
            self.setWindowTitle("topsy: "+self._visualizer.data_loader.get_filename())

        self.call_later(0, populate_quantity_menu)

        self._toolbar.addAction(self._load_script_action)
        self._toolbar.addAction(self._save_script_action)
        self._toolbar.addAction(self._record_action)
        self._toolbar.addAction(self._save_movie_action)

        self._toolbar.addSeparator()
        self._toolbar.addAction(self._save_action)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._colormap_menu)
        self._toolbar.addWidget(self._quantity_menu)
        self._toolbar.addSeparator()

        
        self._toolbar.addAction(self._link_action)
        self._recorder = None
        self._toolbar.addSeparator()

        # adding contrast slider to toolbar
        self._toolbar.addWidget(QtWidgets.QLabel("Contrast"))
        self._toolbar.addWidget(self._contrast_slider)
        self._toolbar.addSeparator()

        # adding vmin/vmax fields to toolbar
        self._toolbar.addAction(self._set_vmin_vmax_action)
        self._toolbar.addSeparator()

        # adding import colormap to toolbar
        self._toolbar.addAction(self._create_colormap_action)
        self._toolbar.addSeparator()

        # now replace the wgpu layout with our own
        layout = self.layout()
        layout.removeWidget(self._subwidget)

        our_layout = PySide6.QtWidgets.QVBoxLayout()
        our_layout.addWidget(self._subwidget)
        our_layout.addWidget(self._toolbar)
        our_layout.setContentsMargins(0, 0, 0, 0)
        our_layout.setSpacing(0)

        self._toolbar.adjustSize()

        self._toolbar_update_timer = QtCore.QTimer(self)
        self._toolbar_update_timer.timeout.connect(self._update_toolbar)
        self._toolbar_update_timer.start(100)

        layout.addLayout(our_layout)
    
    # when slider changes -> connect to visualizer.py to shift the colormap exponent
    def on_contrast_slider_changed(self, value): # value is the value of the slider
        self._visualizer.set_colormap_exponent(value / 100) # value is between 1 and 300, we want it between 0 and 3 for the exponent

    def __del__(self):
        try:
            self._all_instances.remove(self)
        except ValueError:
            pass
        super().__del__()
    def _load_icons(self):
        self._record_icon = _get_icon("record.png")
        self._stop_icon = _get_icon("stop.png")
        self._save_icon = _get_icon("camera.png")
        self._linked_icon = _get_icon("linked.png")
        self._unlinked_icon = _get_icon("unlinked.png")
        self._save_movie_icon = _get_icon("movie.png")
        self._import_icon = _get_icon("load_script.png")
        self._export_icon = _get_icon("save_script.png")
        self._minmax_icon = _get_icon("elevator.png")
        self._import_cmap_icon = _get_icon("pallete.png")

    def _colormap_menu_changed_action(self):
        logger.info("Colormap changed to %s", self._colormap_menu.currentText())
        self._visualizer.colormap_name = self._colormap_menu.currentText()

    def _quantity_menu_changed_action(self):
        logger.info("Quantity changed to %s", self._quantity_menu.currentText())
        if self._quantity_menu.currentText() == self._default_quantity_name:
            self._visualizer.quantity_name = None
        else:
            try:
                self._visualizer.quantity_name = self._quantity_menu.currentText()
            except ValueError as e:
                message = QtWidgets.QMessageBox(self)
                message.setWindowTitle("Invalid quantity")
                message.setText(str(e))
                message.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                message.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
                self._quantity_menu.setCurrentText(self._visualizer.quantity_name or self._default_quantity_name)
                message.exec()

    def on_click_create_colormap(self):
        dialog = CustomColormapDialog(self._visualizer, parent=self)
        dialog.exec()

    def on_click_set_vmin_vmax(self):
        dialog = VminVmaxDialog(self._visualizer, parent=self)
        dialog.exec()

    def on_click_record(self):

        if self._recorder is None or not self._recorder.recording:
            logger.info("Starting recorder")
            self._recorder = VisualizationRecorderWithQtProgressbar(self._visualizer, self)
            self._recorder.record()
            self._record_action.setIconText("Stop")
            self._record_action.setIcon(self._stop_icon)
        else:
            logger.info("Stopping recorder")
            self._recorder.stop()
            self._record_action.setIconText("Record")
            self._record_action.setIcon(self._record_icon)

    def on_click_save_movie(self):
        # show the options dialog first:
        dialog = RecordingSettingsDialog(self)
        dialog.exec()
        if dialog.result() == QtWidgets.QDialog.DialogCode.Accepted:
            fd = QtWidgets.QFileDialog(self)
            fname, _ = fd.getSaveFileName(self, "Save video", "", "MP4 (*.mp4)")
            if fname:
                logger.info("Saving video to %s", fname)
                self._recorder.save_mp4(fname, show_colorbar=dialog.show_colorbar,
                                        show_scalebar=dialog.show_scalebar,
                                        fps=dialog.fps,
                                        resolution=dialog.resolution,
                                        smooth=dialog.smooth,
                                        set_vmin_vmax=dialog.set_vmin_vmax,
                                        set_quantity=dialog.set_quantity)
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(fname))

    def on_click_save(self):
        fd = QtWidgets.QFileDialog(self)
        fname, _ = fd.getSaveFileName(self, "Save snapshot", "", "PNG (*.png);; PDF (*.pdf)")
        if fname:
            logger.info("Saving snapshot to %s", fname)
            self._visualizer.save(fname)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(fname))

    def on_click_save_script(self):
        fd = QtWidgets.QFileDialog(self)
        fname, _ = fd.getSaveFileName(self, "Save camera movements", "", "Python Pickle (*.pickle)")
        if fname:
            logger.info("Saving timestream to %s", fname)
            self._recorder.save_timestream(fname)

    def on_click_load_script(self):
        fd = QtWidgets.QFileDialog(self)
        fname, _ = fd.getOpenFileName(self, "Load camera movements", "", "Python Pickle (*.pickle)")
        if fname:
            logger.info("Loading timestream from %s", fname)
            self._recorder = VisualizationRecorderWithQtProgressbar(self._visualizer, self)
            self._recorder.load_timestream(fname)


    def on_click_link(self):
        if self._visualizer.is_synchronizing():
            logger.info("Stop synchronizing")
            self._visualizer.stop_synchronizing()
        else:
            logger.info("Start synchronizing")
            from .. import view_synchronizer
            synchronizer = view_synchronizer.ViewSynchronizer()
            for instance in self._all_instances:
                synchronizer.add_view(instance._visualizer)

    def _update_toolbar(self):
        if self._recorder is not None or len(self._all_instances)<2:
            self._link_action.setDisabled(True)
        else:
            self._link_action.setDisabled(False)
            if self._visualizer.is_synchronizing():
                self._link_action.setIcon(self._linked_icon)
                self._link_action.setIconText("Unlink")
            else:
                self._link_action.setIcon(self._unlinked_icon)
                self._link_action.setIconText("Link")
        if self._recorder is not None and not self._recorder.recording:
            self._save_movie_action.setDisabled(False)
            self._save_script_action.setDisabled(False)
        else:
            self._save_movie_action.setDisabled(True)
            self._save_script_action.setDisabled(True)




    def request_draw(self, function=None):
        # As a side effect, wgpu gui layer stores our function call, to enable it to be
        # repainted later. But we want to distinguish such repaints and handle them
        # differently, so we need to replace the function with our own
        def function_wrapper():
            function()
            self._subwidget.draw_frame = lambda: self._visualizer.draw(DrawReason.PRESENTATION_CHANGE)

        super().request_draw(function_wrapper)

    @classmethod
    def call_later(cls, delay, fn, *args):
        call_later(delay, fn, *args)
