import logging
import sys
from typing import Any, cast
import pathlib

import napari
import numpy as np
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .parameters import OutputFormat, ScanPattern, StitchingParameters
from .stitcher import ProgressCallbacks, Stitcher

# TODO(colin): this is almost but not quite the same as the map in
# StitchingComputedParameters.get_channel_color. Reconcile the differences?
CHANNEL_COLORS_MAP = {
    "405": {"hex": 0x3300FF, "name": "blue"},
    "488": {"hex": 0x1FFF00, "name": "green"},
    "561": {"hex": 0xFFCF00, "name": "yellow"},
    "638": {"hex": 0xFF0000, "name": "red"},
    "730": {"hex": 0x770000, "name": "dark red"},
    "R": {"hex": 0xFF0000, "name": "red"},
    "G": {"hex": 0x1FFF00, "name": "green"},
    "B": {"hex": 0x3300FF, "name": "blue"},
}


class StitcherThread(QThread):
    def __init__(self, inner: Stitcher) -> None:
        super().__init__()
        self.inner = inner

    def run(self) -> None:
        self.inner.run()


class StitchingGUI(QWidget):
    # Signals for progress indicators. QT dictates these must be defined at the class level.
    update_progress = Signal(int, int)
    getting_flatfields = Signal()
    starting_stitching = Signal()
    starting_saving = Signal(bool)
    finished_saving = Signal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self.stitcher: StitcherThread | None = (
            None  # Stitcher is initialized when needed
        )
        self.inputDirectory: str | None = (
            None  # This will be set by the directory selection
        )
        self.output_path = ""
        self.dtype: np.dtype | None = None
        self.flatfield_manifest: pathlib.Path | None = None
        self.initUI()

    def initUI(self) -> None:
        self.layout = QVBoxLayout(self)  # type: ignore

        # Input Directory Selection (full width at the top)
        self.inputDirectoryBtn = QPushButton("Select Input Directory", self)
        self.inputDirectoryBtn.clicked.connect(self.selectInputDirectory)
        self.layout.addWidget(self.inputDirectoryBtn)  # type: ignore

        # Output format combo box (full width)
        self.outputFormatCombo = QComboBox(self)
        self.outputFormatCombo.addItems(["OME-ZARR", "OME-TIFF"])
        self.outputFormatCombo.currentTextChanged.connect(self.onFormatChange)
        self.layout.addWidget(self.outputFormatCombo)  # type: ignore

        self.compressionLabel = QLabel("Output compression", self)
        self.layout.addWidget(self.compressionLabel)  # type: ignore
        self.outputCompression = QComboBox(self)
        self.outputCompression.addItems(["default", "none"])
        self.layout.addWidget(self.outputCompression)  # type: ignore

        self.pyramidCheckbox = QCheckBox("Infer levels for output image pyramid", self)
        self.pyramidCheckbox.setChecked(True)
        self.pyramidCheckbox.toggled.connect(self.onPyramidChange)
        self.layout.addWidget(self.pyramidCheckbox)  # type: ignore

        # --- Flatfield Correction Options ---
        self.flatfieldModeCombo = QComboBox(self)
        self.flatfieldModeCombo.addItems(
            [
                "No Flatfield Correction",
                "Compute Flatfield Correction",
                "Load Precomputed Flatfield",
            ]
        )
        self.flatfieldModeCombo.currentIndexChanged.connect(self.onFlatfieldModeChanged)
        self.layout.addWidget(self.flatfieldModeCombo)

        self.loadFlatfieldBtn = QPushButton("Select Flatfield Folder", self)
        self.loadFlatfieldBtn.clicked.connect(self.onLoadFlatfield)
        self.loadFlatfieldBtn.setVisible(False)
        self.layout.addWidget(self.loadFlatfieldBtn)  # type: ignore

        self.pyramidLabel = QLabel(
            "Number of output levels for the image pyramid", self
        )
        self.layout.addWidget(self.pyramidLabel)  # type: ignore
        self.pyramidLabel.hide()
        self.pyramidLevels = QSpinBox(self)
        self.pyramidLevels.setMaximum(32)
        self.pyramidLevels.setMinimum(1)
        self.layout.addWidget(self.pyramidLevels)  # type: ignore
        self.pyramidLevels.hide()

        # Status label
        self.statusLabel = QLabel("Status: Ready", self)
        self.layout.addWidget(self.statusLabel)  # type: ignore

        # Start stitching button
        self.startBtn = QPushButton("Start Stitching", self)
        self.startBtn.clicked.connect(self.onStitchingStart)
        self.layout.addWidget(self.startBtn)  # type: ignore

        # Progress bar setup
        self.progressBar = QProgressBar(self)
        self.progressBar.hide()
        self.layout.addWidget(self.progressBar)  # type: ignore

        # Output path QLineEdit
        self.outputPathEdit = QLineEdit(self)
        self.outputPathEdit.setPlaceholderText(
            "Enter Filepath To Visualize (No Stitching Required)"
        )
        self.layout.addWidget(self.outputPathEdit)  # type: ignore

        # View in Napari button
        self.viewBtn = QPushButton("View Output in Napari", self)
        self.viewBtn.clicked.connect(self.onViewOutput)
        self.viewBtn.setEnabled(False)
        self.layout.addWidget(self.viewBtn)  # type: ignore

        self.layout.insertStretch(-1, 1)  # type: ignore
        self.setWindowTitle("Cephla Image Stitcher")
        self.setGeometry(300, 300, 500, 300)
        self.show()

    def selectInputDirectory(self) -> None:
        self.inputDirectory = QFileDialog.getExistingDirectory(
            self, "Select Input Image Folder"
        )
        if self.inputDirectory:
            self.inputDirectoryBtn.setText(f"Selected: {self.inputDirectory}")

    def onStitchingStart(self) -> None:
        """Start stitching from GUI."""
        if not self.inputDirectory:
            QMessageBox.warning(
                self, "Input Error", "Please select an input directory."
            )
            return

        # # In StitchingGUI.onStitchingStart():
        # if self.outputFormatCombo.currentText() == 'OME-TIFF' and (self.mergeTimepointsCheck.isChecked() or self.mergeRegionsCheck.isChecked()):
        #     QMessageBox.warning(self, "Format Warning",
        #                        "Merging operations are only supported for OME-ZARR format. "
        #                        "These operations will be skipped.")

        try:
            # Create parameters from UI state
            mode = self.flatfieldModeCombo.currentIndex()
            apply_flatfield = mode in (1, 2)
            flatfield_manifest = self.flatfield_manifest if mode == 2 else None
            params = StitchingParameters(
                input_folder=self.inputDirectory,
                output_format=OutputFormat(
                    "." + self.outputFormatCombo.currentText().lower().replace("-", ".")
                ),
                scan_pattern=ScanPattern.unidirectional,
                apply_flatfield=apply_flatfield,
                flatfield_manifest=flatfield_manifest,
            )

            if self.outputFormatCombo.currentText() == "OME-ZARR":
                if not self.pyramidCheckbox.isChecked():
                    params.num_pyramid_levels = self.pyramidLevels.value()
                params.output_compression = self.outputCompression.currentText()  # type: ignore

            self.stitcher = StitcherThread(
                Stitcher(
                    params,
                    ProgressCallbacks(
                        update_progress=self.update_progress.emit,
                        getting_flatfields=self.getting_flatfields.emit,
                        starting_stitching=self.starting_stitching.emit,
                        starting_saving=self.starting_saving.emit,
                        finished_saving=self.finished_saving.emit,
                    ),
                )
            )
            self.setupConnections()

            # Start processing
            self.statusLabel.setText("Status: Stitching...")
            self.stitcher.start()
            self.progressBar.show()

        except Exception as e:
            QMessageBox.critical(self, "Stitching Error", str(e))
            self.statusLabel.setText("Status: Error Encountered")

    def onFormatChange(self, format: str) -> None:
        if format == "OME-ZARR":
            self.pyramidCheckbox.show()
            self.compressionLabel.show()
            self.outputCompression.show()
            if not self.pyramidCheckbox.isChecked():
                self.pyramidLabel.show()
                self.pyramidLevels.show()

        else:
            assert format == "OME-TIFF"
            self.compressionLabel.hide()
            self.outputCompression.hide()
            self.pyramidCheckbox.hide()
            self.pyramidLabel.hide()
            self.pyramidLevels.hide()

    def onPyramidChange(self, checked: bool) -> None:
        if checked:
            self.pyramidLevels.hide()
            self.pyramidLabel.hide()
        else:
            self.pyramidLabel.show()
            self.pyramidLevels.show()

    def onFlatfieldModeChanged(self, idx: int) -> None:
        self.loadFlatfieldBtn.setVisible(idx == 2)
        if idx != 2:
            self.flatfield_manifest = None
            self.loadFlatfieldBtn.setText("Select Flatfield Folder")
        elif idx == 2 and self.flatfield_manifest:
            self.loadFlatfieldBtn.setText(f"Loaded: {self.flatfield_manifest.name}")
        else:
            self.loadFlatfieldBtn.setText("Select Flatfield Folder")

    def onLoadFlatfield(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Flatfield Folder")
        if directory:
            self.flatfield_manifest = pathlib.Path(directory)
            self.loadFlatfieldBtn.setText(f"Loaded: {self.flatfield_manifest.name}")
        else:
            self.loadFlatfieldBtn.setText("Select Flatfield Folder")

    def setupConnections(self) -> None:
        assert self.stitcher is not None
        self.update_progress.connect(self.updateProgressBar)
        self.getting_flatfields.connect(
            lambda: self.statusLabel.setText("Status: Calculating Flatfields...")
        )
        self.starting_stitching.connect(
            lambda: self.statusLabel.setText("Status: Stitching FOVS...")
        )
        self.starting_saving.connect(self.onStartingSaving)
        self.finished_saving.connect(self.onFinishedSaving)

    def updateProgressBar(self, value: int, maximum: int) -> None:
        self.progressBar.setRange(0, maximum)
        self.progressBar.setValue(value)

    def onStartingSaving(self, stitch_complete: bool = False) -> None:
        if stitch_complete:
            self.statusLabel.setText("Status: Saving Complete Acquisition Image...")
        else:
            self.statusLabel.setText("Status: Saving Stitched Image...")
        self.progressBar.setRange(0, 0)  # Indeterminate mode
        self.progressBar.show()
        self.statusLabel.show()

    def onFinishedSaving(self, path: str, dtype: Any) -> None:
        self.progressBar.setValue(0)
        self.progressBar.hide()
        self.viewBtn.setEnabled(True)
        self.statusLabel.setText("Saving Completed. Ready to View.")
        self.outputPathEdit.setText(path)
        self.output_path = path
        self.dtype = np.dtype(dtype)
        if dtype == np.uint16:
            c = [0, 65535]
        elif dtype == np.uint8:
            c = [0, 255]
        else:
            c = None
        self.contrast_limits = c
        self.setGeometry(300, 300, 500, 200)

    def onErrorOccurred(self, error: Any) -> None:
        QMessageBox.critical(self, "Error", f"Error while processing: {error}")
        self.statusLabel.setText("Error Occurred!")

    def onViewOutput(self) -> None:
        output_path = self.outputPathEdit.text()
        try:
            viewer = napari.Viewer()
            if ".ome.zarr" in output_path:
                viewer.open(output_path, plugin="napari-ome-zarr")
            else:
                viewer.open(output_path)

            for layer in viewer.layers:
                wavelength = self.extractWavelength(layer.name)
                channel_info = CHANNEL_COLORS_MAP.get(
                    cast(Any, wavelength), {"hex": 0xFFFFFF, "name": "gray"}
                )

                # Set colormap
                if channel_info["name"] in AVAILABLE_COLORMAPS:
                    layer.colormap = AVAILABLE_COLORMAPS[channel_info["name"]]
                else:
                    layer.colormap = self.generateColormap(channel_info)

                # Set contrast limits based on dtype
                if np.issubdtype(layer.data.dtype, np.integer):
                    info = np.iinfo(layer.data.dtype)
                    layer.contrast_limits = (info.min, info.max)
                elif np.issubdtype(layer.data.dtype, np.floating):
                    layer.contrast_limits = (0.0, 1.0)

            napari.run()
        except Exception as e:
            QMessageBox.critical(self, "Error Opening in Napari", str(e))
            logging.error(f"An error occurred while opening output in Napari: {e}")

    def extractWavelength(self, name: str) -> str | None:
        # Split the string and find the wavelength number immediately after "Fluorescence"
        parts = name.split()
        if "Fluorescence" in parts:
            index = parts.index("Fluorescence") + 1
            if index < len(parts):
                return parts[index].split()[0]  # Assuming '488 nm Ex' and taking '488'
        for color in ["R", "G", "B"]:
            if color in parts or "full_" + color in parts:
                return color
        return None

    def generateColormap(self, channel_info: dict[str, Any]) -> Colormap:
        """Convert a HEX value to a normalized RGB tuple."""
        c0 = (0, 0, 0)
        c1 = (
            ((channel_info["hex"] >> 16) & 0xFF) / 255,  # Normalize the Red component
            ((channel_info["hex"] >> 8) & 0xFF) / 255,  # Normalize the Green component
            (channel_info["hex"] & 0xFF) / 255,
        )  # Normalize the Blue component
        return Colormap(colors=[c0, c1], controls=[0, 1], name=channel_info["name"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    gui = StitchingGUI()
    gui.show()
    sys.exit(app.exec_())
