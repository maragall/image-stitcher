#!/usr/bin/env python3
"""
Grid Viewer GUI Module
A PyQt-based GUI that displays a grid of downsampled images on the left
and a napari viewer on the right for viewing individual zarr files.
"""

import os
import sys
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, cast
import pickle

# OME-ZARR imports
from ome_zarr import reader
from ome_zarr.io import ZarrLocation

# PyQt imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QScrollArea, QLabel, QPushButton, QSplitter, QSizePolicy, 
    QMessageBox, QDialog, QFileDialog, QComboBox, QMenuBar, QSplitterHandle
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QRunnable, QThreadPool, QObject, QPoint, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPolygon

# Napari imports
import napari

# Color palette and contrast management
CHANNEL_COLORS_MAP = {
    "405": {"hex": 0x0000FF, "name": "blue"},
    "488": {"hex": 0x00FF00, "name": "green"},
    "561": {"hex": 0xFFCF00, "name": "yellow"},
    "638": {"hex": 0xFF0000, "name": "red"},
    "730": {"hex": 0x770000, "name": "dark_red"},
    "_B": {"hex": 0x0000FF, "name": "blue"},
    "_G": {"hex": 0x00FF00, "name": "green"},
    "_R": {"hex": 0xFF0000, "name": "red"},
}

AVAILABLE_COLORMAPS = {
    "green": "green", 
    "yellow": "yellow",
    "red": "red",
    "dark_red": "red",
}


class CustomSplitterHandle(QSplitterHandle):
    """Minimal custom splitter handle with subtle resize grip"""
    
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.hovered = False
        self.setMouseTracking(True)
        
    def sizeHint(self):
        """Provide minimal size hint for the handle"""
        if self.orientation() == Qt.Orientation.Horizontal:
            return QSize(8, 20)  # Minimal vertical handle
        else:
            return QSize(20, 8)  # Minimal horizontal handle
    
    def resizeEvent(self, event):
        """Ensure handle is properly positioned and centered"""
        super().resizeEvent(event)
        # Force a repaint to ensure centering is correct
        self.update()
            
    def enterEvent(self, event):
        """Handle mouse enter for hover effect"""
        self.hovered = True
        self.update()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle mouse leave for hover effect"""
        self.hovered = False
        self.update()
        super().leaveEvent(event)
        
    def paintEvent(self, event):
        """Paint the minimal resize handle"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Subtle background (almost invisible when not hovered)
        if self.hovered:
            bg_color = QColor("#606060")
            painter.fillRect(self.rect(), bg_color)
        
        # Very minimal grip indicator only when hovered
        if self.hovered:
            grip_color = QColor("#ffffff")
            painter.setPen(grip_color)
            painter.setBrush(grip_color)
            
            # Get true center point
            center_x = self.width() // 2
            center_y = self.height() // 2
            
            if self.orientation() == Qt.Orientation.Horizontal:
                # Vertical handle - single vertical line
                painter.drawLine(center_x, center_y - 6, center_x, center_y + 6)
            else:
                # Horizontal handle - single horizontal line
                painter.drawLine(center_x - 6, center_y, center_x + 6, center_y)


class CustomSplitter(QSplitter):
    """Custom splitter with minimal resize handles"""
    
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(8)  # Minimal width for subtle handles
        
    def createHandle(self):
        """Create custom handle for the splitter"""
        return CustomSplitterHandle(self.orientation(), self)


class ThumbnailWorkerSignals(QObject):
    thumbnail_ready = pyqtSignal(str, np.ndarray, str)  # region_name, thumbnail, channel_name
    error_occurred = pyqtSignal(str, str)
    data_loaded = pyqtSignal(str, dict)  # zarr_path, data_dict for caching

class ThumbnailWorker(QRunnable):
    """Worker runnable for generating thumbnails (thread pool version)"""
    def __init__(self, zarr_path: str, region_name: str, parent_gui=None):
        super().__init__()
        self.zarr_path = zarr_path
        self.region_name = region_name
        self.parent_gui = parent_gui
        self.signals = ThumbnailWorkerSignals()
        # NEW: Support for contrast limits
        self.contrast_limits_per_channel = {}
        # NEW: Support for channel visibility
        self.channel_visibility = {}
        # NEW: Support for cached data
        self.cached_data = None

    def set_contrast_limits(self, contrast_limits_dict):
        """Set contrast limits for thumbnail generation"""
        self.contrast_limits_per_channel = contrast_limits_dict.copy()
    
    def set_channel_visibility(self, visibility_dict):
        """Set channel visibility for thumbnail generation"""
        self.channel_visibility = visibility_dict.copy()
    
    def set_cached_data(self, cached_data_dict):
        """Set cached zarr data to avoid reloading from disk"""
        self.cached_data = cached_data_dict
        print(f"Using cached data for {self.region_name}")

    def run(self):
        try:
            if self.cached_data:
                # Use cached data - much faster!
                channels_data = self.cached_data['channels_data']
                channel_names = self.cached_data['channel_names']
                print(f"Using cached data for {self.region_name} ({len(channels_data)} channels)")
            else:
                # Load data from zarr (slower path)
                channels_data, channel_names = self._load_zarr_data()
                if channels_data:
                    # Cache the loaded data for future use
                    cache_data = {
                        'channels_data': channels_data,
                        'channel_names': channel_names
                    }
                    self.signals.data_loaded.emit(self.zarr_path, cache_data)
            
            if not channels_data:
                self.signals.error_occurred.emit(self.region_name, "No channels could be loaded")
                return
                
            downsampled_channels = []
            for channel_data, channel_name in zip(channels_data, channel_names):
                # NEW: Apply contrast limits before downsampling
                contrast_adjusted_data = self._apply_contrast_limits(channel_data, channel_name)
                downsampled = self._downsample_thumbnail(contrast_adjusted_data, target_size=80)  # Smaller for speed
                downsampled_channels.append(downsampled)
            
            composite_thumbnail = self._create_composite_thumbnail(downsampled_channels, channel_names)
            self.signals.thumbnail_ready.emit(self.region_name, composite_thumbnail, ",".join(channel_names))
            
        except Exception as e:
            self.signals.error_occurred.emit(self.region_name, str(e))
    
    def _load_zarr_data(self):
        """Load zarr data from disk (fallback when no cache available)"""
        zarr_location = ZarrLocation(self.zarr_path)
        zarr_reader = reader.Reader(zarr_location)
        root_node = None
        for node in zarr_reader():
            if node.load(reader.Multiscales):
                root_node = node
                break
        if not root_node:
            return None, None
            
        multiscales = root_node.load(reader.Multiscales)
        if not multiscales:
            return None, None
            
        omero = root_node.load(reader.OMERO)
        try:
            multiscales_list = multiscales.lookup('multiscales', [])
            if not multiscales_list:
                return None, None
            multiscales_metadata = multiscales_list[0]
            version = multiscales_metadata.get('version', '0.4')
            datasets = multiscales_metadata.get('datasets', [])
            if not datasets:
                return None, None
            highest_resolution = datasets[-1]['path']
            data = multiscales.array(resolution=highest_resolution, version=version)
        except Exception as e:
            print(f"Error loading multiscales data: {e}")
            return None, None
            
        channels_data = []
        channel_names = []
        if omero and 'channels' in omero.lookup('omero', {}):
            omero_channels = omero.lookup('omero', {})['channels']
            num_channels = len(omero_channels)
            for ch in range(num_channels):
                try:
                    if len(data.shape) == 5:
                        channel_data = data[0, ch, 0].compute()
                    elif len(data.shape) == 4:
                        channel_data = data[ch, 0].compute() if data.shape[0] == num_channels else data[0, ch].compute()
                    elif len(data.shape) == 3:
                        channel_data = data[ch].compute() if data.shape[0] == num_channels else data[0].compute()
                    else:
                        channel_data = data.compute()
                    channels_data.append(channel_data)
                    channel_info = omero_channels[ch]
                    channel_name = channel_info.get('label', f'ch{ch}')
                    channel_names.append(channel_name)
                except Exception as e:
                    print(f"Error loading channel {ch}: {e}")
                    continue
        else:
            try:
                channel_data = data.compute()
                channels_data.append(channel_data)
                channel_names.append("ch0")
            except Exception as e:
                print(f"Error loading data: {e}")
                return None, None
                
        return channels_data, channel_names
    
    def _apply_contrast_limits(self, image: np.ndarray, channel_name: str) -> np.ndarray:
        """Apply contrast limits to image data before thumbnail generation"""
        # Get contrast limits for this channel
        contrast_limits = None
        
        # Try to get contrast limits from parent GUI
        if self.parent_gui:
            contrast_limits = self.parent_gui.get_contrast_limits_for_channel(channel_name)
        
        # Fallback to stored contrast limits
        if not contrast_limits and channel_name in self.contrast_limits_per_channel:
            contrast_limits = self.contrast_limits_per_channel[channel_name]
        
        # Apply contrast limits if available
        if contrast_limits and len(contrast_limits) == 2:
            min_val, max_val = contrast_limits
            
            # Clip values to contrast range
            clipped_image = np.clip(image, min_val, max_val)
            
            # Normalize to 0-1 range based on contrast limits (same as napari)
            if max_val > min_val:
                normalized_image = (clipped_image - min_val) / (max_val - min_val)
            else:
                normalized_image = np.zeros_like(clipped_image)
                
            # Return normalized float data in 0-1 range for consistent processing
            return normalized_image.astype(np.float32)
        
        # If no contrast limits, normalize using data range (fallback)
        if image.max() > image.min():
            return ((image - image.min()) / (image.max() - image.min())).astype(np.float32)
        else:
            return np.zeros_like(image, dtype=np.float32)
    
    def _get_channel_name(self, omero_spec, channel_idx: int) -> str:
        """Get channel name from OME metadata using ome-zarr."""
        try:
            if omero_spec and 'channels' in omero_spec.lookup('omero', {}):
                omero_channels = omero_spec.lookup('omero', {})['channels']
                if len(omero_channels) > channel_idx:
                    channel_info = omero_channels[channel_idx]
                    return channel_info.get('label', f'ch{channel_idx}')
            return f'ch{channel_idx}'
        except Exception as e:
            print(f"Error reading channel metadata: {e}")
            return f'ch{channel_idx}'
    
    def _downsample_thumbnail(self, image: np.ndarray, target_size: int = 100) -> np.ndarray:
        """Downsample image for thumbnail display"""
        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        # Calculate scale factor
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Simple downsampling using numpy
        if len(image.shape) == 2:
            # For 2D images, use simple downsampling
            h_indices = np.linspace(0, h-1, new_h, dtype=int)
            w_indices = np.linspace(0, w-1, new_w, dtype=int)
            return image[h_indices[:, None], w_indices]
        else:
            # For multi-channel images, process each channel
            result = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
            for c in range(image.shape[2]):
                h_indices = np.linspace(0, h-1, new_h, dtype=int)
                w_indices = np.linspace(0, w-1, new_w, dtype=int)
                result[:, :, c] = image[h_indices[:, None], w_indices, c]
            return result
    
    def _create_composite_thumbnail(self, channels_data: List[np.ndarray], channel_names: List[str]) -> np.ndarray:
        """Create a composite thumbnail by overlaying all channels with their colormaps"""
        if not channels_data:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Get dimensions from first channel
        h, w = channels_data[0].shape
        composite = np.zeros((h, w, 3), dtype=np.float32)
        
        # Define channel colors (same as in main GUI)
        channel_colors = {
            "405": [0, 0, 255],      # Blue
            "488": [0, 255, 0],      # Green
            "561": [255, 207, 0],    # Yellow
            "638": [255, 0, 0],      # Red
            "730": [119, 0, 0],      # Dark Red
            "_B": [0, 0, 255],       # Blue
            "_G": [0, 255, 0],       # Green
            "_R": [255, 0, 0],       # Red
        }
        
        # Overlay each channel (only if visible)
        for channel_data, channel_name in zip(channels_data, channel_names):
            # NEW: Check if this channel is visible
            is_visible = self._is_channel_visible(channel_name)
            if not is_visible:
                print(f"Skipping invisible channel: {channel_name}")
                continue
                
            # Data is already normalized to 0-1 range from contrast application
            # NO additional normalization needed here
            
            # Get color for this channel
            color = [255, 255, 255]  # Default white
            for key, value in channel_colors.items():
                if key in channel_name:
                    color = value
                    break
            
            # Apply color to channel (data is already 0-1)
            for c in range(3):
                composite[:, :, c] += channel_data * (color[c] / 255.0)
        
        # Clip composite to valid range and convert to uint8
        composite = np.clip(composite, 0, 1)
        return (composite * 255).astype(np.uint8)
    
    def _is_channel_visible(self, channel_name: str) -> bool:
        """Check if a channel is visible"""
        # Try to get visibility from parent GUI first
        if self.parent_gui and hasattr(self.parent_gui, 'channel_visibility'):
            if channel_name in self.parent_gui.channel_visibility:
                return self.parent_gui.channel_visibility[channel_name]
        
        # Fallback to stored visibility
        if channel_name in self.channel_visibility:
            return self.channel_visibility[channel_name]
            
        # Default to visible if no information available
        return True
    

class WellPlateWidget(QWidget):
    """A monolithic well plate widget containing all thumbnails in a grid"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thumbnails = {}  # region_name -> thumbnail_data
        self.zarr_paths = {}  # region_name -> zarr_path
        self.base_cell_size = 120  # Base cell size for reference
        self.cell_spacing = 5
        self.rows = 0
        self.cols = 0
        self.parent_gui = parent
        self.hovered_well = None  # Track which well is being hovered
        
        # Allow the widget to shrink and expand as needed
        self.setMinimumSize(100, 100)  # Very small minimum
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
    def set_data(self, zarr_files_dict):
        """Set the zarr files data and calculate well plate format"""
        self.zarr_paths = zarr_files_dict.copy()
        cell_count = len(zarr_files_dict)
        
        # Calculate rows and columns based on well count
        self.rows, self.cols = self._calculate_well_format(cell_count)
        
        # Initialize empty thumbnails for ALL wells in the complete plate
        self.thumbnails = {}
        for row in range(self.rows):
            for col in range(self.cols):
                well_name = self._position_to_well_name(row, col)
                self.thumbnails[well_name] = None
        
        # Map region names to well positions if they don't follow standard naming
        # This ensures regions get assigned to the first available wells
        region_list = sorted(list(zarr_files_dict.keys()))
        for i, region_name in enumerate(region_list):
            if i < self.rows * self.cols:  # Only assign if we have enough wells
                row = i // self.cols
                col = i % self.cols
                well_name = self._position_to_well_name(row, col)
                # Map the region to this well position
                self.zarr_paths[well_name] = zarr_files_dict[region_name]
        
        self.update()
    
    def _calculate_well_format(self, cell_count):
        """Calculate rows and columns based on cell count following standard well plate formats"""
        # Standard well plate formats
        format_map = {
            4: (2, 2),      # 4-well plate: 2x2
            6: (2, 3),      # 6-well plate: 2x3
            12: (3, 4),     # 12-well plate: 3x4
            24: (4, 6),     # 24-well plate: 4x6
            96: (8, 12),    # 96-well plate: 8x12
            384: (16, 24),  # 384-well plate: 16x24
            1536: (32, 48)  # 1536-well plate: 32x48
        }
        
        # Find the appropriate well plate format based on region count
        if cell_count <= 4:
            return format_map[4]  # 4-well plate
        elif cell_count <= 6:
            return format_map[6]  # 6-well plate
        elif cell_count <= 12:
            return format_map[12]  # 12-well plate
        elif cell_count <= 24:
            return format_map[24]  # 24-well plate
        elif cell_count <= 96:
            return format_map[96]  # 96-well plate
        elif cell_count <= 384:
            return format_map[384]  # 384-well plate
        elif cell_count <= 1536:
            return format_map[1536]  # 1536-well plate
        else:
            # For very large datasets, use 1536 format
            return format_map[1536]
    
    def _position_to_well_name(self, row, col):
        """Convert grid position to well name (A1, B3, AA5, etc.)"""
        # Handle row labels (A-Z, then AA-AZ, BA-BZ, etc.)
        if row < 26:
            row_label = chr(ord('A') + row)
        else:
            first_char = chr(ord('A') + (row // 26) - 1)
            second_char = chr(ord('A') + (row % 26))
            row_label = first_char + second_char
            
        # Column labels are 1-based numbers
        col_label = str(col + 1)
        
        return row_label + col_label
    
    def _well_name_to_position(self, well_name):
        """Convert well name back to grid position"""
        # Extract row letters and column number
        import re
        match = re.match(r'([A-Z]+)(\d+)', well_name)
        if not match:
            return 0, 0
            
        row_letters, col_num = match.groups()
        col = int(col_num) - 1
        
        # Calculate row from letters
        if len(row_letters) == 1:
            row = ord(row_letters) - ord('A')
        else:
            # Handle AA, AB, etc.
            row = 0
            for i, char in enumerate(row_letters):
                row = row * 26 + (ord(char) - ord('A'))
            if len(row_letters) > 1:
                row += 26  # Offset for multi-letter combinations
                
        return row, col
    
    def _get_region_position(self, region_name):
        """Get the position for a region, assigning sequentially if not parseable"""
        # Try to parse existing region names first
        try:
            return self._well_name_to_position(region_name)
        except:
            # Assign sequential positions for unparseable names
            region_list = sorted(list(self.zarr_paths.keys()))
            if region_name in region_list:
                idx = region_list.index(region_name)
                return idx // self.cols, idx % self.cols
            return 0, 0
    
    def set_thumbnail(self, region_name, thumbnail_data):
        """Set thumbnail data for a specific region"""
        # First try direct match
        if region_name in self.thumbnails:
            self.thumbnails[region_name] = thumbnail_data
            self.update()
            return
        
        # If no direct match, find the well that contains this region's data
        for well_name in self.thumbnails.keys():
            if well_name in self.zarr_paths and self.zarr_paths[well_name] == self.zarr_paths.get(region_name):
                self.thumbnails[well_name] = thumbnail_data
                self.update()
                return
        
        # If still no match, try to find by original region name in the mapping
        # This handles the case where we mapped regions to wells
        for well_name, zarr_path in self.zarr_paths.items():
            if zarr_path == self.zarr_paths.get(region_name):
                self.thumbnails[well_name] = thumbnail_data
                self.update()
                return
    
    def clear_thumbnails(self):
        """Clear all thumbnails"""
        for well_name in self.thumbnails:
            self.thumbnails[well_name] = None
        self.update()
    
    def leaveEvent(self, event):
        """Handle mouse leaving the widget"""
        if self.hovered_well is not None:
            self.hovered_well = None
            self.update()
    
    def paintEvent(self, event):
        """Paint the well plate with all thumbnails"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor("#262930"))
        
        # Calculate available space and dynamic sizing
        available_width = self.width() - 40  # Leave margin
        available_height = self.height() - 40  # Leave margin
        
        # Calculate label areas based on available space
        label_width = max(30, min(50, available_width // 20))  # Dynamic label width
        label_height = max(20, min(40, available_height // 20))  # Dynamic label height
        
        # Calculate cell size to fit the grid in available space
        grid_width = available_width - label_width
        grid_height = available_height - label_height
        
        if self.cols > 0 and self.rows > 0:
            cell_size = min(
                (grid_width - (self.cols - 1) * self.cell_spacing) // self.cols,
                (grid_height - (self.rows - 1) * self.cell_spacing) // self.rows
            )
            cell_size = max(8, cell_size)  # Very small minimum to ensure everything fits
        else:
            cell_size = 20  # Default cell size
        
        # Calculate total grid dimensions (including labels)
        total_grid_width = label_width + self.cols * (cell_size + self.cell_spacing) - self.cell_spacing
        total_grid_height = label_height + self.rows * (cell_size + self.cell_spacing) - self.cell_spacing
        
        # Calculate centering offsets
        center_x = (self.width() - total_grid_width) // 2
        center_y = (self.height() - total_grid_height) // 2
        
        # Calculate grid position (centered)
        grid_start_x = center_x + label_width
        grid_start_y = center_y + label_height
        
        # Calculate font size based on cell size
        label_font_size = max(6, min(20, cell_size // 4))  # Smaller minimum for very small cells
        hover_font_size = max(6, min(16, cell_size // 4))
        
        # Draw column labels (1, 2, 3, etc.)
        painter.setPen(QColor("#ffffff"))
        font = QFont()
        font.setBold(True)
        font.setPointSize(label_font_size)
        painter.setFont(font)
        
        for col in range(self.cols):
            x = grid_start_x + col * (cell_size + self.cell_spacing)
            label_rect = QRect(x, center_y, cell_size, label_height)
            col_label = str(col + 1)  # 1-based column numbers
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, col_label)
        
        # Draw row labels (A, B, C, etc.)
        for row in range(self.rows):
            y = grid_start_y + row * (cell_size + self.cell_spacing)
            label_rect = QRect(center_x, y, label_width, cell_size)
            row_label = self._position_to_well_name(row, 0)[0]  # Get just the letter part
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, row_label)
        
        # Draw grid lines
        painter.setPen(QColor("#505050"))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        # Vertical grid lines
        for col in range(self.cols + 1):
            x = grid_start_x + col * (cell_size + self.cell_spacing) - self.cell_spacing // 2
            painter.drawLine(x, grid_start_y, x, grid_start_y + self.rows * (cell_size + self.cell_spacing) - self.cell_spacing)
        
        # Horizontal grid lines
        for row in range(self.rows + 1):
            y = grid_start_y + row * (cell_size + self.cell_spacing) - self.cell_spacing // 2
            painter.drawLine(grid_start_x, y, grid_start_x + self.cols * (cell_size + self.cell_spacing) - self.cell_spacing, y)
        
        # Draw outer border
        painter.setPen(QColor("#707070"))
        outer_rect = QRect(grid_start_x - 1, grid_start_y - 1, 
                          self.cols * (cell_size + self.cell_spacing) - self.cell_spacing + 2,
                          self.rows * (cell_size + self.cell_spacing) - self.cell_spacing + 2)
        painter.drawRect(outer_rect)
        
        # Draw each well in the complete plate
        for row in range(self.rows):
            for col in range(self.cols):
                # Calculate cell rectangle
                x = grid_start_x + col * (cell_size + self.cell_spacing)
                y = grid_start_y + row * (cell_size + self.cell_spacing)
                cell_rect = QRect(x, y, cell_size, cell_size)
                
                # Get well name for this position
                well_name = self._position_to_well_name(row, col)
                
                # Check if this well has data
                has_data = well_name in self.zarr_paths
                
                # Check if this well is being hovered
                is_hovered = self.hovered_well == well_name
                
                # Draw cell background
                if has_data:
                    painter.setPen(QColor("#404040"))
                    painter.setBrush(QColor("#1e1e1e"))
                else:
                    painter.setPen(QColor("#303030"))
                    painter.setBrush(QColor("#0f0f0f"))
                
                painter.drawRect(cell_rect)
                
                # Draw cell content
                if has_data:
                    # Check if we have a thumbnail for this well
                    if well_name in self.thumbnails and self.thumbnails[well_name] is not None:
                        # Draw thumbnail
                        pixmap = self._numpy_to_pixmap(self.thumbnails[well_name])
                        if pixmap:
                            scaled_pixmap = pixmap.scaled(cell_rect.width(), cell_rect.height(), 
                                                        Qt.AspectRatioMode.KeepAspectRatio, 
                                                        Qt.TransformationMode.SmoothTransformation)
                            # Center the pixmap in the cell
                            pixmap_x = cell_rect.x() + (cell_rect.width() - scaled_pixmap.width()) // 2
                            pixmap_y = cell_rect.y() + (cell_rect.height() - scaled_pixmap.height()) // 2
                            painter.drawPixmap(pixmap_x, pixmap_y, scaled_pixmap)
                    else:
                        # Show loading text for wells with data
                        painter.setPen(QColor("#cccccc"))
                        font = QFont()
                        font.setPointSize(max(4, cell_size // 8))  # Smaller font for very small cells
                        painter.setFont(font)
                else:
                    # Show subtle dot for empty wells
                    painter.setPen(QColor("#303030"))
                    painter.drawText(cell_rect, Qt.AlignmentFlag.AlignCenter, "·")
                
                # Draw hover effect
                if is_hovered and has_data:
                    # Cyan bounding box
                    painter.setPen(QColor("#00ffff"))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(cell_rect)
                    
                    # Region name overlay
                    if well_name in self.zarr_paths:
                        original_region_name = None
                        for region_name, zarr_path in self.zarr_paths.items():
                            if zarr_path == self.zarr_paths[well_name]:
                                original_region_name = region_name
                                break
                        
                        if original_region_name:
                            painter.setPen(QColor("#ffffff"))
                            font = QFont()
                            font.setPointSize(hover_font_size)
                            painter.setFont(font)
                            painter.drawText(cell_rect, Qt.AlignmentFlag.AlignCenter, original_region_name)
    
    def _numpy_to_pixmap(self, image):
        """Convert numpy array to QPixmap"""
        try:
            if len(image.shape) == 2:
                # Grayscale
                height, width = image.shape
                if image.dtype != np.uint8:
                    if image.max() > 0:
                        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    else:
                        image = np.zeros_like(image, dtype=np.uint8)
                q_image = QImage(image.data, width, height, width, QImage.Format.Format_Grayscale8)
            else:
                # RGB
                height, width = image.shape[:2]
                if image.dtype != np.uint8:
                    if image.max() > 0:
                        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    else:
                        image = np.zeros_like(image, dtype=np.uint8)
                
                if image.shape[2] == 3:
                    bytes_per_line = width * 3
                    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                else:
                    return None
                    
            return QPixmap.fromImage(q_image)
        except Exception as e:
            print(f"Error converting image to pixmap: {e}")
            return None
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for hover effects"""
        # Calculate dynamic sizing (same as in paintEvent)
        available_width = self.width() - 40
        available_height = self.height() - 40
        label_width = max(30, min(50, available_width // 20))
        label_height = max(20, min(40, available_height // 20))
        grid_width = available_width - label_width
        grid_height = available_height - label_height
        
        if self.cols > 0 and self.rows > 0:
            cell_size = min(
                (grid_width - (self.cols - 1) * self.cell_spacing) // self.cols,
                (grid_height - (self.rows - 1) * self.cell_spacing) // self.rows
            )
            cell_size = max(8, cell_size)  # Very small minimum to ensure everything fits
        else:
            cell_size = 20
        
        # Calculate total grid dimensions (including labels)
        total_grid_width = label_width + self.cols * (cell_size + self.cell_spacing) - self.cell_spacing
        total_grid_height = label_height + self.rows * (cell_size + self.cell_spacing) - self.cell_spacing
        
        # Calculate centering offsets
        center_x = (self.width() - total_grid_width) // 2
        center_y = (self.height() - total_grid_height) // 2
        
        # Calculate grid position (centered)
        grid_start_x = center_x + label_width
        grid_start_y = center_y + label_height
        
        # Calculate which cell is being hovered
        x = event.pos().x() - grid_start_x
        y = event.pos().y() - grid_start_y
        
        if x < 0 or y < 0:
            # Mouse is outside the grid area
            if self.hovered_well is not None:
                self.hovered_well = None
                self.update()
            return
            
        col = x // (cell_size + self.cell_spacing)
        row = y // (cell_size + self.cell_spacing)
        
        if row < self.rows and col < self.cols:
            # Get the well name for this position
            well_name = self._position_to_well_name(row, col)
            
            # Check if this well has data and is different from current hover
            if well_name in self.zarr_paths and self.hovered_well != well_name:
                self.hovered_well = well_name
                self.update()
        else:
            # Mouse is outside valid grid cells
            if self.hovered_well is not None:
                self.hovered_well = None
                self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse clicks to identify which cell was clicked"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Calculate dynamic sizing (same as in paintEvent)
            available_width = self.width() - 40
            available_height = self.height() - 40
            label_width = max(30, min(50, available_width // 20))
            label_height = max(20, min(40, available_height // 20))
            grid_width = available_width - label_width
            grid_height = available_height - label_height
            
            if self.cols > 0 and self.rows > 0:
                cell_size = min(
                    (grid_width - (self.cols - 1) * self.cell_spacing) // self.cols,
                    (grid_height - (self.rows - 1) * self.cell_spacing) // self.rows
                )
                cell_size = max(8, cell_size)  # Very small minimum to ensure everything fits
            else:
                cell_size = 20
            
            # Calculate total grid dimensions (including labels)
            total_grid_width = label_width + self.cols * (cell_size + self.cell_spacing) - self.cell_spacing
            total_grid_height = label_height + self.rows * (cell_size + self.cell_spacing) - self.cell_spacing
            
            # Calculate centering offsets
            center_x = (self.width() - total_grid_width) // 2
            center_y = (self.height() - total_grid_height) // 2
            
            # Calculate grid position (centered)
            grid_start_x = center_x + label_width
            grid_start_y = center_y + label_height
            
            # Calculate which cell was clicked
            x = event.pos().x() - grid_start_x
            y = event.pos().y() - grid_start_y
            
            if x < 0 or y < 0:
                return
                
            col = x // (cell_size + self.cell_spacing)
            row = y // (cell_size + self.cell_spacing)
            
            if row < self.rows and col < self.cols:
                # Get the well name for this position
                well_name = self._position_to_well_name(row, col)
                
                # Check if this well has data
                if well_name in self.zarr_paths:
                    # Open zarr in napari
                    if hasattr(self.parent_gui, 'open_zarr_in_napari'):
                        self.parent_gui.open_zarr_in_napari(
                            self.zarr_paths[well_name], 
                            well_name
                        )
    
    def contextMenuEvent(self, event):
        """Handle right-click events to show context menu"""
        from PyQt6.QtWidgets import QMenu
        
        # Create context menu
        context_menu = QMenu(self)
        context_menu.setStyleSheet("""
            QMenu {
                background-color: #262930;
                color: #ffffff;
                border: 1px solid #606060;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 20px;
                background-color: transparent;
            }
            QMenu::item:selected {
                background-color: #404040;
            }
        """)
        
        # Add Switch Layout action
        switch_layout_action = context_menu.addAction("Switch Layout")
        switch_layout_action.triggered.connect(self.switch_layout)
        
        # Show the context menu at the cursor position
        context_menu.exec(event.globalPos())
    
    def switch_layout(self):
        """Switch layout via parent GUI"""
        if self.parent_gui:
            self.parent_gui.toggle_layout()


class DatasetDropDialog(QDialog):
    """Minimal dialog for dropping dataset directory"""
    
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the drop dialog UI"""
        self.setWindowTitle("Drop Dataset")
        self.setFixedSize(300, 200)
        self.setStyleSheet("""
            QDialog {
                background-color: #262930;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #606060;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ffffff;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Drop area
        self.drop_area = QLabel("Drop folder here")
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #606060;
                border-radius: 6px;
                padding: 60px 20px;
                background-color: #1e1e1e;
                font-size: 14px;
                color: #cccccc;
            }
            QLabel:hover {
                border-color: #808080;
                background-color: #252525;
            }
        """)
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.drop_area)
        
        # Browse button
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_directory)
        browse_btn.setFixedWidth(80)
        layout.addWidget(browse_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Status
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #00ff00; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_area.setStyleSheet("""
                QLabel {
                    border: 2px dashed #00ff00;
                    border-radius: 8px;
                    padding: 40px;
                    background-color: #1e1e1e;
                    font-size: 16px;
                    color: #00ff00;
                }
            """)
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event"""
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #606060;
                border-radius: 8px;
                padding: 40px;
                background-color: #1e1e1e;
                font-size: 16px;
                color: #cccccc;
            }
            QLabel:hover {
                border-color: #808080;
                background-color: #252525;
            }
        """)
    
    def dropEvent(self, event):
        """Handle drop event"""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                self.validate_and_set_dataset(path)
            else:
                self.status_label.setText("❌ Please drop a directory, not a file")
                self.status_label.setStyleSheet("color: #ff0000; font-size: 12px;")
        
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #606060;
                border-radius: 8px;
                padding: 40px;
                background-color: #1e1e1e;
                font-size: 16px;
                color: #cccccc;
            }
            QLabel:hover {
                border-color: #808080;
                background-color: #252525;
            }
        """)
    
    def browse_directory(self):
        """Browse for directory"""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Dataset Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.validate_and_set_dataset(directory)
    
    def validate_and_set_dataset(self, directory):
        """Validate the dataset directory and set it if valid"""
        # Check for timepoint directories (e.g., 0_stitched, 1_stitched, etc.)
        timepoint_dirs = []
        for item in Path(directory).iterdir():
            if item.is_dir() and item.name.endswith("_stitched"):
                timepoint_dirs.append(item)
        
        if not timepoint_dirs:
            self.status_label.setText("❌ No timepoint directories found (expected format: 0_stitched, 1_stitched, etc.)")
            self.status_label.setStyleSheet("color: #ff0000; font-size: 12px;")
            return
        
        # Check for OME-ZARR files in timepoint directories
        total_zarr_files = 0
        for timepoint_dir in timepoint_dirs:
            zarr_files = list(timepoint_dir.glob("*.ome.zarr"))
            total_zarr_files += len(zarr_files)
        
        if total_zarr_files == 0:
            self.status_label.setText("❌ No OME-ZARR files found in timepoint directories")
            self.status_label.setStyleSheet("color: #ff0000; font-size: 12px;")
            return
        
        # Valid multi-timepoint dataset
        self.dataset_path = directory
        self.status_label.setText(f"Found {len(timepoint_dirs)} timepoints with {total_zarr_files} total OME-ZARR files")
        self.status_label.setStyleSheet("color: #00ff00; font-size: 12px;")
        
        # Close dialog after a short delay
        QTimer.singleShot(1000, self.accept)


class GridViewerGUI(QMainWindow):
    """Main GUI window with grid on left and napari viewer on right"""
    
    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.timepoints = []
        self.current_timepoint = None
        self.stitched_dir = None
        self.workers = []
        self.contrast_limits = {}  # Store contrast limits per channel
        # NEW: Enhanced contrast management
        self.global_contrast_limits = {}  # Global contrast limits per channel type
        self.current_region_contrast = {}  # Current region's contrast limits
        self.contrast_callbacks = {}  # Store callback connections
        # NEW: Channel mapping and throttling
        self.channel_mapping = {}  # Map full channel names to simplified types
        self.contrast_update_timer = None  # Timer for throttling thumbnail updates
        # NEW: Visibility tracking
        self.channel_visibility = {}  # Store visibility state per channel
        self.visibility_callbacks = {}  # Store visibility callback connections
        
        # NEW: Data caching for faster thumbnail updates
        self.zarr_data_cache = {}  # Cache loaded zarr data per file
        self.last_contrast_limits = {}  # Track last applied contrast limits per channel type
        
        self.discover_timepoints()
        self.init_ui()
        if self.timepoints:
            self.current_timepoint = self.timepoints[0]
        self.load_zarr_files()
        self.create_thumbnails()
        
        # Ensure window is shown
        self.show()
    
    @staticmethod
    def get_channel_color(channel_name: str) -> int:
        """Compute the color for display of a given channel name."""
        color_map = {
            "405": 0x0000FF,  # Blue
            "488": 0x00FF00,  # Green
            "561": 0xFFCF00,  # Yellow
            "638": 0xFF0000,  # Red
            "730": 0x770000,  # Dark Red
            "_B": 0x0000FF,  # Blue
            "_G": 0x00FF00,  # Green
            "_R": 0xFF0000,  # Red
        }
        
        # Check for wavelength patterns first
        for key, value in color_map.items():
            if key in channel_name:
                return value
        
        # If no wavelength found, assign colors based on region name
        region_colors = {
            "A1": 0x0000FF,  # Blue
            "A2": 0x00FF00,  # Green
            "A3": 0xFFCF00,  # Yellow
            "A4": 0xFF0000,  # Red
            "B1": 0x770000,  # Dark Red
            "B2": 0xFF00FF,  # Magenta
            "B3": 0x00FFFF,  # Cyan
            "B4": 0xFFFF00,  # Yellow
        }
        
        return region_colors.get(channel_name, 0xFFFFFF)  # Default to white if no match found
    
    def extract_wavelength(self, channel_name: str) -> Optional[str]:
        """Extract wavelength from channel name."""
        # Look for wavelength patterns in channel name
        for wavelength in ["405", "488", "561", "638", "730"]:
            if wavelength in channel_name:
                return wavelength
        # Look for color indicators
        for color in ["_B", "_G", "_R"]:
            if color in channel_name:
                return color
        return None
    
    def generate_colormap(self, channel_info: Dict[str, Any]) -> str:
        """Generate colormap name for a channel."""
        return channel_info.get("name", "gray")
    
    def calculate_contrast_limits(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate contrast limits based on actual data range."""
        return (float(data.min()), float(data.max()))
    
    def apply_color_and_contrast(self, layer, channel_name: str, data: np.ndarray):
        """Apply color palette and contrast to a napari layer."""
        wavelength = self.extract_wavelength(channel_name)
        channel_info = CHANNEL_COLORS_MAP.get(
            cast(Any, wavelength), {"hex": 0xFFFFFF, "name": "gray"}
        )
        
        # Set colormap
        if channel_info["name"] in AVAILABLE_COLORMAPS:
            layer.colormap = AVAILABLE_COLORMAPS[channel_info["name"]]
        else:
            layer.colormap = self.generate_colormap(channel_info)
        
        # Let napari auto-calculate contrast limits based on data distribution
        # This will give proper histogram-based contrast limits
        
        # Store contrast limits for synchronization
        self.contrast_limits[channel_name] = layer.contrast_limits
    
    def _apply_color_to_thumbnail(self, image: np.ndarray, channel_name: str, omero_channels: List[Dict] = None) -> np.ndarray:
        """Apply color palette to thumbnail image using ome-zarr metadata."""
        if len(image.shape) != 2:
            return image  # Only process 2D images
        
        # Get color for this channel
        color_hex = self.get_channel_color_from_omero(channel_name, omero_channels)
        print(f"Applying color to {channel_name}: {hex(color_hex)}")
        
        # Convert hex to RGB
        r = (color_hex >> 16) & 0xFF
        g = (color_hex >> 8) & 0xFF
        b = color_hex & 0xFF
        
        # Normalize image to 0-1 range
        if image.max() > 0:
            normalized = (image - image.min()) / (image.max() - image.min())
        else:
            normalized = image
        
        # Apply color
        colored = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        colored[:, :, 0] = (normalized * r).astype(np.uint8)
        colored[:, :, 1] = (normalized * g).astype(np.uint8)
        colored[:, :, 2] = (normalized * b).astype(np.uint8)
        
        return colored
    
    def get_channel_color_from_omero(self, channel_name: str, omero_channels: List[Dict] = None) -> int:
        """Get channel color from OME metadata using ome-zarr or fallback to default mapping."""
        if omero_channels:
            try:
                for channel_info in omero_channels:
                    if channel_info.get('label') == channel_name:
                        # Color is stored as hex string in OME metadata
                        color_str = channel_info.get('color', 'FFFFFF')
                        return int(color_str, 16)
            except Exception as e:
                print(f"Error reading color from OME metadata: {e}")
        
        # Fallback to default color mapping
        return self.get_channel_color(channel_name)
    
    def _get_channel_name_from_omero(self, omero_spec, channel_idx: int) -> str:
        """Get channel name from OME metadata using ome-zarr."""
        try:
            if omero_spec and 'channels' in omero_spec.lookup('omero', {}):
                omero_channels = omero_spec.lookup('omero', {})['channels']
                if len(omero_channels) > channel_idx:
                    channel_info = omero_channels[channel_idx]
                    return channel_info.get('label', f'ch{channel_idx}')
            return f'ch{channel_idx}'
        except Exception as e:
            print(f"Error reading channel metadata: {e}")
            return f'ch{channel_idx}'
    
    def apply_color_and_contrast_from_omero(self, layer, channel_name: str, data: np.ndarray, channel_info: Dict):
        """Apply color palette and contrast from OME metadata using ome-zarr."""
        try:
            # Get color from OME metadata
            color_str = channel_info.get('color', 'FFFFFF')
            color_hex = int(color_str, 16)
            
            # Convert hex to RGB and create colormap
            r = (color_hex >> 16) & 0xFF
            g = (color_hex >> 8) & 0xFF
            b = color_hex & 0xFF
            
            # Create a simple colormap from the color
            from napari.utils.colormaps import Colormap
            colors = [[0, 0, 0, 0], [r/255, g/255, b/255, 1]]
            colormap = Colormap(colors, name=f"custom_{channel_name}")
            layer.colormap = colormap
            
            # Let napari auto-calculate contrast limits based on data distribution
            # This will give proper histogram-based contrast limits
            
            # Store contrast limits for synchronization
            self.contrast_limits[channel_name] = layer.contrast_limits
            
        except Exception as e:
            print(f"Error applying color from OME metadata: {e}")
            # Fallback to default method
            self.apply_color_and_contrast(layer, channel_name, data)
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("HCS Grid Viewer")
        self.setGeometry(100, 100, 1400, 800)
        
        # Set custom theme with #262930 color
        self.setStyleSheet("""
            QMainWindow {
                background-color: #262930;
                color: #ffffff;
            }
            QWidget {
                background-color: #262930;
                color: #ffffff;
            }
            QScrollArea {
                background-color: #262930;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #404040;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #606060;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #707070;
            }
        """)
        
        # NEW: Create menu bar for debugging
        self.create_menu_bar()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create splitter with flexible orientation and custom resize handles
        self.splitter = CustomSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(self.splitter)
        
        # Create panels
        self.create_napari_panel(self.splitter)
        self.create_grid_panel(self.splitter)
        
        # Set initial splitter proportions - give more space to napari viewer (top), less to well plate (bottom)
        self.splitter.setSizes([800, 400])
    
    def create_menu_bar(self):
        """Create menu bar with debugging options"""
        menubar = self.menuBar()
        
        # Debug menu
        debug_menu = menubar.addMenu('Debug')
        
        # Test contrast synchronization action
        test_action = debug_menu.addAction('Test Contrast Sync')
        test_action.triggered.connect(self.test_contrast_synchronization)
        test_action.setShortcut('Ctrl+T')
        
        # Debug contrast info action
        debug_action = debug_menu.addAction('Debug Contrast Info') 
        debug_action.triggered.connect(self.debug_contrast_info)
        debug_action.setShortcut('Ctrl+D')
        
        # Compare thumbnail action
        compare_action = debug_menu.addAction('Compare Thumbnail vs Napari')
        compare_action.triggered.connect(lambda: self.compare_thumbnail_with_napari("current"))
        compare_action.setShortcut('Ctrl+C')
        
        # Separator
        debug_menu.addSeparator()
        
        # Sync all thumbnails action
        sync_action = debug_menu.addAction('Sync All Thumbnails with Napari')
        sync_action.triggered.connect(self._sync_all_thumbnails_with_napari)
        sync_action.setShortcut('Ctrl+S')
        
        # Regenerate thumbnails action
        regen_action = debug_menu.addAction('Regenerate Thumbnails')
        regen_action.triggered.connect(self._update_thumbnails_with_contrast)
        regen_action.setShortcut('Ctrl+R')
        
        # Style the menu bar
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #404040;
                color: #ffffff;
                border-bottom: 1px solid #606060;
                padding: 2px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background-color: #606060;
                border-radius: 2px;
            }
            QMenu {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #606060;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                background-color: transparent;
            }
            QMenu::item:selected {
                background-color: #606060;
            }
            QMenu::separator {
                height: 1px;
                background-color: #606060;
                margin: 4px;
            }
        """)


    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up napari viewer
        if hasattr(self, 'viewer'):
            self.viewer.close()
        event.accept()
    
    def create_grid_panel(self, splitter):
        """Create the well plate panel"""
        grid_widget = QWidget()
        grid_widget.setMinimumSize(250, 200)  # Minimum size for both orientations
        
        # Layout
        layout = QVBoxLayout(grid_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Cephla HCS Viewer")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
            }
        """)
        layout.addWidget(title_label)
        
        # Timepoint selector (only show if multiple timepoints exist)
        if len(self.timepoints) > 1:
            timepoint_layout = QHBoxLayout()
            timepoint_layout.setSpacing(5)
            
            timepoint_label = QLabel("Timepoint:")
            timepoint_label.setStyleSheet("""
                QLabel {
                    color: #ffffff;
                    font-size: 12px;
                }
            """)
            timepoint_layout.addWidget(timepoint_label)
            
            self.timepoint_combo = QComboBox()
            self.timepoint_combo.addItems(self.timepoints)
            self.timepoint_combo.setCurrentText(self.current_timepoint)
            self.timepoint_combo.setStyleSheet("""
                QComboBox {
                    background-color: #404040;
                    border: 1px solid #606060;
                    border-radius: 4px;
                    padding: 4px 8px;
                    color: #ffffff;
                    font-size: 11px;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox::down-arrow {
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid #ffffff;
                }
                QComboBox QAbstractItemView {
                    background-color: #404040;
                    border: 1px solid #606060;
                    color: #ffffff;
                    selection-background-color: #606060;
                }
            """)
            self.timepoint_combo.currentTextChanged.connect(self.on_timepoint_changed)
            timepoint_layout.addWidget(self.timepoint_combo)
            
            layout.addLayout(timepoint_layout)
        
        # Well plate widget (no scroll area - always visible)
        self.well_plate = WellPlateWidget(self)
        layout.addWidget(self.well_plate)
        
        splitter.addWidget(grid_widget)
    
    def create_napari_panel(self, splitter):
        """Create the right panel with embedded napari viewer"""
        napari_widget = QWidget()
        
        # Layout
        layout = QVBoxLayout(napari_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Initialize napari viewer with embedded mode
        self.viewer = napari.Viewer(show=False)
        
        # Set napari theme to dark
        try:
            self.viewer.theme = 'dark'
        except:
            pass  # Theme setting might not be available in all versions
        
        # Get the complete Qt window and embed it
        self.viewerWidget = self.viewer.window._qt_window
        
        # Create a layout for the napari container to ensure proper sizing
        layout.addWidget(self.viewerWidget)
        
        splitter.addWidget(napari_widget)
        
        # NEW: Initialize contrast monitoring after viewer is created
        self.setup_contrast_monitoring()
    

    def toggle_layout(self):
        """Toggle between horizontal and vertical splitter orientation"""
        current_orientation = self.splitter.orientation()
        
        if current_orientation == Qt.Orientation.Horizontal:
            # Switch to vertical (well plate on bottom)
            self.splitter.setOrientation(Qt.Orientation.Vertical)
            # Reorder widgets to put well plate on bottom
            self.splitter.insertWidget(0, self.splitter.widget(1))  # Move napari to top
            self.splitter.insertWidget(1, self.splitter.widget(1))  # Move well plate to bottom
            # Adjust proportions for vertical layout (napari on top, well plate on bottom)
            self.splitter.setSizes([800, 400])
        else:
            # Switch to horizontal (well plate on left)
            self.splitter.setOrientation(Qt.Orientation.Horizontal)
            # Reorder widgets to put well plate on left
            self.splitter.insertWidget(0, self.splitter.widget(1))  # Move well plate to left
            self.splitter.insertWidget(1, self.splitter.widget(1))  # Move napari to right
            # Adjust proportions for horizontal layout
            self.splitter.setSizes([300, 1100])
        
        # Force update of splitter handles to match new orientation
        self.splitter.update()
    
    def discover_timepoints(self):
        """Discover available timepoints in the dataset"""
        self.timepoints = []
        
        # Look for timepoint directories (e.g., 0_stitched, 1_stitched, etc.)
        for item in self.dataset_path.iterdir():
            if item.is_dir() and item.name.endswith("_stitched"):
                timepoint_name = item.name.replace("_stitched", "")
                self.timepoints.append(timepoint_name)
        
        # Sort timepoints (handle both numeric and string timepoints)
        try:
            self.timepoints.sort(key=lambda x: int(x) if x.isdigit() else x)
        except:
            self.timepoints.sort()
        
        print(f"Found {len(self.timepoints)} timepoints: {self.timepoints}")
    
    def load_zarr_files(self):
        """Load zarr files from the current timepoint's stitched directory"""
        if not self.current_timepoint:
            raise ValueError("No timepoint selected")
        
        self.stitched_dir = self.dataset_path / f"{self.current_timepoint}_stitched"
        if not self.stitched_dir.exists():
            raise FileNotFoundError(f"Stitched directory not found: {self.stitched_dir}")
        
        self.zarr_files = {}
        for zarr_dir in self.stitched_dir.glob("*.ome.zarr"):
            # Extract region name from directory name
            region_name = zarr_dir.stem.replace("_stitched", "")
            # Remove .ome suffix if present
            if region_name.endswith(".ome"):
                region_name = region_name[:-4]
            self.zarr_files[region_name] = str(zarr_dir)
        
        print(f"Found {len(self.zarr_files)} zarr files in timepoint {self.current_timepoint}:")
        for region, path in self.zarr_files.items():
            print(f"  {region}: {path}")
    

    
    def create_thumbnails(self):
        """Initialize well plate and start thumbnail generation (thread pool version)"""
        self.well_plate.set_data(self.zarr_files)
        if not hasattr(self, 'thread_pool'):
            self.thread_pool = QThreadPool.globalInstance()
            self.thread_pool.setMaxThreadCount(16)  # Limit concurrency
        self.workers = []
        for region_name, zarr_path in self.zarr_files.items():
            worker = ThumbnailWorker(zarr_path, region_name, self)
            # NEW: Pass cached data if available (for initial load)
            if zarr_path in self.zarr_data_cache:
                worker.set_cached_data(self.zarr_data_cache[zarr_path])
            # NEW: Pass current contrast limits to worker
            worker.set_contrast_limits(self.global_contrast_limits)
            # NEW: Pass current channel visibility to worker
            worker.set_channel_visibility(self.channel_visibility)
            worker.signals.thumbnail_ready.connect(self.on_thumbnail_ready)
            worker.signals.error_occurred.connect(self.on_thumbnail_error)
            # NEW: Cache data when loaded
            worker.signals.data_loaded.connect(self._cache_zarr_data)
            self.thread_pool.start(worker)
            self.workers.append(worker)
    
    def on_thumbnail_ready(self, region_name: str, thumbnail: np.ndarray, channel_names: str):
        """Handle thumbnail generation completion"""
        self.well_plate.set_thumbnail(region_name, thumbnail)
    
    def on_thumbnail_error(self, region_name: str, error_msg: str):
        """Handle thumbnail generation error"""
        print(f"Error generating thumbnail for {region_name}: {error_msg}")
        # Could set an error image or leave as loading
    
    def on_timepoint_changed(self, timepoint: str):
        """Handle timepoint selection change"""
        if timepoint != self.current_timepoint:
            self.current_timepoint = timepoint
            print(f"Switching to timepoint: {timepoint}")
            # No need to quit workers; thread pool will manage
            self.workers.clear()
            self.well_plate.clear_thumbnails()
            self.load_zarr_files()
            self.create_thumbnails()
    
    def open_zarr_in_napari(self, zarr_path: str, region_name: str):
        """Open a zarr file in the embedded napari viewer using napari's native OME-ZARR plugin"""
        try:
            print(f"Opening zarr file: {zarr_path}")
            
            # NEW: Capture and preserve current user settings before clearing
            if hasattr(self, 'viewer') and self.viewer.layers:
                self._update_global_settings_from_current_layers()
            
            # Clear existing layers
            self.viewer.layers.clear()
            
            # Use napari's native OME-ZARR plugin for instant loading
            # This is the same method that napari uses when you drag and drop
            from napari_ome_zarr import napari_get_reader
            
            # Get the reader function
            reader_func = napari_get_reader(zarr_path)
            if reader_func is None:
                QMessageBox.critical(self, "Error", "Could not find OME-ZARR reader")
                return
            
            # Read the data using napari's native plugin
            layer_data_list = reader_func(zarr_path)
            print(f"Found {len(layer_data_list)} layers to add")
            
            # Add each layer to the viewer
            for i, layer_data in enumerate(layer_data_list):
                if len(layer_data) >= 2:
                    data, metadata = layer_data[0], layer_data[1]
                    layer_name = metadata.get('name', f"{region_name}_layer")
                    
                    # NEW: Filter metadata to only include hashable values
                    filtered_metadata = self._filter_hashable_metadata(metadata)
                    
                    # Add the layer with filtered metadata
                    try:
                        # For multichannel images, we don't pass 'name' in the kwargs since it's already extracted
                        # and will be used as the name parameter
                        metadata_without_name = {k: v for k, v in filtered_metadata.items() if k != 'name'}
                        
                        layer = self.viewer.add_image(
                            data,
                            name=layer_name,
                            **metadata_without_name
                        )
                        print(f"Successfully added layer with {len(layer_name) if isinstance(layer_name, list) else 1} channel(s)")
                        
                    except Exception as layer_error:
                        print(f"Error adding layer: {layer_error}")
                        raise layer_error
                    
                else:
                    # Fallback for simple data
                    data = layer_data[0]
                    layer_name = f"{region_name}_data"
                    
                    try:
                        layer = self.viewer.add_image(data, name=layer_name)
                        print(f"Successfully added simple layer: {layer_name}")
                        
                    except Exception as layer_error:
                        print(f"Error adding simple layer: {layer_error}")
                        raise layer_error
            
            # NEW: Apply preserved user settings after all layers are loaded
            QTimer.singleShot(50, self._apply_global_settings_to_all_layers)
            
            # NEW: Set up contrast monitoring for all new layers (after settings applied)
            QTimer.singleShot(100, self.setup_contrast_monitoring)
            
            # NEW: Update all thumbnails to match current napari contrast settings
            QTimer.singleShot(150, self._sync_all_thumbnails_with_napari)
            
            print(f"Successfully opened zarr file with preserved user settings")
            
        except Exception as e:
            print(f"Error opening zarr file: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to open zarr file: {str(e)}")
    
    def _update_global_settings_from_current_layers(self):
        """Update global settings from current napari layers before switching wells"""
        print("Preserving current user settings...")
        
        if not hasattr(self, 'viewer') or not self.viewer.layers:
            return
            
        for layer in self.viewer.layers:
            layer_name = layer.name
            
            if isinstance(layer_name, list):
                # Multichannel layer
                for i, individual_name in enumerate(layer_name):
                    channel_type = self._extract_channel_type_from_layer(individual_name)
                    if channel_type:
                        # Store contrast limits
                        if hasattr(layer, 'contrast_limits'):
                            if isinstance(layer.contrast_limits, list) and i < len(layer.contrast_limits):
                                individual_limits = layer.contrast_limits[i]
                            else:
                                individual_limits = layer.contrast_limits
                            self.global_contrast_limits[channel_type] = individual_limits
                            self.current_region_contrast[individual_name] = individual_limits
                            print(f"Preserved contrast for {channel_type}: {individual_limits}")
                        
                        # Store visibility
                        if hasattr(layer, 'visible'):
                            self.channel_visibility[individual_name] = layer.visible
                            print(f"Preserved visibility for {individual_name}: {layer.visible}")
            else:
                # Single channel layer
                channel_type = self._extract_channel_type_from_layer(layer_name)
                if channel_type:
                    # Store contrast limits
                    if hasattr(layer, 'contrast_limits'):
                        self.global_contrast_limits[channel_type] = layer.contrast_limits
                        self.current_region_contrast[layer_name] = layer.contrast_limits
                        print(f"Preserved contrast for {channel_type}: {layer.contrast_limits}")
                    
                    # Store visibility
                    if hasattr(layer, 'visible'):
                        self.channel_visibility[layer_name] = layer.visible
                        print(f"Preserved visibility for {layer_name}: {layer.visible}")
    
    def _apply_global_settings_to_all_layers(self):
        """Apply preserved global settings to all current napari layers"""
        print("Applying preserved settings to new layers...")
        
        if not hasattr(self, 'viewer') or not self.viewer.layers:
            return
            
        for layer in self.viewer.layers:
            layer_name = layer.name
            
            if isinstance(layer_name, list):
                # Multichannel layer
                contrast_limits_list = []
                visibility_to_apply = True  # Start with visible
                
                for i, individual_name in enumerate(layer_name):
                    channel_type = self._extract_channel_type_from_layer(individual_name)
                    
                    # Get preserved contrast limits
                    if channel_type and channel_type in self.global_contrast_limits:
                        contrast_limits_list.append(self.global_contrast_limits[channel_type])
                        print(f"Will apply contrast to {channel_type}: {self.global_contrast_limits[channel_type]}")
                    else:
                        # Use current layer's contrast if no preserved setting
                        if hasattr(layer, 'contrast_limits'):
                            if isinstance(layer.contrast_limits, list) and i < len(layer.contrast_limits):
                                contrast_limits_list.append(layer.contrast_limits[i])
                            else:
                                contrast_limits_list.append(layer.contrast_limits)
                    
                    # Check visibility (if any channel should be invisible, make layer invisible)
                    if individual_name in self.channel_visibility:
                        if not self.channel_visibility[individual_name]:
                            visibility_to_apply = False
                            print(f"Channel {individual_name} should be invisible")
                
                # Apply contrast limits to multichannel layer
                if contrast_limits_list:
                    try:
                        layer.contrast_limits = contrast_limits_list
                        print(f"Applied multichannel contrast limits: {contrast_limits_list}")
                    except Exception as e:
                        print(f"Could not apply multichannel contrast limits: {e}")
                
                # Apply visibility
                try:
                    layer.visible = visibility_to_apply
                    print(f"Applied visibility to multichannel layer: {visibility_to_apply}")
                except Exception as e:
                    print(f"Could not apply visibility: {e}")
                    
            else:
                # Single channel layer
                channel_type = self._extract_channel_type_from_layer(layer_name)
                
                # Apply contrast limits
                if channel_type and channel_type in self.global_contrast_limits:
                    try:
                        layer.contrast_limits = self.global_contrast_limits[channel_type]
                        print(f"Applied contrast to {channel_type}: {self.global_contrast_limits[channel_type]}")
                    except Exception as e:
                        print(f"Could not apply contrast to {channel_type}: {e}")
                
                # Apply visibility
                if layer_name in self.channel_visibility:
                    try:
                        layer.visible = self.channel_visibility[layer_name]
                        print(f"Applied visibility to {layer_name}: {self.channel_visibility[layer_name]}")
                    except Exception as e:
                        print(f"Could not apply visibility to {layer_name}: {e}")
        
        # Update our internal state to match what we just applied
        self._update_global_settings_from_current_layers()
    
    def _sync_all_thumbnails_with_napari(self):
        """Synchronize all thumbnails with current napari contrast and visibility settings"""
        print("Synchronizing all thumbnails with current napari settings...")
        
        # Update global contrast limits and visibility from current napari layers
        if hasattr(self, 'viewer') and self.viewer.layers:
            for layer in self.viewer.layers:
                layer_name = layer.name
                
                # Update contrast limits
                if hasattr(layer, 'contrast_limits'):
                    if isinstance(layer_name, list):
                        # Multichannel layer
                        for i, individual_name in enumerate(layer_name):
                            channel_type = self._extract_channel_type_from_layer(individual_name)
                            if channel_type:
                                if isinstance(layer.contrast_limits, list) and i < len(layer.contrast_limits):
                                    individual_limits = layer.contrast_limits[i]
                                else:
                                    individual_limits = layer.contrast_limits
                                self.global_contrast_limits[channel_type] = individual_limits
                                self.current_region_contrast[individual_name] = individual_limits
                                # NEW: Update last known limits to prevent unnecessary updates
                                self.last_contrast_limits[channel_type] = individual_limits
                    else:
                        # Single channel layer
                        channel_type = self._extract_channel_type_from_layer(layer_name)
                        if channel_type and hasattr(layer, 'contrast_limits'):
                            self.global_contrast_limits[channel_type] = layer.contrast_limits
                            self.current_region_contrast[layer_name] = layer.contrast_limits
                            # NEW: Update last known limits to prevent unnecessary updates
                            self.last_contrast_limits[channel_type] = layer.contrast_limits
                
                # Update visibility
                if hasattr(layer, 'visible'):
                    if isinstance(layer_name, list):
                        # Multichannel layer
                        for individual_name in layer_name:
                            self.channel_visibility[individual_name] = layer.visible
                    else:
                        # Single channel layer
                        self.channel_visibility[layer_name] = layer.visible
        
        # Trigger thumbnail regeneration for all wells using cached data
        self._create_thumbnails_from_cache()
    
    def _filter_hashable_metadata(self, metadata: Dict) -> Dict:
        """Filter metadata to only include values that are hashable and safe for napari"""
        filtered = {}
        
        # Special handling for multichannel image metadata
        # These properties are lists for multichannel images and napari expects them
        multichannel_list_properties = ['name', 'visible', 'contrast_limits', 'colormap']
        
        for key, value in metadata.items():
            # Handle multichannel properties specially
            if key in multichannel_list_properties and isinstance(value, list):
                # For multichannel properties, keep the lists as napari expects them
                filtered[key] = value
                continue
                
            try:
                # Test if the value is hashable
                hash(value)
                
                # Additional check for types that napari expects
                if isinstance(value, (str, int, float, bool, tuple, type(None))):
                    filtered[key] = value
                elif isinstance(value, (list, dict, np.ndarray)):
                    # Skip other lists, dicts, and arrays that aren't multichannel properties
                    continue
                else:
                    # For other types, try to include them but be cautious
                    filtered[key] = value
                    
            except TypeError:
                # Value is not hashable, skip it (unless it's a multichannel property)
                continue
        
        return filtered

    def setup_contrast_monitoring(self):
        """Set up monitoring of napari layer contrast changes"""
        if not hasattr(self, 'viewer'):
            return
            
        # Monitor when layers are added/removed
        self.viewer.layers.events.inserted.connect(self._on_layer_added)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        
        # Set up monitoring for existing layers
        for layer in self.viewer.layers:
            if hasattr(layer, 'contrast_limits'):
                self._setup_layer_contrast_monitoring(layer)
    
    def _on_layer_added(self, event):
        """Handle when a new layer is added to napari"""
        layer = event.value
        if hasattr(layer, 'contrast_limits'):
            self._setup_layer_contrast_monitoring(layer)
    
    def _on_layer_removed(self, event):
        """Handle when a layer is removed from napari"""
        layer = event.value
        layer_name = layer.name
        
        # Clean up stored callbacks and data
        if layer_name in self.contrast_callbacks:
            del self.contrast_callbacks[layer_name]
        if layer_name in self.visibility_callbacks:
            del self.visibility_callbacks[layer_name]
        if layer_name in self.current_region_contrast:
            del self.current_region_contrast[layer_name]
        if layer_name in self.channel_visibility:
            del self.channel_visibility[layer_name]
    
    def _setup_layer_contrast_monitoring(self, layer):
        """Set up contrast limit monitoring for a specific layer"""
        layer_name = layer.name
        
        # Store initial contrast limits and visibility
        self.current_region_contrast[layer_name] = layer.contrast_limits
        self.channel_visibility[layer_name] = layer.visible
        
        # Connect to contrast limit changes
        contrast_callback = lambda event, layer=layer: self._on_contrast_limits_changed(layer.name, layer.contrast_limits)
        layer.events.contrast_limits.connect(contrast_callback)
        self.contrast_callbacks[layer_name] = contrast_callback
        
        # NEW: Connect to visibility changes
        visibility_callback = lambda event, layer=layer: self._on_visibility_changed(layer.name, layer.visible)
        layer.events.visible.connect(visibility_callback)
        self.visibility_callbacks[layer_name] = visibility_callback
        
        print(f"Set up contrast and visibility monitoring for layer: {layer_name}")
    
    def _on_contrast_limits_changed(self, layer_name: str, new_limits: tuple):
        """Handle when contrast limits change in napari"""
        print(f"Contrast limits changed for {layer_name}: {new_limits}")
        
        # Handle both single layer names and lists of layer names
        if isinstance(layer_name, list):
            # For multichannel images, store contrast for each channel
            for i, individual_name in enumerate(layer_name):
                if isinstance(new_limits, list) and i < len(new_limits):
                    individual_limits = new_limits[i]
                else:
                    individual_limits = new_limits
                    
                self.current_region_contrast[individual_name] = individual_limits
                
                # Extract channel type and update global limits
                channel_type = self._extract_channel_type_from_layer(individual_name)
                if channel_type:
                    self.global_contrast_limits[channel_type] = individual_limits
                    self.channel_mapping[individual_name] = channel_type
                    print(f"Updated global contrast for channel type {channel_type}: {individual_limits}")
        else:
            # Store the new contrast limits for single channel
            self.current_region_contrast[layer_name] = new_limits
            
            # Extract channel type from layer name for global mapping
            channel_type = self._extract_channel_type_from_layer(layer_name)
            if channel_type:
                self.global_contrast_limits[channel_type] = new_limits
                print(f"Updated global contrast for channel type {channel_type}: {new_limits}")
                
                # Update channel mapping
                self.channel_mapping[layer_name] = channel_type
        
        # Trigger thumbnail regeneration with throttling
        self._schedule_thumbnail_update()
    
    def _schedule_thumbnail_update(self):
        """Schedule thumbnail update with throttling to prevent excessive regeneration"""
        # Cancel existing timer if any
        if self.contrast_update_timer is not None:
            self.contrast_update_timer.stop()
        
        # Create new timer for delayed update (reduced delay for responsiveness)
        self.contrast_update_timer = QTimer()
        self.contrast_update_timer.setSingleShot(True)
        self.contrast_update_timer.timeout.connect(self._smart_thumbnail_update)
        self.contrast_update_timer.start(100)  # Reduced from 500ms to 100ms
    
    def _smart_thumbnail_update(self):
        """Smart thumbnail update that only regenerates when necessary"""
        print("Smart thumbnail update: checking what needs updating...")
        
        # Check which channel types actually changed
        changed_channel_types = set()
        for channel_type, current_limits in self.global_contrast_limits.items():
            last_limits = self.last_contrast_limits.get(channel_type)
            if last_limits != current_limits:
                changed_channel_types.add(channel_type)
                self.last_contrast_limits[channel_type] = current_limits
                print(f"Channel type {channel_type} contrast changed: {current_limits}")
        
        # If no contrast changes, check visibility changes
        if not changed_channel_types:
            print("No contrast changes detected, updating for visibility changes...")
            self._update_thumbnails_with_contrast()
            return
        
        print(f"Updating thumbnails for changed channel types: {changed_channel_types}")
        self._update_thumbnails_with_contrast()

    def _update_thumbnails_with_contrast(self):
        """Regenerate thumbnails with current contrast limits using cached data"""
        print("Updating thumbnails with new contrast limits...")
        # Clear existing thumbnails
        self.well_plate.clear_thumbnails()
        
        # Use cached data for faster regeneration
        self._create_thumbnails_from_cache()
    
    def _create_thumbnails_from_cache(self):
        """Create thumbnails using cached data when available"""
        if not hasattr(self, 'thread_pool'):
            self.thread_pool = QThreadPool.globalInstance()
            self.thread_pool.setMaxThreadCount(16)
        
        self.workers = []
        for region_name, zarr_path in self.zarr_files.items():
            worker = ThumbnailWorker(zarr_path, region_name, self)
            # NEW: Pass cached data if available
            if zarr_path in self.zarr_data_cache:
                worker.set_cached_data(self.zarr_data_cache[zarr_path])
            worker.set_contrast_limits(self.global_contrast_limits)
            worker.set_channel_visibility(self.channel_visibility)
            worker.signals.thumbnail_ready.connect(self.on_thumbnail_ready)
            worker.signals.error_occurred.connect(self.on_thumbnail_error)
            # NEW: Cache data when loaded
            worker.signals.data_loaded.connect(self._cache_zarr_data)
            self.thread_pool.start(worker)
            self.workers.append(worker)
    
    def _cache_zarr_data(self, zarr_path: str, data_dict: dict):
        """Cache loaded zarr data for faster future access"""
        self.zarr_data_cache[zarr_path] = data_dict
        print(f"Cached data for {zarr_path}")

    def _extract_channel_type_from_layer(self, layer_name: str) -> Optional[str]:
        """Extract channel type from napari layer name with enhanced mapping"""
        # Check if we already have this mapped
        if layer_name in self.channel_mapping:
            return self.channel_mapping[layer_name]
        
        # Look for wavelength patterns (prioritize these)
        for wavelength in ["405", "488", "561", "638", "730"]:
            if wavelength in layer_name:
                self.channel_mapping[layer_name] = wavelength
                return wavelength
        
        # Look for color indicators  
        for color in ["_B", "_G", "_R"]:
            if color in layer_name:
                self.channel_mapping[layer_name] = color
                return color
                
        # Try to extract from channel patterns like "ch0", "ch1", etc.
        import re
        ch_match = re.search(r'ch(\d+)', layer_name.lower())
        if ch_match:
            channel_type = f"ch{ch_match.group(1)}"
            self.channel_mapping[layer_name] = channel_type
            return channel_type
        
        # Try to extract from common fluorescence names
        fluorescence_map = {
            'dapi': '405',
            'hoechst': '405', 
            'gfp': '488',
            'fitc': '488',
            'alexa488': '488',
            'cy3': '561',
            'tritc': '561',
            'alexa568': '561',
            'alexa594': '561',
            'cy5': '638',
            'alexa647': '638',
            'alexa680': '730',
            'cy7': '730'
        }
        
        layer_lower = layer_name.lower()
        for fluor_name, wavelength in fluorescence_map.items():
            if fluor_name in layer_lower:
                self.channel_mapping[layer_name] = wavelength
                return wavelength
        
        # If no specific pattern found, use the layer name itself as the type
        # but clean it up first
        clean_name = re.sub(r'[^\w]', '_', layer_name)
        self.channel_mapping[layer_name] = clean_name
        return clean_name
    
    def get_contrast_limits_for_channel(self, channel_name) -> Optional[tuple]:
        """Get stored contrast limits for a channel type with enhanced lookup"""
        # Handle case where channel_name is a list (multichannel)
        if isinstance(channel_name, list):
            return None
            
        # First try direct lookup
        if channel_name in self.current_region_contrast:
            return self.current_region_contrast[channel_name]
        
        # Try to find by mapped channel type
        channel_type = self._extract_channel_type_from_layer(channel_name)
        if channel_type and channel_type in self.global_contrast_limits:
            return self.global_contrast_limits[channel_type]
        
        # Try reverse lookup through channel mapping
        for mapped_name, mapped_type in self.channel_mapping.items():
            if mapped_type == channel_name and mapped_name in self.current_region_contrast:
                return self.current_region_contrast[mapped_name]
        
        # Try partial matching for channel names
        for stored_name, limits in self.current_region_contrast.items():
            stored_type = self._extract_channel_type_from_layer(stored_name)
            if stored_type == channel_name:
                return limits
                
        return None
    
    def get_all_stored_contrast_limits(self) -> Dict[str, tuple]:
        """Get all stored contrast limits organized by channel type"""
        result = {}
        
        # Add global contrast limits
        result.update(self.global_contrast_limits)
        
        # Add current region contrast limits mapped to their types
        for layer_name, limits in self.current_region_contrast.items():
            channel_type = self._extract_channel_type_from_layer(layer_name)
            if channel_type:
                result[channel_type] = limits
                
        return result
    
    def test_contrast_synchronization(self):
        """Test method to verify contrast synchronization is working"""
        print("\n=== Testing Contrast Synchronization ===")
        
        # Test 1: Check if contrast monitoring is set up
        print(f"Napari viewer layers: {len(self.viewer.layers)}")
        for layer in self.viewer.layers:
            print(f"  Layer: {layer.name}, contrast_limits: {getattr(layer, 'contrast_limits', 'N/A')}")
        
        # Test 2: Check stored contrast limits
        print(f"Global contrast limits: {self.global_contrast_limits}")
        print(f"Current region contrast: {self.current_region_contrast}")
        print(f"Channel mapping: {self.channel_mapping}")
        
        # Test 3: Check contrast callbacks
        print(f"Active contrast callbacks: {list(self.contrast_callbacks.keys())}")
        
        # Test 4: Test channel type extraction
        test_names = [
            "DAPI_405nm_ch0",
            "GFP_488nm_ch1", 
            "RFP_561nm_ch2",
            "Cy5_638nm_ch3",
            "ch0",
            "ch1_FITC",
            "region_A1_405"
        ]
        
        print("Channel type extraction test:")
        for name in test_names:
            channel_type = self._extract_channel_type_from_layer(name)
            print(f"  {name} -> {channel_type}")
        
        print("=== Test Complete ===\n")
    
    def debug_contrast_info(self):
        """Debug method to print current contrast information"""
        print("\n=== Debug: Current Contrast Information ===")
        print(f"Global contrast limits: {self.global_contrast_limits}")
        print(f"Current region contrast: {self.current_region_contrast}")
        print(f"Channel mapping: {self.channel_mapping}")
        print(f"Channel visibility: {self.channel_visibility}")
        
        if hasattr(self, 'viewer') and self.viewer.layers:
            print("Napari layers:")
            for layer in self.viewer.layers:
                if hasattr(layer, 'contrast_limits'):
                    print(f"  {layer.name}: contrast={layer.contrast_limits}, visible={layer.visible}")
                    if hasattr(layer, 'data'):
                        data = layer.data
                        if isinstance(data, list):
                            print(f"    Multichannel data: {len(data)} channels")
                            for i, channel_data in enumerate(data):
                                if hasattr(channel_data, 'shape'):
                                    print(f"      Channel {i}: shape={channel_data.shape}, "
                                          f"min={channel_data.min()}, max={channel_data.max()}")
                        else:
                            print(f"    Data shape: {data.shape}, min={data.min()}, max={data.max()}")
        else:
            print("No napari layers found")
        print("=== End Debug ===\n")
    
    def compare_thumbnail_with_napari(self, region_name: str):
        """Compare thumbnail generation with napari display for debugging"""
        print(f"\n=== Comparing Thumbnail vs Napari for {region_name} ===")
        
        if not hasattr(self, 'viewer') or not self.viewer.layers:
            print("No napari layers to compare with")
            return
            
        # Get the current layer information
        for layer in self.viewer.layers:
            layer_name = layer.name
            if hasattr(layer, 'contrast_limits') and hasattr(layer, 'data'):
                print(f"Layer: {layer_name}")
                print(f"  Napari contrast limits: {layer.contrast_limits}")
                print(f"  Napari visibility: {layer.visible}")
                
                if isinstance(layer_name, list):
                    # Multichannel layer
                    for i, channel_name in enumerate(layer_name):
                        channel_type = self._extract_channel_type_from_layer(channel_name)
                        stored_contrast = self.get_contrast_limits_for_channel(channel_name)
                        stored_visibility = self.channel_visibility.get(channel_name, True)
                        print(f"  Channel {i} ({channel_name}):")
                        print(f"    Channel type: {channel_type}")
                        print(f"    Stored contrast: {stored_contrast}")
                        print(f"    Stored visibility: {stored_visibility}")
                        
                        if hasattr(layer.data, '__len__') and i < len(layer.data):
                            channel_data = layer.data[i]
                            print(f"    Data range: {channel_data.min()} - {channel_data.max()}")
                else:
                    # Single channel layer
                    channel_type = self._extract_channel_type_from_layer(layer_name)
                    stored_contrast = self.get_contrast_limits_for_channel(layer_name)
                    stored_visibility = self.channel_visibility.get(layer_name, True)
                    print(f"  Channel type: {channel_type}")
                    print(f"  Stored contrast: {stored_contrast}")
                    print(f"  Stored visibility: {stored_visibility}")
                    print(f"  Data range: {layer.data.min()} - {layer.data.max()}")
        
        print("=== End Comparison ===\n")

    def _on_visibility_changed(self, layer_name, is_visible: bool):
        """Handle when layer visibility changes in napari"""
        print(f"Visibility changed for {layer_name}: {is_visible}")
        
        # Handle both single layer names and lists of layer names
        if isinstance(layer_name, list):
            # For multichannel images, store visibility for each channel
            for individual_name in layer_name:
                self.channel_visibility[individual_name] = is_visible
                print(f"Updated visibility for channel {individual_name}: {is_visible}")
        else:
            # Store the new visibility for single channel
            self.channel_visibility[layer_name] = is_visible
            print(f"Updated visibility for channel {layer_name}: {is_visible}")
        
        # Trigger thumbnail regeneration with throttling
        self._schedule_thumbnail_update()

    def _update_global_settings_from_layer(self, layer, layer_name):
        """Update global settings dictionaries from a layer"""
        if isinstance(layer_name, list):
            # Multichannel layer
            for i, individual_name in enumerate(layer_name):
                channel_type = self._extract_channel_type_from_layer(individual_name)
                if channel_type:
                    if hasattr(layer, 'contrast_limits'):
                        if isinstance(layer.contrast_limits, list) and i < len(layer.contrast_limits):
                            self.global_contrast_limits[channel_type] = layer.contrast_limits[i]
                            self.current_region_contrast[individual_name] = layer.contrast_limits[i]
                        else:
                            self.global_contrast_limits[channel_type] = layer.contrast_limits
                            self.current_region_contrast[individual_name] = layer.contrast_limits
                    
                    if hasattr(layer, 'visible'):
                        self.channel_visibility[individual_name] = layer.visible
        else:
            # Single channel layer
            channel_type = self._extract_channel_type_from_layer(layer_name)
            if channel_type:
                if hasattr(layer, 'contrast_limits'):
                    self.global_contrast_limits[channel_type] = layer.contrast_limits
                    self.current_region_contrast[layer_name] = layer.contrast_limits
                
                if hasattr(layer, 'visible'):
                    self.channel_visibility[layer_name] = layer.visible

    def _apply_stored_contrast_to_layer(self, layer, layer_name):
        """Apply stored contrast limits to a napari layer"""
        # Handle multichannel layers where layer_name is a list
        if isinstance(layer_name, list):
            print(f"Storing contrast for multichannel layer with {len(layer_name)} channels")
            # For multichannel layers, we need to handle each channel separately
            # But for now, let's just store the auto-calculated contrast limits
            if hasattr(layer, 'contrast_limits'):
                contrast_limits = layer.contrast_limits
                for i, individual_name in enumerate(layer_name):
                    if isinstance(contrast_limits, list) and i < len(contrast_limits):
                        individual_limits = contrast_limits[i]
                    else:
                        individual_limits = contrast_limits
                        
                    channel_type = self._extract_channel_type_from_layer(individual_name)
                    if channel_type:
                        self.global_contrast_limits[channel_type] = individual_limits
        else:
            # Handle single channel layers
            # Try to get stored contrast limits for this channel
            stored_contrast = self.get_contrast_limits_for_channel(layer_name)
            
            if stored_contrast:
                try:
                    layer.contrast_limits = stored_contrast
                    print(f"Applied stored contrast to {layer_name}")
                except Exception as e:
                    print(f"Could not apply contrast limits to {layer_name}: {e}")
            else:
                # Store the layer's auto-calculated contrast limits
                channel_type = self._extract_channel_type_from_layer(layer_name)
                if channel_type and hasattr(layer, 'contrast_limits'):
                    self.global_contrast_limits[channel_type] = layer.contrast_limits


def main():
    """Main function to run the GUI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HCS Grid Viewer")
    parser.add_argument("dataset_path", nargs="?", 
                       default=None,
                       help="Path to the dataset directory")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    try:
        # If dataset path provided as argument, use it directly
        if args.dataset_path and os.path.exists(args.dataset_path):
            dataset_path = args.dataset_path
            # Validate dataset path has timepoint directories
            timepoint_dirs = [item for item in Path(dataset_path).iterdir() 
                            if item.is_dir() and item.name.endswith("_stitched")]
            if not timepoint_dirs:
                print(f"Error: No timepoint directories found in: {dataset_path}")
                print("Please make sure the dataset contains directories like '0_stitched', '1_stitched', etc.")
                sys.exit(1)
        else:
            # Show drop dialog to select dataset
            drop_dialog = DatasetDropDialog()
            if drop_dialog.exec() == QDialog.DialogCode.Accepted and drop_dialog.dataset_path:
                dataset_path = drop_dialog.dataset_path
            else:
                print("No dataset selected. Exiting.")
                sys.exit(0)
        
        # Create and show the main window
        window = GridViewerGUI(dataset_path)
        window.show()
        
        # Run the application
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
