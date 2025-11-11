#!/usr/bin/env python3
"""
Launcher script for the Grid Viewer GUI
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the GUI
from grid_viewer_gui import main

if __name__ == "__main__":
    main() 
