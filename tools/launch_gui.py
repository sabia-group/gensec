#!/usr/bin/env python
"""
Launch script for GenSec Parameter GUI
Usage: python launch_gui.py [optional_json_file]
"""

import sys
import os

# Add parent directory to path so we can import from tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PyQt5.QtWidgets import QApplication
    from tools.parameter_gui.gui import ParameterGUI
    
    app = QApplication(sys.argv)
    gui = ParameterGUI()
    
    # If a file was passed as argument, open it
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            gui.load_file(file_path)
    
    gui.show()
    sys.exit(app.exec_())
    
except ImportError as e:
    print(f"Error: {e}")
    print("\nMissing dependencies. Install with:")
    print("  pip install PyQt5")
    sys.exit(1)
except Exception as e:
    print(f"Error starting GUI: {e}")
    sys.exit(1)
