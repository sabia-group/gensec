"""
PyQt5 GUI for editing GenSec parameters.json files.
Features:
- Collapsible sections by category
- Type-aware input widgets
- (?) info tooltips for each parameter
- Mandatory field indicators
- Load/Save JSON
- Preview of defaults
"""

import sys
import json
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, 
    QPushButton, QFileDialog, QScrollArea, QMessageBox,
    QGroupBox, QFormLayout, QDictEdit, QFrame
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QFont, QColor

from schema import (
    PARAMETER_SCHEMA, get_all_categories, get_category_label,
    get_category_description, get_schema_field
)


class CollapsibleSection(QWidget):
    """A collapsible group box for parameter sections."""
    
    def __init__(self, title, description=""):
        super().__init__()
        self.title = title
        self.description = description
        self.is_expanded = True
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header_layout = QHBoxLayout()
        self.toggle_btn = QPushButton("▼")
        self.toggle_btn.setMaximumWidth(30)
        self.toggle_btn.clicked.connect(self.toggle)
        
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        
        header_layout.addWidget(self.toggle_btn)
        header_layout.addWidget(title_label)
        if description:
            desc_label = QLabel(f"({description})")
            desc_label.setStyleSheet("color: gray; font-style: italic;")
            header_layout.addWidget(desc_label)
        header_layout.addStretch()
        
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)
        
        # Content area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 10, 0, 10)
        layout.addWidget(self.content_widget)
        
    def add_field(self, widget):
        """Add a field widget to the section."""
        self.content_layout.addWidget(widget)
        
    def toggle(self):
        """Toggle expansion state."""
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        self.toggle_btn.setText("▼" if self.is_expanded else "▶")


class ParameterWidget(QWidget):
    """Widget for a single parameter with label, input, and help."""
    
    def __init__(self, name, field_info):
        super().__init__()
        self.name = name
        self.field_info = field_info
        self.field_type = field_info.get("type", "string")
        self.value_widget = None
        
        layout = QHBoxLayout(self)
        
        # Mandatory indicator + label
        label_text = name.replace("_", " ").title()
        if field_info.get("mandatory", False):
            label_text += " *"
        
        label = QLabel(label_text)
        label.setMinimumWidth(200)
        layout.addWidget(label)
        
        # Input widget based on type
        self.value_widget = self._create_input_widget()
        layout.addWidget(self.value_widget)
        
        # Help button
        help_btn = QPushButton("?")
        help_btn.setMaximumWidth(30)
        help_btn.setToolTip(field_info.get("description", "No description available"))
        help_btn.clicked.connect(lambda: QMessageBox.information(
            self, f"Help: {label_text}",
            field_info.get("description", "No description available")
        ))
        layout.addWidget(help_btn)
        
        # Default indicator
        default_val = field_info.get("default")
        if default_val is not None:
            default_label = QLabel(f"(default: {str(default_val)[:30]})")
            default_label.setStyleSheet("color: gray; font-size: 9pt;")
            layout.addWidget(default_label)
        
        layout.addStretch()
        
    def _create_input_widget(self):
        """Create appropriate input widget for field type."""
        default = self.field_info.get("default")
        
        if self.field_type == "string":
            widget = QLineEdit()
            if isinstance(default, str):
                widget.setText(default)
            return widget
            
        elif self.field_type == "int":
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            if isinstance(default, int):
                widget.setValue(default)
            return widget
            
        elif self.field_type == "float":
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            widget.setDecimals(6)
            if isinstance(default, (int, float)):
                widget.setValue(float(default))
            return widget
            
        elif self.field_type == "bool":
            widget = QCheckBox()
            if isinstance(default, bool):
                widget.setChecked(default)
            return widget
            
        elif self.field_type == "dict":
            # For dicts, show JSON editor in text field
            widget = QLineEdit()
            widget.setPlaceholderText("JSON dict (e.g., {\"key\": \"value\"})")
            if isinstance(default, dict):
                widget.setText(json.dumps(default))
            return widget
            
        elif self.field_type == "list":
            widget = QLineEdit()
            widget.setPlaceholderText("Comma-separated values")
            if isinstance(default, list):
                widget.setText(",".join(str(x) for x in default))
            return widget
            
        else:
            return QLineEdit()
    
    def get_value(self):
        """Extract value from widget."""
        if self.field_type == "string":
            return self.value_widget.text() or None
        elif self.field_type == "int":
            return self.value_widget.value()
        elif self.field_type == "float":
            return self.value_widget.value()
        elif self.field_type == "bool":
            return self.value_widget.isChecked()
        elif self.field_type == "dict":
            text = self.value_widget.text().strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {}
        elif self.field_type == "list":
            text = self.value_widget.text().strip()
            if not text:
                return []
            return [x.strip() for x in text.split(",")]
        return None
    
    def set_value(self, value):
        """Set value in widget."""
        if value is None:
            return
            
        if self.field_type == "string":
            self.value_widget.setText(str(value))
        elif self.field_type == "int":
            self.value_widget.setValue(int(value))
        elif self.field_type == "float":
            self.value_widget.setValue(float(value))
        elif self.field_type == "bool":
            self.value_widget.setChecked(bool(value))
        elif self.field_type == "dict":
            self.value_widget.setText(json.dumps(value))
        elif self.field_type == "list":
            if isinstance(value, list):
                self.value_widget.setText(",".join(str(x) for x in value))


class ParameterGUI(QMainWindow):
    """Main GUI window for parameter editing."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenSec Parameter Editor")
        self.setGeometry(100, 100, 900, 700)
        
        self.param_widgets = {}  # Store references to all parameter widgets
        self.current_file = None
        
        self._create_ui()
        
    def _create_ui(self):
        """Create the main UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # File operations toolbar
        file_layout = QHBoxLayout()
        
        open_btn = QPushButton("📂 Open")
        open_btn.clicked.connect(self.open_file)
        file_layout.addWidget(open_btn)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_file)
        file_layout.addWidget(save_btn)
        
        save_as_btn = QPushButton("💾 Save As")
        save_as_btn.clicked.connect(self.save_file_as)
        file_layout.addWidget(save_as_btn)
        
        file_layout.addStretch()
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: blue;")
        file_layout.addWidget(self.file_label)
        
        main_layout.addLayout(file_layout)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Create sections by category
        for category in get_all_categories():
            cat_label = get_category_label(category)
            cat_desc = get_category_description(category)
            section = CollapsibleSection(cat_label, cat_desc)
            
            schema_cat = PARAMETER_SCHEMA.get(category, {})
            fields = schema_cat.get("fields", {})
            
            for field_name, field_info in fields.items():
                param_widget = ParameterWidget(field_name, field_info)
                section.add_field(param_widget)
                self.param_widgets[field_name] = param_widget
            
            scroll_layout.addWidget(section)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def open_file(self):
        """Open a parameters.json file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open parameters.json", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            self._populate_from_dict(data)
            self.current_file = file_path
            self.file_label.setText(f"Loaded: {Path(file_path).name}")
            self.statusBar().showMessage(f"Loaded: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
    
    def save_file(self):
        """Save to current file or prompt for new file."""
        if not self.current_file:
            self.save_file_as()
            return
        
        try:
            data = self._collect_form_data()
            with open(self.current_file, "w") as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"Saved: {self.current_file}")
            QMessageBox.information(self, "Success", "Parameters saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")
    
    def save_file_as(self):
        """Save to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save parameters.json", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self.current_file = file_path
        self.save_file()
    
    def _populate_from_dict(self, data):
        """Populate form from a dictionary."""
        for field_name, param_widget in self.param_widgets.items():
            if field_name in data:
                param_widget.set_value(data[field_name])
    
    def _collect_form_data(self):
        """Collect all form values into a dictionary."""
        data = {}
        for field_name, param_widget in self.param_widgets.items():
            value = param_widget.get_value()
            if value is not None:
                data[field_name] = value
        return data


def main():
    app = QApplication(sys.argv)
    gui = ParameterGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
