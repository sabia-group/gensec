"""
Advanced PyQt5 GUI for editing GenSec parameters.json files.
Features:
- Hierarchical nested sections with indentation
- Collapsible subsections for nested dicts
- Table editors for complex dicts (mace_args, E0s)
- Full type support with validation
- Load/Save JSON functionality
"""

import sys
import json
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QCheckBox, QToolTip,
    QPushButton, QFileDialog, QScrollArea, QMessageBox,
    QGroupBox, QFormLayout, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor

from tools.parameter_gui.schema import (
    PARAMETER_SCHEMA, get_all_categories, get_category_label,
    get_category_description, get_schema_field, is_nested_field,
    is_mace_args_field
)


class E0sTableEditor(QWidget):
    """Specialized table editor for E0s (atomic_number → energy)."""
    
    def __init__(self, initial_dict=None):
        super().__init__()
        self.data = initial_dict or {}
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Table with Atomic Number and Energy columns
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Atomic Number", "Energy"])
        self.table.setMaximumHeight(250)
        layout.addWidget(self.table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+")
        add_btn.setMaximumWidth(40)
        add_btn.clicked.connect(self.add_row)
        btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("-")
        remove_btn.setMaximumWidth(40)
        remove_btn.clicked.connect(self.remove_row)
        btn_layout.addWidget(remove_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self._populate_table()
        
        # Resize columns
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
    
    def _populate_table(self):
        """Fill table with E0s data."""
        self.table.setRowCount(0)
        for atomic_num, energy in sorted(self.data.items(), key=lambda x: int(x[0])):
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            atomic_item = QTableWidgetItem(str(atomic_num))
            energy_item = QTableWidgetItem(str(energy))
            
            self.table.setItem(row, 0, atomic_item)
            self.table.setItem(row, 1, energy_item)
    
    def add_row(self):
        """Add new element-energy pair."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setItem(row, 1, QTableWidgetItem(""))
    
    def remove_row(self):
        """Remove selected row."""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)
    
    def get_dict(self):
        """Extract E0s dict from table."""
        result = {}
        for row in range(self.table.rowCount()):
            atomic_item = self.table.item(row, 0)
            energy_item = self.table.item(row, 1)
            
            if atomic_item and energy_item:
                atomic_num = atomic_item.text().strip()
                energy_text = energy_item.text().strip()
                
                if atomic_num and energy_text:
                    try:
                        result[atomic_num] = float(energy_text)
                    except ValueError:
                        pass
        return result


class KeyValueTableEditor(QWidget):
    """Table editor for generic key-value dicts (like mace_args)."""
    
    def __init__(self, initial_dict=None):
        super().__init__()
        self.data = initial_dict or {}
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Table for key-value pairs
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.setMaximumHeight(250)
        layout.addWidget(self.table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+")
        add_btn.setMaximumWidth(40)
        add_btn.clicked.connect(self.add_row)
        btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("-")
        remove_btn.setMaximumWidth(40)
        remove_btn.clicked.connect(self.remove_row)
        btn_layout.addWidget(remove_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self._populate_table()
        
        # Resize columns
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
    
    def _populate_table(self):
        """Fill table with data."""
        self.table.setRowCount(0)
        for key, value in self.data.items():
            # Skip E0s as it gets its own editor
            if key == "E0s":
                continue
            
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            key_item = QTableWidgetItem(str(key))
            val_item = QTableWidgetItem(str(value))
            
            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, val_item)
    
    def add_row(self):
        """Add a new key-value row."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setItem(row, 1, QTableWidgetItem(""))
    
    def remove_row(self):
        """Remove selected row."""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)
    
    def get_dict(self):
        """Extract dictionary from table (excluding E0s)."""
        result = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            val_item = self.table.item(row, 1)
            
            if key_item and val_item:
                key = key_item.text().strip()
                val_text = val_item.text().strip()
                
                if key:
                    # Try to convert value to appropriate type
                    try:
                        val = int(val_text)
                    except ValueError:
                        try:
                            val = float(val_text)
                        except ValueError:
                            if val_text.lower() == "true":
                                val = True
                            elif val_text.lower() == "false":
                                val = False
                            elif val_text.startswith('[') or val_text.startswith('{'):
                                try:
                                    val = json.loads(val_text)
                                except:
                                    val = val_text
                            else:
                                val = val_text
                    result[key] = val
        return result


class NestedSectionWidget(QWidget):
    """Collapsible section for nested_dict fields."""
    
    def __init__(self, field_name, field_info, data=None, level=0):
        super().__init__()
        self.field_name = field_name
        self.field_info = field_info
        self.level = level
        self.data = data or field_info.get("default", {})
        self.is_expanded = True
        self.subfield_widgets = {}
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(level * 20, 2, 0, 2)
        
        # Header with toggle button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.toggle_btn = QPushButton("▼")
        self.toggle_btn.setMaximumWidth(25)
        self.toggle_btn.setMaximumHeight(22)
        self.toggle_btn.clicked.connect(self.toggle)
        header_layout.addWidget(self.toggle_btn)
        
        # Title
        title = field_info.get("label", field_name.replace("_", " ").title())
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(9)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        # Description
        desc = field_info.get("description", "")
        if desc:
            desc_label = QLabel(f"· {desc}")
            desc_label.setStyleSheet("color: gray; font-size: 8pt;")
            header_layout.addWidget(desc_label)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Content area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 5, 0, 5)
        
        # Create subfield widgets
        subfields = field_info.get("subfields", {})
        for subfield_name, subfield_info in subfields.items():
            if subfield_info.get("type") == "nested_dict":
                # Recursive nested dict
                subvalue = self.data.get(subfield_name, subfield_info.get("default", {}))
                widget = NestedSectionWidget(subfield_name, subfield_info, subvalue, level + 1)
                self.subfield_widgets[subfield_name] = widget
                self.content_layout.addWidget(widget)
            elif subfield_info.get("type") == "mace_args":
                # mace_args special editor
                lbl = QLabel("MACE Args")
                lbl_font = QFont()
                lbl_font.setBold(True)
                lbl.setFont(lbl_font)
                lbl.setToolTip(subfield_info.get("description", ""))
                self.content_layout.addWidget(lbl)
                hint = QLabel(subfield_info.get("description", ""))
                hint.setStyleSheet("color: gray; font-size: 8pt; font-style: italic;")
                hint.setWordWrap(True)
                self.content_layout.addWidget(hint)
                widget = MaceArgsEditor(subfield_info, self.data.get(subfield_name, {}))
                self.subfield_widgets[subfield_name] = widget
                self.content_layout.addWidget(widget)
            else:
                # Simple field within nested section
                widget = SimpleFieldWidget(subfield_name, subfield_info)
                widget.set_value(self.data.get(subfield_name))
                self.subfield_widgets[subfield_name] = widget
                self.content_layout.addWidget(widget)
        
        self.content_layout.addStretch()
        layout.addWidget(self.content_widget)
    
    def toggle(self):
        """Toggle expansion."""
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        self.toggle_btn.setText("▼" if self.is_expanded else "▶")
    
    def get_value(self):
        """Collect all subfield values."""
        result = {}
        for field_name, widget in self.subfield_widgets.items():
            result[field_name] = widget.get_value()
        return result
    
    def set_value(self, value):
        """Set all subfield values."""
        if not isinstance(value, dict):
            return
        for field_name, widget in self.subfield_widgets.items():
            if field_name in value:
                widget.set_value(value[field_name])


class SimpleFieldWidget(QWidget):
    """Widget for a single non-nested field."""
    
    def __init__(self, name, field_info):
        super().__init__()
        self.name = name
        self.field_info = field_info
        self.field_type = field_info.get("type", "string")
        self.value_widget = None
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        
        # Label
        label_text = name.replace("_", " ").title()
        
        label = QLabel(label_text)
        label.setMinimumWidth(150)
        layout.addWidget(label)
        
        # Input widget
        self.value_widget = self._create_input_widget()
        layout.addWidget(self.value_widget, 1)
        
        # Help button
        help_btn = QPushButton("?")
        help_btn.setMaximumWidth(25)
        help_btn.setMaximumHeight(22)
        help_btn.setToolTip(field_info.get("description", ""))
        help_btn.clicked.connect(lambda: self._show_help_overlay(help_btn, label_text))
        layout.addWidget(help_btn)

        # Inline default hint (requested behavior)
        default_val = field_info.get("default", None)
        if default_val is not None:
            default_text = str(default_val)
            if len(default_text) > 40:
                default_text = default_text[:37] + "..."
            default_lbl = QLabel(f"(default: {default_text})")
            default_lbl.setStyleSheet("color: gray; font-size: 8pt; font-style: italic;")
            layout.addWidget(default_lbl)
        
        layout.addStretch()
    
    def _create_input_widget(self):
        """Create appropriate input widget."""
        default = self.field_info.get("default")
        
        if self.field_type == "string":
            widget = QLineEdit()
            if isinstance(default, str):
                widget.setText(default)
            return widget
        
        elif self.field_type == "int":
            widget = QLineEdit()
            widget.setPlaceholderText("Integer")
            if isinstance(default, int):
                widget.setText(str(default))
            return widget
        
        elif self.field_type == "float":
            widget = QLineEdit()
            widget.setPlaceholderText("Float")
            if isinstance(default, (int, float)):
                widget.setText(str(float(default)))
            return widget
        
        elif self.field_type == "bool":
            widget = QCheckBox()
            if isinstance(default, bool):
                widget.setChecked(default)
            return widget
        
        elif self.field_type == "list":
            widget = QLineEdit()
            widget.setPlaceholderText("Comma-separated or JSON list")
            if isinstance(default, list):
                widget.setText(json.dumps(default))
            return widget
        
        else:
            return QLineEdit()
    
    def get_value(self):
        """Extract value."""
        if self.field_type == "string":
            return self.value_widget.text() or None
        elif self.field_type == "int":
            text = self.value_widget.text().strip()
            if not text:
                return None
            try:
                return int(text)
            except ValueError:
                return None
        elif self.field_type == "float":
            text = self.value_widget.text().strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        elif self.field_type == "bool":
            return self.value_widget.isChecked()
        elif self.field_type == "list":
            text = self.value_widget.text().strip()
            if not text:
                return []
            try:
                return json.loads(text)
            except:
                return [x.strip() for x in text.split(",")]
        return None
    
    def set_value(self, value):
        """Set value."""
        if value is None:
            return
        
        if self.field_type == "string":
            self.value_widget.setText(str(value))
        elif self.field_type == "int":
            self.value_widget.setText(str(int(value)))
        elif self.field_type == "float":
            self.value_widget.setText(str(float(value)))
        elif self.field_type == "bool":
            self.value_widget.setChecked(bool(value))
        elif self.field_type == "list":
            if isinstance(value, list):
                self.value_widget.setText(json.dumps(value))

    def _show_help_overlay(self, anchor_widget, label_text):
        """Show a floating tooltip-style help overlay near the ? button."""
        desc = self.field_info.get("description", "No help available")
        pos = anchor_widget.mapToGlobal(anchor_widget.rect().bottomLeft())
        QToolTip.showText(pos, f"{label_text}\n\n{desc}", anchor_widget)


class MaceArgsEditor(QWidget):
    """
    Editor for mace_args: known fields as individual form rows (same style as the rest),
    followed by E0s table and an 'Additional Args' table for free-form key-value pairs.
    """

    def __init__(self, field_info, initial_dict=None):
        super().__init__()
        self.field_info = field_info
        self.known_fields = field_info.get("known_fields", {})
        self.data = initial_dict or {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(2)

        self.known_widgets = {}  # field_name → SimpleFieldWidget

        # ---- Known fields as individual form rows ----
        for fname, finfo in self.known_fields.items():
            w = SimpleFieldWidget(fname, finfo)
            # Populate from initial data if present, else leave default
            if fname in self.data:
                w.set_value(self.data[fname])
            self.known_widgets[fname] = w
            layout.addWidget(w)

        # ---- E0s section ----
        e0s_header = QLabel("Reference Energies (E0s)")
        e0s_font = QFont()
        e0s_font.setBold(True)
        e0s_header.setFont(e0s_font)
        e0s_header.setToolTip("Per-element reference energies used by MACE. Key = atomic number (string), Value = energy in eV.")
        layout.addWidget(e0s_header)

        e0s_hint = QLabel("Key: atomic number (e.g. 6 for C)  ·  Value: energy in eV")
        e0s_hint.setStyleSheet("color: gray; font-size: 8pt; font-style: italic;")
        layout.addWidget(e0s_hint)

        self.e0s_editor = E0sTableEditor(self.data.get("E0s", {}))
        layout.addWidget(self.e0s_editor)

        # ---- Additional MACE args (free-form key-value) ----
        extra_header = QLabel("Additional MACE Args")
        extra_font = QFont()
        extra_font.setBold(True)
        extra_header.setFont(extra_font)
        extra_header.setToolTip("Any extra mace_run_train CLI argument not listed above. Use (+) to add.")
        layout.addWidget(extra_header)

        extra_hint = QLabel("Any other mace_run_train argument (see mace_run_train --help)")
        extra_hint.setStyleSheet("color: gray; font-size: 8pt; font-style: italic;")
        layout.addWidget(extra_hint)

        # Extra args = keys in data that are not known_fields and not E0s
        extra_data = {k: v for k, v in self.data.items()
                      if k not in self.known_fields and k != "E0s"}
        self.extra_editor = KeyValueTableEditor(extra_data)
        layout.addWidget(self.extra_editor)

    def get_dict(self):
        """Collect known fields + E0s + extra args into one dict."""
        result = {}
        # Known fields (skip None / empty string so file stays clean)
        for fname, widget in self.known_widgets.items():
            val = widget.get_value()
            if val is not None and val != "":
                result[fname] = val
        # E0s
        e0s = self.e0s_editor.get_dict()
        if e0s:
            result["E0s"] = e0s
        # Extra
        result.update(self.extra_editor.get_dict())
        return result
    def get_value(self):
        """Alias so NestedSectionWidget can call .get_value() uniformly."""
        return self.get_dict()
    def set_value(self, value):
        """Populate from a dict (e.g. when loading a JSON file)."""
        if not isinstance(value, dict):
            return
        for fname, widget in self.known_widgets.items():
            if fname in value:
                widget.set_value(value[fname])
        self.e0s_editor.data = value.get("E0s", {})
        self.e0s_editor._populate_table()
        extra_data = {k: v for k, v in value.items()
                      if k not in self.known_fields and k != "E0s"}
        self.extra_editor.data = extra_data
        self.extra_editor._populate_table()


class CategorySectionWidget(QWidget):
    """Collapsible top-level category section (collapsed by default)."""

    def __init__(self, title, description="", collapsed=True):
        super().__init__()
        self.title = title
        self.description = description
        self.is_expanded = not collapsed
        self.desc_label = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 4)

        # Header as two lines:
        # 1) toggle + title
        # 2) description (only visible when expanded)
        header_box = QVBoxLayout()
        header_box.setContentsMargins(0, 0, 0, 0)
        header_box.setSpacing(2)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        self.toggle_btn = QPushButton("▼" if self.is_expanded else "▶")
        self.toggle_btn.setMaximumWidth(25)
        self.toggle_btn.setMaximumHeight(24)
        self.toggle_btn.clicked.connect(self.toggle)
        header_row.addWidget(self.toggle_btn)

        title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        header_row.addWidget(title_label)
        header_row.addStretch()
        header_box.addLayout(header_row)

        if description:
            self.desc_label = QLabel(description)
            self.desc_label.setStyleSheet("color: gray; font-size: 9pt; font-style: italic;")
            self.desc_label.setWordWrap(True)
            self.desc_label.setVisible(self.is_expanded)
            self.desc_label.setContentsMargins(30, 0, 0, 0)
            header_box.addWidget(self.desc_label)

        layout.addLayout(header_box)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(12, 4, 0, 4)
        self.content_widget.setVisible(self.is_expanded)
        layout.addWidget(self.content_widget)

    def add_field(self, widget):
        self.content_layout.addWidget(widget)

    def toggle(self):
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        self.toggle_btn.setText("▼" if self.is_expanded else "▶")
        if self.desc_label is not None:
            self.desc_label.setVisible(self.is_expanded)


class ParameterGUI(QMainWindow):
    """Main GUI window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenSec Parameter Editor - Complete")
        self.setGeometry(50, 50, 1000, 800)
        
        self.param_widgets = {}
        self.param_source_keys = {}
        self.current_file = None
        
        self._create_ui()
    
    def _create_ui(self):
        """Create main UI."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        
        # File toolbar
        file_layout = QHBoxLayout()
        
        open_btn = QPushButton("📂 Open")
        open_btn.clicked.connect(self.open_file_browser)
        file_layout.addWidget(open_btn)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_file)
        file_layout.addWidget(save_btn)
        
        save_as_btn = QPushButton("💾 Save As")
        save_as_btn.clicked.connect(self.save_file_as)
        file_layout.addWidget(save_as_btn)
        
        file_layout.addStretch()
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: blue; font-weight: bold;")
        file_layout.addWidget(self.file_label)
        
        main_layout.addLayout(file_layout)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        
        # Build form from schema
        self._build_form()
        
        self.scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def _build_form(self):
        """Build form from schema."""
        for category in get_all_categories():
            cat_label = get_category_label(category)
            cat_desc = get_category_description(category)

            # Top-level collapsible category (collapsed by default)
            cat_section = CategorySectionWidget(cat_label, cat_desc, collapsed=True)
            
            # Fields in category
            schema_cat = PARAMETER_SCHEMA.get(category, {})
            fields = schema_cat.get("fields", {})
            generation_hint_added = False
            
            for field_name, field_info in fields.items():
                # Small descriptive block for generation-attempt controls in Core
                if (
                    category == "core"
                    and not generation_hint_added
                    and field_name == "trials"
                ):
                    hint = QLabel("Generation attempts: controls how many placement trials run and how many successful structures are targeted.")
                    hint.setStyleSheet("color: gray; font-size: 8pt; font-style: italic;")
                    hint.setWordWrap(True)
                    cat_section.add_field(hint)
                    generation_hint_added = True

                source_key = field_info.get("source_key", field_name)
                if field_info.get("type") == "nested_dict":
                    widget = NestedSectionWidget(field_name, field_info)
                    cat_section.add_field(widget)
                    self.param_widgets[field_name] = widget
                    self.param_source_keys[field_name] = source_key

                else:
                    widget = SimpleFieldWidget(field_name, field_info)
                    cat_section.add_field(widget)
                    self.param_widgets[field_name] = widget
                    self.param_source_keys[field_name] = source_key
            
            self.scroll_layout.addWidget(cat_section)
    
    def open_file_browser(self):
        """Open file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open parameters.json", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path):
        """Load parameters file."""
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
        """Save to current file."""
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
        """Save to new file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save parameters.json", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.current_file = file_path
            self.save_file()
    
    def _populate_from_dict(self, data):
        """Populate form from data."""
        for field_name, widget in self.param_widgets.items():
            source_key = self.param_source_keys.get(field_name, field_name)
            if source_key in data:
                widget.set_value(data[source_key])
    
    def _collect_form_data(self):
        """Collect all field values."""
        result = {}
        for field_name, widget in self.param_widgets.items():
            source_key = self.param_source_keys.get(field_name, field_name)
            value = widget.get_value()
            if source_key in result and isinstance(result[source_key], dict) and isinstance(value, dict):
                result[source_key].update(value)
            else:
                result[source_key] = value
        return result


def main():
    app = QApplication(sys.argv)
    gui = ParameterGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
