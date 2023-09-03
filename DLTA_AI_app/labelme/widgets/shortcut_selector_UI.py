import yaml
from PyQt6 import QtWidgets, QtGui, QtCore


def PopUp():
    """
    Displays a dialog box for selecting and editing keyboard shortcuts for the application.

    Parameters:
    None

    Returns:
    None
    """

    # Load the default shortcuts from the config file
    shortcuts = {}
    with open("labelme/config/default_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        shortcuts = config.get("shortcuts", {})

    # Encode the shortcut names for display in the table
    shortcuts_names_encode = {name: name.lower().capitalize().replace("_", " ").replace("Sam", "SAM").replace("sam", "SAM") for name in shortcuts.keys()}

    # Decode the shortcut names back to their original form
    shortcuts_names_decode = {value: key for key, value in shortcuts_names_encode.items()}

    # Change the keys of the shortcuts dictionary to use the encoded names
    shortcuts = {shortcuts_names_encode[key]: value for key, value in shortcuts.items()}

    # Create a table to display the shortcuts
    shortcut_table = QtWidgets.QTableWidget()
    shortcut_table.setColumnCount(2)
    shortcut_table.setHorizontalHeaderLabels(['Function', 'Shortcut'])
    shortcut_table.setRowCount(len(shortcuts))
    shortcut_table.verticalHeader().setVisible(False)

    # Populate the table with the shortcut names and keys
    row = 0
    for name, key in shortcuts.items():
        name_item = QtWidgets.QTableWidgetItem(name)
        shortcut_item = QtWidgets.QTableWidgetItem(key)
        shortcut_table.setItem(row, 0, name_item)
        shortcut_table.setItem(row, 1, shortcut_item)
        row += 1

    # Define a function to handle clicks on the shortcut table
    def on_shortcut_table_clicked(item):
        row = item.row()
        name_item = shortcut_table.item(row, 0)
        name = name_item.text()
        current_key = shortcuts[name]
        key_edit = QtWidgets.QKeySequenceEdit(QtGui.QKeySequence(current_key))
        key_edit.setWindowTitle(f"Edit Shortcut for {name}")
        key_edit_label = QtWidgets.QLabel("Enter new shortcut for " + name)
        dialog = QtWidgets.QDialog()
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        dialog.setWindowTitle("Shortcut Selector")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(key_edit_label)
        layout.addWidget(key_edit)
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        null_hint_label = QtWidgets.QLabel("to remove shortcut, press 'Ctrl' only then click 'OK") 
        layout.addWidget(ok_button)
        layout.addWidget(null_hint_label)
        dialog.setLayout(layout)

        # If the user clicks OK, update the shortcut and table
        if dialog.exec():
            key = key_edit.keySequence().toString(QtGui.QKeySequence.SequenceFormat.NativeText)
            if key in shortcuts.values() and list(shortcuts.keys())[list(shortcuts.values()).index(key)] != name:
                conflicting_shortcut = list(shortcuts.keys())[list(shortcuts.values()).index(key)]
                QtWidgets.QMessageBox.warning(None, "Error", f"{key} is already assigned to {conflicting_shortcut}.")
            else:
                if key == "":
                    key = None
                shortcuts[name] = key
                shortcut_table.item(row, 1).setText(key)

    def write_shortcuts_to_ui(config):
        shortcuts = config.get("shortcuts", {})
        # Encode the shortcut names for display in the table
        shortcuts_names_encode = {name: name.lower().capitalize().replace("_", " ").replace("Sam", "SAM").replace("sam", "SAM") for name in shortcuts.keys()}

        # Change the keys of the shortcuts dictionary to use the encoded names
        shortcuts = {shortcuts_names_encode[key]: value for key, value in shortcuts.items()}
        
        row = 0
        for name, key in shortcuts.items():
            name_item = QtWidgets.QTableWidgetItem(name)
            shortcut_item = QtWidgets.QTableWidgetItem(key)
            shortcut_table.setItem(row, 0, name_item)
            shortcut_table.setItem(row, 1, shortcut_item)
            row += 1

    def on_reset_button_clicked():
        
        with open("labelme/config/default_config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        write_shortcuts_to_ui(config)

    def on_restore_button_clicked():
        
        with open("labelme/config/default_config_base.yaml", "r") as f:
            configBase = yaml.load(f, Loader=yaml.FullLoader)
        with open("labelme/config/default_config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["shortcuts"] = configBase["shortcuts"]
        
        write_shortcuts_to_ui(config)

    # Connect the on_shortcut_table_clicked function to the itemClicked signal of the shortcut table
    shortcut_table.itemClicked.connect(on_shortcut_table_clicked)

    # Create a dialog box to display the shortcut table
    dialog = QtWidgets.QDialog()
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
    dialog.setWindowTitle("Shortcuts")
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(shortcut_table)
    
    ok_button = QtWidgets.QPushButton("OK")
    ok_button.clicked.connect(dialog.accept)
    layout.addWidget(ok_button)
    
    reset_button = QtWidgets.QPushButton("Reset")
    reset_button.clicked.connect(on_reset_button_clicked)
    layout.addWidget(reset_button)
    
    restore_button = QtWidgets.QPushButton("Restore Default Shortcuts")
    restore_button.clicked.connect(on_restore_button_clicked)
    layout.addWidget(restore_button)
    
    note_label = QtWidgets.QLabel("Shortcuts will be updated after restarting the app.")
    layout.addWidget(note_label)
    dialog.setLayout(layout)

    # Set the size of the dialog box
    dialog.setMinimumWidth(shortcut_table.sizeHintForColumn(0) + shortcut_table.sizeHintForColumn(1) + 55)
    dialog.setMinimumHeight(shortcut_table.rowHeight(0) * 10 + 50)

    # Set the size policy to allow vertical resizing
    dialog.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)

    # Display the dialog box
    dialog.exec()
    
    # load shortcuts from shortcut table to be updated
    shortcuts = {}
    for row in range(shortcut_table.rowCount()):
        name_item = shortcut_table.item(row, 0)
        name = name_item.text()
        shortcut_item = shortcut_table.item(row, 1)
        shortcut = shortcut_item.text()
        shortcuts[name] = shortcut if shortcut != "" else None

    # Decode the shortcut names back to their original form
    shortcuts = {shortcuts_names_decode[key]: value for key, value in shortcuts.items()}

    # Write the updated shortcuts to the config file
    with open("labelme/config/default_config.yaml", "w") as f:
        config["shortcuts"] = shortcuts
        yaml.dump(config, f)

