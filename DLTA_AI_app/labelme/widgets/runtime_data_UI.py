from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout
from PyQt6.QtGui import QFont
from PyQt6 import QtCore
import psutil
import torch



def PopUp():
    """

    Description:
    This function displays a dialog box with information about the runtime data of the system, including GPU and RAM stats.

    Parameters:
    This function takes no parameters.

    Returns:
    This function does not return anything.

    Libraries:
    This function requires the following libraries to be installed:
    - PyQt6
    - psutil
    - torch
    """

    # Create a dialog box to display the runtime data
    dialog = QDialog()
    dialog.setWindowTitle("Runtime data")
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(10)

    # Set font styles for the title and normal text
    title_font = QFont()
    title_font.setPointSize(12)
    title_font.setBold(True)

    normal_font = QFont()
    normal_font.setPointSize(10)

    # If CUDA is available, display GPU stats
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_title_label = QLabel("Device Stats")
        gpu_title_label.setFont(title_font)
        layout.addWidget(gpu_title_label)

        gpu_name_label = QLabel(f"GPU Name: {device_name}")
        gpu_name_label.setFont(normal_font) 
        layout.addWidget(gpu_name_label)

        total_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        used_vram = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        gpu_vram_label = QLabel(f"Total GPU VRAM: {total_vram} GB\nUsed: {used_vram} GB")
        gpu_vram_label.setFont(normal_font)
        layout.addWidget(gpu_vram_label)

    # If CUDA is not available, display CPU stats
    else:
        cpu_label = QLabel("DLTA-AI is Using CPU")
        cpu_label.setFont(title_font)
        layout.addWidget(cpu_label)

    # Display RAM stats
    ram_title_label = QLabel("RAM Stats")
    ram_title_label.setFont(title_font)
    layout.addWidget(ram_title_label)

    total_ram = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    used_ram = round(psutil.virtual_memory().used / (1024 ** 3), 2)
    ram_label = QLabel(f"Total RAM: {total_ram} GB\nUsed: {used_ram} GB")
    ram_label.setFont(normal_font)
    layout.addWidget(ram_label)

    # Display the dialog box
    dialog.exec_()