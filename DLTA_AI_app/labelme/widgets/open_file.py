import os
import subprocess
import platform




def PopUp():
    """
    Open a file with the default application for the file type.

    Args:
        filename (str): The name of the file to open.

    Raises:
        OSError: If the file cannot be opened.

    Returns:
        None
    """
    filename = os.path.join(os.getcwd(), 'labelme/utils/custom_exports.py')
    print(filename)
    # Determine the platform and use the appropriate command to open the file
    # Windows
    if platform.system() == 'Windows':
        os.startfile(filename)
    # macOS
    elif platform.system() == 'Darwin':
        os.system(f'open {filename}')
    else:
        try:
            opener = "open" if platform.system() == "Darwin" else "xdg-open"
            subprocess.call([opener, filename])
        except OSError:
            print(f"Could not open file: {filename}")


