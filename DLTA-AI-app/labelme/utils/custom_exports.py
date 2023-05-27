# Don't Modify These Lines
# =========================================
custom_exports_list = []

# custom export class blueprint
class CustomExport:
    """
    A blueprint for defining custom exports.

    Attributes:
        file_name (str): The name of the file to export to.
        button_name (str): The name of the button that triggers the export.
        format (str): The format of the exported file.
        function (callable): The function that generates the export data.
        type (str): The type of the export, either "video" or "image".

    Methods:
        __call__(*args): Calls the function with the given arguments and returns the result.
    """
    def __init__(self, file_name, button_name, format, function, type = "video"):
        """
        Initializes a new instance of the CustomExport class.

        Args:
            file_name (str): The name of the file to export to.
            button_name (str): The name of the button that triggers the export.
            format (str): The format of the exported file.
            function (callable): The function that generates the export data.
            type (str): The type of the export, either "video" or "image".
        """
        self.file_name = file_name
        self.button_name = button_name
        self.format = format
        self.function = function
        self.type = type
    
    def __call__(self, *args):
        """
        Calls the function with the given arguments and returns the result.

        Args:
            *args: The arguments to pass to the function.

        Returns:
            The result of calling the function with the given arguments.
        """
        return self.function(*args)
    
# =========================================


# Add your functions here
# =========================================
# def foo():
#     pass

# =========================================


# append the custom exports to the list as new CustomExport objects
# =========================================

# custom_exports_list.append(CustomExport("file_name", "button_name", "format", foo))

# =========================================
