# Don't Modify These Lines
# =========================================
custom_exports_list = []

# custom export class blueprint
class CustomExport:
    def __init__(self, file_name, button_name, format, function):
        self.file_name = file_name
        self.button_name = button_name
        self.format = format
        self.function = function
    
    def __call__(self, *args):
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
