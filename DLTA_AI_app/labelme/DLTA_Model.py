DLTA_Model_list = []

class DLTA_Model():
    def __init__(self, model_name, model_family, config, checkpoint, task, classes, inference_function):
        """
        Initializes a new instance of DLTA_Model class.

        Args:
            model_name (str): The name of the model.
            model_family (str): The family of the model. (e.g., YOLOv8, MMdetection, etc.)
            config (str): The path to the model's configuration file.
            checkpoint (str): The path to the model's checkpoint file.
            task (str): The task of the model. (e.g., object detection, instance segmentation, etc.)
            classes (list): The list of classes that the model can detect. (e.g., coco, ImageNet, custom, etc.)

        Methods:
            __call__(*args): Calls the function with the given arguments and returns the result.
        """
        self.model_name = model_name
        self.model_family = model_family
        self.config = config
        self.checkpoint = checkpoint
        self.task = task
        self.classes = classes
        self.inference_function = inference_function
        self.model = None

        print(f"{self.model_name} successfully initialized")

        DLTA_Model_list.append(self)

    def __call__(self, *args):
        """
        Calls the function with the given arguments and returns the result.

        Args:
            *args: The arguments to pass to the function.

        Returns:
            The result of calling the function with the given arguments.
        """
        return self.inference_function(*args)
    def __str__(self):
        """
        Returns a string representation of the model.
        """
        return (f"Model Name: {self.model_name} | Model Family: {self.model_family} | Task: {self.task} | Classes: {self.classes}")
    
    def initialize(self):
        """
        Initializes the model.

        Returns:
            model: The initialized model.
        """
        pass  
    def imports(self, type = "import"):
        """
        Imports the model's dependencies.

        Args:
            type (str): import or install, if import, then import the model's dependencies, if install, then install the model's dependencies
        
        Returns:
            Imported model's dependencies.
        """
    def install(self):
        """
        Installs the model and its dependencies if they are not already installed.

        Returns:
            None.
        """
        try:
            return self.imports()
        except ImportError:
            print(f"{self.model_name} is not installed, installing...")
            self.imports("install")
            print(f"{self.model_name} successfully installed")
            return self.imports()
        except Exception as e:
            print(f"Error installing {self.model_name}: {e}")
            return

    def parse(self):
        """
        Parses the model's configuration file.

        Returns:
            None.
        """
        pass

    def inference(self):
        """
        Performs inference on the given image.

        Args:
            image_path (str): The path to the image.

        Returns:
            None.
        """
        pass

    def postprocess(self):
        """
        Postprocesses the model's inference results.

        Returns:
            None.
        """
        pass