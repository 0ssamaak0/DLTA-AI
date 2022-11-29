0--- install anaconda or miniconda (https://docs.conda.io/en/latest/miniconda.html)   (https://www.anaconda.com)
     and create a virtual enviroment 

	conda create --name auto_annotate python=3.8 -y

	conda activate auto_annotate

	
---------------------------------------------------------------------------------------------------------

1--- install pytorch (from https://pytorch.org) depending on your OS and cuda version


	conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

---------------------------------------------------------------------------------------------------------


2--- install mmdetection (https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation)


	pip install -U openmim

	mim install mmcv-full

	pip install mmdet


---------------------------------------------------------------------------------------------------------

3--- install labelme (https://github.com/wkentaro/labelme#installation)


	conda install -c conda-forge pyside2

	pip install pyqt5

	pip install labelme

---------------------------------------------------------------------------------------------------------

4--- install cv2 

	pip install opencv-python

---------------------------------------------------------------------------------------------------------


5--- run the tool


	python __main__.py