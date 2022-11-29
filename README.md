# Labelme (Automated)
Supporting Auto Annotation using mmdetection frameworks for [Labelme](https://github.com/wkentaro/labelme).


# Installation
## 1. Install [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html)

create a virtual enviroment
```
conda create --name auto_annotate python=3.8 -y

conda activate auto_annotate
```

## 2. Install [Pytorch](https://pytorch.org/)
depending on your OS and CUDA version 

```
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```

## 3. Install [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation)

you can simply use pip

```
	pip install -U openmim

	mim install mmcv-full

	pip install mmdet
```

## 4. Install [labelme](https://github.com/wkentaro/labelme#installation)

```
	conda install -c conda-forge pyside2

	pip install pyqt5

	pip install labelme
```

## 5. Install [OpenCV-python](https://pypi.org/project/opencv-python/)

```
pip install opencv-python
```
# Running
In the tool's directory, run using

```
python __main__.py
```
