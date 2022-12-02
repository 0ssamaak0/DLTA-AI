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

## 4. download model checkpoint from [mmdetection](https://github.com/open-mmlab/mmdetection) (in checkpoint directory) you need to run cmd as adminstrator.

```
	wget https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth -O /mmdetection/checkpoints
```


## 5. Install [labelme](https://github.com/wkentaro/labelme#installation)

```
	conda install -c conda-forge pyside2

	pip install pyqt5

	pip install labelme
```

## 6. Install [OpenCV-python](https://pypi.org/project/opencv-python/)

```
pip install opencv-python
```
# 7.Running
In the tool's directory, run using

```
python __main__.py
```
