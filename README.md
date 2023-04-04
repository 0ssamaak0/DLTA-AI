<h1 align = "center">
<img src = "imgs/icon.png" width = 200 height = 200>
<br>
Labelmm
</h1>

<h3 align = "center">
Auto Annotation Tool for Computer Vision Tasks
</h3>

<p align = "center">
Labelmm is the integrates the power of SOTA models in Computer vision to <a href = "https://github.com/wkentaro/labelme">Labelme</a> providing a tool for auto annotating images for different computer vision tasks with minimal effort in a user friendly interface and the ability to export the annotations in the common formats


![pic5](https://user-images.githubusercontent.com/85165808/227048466-e354d66c-9072-41a9-bef5-dcd88988500c.png)


# Installation
## 1. Install [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

then create a virtual enviroment
```
conda create --name auto_annotate python=3.8 -y

conda activate auto_annotate
```

## 2. Install [Pytorch](https://pytorch.org/)
depending on your OS & CUDA version 

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

```

## 3. Install [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation)

you can simply use pip

```
pip install -U openmim

mim install mmcv-full

pip install mmdet
```

## 4. download models checkpoints from [mmdetection](https://github.com/open-mmlab/mmdetection) you may need to run cmd as adminstrator.

```
wget https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth  -P mmdetection\checkpoints
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

## 7. Install other packages 

```
pip install pyqtdarktheme
pip install loguru
pip install yolox
pip install cython
pip install cython-bbox
pip install onemetric
```

inside Bytetrack directory, run
```
pip install -e .
```
# Running
run the tool using

```
cd labelme-master
python __main__.py
```
