<h1 align = "center">
<img src = "imgs/icon.png" width = 200 height = 200>
<br>
Labelmm
</h1>

<h3 align = "center">
Auto Annotation Tool for Computer Vision Tasks
</h3>

<p align = "center">
Labelmm is the integrates the power of SOTA models in Computer vision to <a href = "https://github.com/wkentaro/labelme">Labelme</a> providing a tool for auto annotating images for different computer vision tasks with minimal effort with the ability to export the annotations in the common formats to train your models


![pic5](https://user-images.githubusercontent.com/85165808/227048466-e354d66c-9072-41a9-bef5-dcd88988500c.png)


# Installation
## 1. Install [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and Install [Pytorch](https://pytorch.org/)

create conda virtual environment and install pytorch
```
conda create --name <env_name> python=3.8 -y

conda activate <env_name>
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
## 2. install requirements

```
pip install openmim mmdet
mim install mmcv-full

pip install -r requirements.txt
```

## 3. download models checkpoints from [mmdetection](https://github.com/open-mmlab/mmdetection)
Example: Yolact installation
```
wget https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth  -P mmdetection\checkpoints
```
Run as adminsitrator if not working

### Common issue (linux only 🐧) 
some linux machines may have this problem 
```
Could not load the Qt platform plugin "xcb" in "/home/<username>/miniconda3/envs/test/lib/python3.8/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl.
```
it can be solved simply be installing opencv-headless
```
pip3 install opencv-python-headless
```
# Running

```
cd labelme-master
python __main__.py
```
