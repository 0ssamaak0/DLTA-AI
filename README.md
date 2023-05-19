<div align = "center">
<h1>
    <img src = "assets/icon.png" width = 200 height = 200>
<br>

Labelmm 
</h1>

<h3>
Auto Annotation Tool for Computer Vision Tasks
</h3>

Labelmm is the next generation of annotation tools, harnessing the power of Computer Vision SOTA models from <a href = "https://github.com/open-mmlab/mmdetection/tree/2.x">mmdetection</a> to <a href = "https://github.com/wkentaro/labelme">Labelme</a> in a seamless expirence with modern user interface and intuitive workflow


[![python](https://img.shields.io/static/v1?label=python&message=3.8&color=blue&logo=python)](https://pytorch.org/)
[![pytorch](https://img.shields.io/static/v1?label=pytorch&message=1.13.1&color=violet&logo=pytorch)](https://pytorch.org/)
[![mmdetection](https://img.shields.io/static/v1?label=mmdetection&message=v2&color=blue)](https://github.com/open-mmlab/mmdetection/tree/2.x)
[![GitHub License](https://img.shields.io/github/license/0ssamaak0/labelmm)](https://github.com/0ssamaak0/labelmm/blob/master/LICENSE)
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/0ssamaak0/labelmm?include_prereleases)](https://github.com/0ssamaak0/labelmm/releases)
[![GitHub issues](https://img.shields.io/github/issues/0ssamaak0/labelmm)](https://github.com/0ssamaak0/labelmm/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/0ssamaak0/labelmm)](https://github.com/0ssamaak0/labelmm/commits)
![gif_main](assets/gif_main2.gif)

<!-- make p with larger font size -->
[Installation](#installation-%EF%B8%8F)  🛠️ | [Segment Anything](#Segment-Anything-) 🪄 |[Input Modes](#input-modes-%EF%B8%8F) 🎞️ | [Model Selection](#model-selection-) 🤖 | [Object Tracking](#object-tracking-) 🚗 | [Export](#export-) 📤 | [Other Features](#other-features-) 🌟| [Contributing](#contributing-) 🤝| [Acknowledgements](#acknowledgements-)🙏| [Resources](#resources-) 🌐 | [License](#license-) 📜

</div>


# Installation 🛠️
## 1. Install [Pytorch](https://pytorch.org/)
preferably using anaconda

```
conda create --name labelmm python=3.8 -y
conda activate labelmm

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
## 2. install requirements

```
pip install -r requirements.txt
mim install mmcv-full==1.7.0
```
# 3. Running
Run the tool from `labelmm-master` directory
```
cd labelme-master
python __main__.py
```
### Solutions to possible problems
<details>

<summary>click to expand </summary>

#### 1. (linux only 🐧) 
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
#### 2. (windows only 🪟)
some windows machines may have this problem when installing **mmdet**
```
Building wheel for pycocotools (setup.py) ... error
...
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```
You can try
```
conda install -c conda-forge pycocotools
```
or just use Visual Studio installer to Install `MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)**`

</details>

# Segment Anything 🪄
Labelmm provides a seamless expirence to use lastest Meta models [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) and options to modify the results and export them

![Segment Anything](assets/SAM.gif)

# Input Modes 🎞️

labelmm provides 3 Input modes:

- **Image** : for image annotation
- **Directory** : for annotating images in a directory
- **Video** : for annotating videos

![Input Modes](assets/input_modes.png)


# Model Selection 🤖
For model selection, Labelmm provides the **Model Explorer** to utilize the power of the numerous models in [mmdetection](https://github.com/open-mmlab/mmdetection/tree/2.x) and [ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) the to give the user the ability to compare, download and select his library of models

![Model Explorer](assets/model_explorer.gif)
for Object Tracking, Labelmm offers 5 different tracking models with the ability to select between them


# Object Tracking 🚗
In Object Detection, Labelmm provides a seamless expirence for video navigation, tracking settings and different visualization options with the ability to export the tracking results to a video file

Beside this, Labelmm provides a completely new way to modify the tracking results, including edit and delete propagation across frames and different interpolation methods

![Object Tracking](assets/tracking.gif)

# Export 📤
For Instance Segmentation, Labelmm provides to option to export all input modes to COCO format
while for Object Detection on videos, Labelmm provides the ability to export the tracking results to MOT format


![Export](assets/Export.png)

# Other Features 🌟

- Threshold Selection
- Select Classes (from 80 COCO classes)
- Track assigned objects only
- Merging models (Run both models and merge the results)
- Show Runtime Type (CPU/GPU)
- Show GPU Memory Usage
- Video Navigation (Frame by Frame, Fast Forward, Fast Backward, Play/Pause)
- Light / Dark Theme Support (syncs with OS theme)
- Customizable UI Elements (Hide/Show and Change Position)

# Contributing 🤝
Labelmm is an open source project and contributions are very welcome, specially in this early stage of development.

you can contribute by:
- Create an [issue](https://github.com/0ssamaak0/labelmm/issues) Reporting bugs 🐞 or suggesting new features 🌟 or just give your feedback 📝

- Create a [pull request](https://github.com/0ssamaak0/labelmm/pulls) to fix bugs or add new features, or just to improve the code quality, optimize performance, documentation, or even just to fix typos

- Review [pull requests](https://github.com/0ssamaak0/labelmm/pulls) and help with the code review process

- Spread the word about Labelmm and help us grow the community 🌎, by sharing the project on social media, or just by telling your friends about it

# Acknowledgements 🙏
This tool is part of a Graduation Project at [Faculty of Engineering, Ain Shams University](https://eng.asu.edu.eg/) under the supervision of:

- [Dr. Karim Ismail](karim.ismail@carleton.ca)

- [Dr. Ahmed Osama](ahmed.osama@eng.asu.edu.eg)

- Dr. Watheq El-Kharashy

- [Eng. Yousra El-Qattan](https://www.linkedin.com/in/youssra-elqattan/)


we want also to thank our friends who helped us with testing, feedback and suggestions:

- Eng. Houssam Siyoufi

- [Amin Mohamed](https://github.com/AminMohamed-3)

- [Badr Mohamed](https://github.com/Badr-1)

- [Ahmed Mahmoud](https://github.com/ahmedd-mahmoudd)

- [Youssef Ashraf](https://github.com/0xNine9)



# Resources 🌐
- [Labelme](https://github.com/wkentaro/labelme)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [MMDetection](https://github.com/open-mmlab/mmdetection/tree/2.x)
- [ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [mikelbrostrom yolov8_tracking](https://github.com/mikel-brostrom/yolov8_tracking)
- [icons8](https://icons8.com/)

# License 📜
Labelmm is released under the [GPLv3 license](https://github.com/0ssamaak0/labelmm/blob/master/LICENSE). 
