<div align = "center">
<h1>
    <img src = "https://github.com/0ssamaak0/DLTA-AI/blob/master/assets/icon.png?raw=true" width = 200 height = 200>
<br>

</h1>

<h3>
Data Labeling, Tracking and Annotation with AI
</h3>

DLTA-AI is the next generation of annotation tools, integrating the power of Computer Vision SOTA models to <a href = "https://github.com/wkentaro/labelme">Labelme</a> in a seamless expirence and intuitive workflow to make creating image datasets easier than ever before


[![User Guide](https://img.shields.io/badge/User%20Guide-blue)](https://0ssamaak0.github.io/DLTA-AI/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/DLTA-AI)](https://pypi.org/project/DLTA-AI/)
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/0ssamaak0/DLTA-AI?include_prereleases)](https://github.com/0ssamaak0/DLTA-AI/releases)
[![GitHub issues](https://img.shields.io/github/issues/0ssamaak0/DLTA-AI)](https://github.com/0ssamaak0/DLTA-AI/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/0ssamaak0/DLTA-AI)](https://github.com/0ssamaak0/DLTA-AI/commits)
[![GitHub License](https://img.shields.io/github/license/0ssamaak0/DLTA-AI)](https://github.com/0ssamaak0/DLTA-AI/blob/master/LICENSE)

![gif_main](https://github.com/0ssamaak0/DLTA-AI/blob/master/assets/gif_main.gif?raw=true)

<!-- make p with larger font size -->
[Installation](#installation-%EF%B8%8F)  🛠️ | [Segment Anything](#Segment-Anything-) 🪄 | [Model Selection](#model-selection-) 🤖 | [Segmentation](#segmentation-) 🎨 | [Object Tracking](#object-tracking-) 🚗 | [Export](#export-) 📤 | [Other Features](#other-features-) 🌟| [Contributing](#contributing-) 🤝| [Acknowledgements](#acknowledgements-)🙏| [Resources](#resources-) 🌐 | [License](#license-) 📜

</div>

# Installation 🛠️
After creating a new environment, installing Pytorch to it, you can install DLTA-AI using pip
```
pip install DLTA-AI
```
and run it using
```
DLTA-AI
```
Check the [Installation section in User Guide](https://0ssamaak0.github.io/DLTA-AI/installation/full-installation/) for more details, different installation options and solutions for common issues.
# Segment Anything 🪄
DLTA-AI takes the Annotation to the next level by integrating lastest Meta models [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) to support zero-shot segmentation for any class

**SAM** can be used also to improve the quality of Segmentation, even inaccurate polygons around the object is enough to be segmented correctly

**SAM** doesn't only work for Segmentation tasks, it's build in the video mode to support **Object Tracking** as well for any class

<div align = "center">

![Segment Anything](https://github.com/0ssamaak0/DLTA-AI/blob/master/assets/SAM.gif?raw=true)
</div>

# Model Selection 🤖
For model selection, DLTA-AI provides the **Model Explorer** to utilize the power of the numerous models in [mmdetection](https://github.com/open-mmlab/mmdetection/tree/2.x) and [ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) as well as the models of [SAM](https://github.com/facebookresearch/segment-anything)

the to give the user the ability to compare, download and select from the library of models
<div align = "center">

![Model Explorer](https://github.com/0ssamaak0/DLTA-AI/blob/master/assets/model_explorer.png?raw=true)
</div>

# Segmentation 🎨
Using the models from the **Model Explorer**, DLTA-AI provides a seamless expirence to annotate single image or batch of images, with options to select classes, modify threshold, and full control to edit the segmentation results.

<div align = "center">

![Segmentation](https://github.com/0ssamaak0/DLTA-AI/blob/master/assets/segmentation.png?raw=true)
</div>
and as mentioned before, **SAM** is fully integrated in DLTA-AI to provide zero-shot segmentation for any class, and to improve the quality of segmentation

# Object Tracking 🚗
Built on top of the segmentation and detection models, DLTA-AI provides a complete solution for Object Tracking, with 5 different models for tracking

To impr DLTA-AI have options for video navigation, tracking settings and different visualization options with the ability to export the tracking results to a video file

Beside this, DLTA-AI provides a completely new way to modify the tracking results, including edit and delete propagation across frames

<div align = "center">

![Object Tracking](https://github.com/0ssamaak0/DLTA-AI/blob/master/assets/tracking.gif?raw=true)

</div>

Beside automatic tracking models, DLTA-AI provides different methods of interpolation and filling gaps between frames to fix occlusions and unpredicted behaviors in a semi-automatic way

<div align = "center">

![Interpolation](https://github.com/0ssamaak0/DLTA-AI/blob/master/assets/interpolation.png?raw=true)
</div>

# Export 📤
For Instance Segmentation, DLTA-AI provides to option to export the segmentation to standard COCO format, and the results of tracking to MOT format, and a video file for the tracking results with desired visualization options e.g., show id, bbox, class name, etc.

<div align = "center">

![Export](https://github.com/0ssamaak0/DLTA-AI/blob/master/assets/Export.png?raw=true)

</div>

DLTA-AI provides also the ability to add user-defined or custom export formats that can be used for any purpose, once the user defines his own format, it will be available in the export menu.

# Other Features 🌟

- Threshold Selection (Confidence and IoU)
- Select Classes (from 80 COCO classes) with option to save default classes
- Track assigned objects only
- Merging models (Run both models and merge the results)
- Show Runtime Type (CPU/GPU)
- Show GPU Memory Usage
- Video Navigation (Frame by Frame, Fast Forward, Fast Backward, Play/Pause)
- Light / Dark Theme Support (syncs with OS theme)
- Fully Customizable UI (drag and drop, show/hide)
- OS Notifications (for long running tasks)
- using orjson for faster json serialization
- additional script (external) to evaluate the results of segmentation (COCO)
- additional script (external) to extract frames from a video file for future use
- User shortcuts and preferences settings

# Contributing 🤝
DLTA-AI is an open source project and contributions are very welcome, specially in this early stage of development.

You can contribute in many ways:
- Create an [issue](https://github.com/0ssamaak0/DLTA-AI/issues) Reporting bugs 🐞 or suggesting new features 🌟 or just give your feedback 📝

- Create a [pull request](https://github.com/0ssamaak0/DLTA-AI/pulls) to fix bugs or add new features, or just to improve the code quality, optimize performance, documentation, or even just to fix typos

- Review [pull requests](https://github.com/0ssamaak0/DLTA-AI/pulls) and help with the code review process

- Spread the word about DLTA-AI and help us grow the community 🌎, by sharing the project on social media, or just by telling your friends about it

# Acknowledgements 🙏
This tool is part of a Graduation Project at [Faculty of Engineering, Ain Shams University](https://eng.asu.edu.eg/) under the supervision of:

- [Dr. Karim Ismail](https://carleton.ca/cee/profile/karim-ismail/)
- [Dr. Ahmed Osama](ahmed.osama@eng.asu.edu.eg)
- Dr. Watheq El-Kharashy
- [Eng. Yousra El-Qattan](https://www.linkedin.com/in/youssra-elqattan/)

we want also to thank our friends who helped us with testing, feedback and suggestions:

- [Eng. Houssam Siyoufi](https://www.linkedin.com/in/houssam-siyoufi-163627110/)
- [Amin Mohamed](https://github.com/AminMohamed-3)
- [Badr Mohamed](https://github.com/Badr-1)
- [Ahmed Mahmoud](https://github.com/ahmedd-mahmoudd)
- [Youssef Ashraf](https://github.com/0xNine9)
- [Chadi Ashraf](https://github.com/Chady00)


# Resources 🌐
- [Labelme](https://github.com/wkentaro/labelme)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [MMDetection](https://github.com/open-mmlab/mmdetection/tree/2.x)
- [ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [mikelbrostrom yolov8_tracking](https://github.com/mikel-brostrom/yolov8_tracking)
- [orjson](https://github.com/ijl/orjson)
- [icons8](https://icons8.com/)

# License 📜
DLTA-AI is released under the [GPLv3 license](https://github.com/0ssamaak0/DLTA-AI/blob/master/LICENSE). 
