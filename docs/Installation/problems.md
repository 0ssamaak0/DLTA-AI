---
label: possible problems
order: 1
icon: ":interrobang:"
---

# Solutions to possible problems

## 1. (linux devices 🐧) 
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
## 2. (windows devices 🪟)
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
## 3. Problem in installing mmcv-full
you may often stuck in installing `mmcv-full` with this message
```
Building wheels for collected packages: mmcv-full
  Building wheel for mmcv-full (setup.py) ...
```
you can try installing [pytorch 1.13.1](https://pytorch.org/get-started/previous-versions/#v1131), instead of the lastest version, you can also refer to [this isse](https://github.com/open-mmlab/mmcv/issues/1386)