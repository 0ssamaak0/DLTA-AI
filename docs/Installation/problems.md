---
label: possible problems
order: 1
icon: ":interrobang:"
---

# Solutions to possible problems

## Qt Platform Plugin Error in OpenCV on Linux Machines 🐧
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
## Microsoft Visual C++ Build Tools Error When Installing MMDetection on Windows Machines 🪟
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
## Problem in installing mmcv-full
you may often stuck in installing `mmcv-full` with this message
```
Building wheels for collected packages: mmcv-full
  Building wheel for mmcv-full (setup.py) ...
```
you can try installing [pytorch 1.13.1](https://pytorch.org/get-started/previous-versions/#v1131), instead of the lastest version, you can also refer to [this issue](https://github.com/open-mmlab/mmcv/issues/1386)

## Multiple copies of the OpenMP runtime have been linked into the program
you may encounter this problem
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
```
you can solve this by upgrading numpy
```
pip install numpy==1.23.3
```

Thanks for [mohamedraafat96's issue](https://github.com/0ssamaak0/DLTA-AI/issues/52), you can check this [stackoverflow answer](https://stackoverflow.com/questions/64209238/error-15-initializing-libiomp5md-dll-but-found-libiomp5md-dll-already-initial) for more details