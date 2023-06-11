---
label: full installation
icon: ":inbox_tray:"
order: 3
---

# Full Installation
## Create a Virtual Environment
It is highly recommended to install DTLA-AI in virtual environment using conda. This will ensure a clean and isolated environment for the installation process. use `python=3.8` to avoid any compatibility issues
```
conda create -n DLTA-AI python=3.8
conda activate DLTA-AI
```


## Install Pytorch

First, you need to install [pytorch](https://pytorch.org/get-started/locally/) according to your device and your OS, if you have GPU, choose CUDA version, otherwise choose CPU version

Example:
```
conda install pytorch torchvision torchaudio .... -c pytorch>
```

## Option 1: Using pip
Installation using pip is more easier since it handles all dependencies
```
pip install DLTA-AI
```
then run it from anywhere using
```
DLTA-AI
```
note that first time running DLTA-AI, it will download a required module, it may take some time

you can also use pip for updating DLTA-AI
```
pip install DLTA-AI -U
```



## Option 2: Manual Installation
Download the lastest release from [here](https://github.com/0ssamaak0/DLTA-AI/releases)

install requirements

```
pip install -r requirements.txt
mim install mmcv-full==1.7.0
```
then 
Run the tool from `DLTA_AI_app` directory
```
cd DLTA_AI_app
python __main__.py
```






