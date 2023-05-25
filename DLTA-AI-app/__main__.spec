# -*- mode: python -*-
# vim: ft=python

from glob import glob


block_cipher = None

datas_list = [ 
    ('models_menu/*.json', 'models_menu'),
    ('models_menu/*.py', 'models_menu'),
    ('ultralytics/' , 'ultralytics'),
    ('labelme/' , 'labelme'),
    ('mmdetection/' , 'mmdetection'),
    ('trackers/' , 'trackers')
]

hiddenimports_list = [
    'mmcv' ,
    'mmcv._ext',
    'torchvision']

a = Analysis(
    ['__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas_list,
    hiddenimports=hiddenimports_list,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DLTA-AI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon = "C:\Graduation Project\Auto Annotation Tool\DLTA-AI\DLTA-AI-app\labelme\icons\icon.png"
    
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DLTA-AI',
)
