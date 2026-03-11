# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect MediaPipe data files
mediapipe_datas = collect_data_files('mediapipe', include_py_files=True)

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('hand_landmarker.task', '.'),
        ('firebase_service.py', '.'),
        ('firebase_config.json', '.'),
    ] + mediapipe_datas,
    hiddenimports=[
        'firebase_service',
        'mediapipe',
        'mediapipe.tasks',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.core',
        'mediapipe.tasks.python.vision',
        'mediapipe.tasks.c',
        'cv2',
        'PIL',
        'customtkinter',
        'qrcode',
        'pyperclip',
        'pyrebase',
        'requests',
        'google.oauth2',
        'google.auth',
    ] + collect_submodules('mediapipe'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HandCopyPaste',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
