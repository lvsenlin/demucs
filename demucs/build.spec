# -*- mode: python -*-
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

a = Analysis(
    ['demucs/separate.py'],
    pathex=[],
    binaries=[],
    datas=[
        *collect_data_files('demucs'),
        ('model_files/htdemucs_ft/checkpoint.th', 'demucs/pretrained_models/htdemucs_ft'),
        ('model_files/htdemucs_ft/model.yaml', 'demucs/pretrained_models/htdemucs_ft')
    ],
    hiddenimports=['torch', 'torchaudio', 'demucs'],
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='demucs_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
