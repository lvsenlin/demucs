# Hook for demucs module
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('demucs')
datas = collect_data_files('demucs')
