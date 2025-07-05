import sys
import os
import tempfile
import shutil
from pathlib import Path

# 解决PyInstaller打包后的导入问题
if getattr(sys, 'frozen', False):
    # 添加必要的路径到系统路径
    base_path = Path(sys._MEIPASS)
    sys.path.insert(0, str(base_path))
    
    # 处理模型资源
    resource_dir = base_path / 'demucs_resources'
    if resource_dir.exists():
        temp_dir = Path(tempfile.mkdtemp())
        print(f"[INFO] Created temp model dir: {temp_dir}")
        
        # 复制模型文件
        pretrained_dir = resource_dir / 'pretrained'
        for item in pretrained_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, temp_dir / item.name)
                print(f"[INFO] Copied model file: {item.name}")
        
        # 设置模型仓库参数
        sys.argv += ['--repo', str(temp_dir)]
    
    # 确保numpy正确初始化
    try:
        import numpy as np
        np.finfo(np.dtype("float32"))
        np.finfo(np.dtype("float64"))
    except Exception as e:
        print(f"[WARNING] Numpy initialization error: {e}")
        # 手动设置必要的numpy属性
        class finfo:
            eps = 1.1920929e-07
            min = -3.4028235e+38
            max = 3.4028235e+38
        np.finfo = lambda dtype: finfo()

# 使用绝对导入
try:
    from demucs.api import Separator, save_audio, list_models
    from demucs.apply import BagOfModels
    from demucs.htdemucs import HTDemucs
    from demucs.pretrained import add_model_flags, ModelLoadingError
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("[DEBUG] sys.path:", sys.path)
    raise

# 原始代码保持不变
# Copyright (c) Meta Platforms, Inc. and affiliates.
# ... [原始文件内容] ...
