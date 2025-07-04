# prepare_models.py
from demucs.pretrained import get_model
import torch
import shutil
import os
from pathlib import Path

def prepare_models():
    # 确保模型目录存在
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # 下载htdemucs_ft模型
    print("Downloading htdemucs_ft model...")
    model = get_model('htdemucs_ft')
    
    # 复制模型文件到本地目录
    torch_hub_dir = torch.hub.get_dir()
    src_path = Path(torch_hub_dir) / "facebookresearch_demucs_main"
    for model_file in src_path.glob("htdemucs_ft*"):
        if model_file.is_dir():
            dest = model_dir / model_file.name
            shutil.copytree(model_file, dest, dirs_exist_ok=True)
            print(f"Copied {model_file.name} to models directory")

if __name__ == "__main__":
    prepare_models()
