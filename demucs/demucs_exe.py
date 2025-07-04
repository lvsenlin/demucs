# demucs_exe.py
import sys
import os
import argparse
from pathlib import Path
from demucs.separate import main as separate_main
from demucs.separate import get_parser

def main():
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置模型存储路径
    models_dir = os.path.join(base_dir, "models")
    os.environ["DEMUCS_MODEL_DIR"] = models_dir
    
    # 解析命令行参数
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    
    # 处理输出路径的Windows格式
    if hasattr(args, 'out'):
        args.out = Path(args.out).resolve()
        sys.argv[sys.argv.index('--out') + 1] = str(args.out)
    
    # 如果没有指定模型则使用htdemucs_ft
    if not args.name:
        sys.argv.extend(['--name', 'htdemucs_ft'])
    
    separate_main()

if __name__ == "__main__":
    main()
