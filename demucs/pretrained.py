import logging
from pathlib import Path
import typing as tp
import sys

from dora.log import fatal

from hdemucs import HDemucs
from repo import RemoteRepo, LocalRepo, ModelOnlyRepo, BagOnlyRepo, AnyModelRepo, ModelLoadingError  # noqa

logger = logging.getLogger(__name__)
ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/mdx_final/"
REMOTE_ROOT = Path(__file__).parent / 'remote'

SOURCES = ["drums", "bass", "other", "vocals"]


def demucs_unittest():
    model = HDemucs(channels=4, sources=SOURCES)
    return model


def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-s", "--sig", help="Locally trained XP signature.")
    group.add_argument("-n", "--name", default="mdx_extra_q",
                       help="Pretrained model name or signature. Default is mdx_extra_q.")
    parser.add_argument("--repo", type=Path,
                        help="Folder containing all pre-trained models for use with -n.")


def resolve_default_repo():
    """自动解析打包环境下的 models 文件夹"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 打包后的临时目录
        base_path = Path(sys._MEIPASS)
        return base_path / "models"
    else:
        # 本地运行时，使用源码目录下 models（可根据需求改为其他默认位置）
        return Path(__file__).parent.parent / "models"


def get_model(name: str,
              repo: tp.Optional[Path] = None):
    """
    `name` 可为模型签名、模型文件名或本地训练名称。
    若 repo 为 None，自动适配打包路径或默认路径。
    """
    if name == 'demucs_unittest':
        return demucs_unittest()
    model_repo: ModelOnlyRepo

    # 自动补充 repo
    if repo is None:
        repo = resolve_default_repo()

    if not repo.exists() or not repo.is_dir():
        logger.warning(f"默认模型目录不存在或无效：{repo}，尝试远程加载。")
        remote_files = [line.strip()
                        for line in (REMOTE_ROOT / 'files.txt').read_text().split('\n')
                        if line.strip()]
        model_repo = RemoteRepo(ROOT_URL, remote_files)
        bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
    else:
        model_repo = LocalRepo(repo)
        bag_repo = BagOnlyRepo(repo, model_repo)

    any_repo = AnyModelRepo(model_repo, bag_repo)
    return any_repo.get_model(name)


def get_model_from_args(args):
    """
    Load local model package or pre-trained model.
    自动适配打包路径中 models 文件夹。
    """
    return get_model(name=args.name, repo=args.repo)
