# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Loading pretrained models.
"""

import logging
from pathlib import Path
import typing as tp

from dora.log import fatal, bold

from hdemucs import HDemucs
from repo import RemoteRepo, LocalRepo, ModelOnlyRepo, BagOnlyRepo, AnyModelRepo, ModelLoadingError  # noqa
from states import _check_diffq


import sys

def resolve_default_repo():
    import sys
    from pathlib import Path
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "models"
    else:
        return Path(__file__).parent / "models"



logger = logging.getLogger(__name__)
ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/"
REMOTE_ROOT = Path(__file__).parent / 'remote'

SOURCES = ["drums", "bass", "other", "vocals"]
DEFAULT_MODEL = 'htdemucs'


def demucs_unittest():
    model = HDemucs(channels=4, sources=SOURCES)
    return model


def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-s", "--sig", help="Locally trained XP signature.")
    group.add_argument("-n", "--name", default="htdemucs",
                       help="Pretrained model name or signature. Default is htdemucs.")
    parser.add_argument("--repo", type=Path,
                        help="Folder containing all pre-trained models for use with -n.")


def _parse_remote_files(remote_file_list) -> tp.Dict[str, str]:
    root: str = ''
    models: tp.Dict[str, str] = {}
    for line in remote_file_list.read_text().split('\n'):
        line = line.strip()
        if line.startswith('#'):
            continue
        elif len(line) == 0:
            continue
        elif line.startswith('root:'):
            root = line.split(':', 1)[1].strip()
        else:
            sig = line.split('-', 1)[0]
            assert sig not in models
            models[sig] = ROOT_URL + root + line
    return models


def get_model(name: str, repo: tp.Optional[Path] = None):
    if name == 'demucs_unittest':
        return demucs_unittest()

    if repo is None:
        repo = resolve_default_repo()

    if not repo.is_dir():
        fatal(f"{repo} must exist and be a directory.")

    # ✅ 模拟 remote repo，通过目录结构硬编码签名映射
    folder = repo / name
    if folder.is_dir():
        model_map: tp.Dict[str, str] = {}
        for file in sorted(folder.glob("*.th")):
            sig = name  # 使用 folder 名作为统一签名
            rel_path = str(file.relative_to(repo)).replace("\\", "/")
            model_map[sig] = rel_path  # 只需提供一个代表文件即可
            break  # 不用管 BagOnlyRepo 的多个文件逻辑，它会接着找同签名前缀
        if not model_map:
            fatal(f"No .th files found under folder: {folder}")
        model_repo = RemoteRepo(model_map, root=repo)
        bag_repo = BagOnlyRepo(repo, model_repo)
        any_repo = AnyModelRepo(model_repo, bag_repo)
        model = any_repo.get_model(name)
        model.eval()
        return model

    # ✅ 退回旧逻辑：repo 下存在 *.th 文件，且文件名以签名为前缀
    model_repo = LocalRepo(repo)
    bag_repo = BagOnlyRepo(repo, model_repo)
    any_repo = AnyModelRepo(model_repo, bag_repo)

    try:
        model = any_repo.get_model(name)
        model.eval()
        return model
    except ImportError as exc:
        if 'diffq' in str(exc):
            _check_diffq()
        raise


def get_model_from_args(args):
    """
    Load local model package or pre-trained model from `models/` folder next to the executable or source.
    """
    if args.name is None:
        args.name = DEFAULT_MODEL
        print(bold("Important: the default model was recently changed to `htdemucs`"),
              "the latest Hybrid Transformer Demucs model. In some cases, this model can "
              "actually perform worse than previous models. To get back the old default model "
              "use `-n mdx_extra_q`.")

    # 自动设置 repo 路径
    if args.repo is None:
        args.repo = resolve_default_repo()

    return get_model(name=args.name, repo=args.repo)

