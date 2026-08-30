import importlib.metadata
import importlib.util
import logging
from typing import Optional, Tuple, Union

import torch
from packaging import version


def copy_parameters(source: torch.nn.Module, dest: torch.nn.Module):
    dest.load_state_dict(source.state_dict())
    dest.requires_grad_(False)


def setup_logging(log_level: str = "WARN", log_file: str = None):
    # set the logger
    log_handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="[%(asctime)s] MoE-PEFT: %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )

import importlib.metadata
from packaging import version
import logging
import importlib


def is_package_available(pkg_name: str, pkg_version: str = None) -> bool:
    """
    检查包是否安装，并可选检查版本。
    优先使用 importlib.metadata.version；
    如果失败，则回退到 pkg.__version__。
    """
    installed_version = None

    # 先尝试 importlib.metadata
    try:
        installed_version = importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return False
    except Exception as e:
        logging.warning(f"metadata.version() failed for {pkg_name}: {e}")

    # 如果 metadata 拿不到版本，再尝试 __version__
    if not installed_version or installed_version in ("None", None):
        try:
            pkg = importlib.import_module(pkg_name)
            installed_version = getattr(pkg, "__version__", None)
        except Exception as e:
            logging.warning(f"import {pkg_name} failed: {e}")
            return False

    if not installed_version:
        return False

    installed_version = str(installed_version)
    logging.debug(f"Detected {pkg_name} version {installed_version}")

    # 如果不要求最低版本，只要安装了就返回 True
    if pkg_version is None:
        return True

    try:
        return version.parse(installed_version) >= version.parse(str(pkg_version))
    except Exception as e:
        logging.warning(f"Version check failed for {pkg_name}: {e}")
        return False


class Unsubscribable:
    def __init__(self) -> None:
        raise RuntimeError(f"Instant unsubscribable class {__class__}")


# Class Placeholder for Bitsandbytes
class Linear8bitLt(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class Linear4bit(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class BitsAndBytesConfig:
    def __init__(self, **kwargs) -> None:
        raise RuntimeError("Quantization not supported.")


class NoneContexts(object):
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
