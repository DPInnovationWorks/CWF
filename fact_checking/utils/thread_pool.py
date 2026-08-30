"""
通用线程池工具：对一批任务并行执行，可配置最大工作线程数，可选 tqdm 进度条。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, TypeVar

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

T = TypeVar("T")
R = TypeVar("R")


def run_in_thread_pool(
    func: Callable[[T], R],
    tasks: Sequence[T],
    *,
    max_workers: int,
    desc: Optional[str] = "任务",
    show_progress: bool = True,
) -> List[R]:
    """
    使用线程池对 tasks 中每个元素调用 func(task)，返回与 tasks **同序** 的结果列表。

    :param desc: 进度条描述；``show_progress=False`` 时忽略。
    :param show_progress: 为 True 且已安装 tqdm 时显示进度条。
    """
    if not tasks:
        return []
    task_list = list(tasks)
    n = len(task_list)
    workers = max(1, min(int(max_workers), n))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        it = pool.map(func, task_list)
        if show_progress and tqdm is not None:
            it = tqdm(it, total=n, desc=desc or "任务", unit="条")
        return list(it)
