from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

from _shared import (
    create_client_from_env,
    load_tqdm,
    read_json_file,
    should_disable_tqdm,
    utc_now_iso,
    write_json,
)


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="轮询 Batch 任务状态，并在完成后下载输出文件。")
    parser.add_argument(
        "--job-file",
        type=Path,
        default=Path("artifacts/jobs/latest_batch_job.json"),
        help="create_batch_job.py 生成的任务元数据文件",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("artifacts/batch_outputs"),
        help="输出文件下载目录",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="轮询间隔秒数，默认 60",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=7200,
        help="最长等待时间，默认 7200 秒",
    )
    return parser.parse_args()


def safe_get(value: Any, field: str) -> Any:
    if value is None:
        return None
    return getattr(value, field, None)


def download_file(client: Any, file_id: str | None, output_path: Path) -> str | None:
    if not file_id:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    content.write_to_file(str(output_path))
    return str(output_path)


def serialize_request_counts(request_counts: Any) -> dict[str, Any] | None:
    if request_counts is None:
        return None
    if hasattr(request_counts, "model_dump"):
        return request_counts.model_dump()
    if isinstance(request_counts, dict):
        return request_counts
    output: dict[str, Any] = {}
    for field in ("total", "completed", "failed"):
        value = getattr(request_counts, field, None)
        if value is not None:
            output[field] = value
    return output or None


def main() -> None:
    args = parse_args()
    tqdm = load_tqdm()
    if not args.job_file.exists():
        raise SystemExit(f"任务文件不存在: {args.job_file}")

    job_payload = read_json_file(args.job_file)
    batch_id = job_payload.get("batch_id")
    if not batch_id:
        raise SystemExit(f"任务文件中缺少 batch_id: {args.job_file}")

    client = create_client_from_env()
    elapsed = 0
    batch = None
    progress = tqdm(
        total=args.timeout_seconds,
        desc="轮询 Batch 状态",
        unit="s",
        disable=should_disable_tqdm(),
    )
    while elapsed <= args.timeout_seconds:
        batch = client.batches.retrieve(batch_id)
        status = safe_get(batch, "status")
        progress.set_postfix(status=status)
        if status in TERMINAL_STATUSES:
            break
        sleep_seconds = min(args.poll_interval, max(0, args.timeout_seconds - elapsed))
        if sleep_seconds == 0:
            break
        time.sleep(sleep_seconds)
        elapsed += sleep_seconds
        progress.update(sleep_seconds)
    progress.close()

    if batch is None:
        raise SystemExit(f"未能成功获取 batch 状态，batch_id={batch_id}")

    status = safe_get(batch, "status")
    if status not in TERMINAL_STATUSES:
        raise SystemExit(
            f"轮询超时，batch_id={batch_id}，最后状态={status}。"
            "可稍后重新运行 poll_batch_job.py 继续轮询。"
        )

    output_file_id = safe_get(batch, "output_file_id")
    error_file_id = safe_get(batch, "error_file_id")
    args.download_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.download_dir / f"{batch_id}.output.jsonl"
    error_path = args.download_dir / f"{batch_id}.error.jsonl"

    downloaded_output = None
    downloaded_error = None
    if status == "completed":
        downloaded_output = download_file(client, output_file_id, output_path)
    if error_file_id:
        downloaded_error = download_file(client, error_file_id, error_path)

    latest_output_path = None
    latest_error_path = None
    if downloaded_output:
        latest_output = args.download_dir / "latest.output.jsonl"
        shutil.copyfile(output_path, latest_output)
        latest_output_path = str(latest_output)
    if downloaded_error:
        latest_error = args.download_dir / "latest.error.jsonl"
        shutil.copyfile(error_path, latest_error)
        latest_error_path = str(latest_error)

    job_payload.update(
        {
            "polled_at": utc_now_iso(),
            "batch_status": status,
            "output_file_id": output_file_id,
            "error_file_id": error_file_id,
            "output_file_path": downloaded_output,
            "error_file_path": downloaded_error,
            "latest_output_file_path": latest_output_path,
            "latest_error_file_path": latest_error_path,
            "request_counts": serialize_request_counts(safe_get(batch, "request_counts")),
        }
    )
    write_json(args.job_file, job_payload)

    print(f"Batch 任务结束，状态={status}，batch_id={batch_id}")
    if downloaded_output:
        print(f"已下载输出文件: {downloaded_output}")
    if downloaded_error:
        print(f"已下载错误文件: {downloaded_error}")


if __name__ == "__main__":
    main()
