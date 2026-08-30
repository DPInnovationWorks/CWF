from __future__ import annotations

import argparse
from pathlib import Path

from _shared import create_client_from_env, load_tqdm, should_disable_tqdm, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="上传 batch JSONL 并创建 qwen-plus Batch 任务。")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("batch_example.jsonl"),
        help="待上传的 batch JSONL 文件",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("artifacts/manifests/batch_manifest.jsonl"),
        help="与 custom_id 对应的 manifest 文件路径",
    )
    parser.add_argument(
        "--job-file",
        type=Path,
        default=Path("artifacts/jobs/latest_batch_job.json"),
        help="本地保存 batch 任务元数据的路径",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="Batch 完成窗口，默认 24h",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tqdm = load_tqdm()
    if not args.input_file.exists():
        raise SystemExit(f"输入文件不存在: {args.input_file}")

    client = create_client_from_env()
    progress = tqdm(
        total=3,
        desc="创建 Batch 任务",
        unit="step",
        disable=should_disable_tqdm(),
    )

    file_size = args.input_file.stat().st_size
    progress.set_postfix(file_size=file_size)
    progress.update(1)

    with args.input_file.open("rb") as input_fp:
        uploaded_file = client.files.create(file=input_fp, purpose="batch")
    progress.set_postfix(input_file_id=uploaded_file.id)
    progress.update(1)

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=args.completion_window,
    )
    progress.set_postfix(batch_id=batch.id)
    progress.update(1)
    progress.close()

    payload = {
        "created_at": utc_now_iso(),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "endpoint": "/v1/chat/completions",
        "input_file": str(args.input_file),
        "manifest_file": str(args.manifest_file),
        "job_file": str(args.job_file),
        "input_file_size_bytes": file_size,
        "input_file_id": uploaded_file.id,
        "batch_id": batch.id,
        "batch_status": getattr(batch, "status", None),
        "completion_window": args.completion_window,
    }
    write_json(args.job_file, payload)
    print(f"Batch 任务已创建，batch_id={batch.id}")
    print(f"任务元数据已写入: {args.job_file}")


if __name__ == "__main__":
    main()
