import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


TRAINING_DIR = Path(__file__).resolve().parent
MOE_PEFT_DIR = TRAINING_DIR.parent

DEFAULT_CHILD_LITERATURE_DATA = [
    MOE_PEFT_DIR / "train_data/more_child.json",
]

DEFAULT_ADULT_DATA = [
    MOE_PEFT_DIR / "train_data/adult_popular_science.json",
]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_path_list(raw: str) -> List[Path]:
    return [Path(item).expanduser() for item in raw.split(";") if item.strip()]


def clean_record(instruction: str, output: str):
    return {"instruction": instruction.strip(), "output": output.strip()}


def load_instruction_output_records(
    paths: Iterable[Path], max_samples: int = 0
) -> List[dict]:
    records = []
    for path in paths:
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list: {path}")
        for item in data:
            if not isinstance(item, dict):
                continue
            instruction = item.get("instruction") or item.get("input") or ""
            output = item.get("output") or item.get("answer") or ""
            if not instruction or not output:
                continue
            if str(output).startswith("处理错误:"):
                continue
            records.append(clean_record(str(instruction), str(output)))
            if max_samples and len(records) >= max_samples:
                return records
    return records


def target_modules():
    return {
        "q_proj": True,
        "k_proj": True,
        "v_proj": True,
        "o_proj": True,
        "gate_proj": True,
        "down_proj": True,
        "up_proj": True,
    }


def adapter_template(adapter_name: str, data_path: Path, args):
    adapter = {
        "name": adapter_name,
        "task_name": "casual",
        "optim": "adamw",
        "scheduler_type": args.scheduler_type,
        "warmup_steps": 0,
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "evaluate_batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "r": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules(),
        "group_by_length": args.group_by_length,
        "data": str(data_path),
        "prompt": args.prompt_template,
    }
    if args.adapter_type == "mixlora":
        adapter.update(
            {
                "routing_strategy": "mixlora",
                "num_experts": args.num_experts,
                "top_k": args.top_k,
            }
        )
    return adapter


def build_config(adapter_name: str, data_path: Path, args):
    return {
        "cutoff_len": args.cutoff_len,
        "save_step": args.save_step,
        "train_lora_candidate_num": 1,
        "train_lora_simultaneously_num": 1,
        "train_strategy": "optim",
        "lora": [adapter_template(adapter_name, data_path, args)],
    }


def training_command(config_path: Path, args, device: str):
    cmd = [
        sys.executable,
        str(args.moe_peft_dir / "moe_peft.py"),
        "--base_model",
        args.base_model,
        "--config",
        str(config_path),
        "--dir",
        str(args.adapter_output_dir),
        "--overwrite",
        "--attn_impl",
        args.attn_impl,
    ]
    if device:
        cmd.extend(["--device", device])
    if args.dtype == "bf16":
        cmd.append("--bf16")
    elif args.dtype == "fp16":
        cmd.append("--fp16")
    if args.load_8bit:
        cmd.append("--load_8bit")
    if args.load_4bit:
        cmd.append("--load_4bit")
    if args.tf32:
        cmd.append("--tf32")
    return cmd


def process_env(args, gpu_id: str = None):
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if args.silence_batch_logs:
        env["MOE_PEFT_SILENCE_BATCH_LOGS"] = "1"
    if args.disable_tqdm:
        env["MOE_PEFT_DISABLE_TQDM"] = "1"
    return env


def run_training(name: str, config_path: Path, args, gpu_id: str = None):
    device = "cuda:0" if gpu_id is not None else args.device
    cmd = training_command(config_path, args, device)
    gpu_msg = f" on physical GPU {gpu_id}" if gpu_id is not None else ""
    print(f"Running {name}{gpu_msg}: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=args.moe_peft_dir,
        env=process_env(args, gpu_id),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a child-literature style adapter and an adult style adapter."
    )
    parser.add_argument(
        "--base_model",
        default="/mnt/ssd/weicheng/data_interns/ruibiao/RAGEN/models/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--moe_peft_dir", type=Path, default=MOE_PEFT_DIR)
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=TRAINING_DIR / "child_adult_adapter_training",
    )
    parser.add_argument("--adapter_output_dir", type=Path, default=TRAINING_DIR / "adapters")
    parser.add_argument("--adapter_type", choices=["mixlora", "lora"], default="mixlora")
    parser.add_argument("--child_adapter_name", default="child_literature_mixlora")
    parser.add_argument("--adult_adapter_name", default="adult_mixlora")
    parser.add_argument(
        "--child_literature_data",
        default=";".join(str(path) for path in DEFAULT_CHILD_LITERATURE_DATA),
        help="Semicolon-separated JSON files with child-literature instruction/output records.",
    )
    parser.add_argument(
        "--adult_data",
        default=";".join(str(path) for path in DEFAULT_ADULT_DATA),
        help="Semicolon-separated JSON files with adult instruction/output records.",
    )
    parser.add_argument("--max_child_samples", type=int, default=0)
    parser.add_argument("--max_adult_samples", type=int, default=0)
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--scheduler_type", default="constant")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--prompt_template", default="alpaca")
    parser.add_argument("--group_by_length", action="store_true")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--attn_impl", choices=["eager", "flash_attn"], default="eager")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--child_gpu", default="2")
    parser.add_argument("--adult_gpu", default="3")
    parser.add_argument("--only_child", action="store_true")
    parser.add_argument("--only_adult", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument(
        "--silence_batch_logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide per-batch dispatcher logs so tqdm ETA stays readable.",
    )
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    args = parser.parse_args()
    if args.only_child and args.only_adult:
        raise ValueError("Use at most one of --only_child and --only_adult.")

    args.moe_peft_dir = args.moe_peft_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.adapter_output_dir = args.adapter_output_dir.resolve()

    child_records = []
    adult_records = []
    if not args.only_adult:
        child_records = load_instruction_output_records(
            parse_path_list(args.child_literature_data), args.max_child_samples
        )
        if not child_records:
            raise ValueError("No child-literature records were loaded.")
    if not args.only_child:
        adult_records = load_instruction_output_records(
            parse_path_list(args.adult_data), args.max_adult_samples
        )
        if not adult_records:
            raise ValueError("No adult records were loaded.")

    data_dir = args.work_dir / "data"
    config_dir = args.work_dir / "configs"
    child_data_path = data_dir / "child_literature_train.json"
    adult_data_path = data_dir / "adult_train.json"
    child_config_path = config_dir / "child_literature_adapter.json"
    adult_config_path = config_dir / "adult_adapter.json"

    if child_records:
        write_json(child_data_path, child_records)
        write_json(
            child_config_path,
            build_config(args.child_adapter_name, child_data_path, args),
        )
        print(f"Child-literature samples: {len(child_records)} -> {child_data_path}")
        print(f"Child-literature config: {child_config_path}")
    if adult_records:
        write_json(adult_data_path, adult_records)
        write_json(
            adult_config_path,
            build_config(args.adult_adapter_name, adult_data_path, args),
        )
        print(f"Adult samples: {len(adult_records)} -> {adult_data_path}")
        print(f"Adult config: {adult_config_path}")
    print(f"Adapter output dir: {args.adapter_output_dir}")

    if args.prepare_only:
        return

    jobs = []
    if child_records:
        jobs.append(("child_literature", child_config_path, args.child_gpu))
    if adult_records:
        jobs.append(("adult", adult_config_path, args.adult_gpu))
    if args.sequential:
        for name, config_path, gpu_id in jobs:
            proc = run_training(name, config_path, args, gpu_id)
            return_code = proc.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, name)
    else:
        procs = [
            (name, run_training(name, config_path, args, gpu_id))
            for name, config_path, gpu_id in jobs
        ]
        failed = []
        for name, proc in procs:
            return_code = proc.wait()
            if return_code != 0:
                failed.append((name, return_code))
        if failed:
            raise subprocess.CalledProcessError(
                failed[0][1], f"parallel training failed: {failed}"
            )


if __name__ == "__main__":
    main()
