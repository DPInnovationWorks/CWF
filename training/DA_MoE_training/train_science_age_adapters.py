import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


TRAINING_DIR = Path(__file__).resolve().parent
MOE_PEFT_DIR = TRAINING_DIR.parent

DEFAULT_SCIENCE_DATA = [
    MOE_PEFT_DIR / "train_data/别的专业的合集/biology_lima.json",
    MOE_PEFT_DIR / "train_data/别的专业的合集/medicine.json",
    MOE_PEFT_DIR / "train_data/别的专业的合集/chemistry(1).json",
]

DEFAULT_AGE_DATA = {
    "child": MOE_PEFT_DIR / "train_data/processed/child.json",
    "teens": MOE_PEFT_DIR / "train_data/processed/teens.json",
    "adult": MOE_PEFT_DIR / "train_data/processed/adult.json",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_path_list(raw: str) -> List[Path]:
    if not raw:
        return []
    return [Path(item).expanduser() for item in raw.split(";") if item.strip()]


def clean_record(instruction: str, output: str) -> Dict[str, str]:
    return {"instruction": instruction.strip(), "output": output.strip()}


def load_science_records(paths: Iterable[Path], max_samples: int = 0) -> List[Dict[str, str]]:
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
            if instruction and output:
                records.append(clean_record(instruction, output))
                if max_samples and len(records) >= max_samples:
                    return records
    return records


def age_name_to_zh(age_name: str) -> str:
    return {
        "child": "儿童",
        "children": "儿童",
        "teens": "青少年",
        "teen": "青少年",
        "adult": "成年人",
        "adults": "成年人",
    }.get(age_name, age_name)


def age_prompt(audience: str, source: str) -> str:
    return (
        f"请将下面的科普内容改写为适合{audience}阅读的表达，"
        f"保持科学含义准确：\n{source}"
    )


def get_source_target(item: Dict[str, str], reverse: bool):
    if "original" in item and "rewritten" in item:
        source = item["rewritten"] if reverse else item["original"]
        target = item["original"] if reverse else item["rewritten"]
    else:
        instruction = item.get("instruction") or item.get("input") or ""
        output = item.get("output") or item.get("answer") or ""
        source = output if reverse else instruction
        target = instruction if reverse else output
    return str(source), str(target)


def load_age_records(
    age_paths: Dict[str, Path], max_samples: int = 0, reverse: bool = True
) -> List[Dict[str, str]]:
    records = []
    for age_name, path in age_paths.items():
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list: {path}")
        audience = age_name_to_zh(age_name)
        for item in data:
            if not isinstance(item, dict):
                continue
            source, target = get_source_target(item, reverse)
            instruction = age_prompt(audience, source)
            output = target
            if instruction and output:
                records.append(clean_record(instruction, output))
                if max_samples and len(records) >= max_samples:
                    return records
    return records


def adapter_template(adapter_type: str, adapter_name: str, data_path: Path, args):
    if adapter_type == "mixlora":
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
            "routing_strategy": "mixlora",
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "group_by_length": args.group_by_length,
            "data": str(data_path),
            "prompt": args.prompt_template,
        }
    elif adapter_type == "lora":
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
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
    return adapter


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


def build_config(adapter_name: str, data_path: Path, args):
    return {
        "cutoff_len": args.cutoff_len,
        "save_step": args.save_step,
        "train_lora_candidate_num": 1,
        "train_lora_simultaneously_num": 1,
        "train_strategy": "optim",
        "lora": [
            adapter_template(args.adapter_type, adapter_name, data_path, args),
        ],
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


def run_training(config_path: Path, args, name: str, gpu_id: str = None):
    device = "cuda:0" if gpu_id is not None else args.device
    cmd = training_command(config_path, args, device)
    gpu_msg = f" on physical GPU {gpu_id}" if gpu_id is not None else ""
    print(f"Running {name}{gpu_msg}: {' '.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=args.moe_peft_dir,
        env=process_env(args, gpu_id),
        check=True,
    )


def run_training_parallel(science_config_path: Path, age_config_path: Path, args):
    jobs = [
        ("science", science_config_path, args.science_gpu),
        ("age", age_config_path, args.age_gpu),
    ]
    procs = []
    for name, config_path, gpu_id in jobs:
        device = "cuda:0" if gpu_id is not None else args.device
        cmd = training_command(config_path, args, device)
        print(f"Running {name} on physical GPU {gpu_id}: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=args.moe_peft_dir,
            env=process_env(args, gpu_id),
        )
        procs.append((name, proc))

    failed = []
    for name, proc in procs:
        return_code = proc.wait()
        if return_code != 0:
            failed.append((name, return_code))

    if failed:
        raise subprocess.CalledProcessError(
            failed[0][1],
            f"parallel training failed: {failed}",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train one science-knowledge adapter and one age-style adapter."
    )
    parser.add_argument(
        "--base_model",
        default="/mnt/ssd/weicheng/data_interns/ruibiao/RAGEN/models/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--moe_peft_dir", type=Path, default=MOE_PEFT_DIR)
    parser.add_argument("--work_dir", type=Path, default=TRAINING_DIR / "adapter_training")
    parser.add_argument("--adapter_output_dir", type=Path, default=TRAINING_DIR / "adapters")
    parser.add_argument("--adapter_type", choices=["mixlora", "lora"], default="mixlora")
    parser.add_argument("--science_adapter_name", default="science_mixlora")
    parser.add_argument("--age_adapter_name", default="age_mixlora")
    parser.add_argument(
        "--science_data",
        default=";".join(str(path) for path in DEFAULT_SCIENCE_DATA),
        help="Semicolon-separated JSON files with instruction/output records.",
    )
    parser.add_argument("--age_child_data", type=Path, default=DEFAULT_AGE_DATA["child"])
    parser.add_argument("--age_teens_data", type=Path, default=DEFAULT_AGE_DATA["teens"])
    parser.add_argument("--age_adult_data", type=Path, default=DEFAULT_AGE_DATA["adult"])
    parser.add_argument(
        "--age_reverse_pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Train age adapter as neutral/formal text -> age-style text. "
            "Use --no-age_reverse_pairs to keep JSON instruction/original -> output/rewritten."
        ),
    )
    parser.add_argument("--max_science_samples", type=int, default=0)
    parser.add_argument("--max_age_samples", type=int, default=0)
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
    parser.add_argument("--science_gpu", default="2")
    parser.add_argument("--age_gpu", default="3")
    parser.add_argument("--only_science", action="store_true")
    parser.add_argument("--only_age", action="store_true")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Train science and age adapters one after another instead of using two GPUs.",
    )
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
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only write merged data/config files, do not launch training.",
    )
    args = parser.parse_args()
    if args.only_science and args.only_age:
        raise ValueError("Use at most one of --only_science and --only_age.")

    args.moe_peft_dir = args.moe_peft_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.adapter_output_dir = args.adapter_output_dir.resolve()

    science_paths = parse_path_list(args.science_data)
    age_paths = {
        "child": args.age_child_data,
        "teens": args.age_teens_data,
        "adult": args.age_adult_data,
    }

    science_records = []
    age_records = []
    if not args.only_age:
        science_records = load_science_records(science_paths, args.max_science_samples)
        if not science_records:
            raise ValueError("No science records were loaded.")
    if not args.only_science:
        age_records = load_age_records(
            age_paths,
            args.max_age_samples,
            reverse=args.age_reverse_pairs,
        )
        if not age_records:
            raise ValueError("No age records were loaded.")

    data_dir = args.work_dir / "data"
    config_dir = args.work_dir / "configs"
    science_data_path = data_dir / "science_train.json"
    age_data_path = data_dir / "age_train.json"
    science_config_path = config_dir / "science_adapter.json"
    age_config_path = config_dir / "age_adapter.json"

    if science_records:
        write_json(science_data_path, science_records)
        write_json(
            science_config_path,
            build_config(args.science_adapter_name, science_data_path, args),
        )
        print(f"Science samples: {len(science_records)} -> {science_data_path}")
        print(f"Science config: {science_config_path}")
    if age_records:
        write_json(age_data_path, age_records)
        write_json(
            age_config_path,
            build_config(args.age_adapter_name, age_data_path, args),
        )
        print(f"Age samples: {len(age_records)} -> {age_data_path}")
        print(f"Age config: {age_config_path}")
    print(f"Adapter output dir: {args.adapter_output_dir}")

    if args.prepare_only:
        return

    if args.only_science:
        run_training(science_config_path, args, "science", args.science_gpu)
    elif args.only_age:
        run_training(age_config_path, args, "age", args.age_gpu)
    elif args.sequential:
        run_training(science_config_path, args, "science", args.science_gpu)
        run_training(age_config_path, args, "age", args.age_gpu)
    else:
        run_training_parallel(science_config_path, age_config_path, args)


if __name__ == "__main__":
    main()
