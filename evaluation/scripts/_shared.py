from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from itertools import combinations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_ENCODINGS = ("utf-8", "utf-8-sig")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KNOWN_DOMAINS = {"ai", "biology", "medicine"}
AUDIENCE_MAP = {
    "adult": "adult",
    "adults": "adult",
    "child": "child",
    "children": "child",
    "teen": "teen",
    "teens": "teen",
}
METRIC_ORDER = ("cl", "pa", "ra")
METRIC_SUMMARY_COLUMNS = {
    "cl": "cognitive_load_score",
    "pa": "personalization_alignment_score",
    "ra": "reader_attitude_score",
}
METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "cl": {
        "name": "Cognitive Load",
        "score_label": "cognitive_load_score",
        "dimensions": {
            "INTR": (
                "Intrinsic Fit: 文章的概念难度、信息密度和前置知识假设是否适合目标读者，"
                "既不过难也不过度浅化。"
            ),
            "EXTR": (
                "Extraneous Burden Control: 是否减少不必要的认知负担，例如术语未解释、"
                "结构混乱、冗余细节或突兀跳转。"
            ),
            "GERM": (
                "Germane Support: 是否通过类比、例子、分步解释、局部总结、概念搭脚手架等方式"
                "主动促进理解。"
            ),
        },
    },
    "pa": {
        "name": "Personalization Alignment",
        "score_label": "personalization_alignment_score",
        "dimensions": {
            "CONT": (
                "Content Relevance: 内容选择和强调点是否贴合目标读者真正关心的方面，"
                "而不是泛泛而谈。"
            ),
            "KNOW": (
                "Knowledge-Level Fit: 解释深度、抽象层次和默认前置知识是否匹配目标读者。"
            ),
            "STYLE": (
                "Style Consistency: 措辞、语气、节奏和说明方式是否与目标读者一致。"
            ),
            "CONTX": (
                "Contextual Resonance: 例子、类比、场景或生活连接是否让该读者更容易代入和理解。"
            ),
        },
    },
    "ra": {
        "name": "Reader Attitude",
        "score_label": "reader_attitude_score",
        "dimensions": {
            "ENG": "Engagement Appeal: 文章是否有趣、生动，能吸引目标读者继续阅读。",
            "TRU": (
                "Trust and Credibility: 文章是否给人可靠、审慎、科学 grounded 的感觉，"
                "从而提升读者信任。"
            ),
            "CONTI": (
                "Continuance Intention: 文章是否鼓励读者继续阅读、继续探索相关内容或继续学习。"
            ),
        },
    },
}


class DatasetError(ValueError):
    """Raised when the dataset format violates the agreed contract."""


@dataclass(frozen=True)
class SampleRecord:
    source_file: str
    domain: str
    baseline: str
    audience: str
    index: int
    topic_text: str
    source_prompt_field: str
    final_instruction: str
    output: str
    status: str


@dataclass(frozen=True)
class PairwiseComparisonRecord:
    comparison_id: str
    domain: str
    audience: str
    index: int
    topic_text: str
    baseline_a: str
    baseline_b: str
    left_baseline: str
    right_baseline: str
    left_text: str
    right_text: str
    left_source_file: str
    right_source_file: str


@dataclass(frozen=True)
class ModelEndpointConfig:
    model: str
    provider: str
    api_key_env: str
    base_url: str | None
    direct_supported: bool
    batch_supported: bool
    thinking_mode: str
    direct_token_param: str
    direct_request_extras: dict[str, Any]
    batch_body_extras: dict[str, Any]
    note: str = ""


MODEL_ENDPOINT_CONFIGS: dict[str, ModelEndpointConfig] = {
    "qwen-plus": ModelEndpointConfig(
        model="qwen-plus",
        provider="dashscope",
        api_key_env="DASHSCOPE_API_KEY",
        base_url=DASHSCOPE_BASE_URL,
        direct_supported=True,
        batch_supported=True,
        thinking_mode="explicit_off",
        direct_token_param="max_tokens",
        direct_request_extras={"extra_body": {"enable_thinking": False}},
        batch_body_extras={"enable_thinking": False},
        note="Qwen Plus supports toggling thinking mode; this project forces it off.",
    ),
    "qwen3.6-plus": ModelEndpointConfig(
        model="qwen3.6-plus",
        provider="dashscope",
        api_key_env="DASHSCOPE_API_KEY",
        base_url=DASHSCOPE_BASE_URL,
        direct_supported=True,
        batch_supported=True,
        thinking_mode="explicit_off",
        direct_token_param="max_tokens",
        direct_request_extras={"extra_body": {"enable_thinking": False}},
        batch_body_extras={"enable_thinking": False},
        note="Qwen 3.6 Plus supports toggling thinking mode; this project forces it off.",
    ),
    "deepseek-v3.2": ModelEndpointConfig(
        model="deepseek-v3.2",
        provider="dashscope",
        api_key_env="DASHSCOPE_API_KEY",
        base_url=DASHSCOPE_BASE_URL,
        direct_supported=True,
        batch_supported=True,
        thinking_mode="explicit_off",
        direct_token_param="max_tokens",
        direct_request_extras={"extra_body": {"enable_thinking": False}},
        batch_body_extras={"enable_thinking": False},
        note="DeepSeek V3.2 supports toggling thinking mode; this project forces it off.",
    ),
    "deepseek-v4-pro": ModelEndpointConfig(
        model="deepseek-v4-pro",
        provider="dashscope",
        api_key_env="DASHSCOPE_API_KEY",
        base_url=DASHSCOPE_BASE_URL,
        direct_supported=True,
        batch_supported=False,
        thinking_mode="explicit_off",
        direct_token_param="max_tokens",
        direct_request_extras={"extra_body": {"enable_thinking": False}},
        batch_body_extras={"enable_thinking": False},
        note="DeepSeek V4-Pro supports toggling thinking mode; this project forces it off.",
    ),
    "MiniMax-M2.5": ModelEndpointConfig(
        model="MiniMax-M2.5",
        provider="dashscope",
        api_key_env="DASHSCOPE_API_KEY",
        base_url=DASHSCOPE_BASE_URL,
        direct_supported=True,
        batch_supported=False,
        thinking_mode="forced_on",
        direct_token_param="max_tokens",
        direct_request_extras={},
        batch_body_extras={},
        note=(
            "According to DashScope documentation dated 2026-04-26, MiniMax-M2.5 is a "
            "thinking-only model and cannot disable thinking mode."
        ),
    ),
    "gpt-4o-mini": ModelEndpointConfig(
        model="gpt-4o-mini",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        direct_supported=True,
        batch_supported=False,
        thinking_mode="implicit_off",
        direct_token_param="max_tokens",
        direct_request_extras={},
        batch_body_extras={},
        note="GPT-4o mini uses the standard OpenAI Chat Completions API without reasoning mode.",
    ),
    "gpt-5.5": ModelEndpointConfig(
        model="gpt-5.5",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        direct_supported=True,
        batch_supported=False,
        thinking_mode="explicit_off",
        direct_token_param="max_completion_tokens",
        direct_request_extras={"reasoning_effort": "none"},
        batch_body_extras={},
        note="GPT-5.5 uses Chat Completions with max_completion_tokens and reasoning_effort=none.",
    ),
}


def require_package(import_name: str, install_hint: str) -> Any:
    try:
        module = __import__(import_name, fromlist=["__name__"])
    except ImportError as exc:
        raise SystemExit(install_hint) from exc
    return module


def load_tqdm():
    tqdm_module = require_package(
        "tqdm",
        "缺少依赖 tqdm，请先运行: pip install tqdm",
    )
    return tqdm_module.tqdm


def should_disable_tqdm() -> bool:
    return not sys.stderr.isatty()


def load_openai():
    return require_package(
        "openai",
        "缺少依赖 openai，请先运行: pip install openai",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env_file(env_path: Path | None = None) -> None:
    target = env_path or (project_root() / ".env")
    if not target.exists():
        return
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def read_json_file(path: Path) -> Any:
    last_error: Exception | None = None
    for encoding in DATASET_ENCODINGS:
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise DatasetError(f"无法读取 JSON 文件 {path}: {last_error}")


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path} 第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise DatasetError(f"{path} 第 {line_number} 行不是 JSON 对象")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_audience(raw_value: str) -> str:
    normalized = AUDIENCE_MAP.get(raw_value.strip())
    if not normalized:
        raise DatasetError(f"无法识别人群标签: {raw_value!r}")
    return normalized


def parse_baseline_and_audience(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_")
    if not parts:
        raise DatasetError(f"无法解析文件名: {path.name}")
    baseline = parts[0]
    if not baseline.strip():
        raise DatasetError(f"文件名中缺少有效 baseline 前缀: {path.name}")
    audience = next((part for part in parts if part in {"adult", "child", "teen"}), None)
    if not audience:
        raise DatasetError(f"无法从文件名识别人群: {path.name}")
    return baseline, audience


def normalize_model_slug(model_name: str) -> str:
    return (
        model_name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def resolve_model_endpoint(model_name: str) -> ModelEndpointConfig:
    if model_name in MODEL_ENDPOINT_CONFIGS:
        return MODEL_ENDPOINT_CONFIGS[model_name]
    if model_name.startswith("gpt-"):
        return ModelEndpointConfig(
            model=model_name,
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            base_url=None,
            direct_supported=True,
            batch_supported=False,
            thinking_mode="implicit_off",
            direct_token_param="max_tokens",
            direct_request_extras={},
            batch_body_extras={},
            note="Fallback OpenAI model config inferred from the model prefix.",
        )
    return ModelEndpointConfig(
        model=model_name,
        provider="dashscope",
        api_key_env="DASHSCOPE_API_KEY",
        base_url=DASHSCOPE_BASE_URL,
        direct_supported=True,
        batch_supported=True,
        thinking_mode="unknown",
        direct_token_param="max_tokens",
        direct_request_extras={},
        batch_body_extras={},
        note="Fallback DashScope model config inferred from the current project defaults.",
    )


def create_client_for_model(model_name: str):
    config = resolve_model_endpoint(model_name)
    load_env_file()
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise SystemExit(
            f"未找到 {config.api_key_env}。请在仓库根目录创建 .env 文件，或在系统环境变量中设置它。"
        )
    openai_module = load_openai()
    kwargs: dict[str, Any] = {"api_key": api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return openai_module.OpenAI(**kwargs)


def create_client_from_env(api_key_env: str = "DASHSCOPE_API_KEY"):
    if api_key_env != "DASHSCOPE_API_KEY":
        load_env_file()
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise SystemExit(
                f"未找到 {api_key_env}。请在仓库根目录创建 .env 文件，或在系统环境变量中设置它。"
            )
        openai_module = load_openai()
        return openai_module.OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
    return create_client_for_model("qwen-plus")


def ensure_model_supports_batch(model_name: str) -> ModelEndpointConfig:
    config = resolve_model_endpoint(model_name)
    if not config.batch_supported:
        raise SystemExit(f"模型 {model_name} 当前未配置为 Batch 路径。{config.note}")
    return config


def ensure_model_supports_direct(
    model_name: str,
    *,
    allow_forced_thinking: bool = False,
) -> ModelEndpointConfig:
    config = resolve_model_endpoint(model_name)
    if not config.direct_supported:
        raise SystemExit(f"模型 {model_name} 当前未配置为直接调用路径。{config.note}")
    if config.thinking_mode == "forced_on" and not allow_forced_thinking:
        raise SystemExit(
            f"模型 {model_name} 官方文档标注为仅思考模式，无法严格关闭 thinking。{config.note} "
            "如仍需纳入对比实验，请重新运行并加上 --allow-thinking-models。"
        )
    return config


def build_batch_request_body(model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    config = ensure_model_supports_batch(model)
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body.update(deepcopy(config.batch_body_extras))
    return body


def build_direct_chat_kwargs(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 8,
    allow_forced_thinking: bool = False,
) -> dict[str, Any]:
    config = ensure_model_supports_direct(
        model,
        allow_forced_thinking=allow_forced_thinking,
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    kwargs[config.direct_token_param] = max_tokens
    for key, value in deepcopy(config.direct_request_extras).items():
        kwargs[key] = value
    return kwargs


def select_topic_text(item: dict[str, Any], baseline: str, source_file: str, row_index: int) -> tuple[str, str]:
    if baseline in {"domain", "revised"}:
        field_name = "input"
    else:
        field_name = "original_input"
    raw_value = item.get(field_name)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise DatasetError(
            f"{source_file} 第 {row_index} 条缺少有效字段 {field_name!r}，"
            "按照约定这里不做自动兜底。"
        )
    return raw_value.strip(), field_name


def select_output_text(
    item: dict[str, Any],
    baseline: str,
    source_file: str,
    row_index: int,
) -> str:
    if baseline == "revised":
        for field_name in ("revised_ouput", "revised_output"):
            raw_value = item.get(field_name)
            if isinstance(raw_value, str) and raw_value.strip():
                return raw_value.strip()
        raise DatasetError(
            f"{source_file} 第 {row_index} 条缺少有效 revised_ouput/revised_output，"
            "revised baseline 按约定必须使用事实核查后的输出字段。"
        )
    raw_value = item.get("output")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise DatasetError(f"{source_file} 第 {row_index} 条缺少有效 output")
    return raw_value.strip()


def infer_domain_from_path(path: Path, data_dir: Path) -> str:
    try:
        relative_parts = path.relative_to(data_dir).parts
    except ValueError as exc:
        raise DatasetError(f"文件不在数据目录下: {path}") from exc
    if len(relative_parts) == 2 and path.parent.name in KNOWN_DOMAINS:
        return path.parent.name
    if len(relative_parts) == 1:
        stem_parts = path.stem.split("_")
        domain = next((part for part in stem_parts if part in KNOWN_DOMAINS), None)
        if domain:
            return domain
    raise DatasetError(f"无法为文件推断领域: {path}")


def build_sample_record(path: Path, item: dict[str, Any], data_dir: Path) -> SampleRecord:
    baseline, audience_from_name = parse_baseline_and_audience(path)
    domain = infer_domain_from_path(path, data_dir)
    raw_age_group = item.get("age_group")
    if isinstance(raw_age_group, str) and raw_age_group.strip():
        audience_from_data = normalize_audience(raw_age_group)
        if audience_from_data != audience_from_name:
            raise DatasetError(
                f"{path.name} 第 {item.get('index')} 条的人群字段与文件名不一致: "
                f"{audience_from_data} != {audience_from_name}"
            )
    topic_text, source_prompt_field = select_topic_text(
        item=item,
        baseline=baseline,
        source_file=path.name,
        row_index=int(item.get("index", -1)),
    )
    output = select_output_text(
        item=item,
        baseline=baseline,
        source_file=path.name,
        row_index=int(item.get("index", -1)),
    )
    status = item.get("status")
    if not isinstance(status, str) or not status.strip():
        raise DatasetError(f"{path.name} 第 {item.get('index')} 条缺少有效 status")
    index_value = item.get("index")
    if not isinstance(index_value, int):
        raise DatasetError(f"{path.name} 中发现非整数 index: {index_value!r}")
    final_instruction = item.get("final_instruction", "")
    if final_instruction is None:
        final_instruction = ""
    if not isinstance(final_instruction, str):
        raise DatasetError(f"{path.name} 第 {index_value} 条的 final_instruction 不是字符串")
    return SampleRecord(
        source_file=path.name,
        domain=domain,
        baseline=baseline,
        audience=audience_from_name,
        index=index_value,
        topic_text=topic_text,
        source_prompt_field=source_prompt_field,
        final_instruction=final_instruction.strip(),
        output=output,
        status=status.strip(),
    )


def is_evaluation_dataset_file(path: Path, data_dir: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        relative_parts = path.relative_to(data_dir).parts
    except ValueError:
        return False
    if path.parent.name == "generation":
        return False
    if len(relative_parts) == 2:
        if path.parent.name not in KNOWN_DOMAINS:
            return False
    elif len(relative_parts) == 1:
        stem_parts = path.stem.split("_")
        if not any(part in KNOWN_DOMAINS for part in stem_parts):
            return False
    else:
        return False
    stem_parts = path.stem.split("_")
    return any(part in {"adult", "child", "teen"} for part in stem_parts)


def collect_samples(data_dir: Path, baseline_filters: set[str] | None = None) -> list[SampleRecord]:
    if not data_dir.exists():
        raise DatasetError(f"数据目录不存在: {data_dir}")
    samples: list[SampleRecord] = []
    for path in sorted(data_dir.rglob("*.json")):
        if not is_evaluation_dataset_file(path, data_dir):
            continue
        baseline, _ = parse_baseline_and_audience(path)
        if baseline_filters and baseline not in baseline_filters:
            continue
        payload = read_json_file(path)
        if not isinstance(payload, list):
            raise DatasetError(f"{path} 不是 JSON 数组")
        for item in payload:
            if not isinstance(item, dict):
                raise DatasetError(f"{path} 中存在非对象样本")
            samples.append(build_sample_record(path, item, data_dir))
    if not samples:
        raise DatasetError(f"在 {data_dir} 下没有找到任何样本")
    return samples


def stable_topic_hash(topic_text: str) -> str:
    return hashlib.sha256(topic_text.encode("utf-8")).hexdigest()[:16]


def build_weight_id(metric: str, domain: str, audience: str, topic_text: str) -> str:
    return f"{metric}::{domain}::{audience}::{stable_topic_hash(topic_text)}"


def build_custom_id(metric: str, sample: SampleRecord) -> str:
    return f"{metric}::{sample.domain}::{sample.baseline}::{sample.audience}::{sample.index}"


def build_pairwise_comparison_id(
    domain: str,
    audience: str,
    index: int,
    baseline_a: str,
    baseline_b: str,
) -> str:
    left, right = sorted((baseline_a, baseline_b))
    return f"{domain}::{audience}::{index}::{left}::{right}"


def deterministic_left_right(
    seed: int,
    comparison_id: str,
    baseline_a: str,
    baseline_b: str,
) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{comparison_id}".encode("utf-8")).hexdigest()
    if int(digest[:2], 16) % 2 == 0:
        return baseline_a, baseline_b
    return baseline_b, baseline_a


def iter_baseline_pairs(baselines: list[str]) -> list[tuple[str, str]]:
    return list(combinations(sorted(baselines), 2))


def build_pairwise_prompt(record: PairwiseComparisonRecord) -> tuple[str, str]:
    system_prompt = (
        "You are a strict comparative judge for personalized popular science writing.\n"
        "Compare candidate A and candidate B for the same topic, domain, and target audience.\n"
        "Choose the one that is better overall for this audience.\n"
        "Output exactly one uppercase letter: A or B.\n"
        "Do not output explanations, punctuation, JSON, markdown, or any extra words."
    )
    user_prompt = (
        f"Domain: {record.domain}\n"
        f"Audience: {record.audience}\n"
        f"Topic: {record.topic_text}\n\n"
        "Judge which candidate is better overall for this target audience in this domain.\n"
        "Use overall personalized effect as the criterion: which text is more appropriate, clearer, more trustworthy, and more worth continuing to read.\n"
        "Even if both are imperfect, choose the relatively better one.\n"
        "Return only A or B.\n\n"
        f"Candidate A (baseline={record.left_baseline}):\n{record.left_text}\n\n"
        f"Candidate B (baseline={record.right_baseline}):\n{record.right_text}"
    )
    return system_prompt, user_prompt


def metric_dimensions(metric: str) -> dict[str, str]:
    try:
        return METRIC_DEFINITIONS[metric]["dimensions"]
    except KeyError as exc:
        raise DatasetError(f"不支持的指标: {metric}") from exc


def build_weight_prompt(metric: str, domain: str, audience: str, topic_text: str) -> tuple[str, str]:
    metric_info = METRIC_DEFINITIONS[metric]
    dimension_lines = "\n".join(
        f"- {code}: {desc}" for code, desc in metric_info["dimensions"].items()
    )
    system_prompt = (
        "你是一个严格的评估协议元评估器。"
        "你需要根据科学主题、领域和目标读者，为给定指标分配维度权重。"
        "只输出一个紧凑 JSON 对象，键必须完整、大小写完全一致，值必须是数字。"
        "不要输出解释、Markdown、代码块或额外文本。"
    )
    user_prompt = (
        f"指标: {metric_info['name']} ({metric})\n"
        f"领域: {domain}\n"
        f"目标读者: {audience}\n"
        f"科学主题: {topic_text}\n\n"
        "请依据《指标拆解v2_en.pdf》的共享协议，只为当前指标分配维度权重。\n"
        "要求:\n"
        "1. 权重越大，代表该维度对这个 topic + persona 更重要。\n"
        "2. 所有维度都必须出现。\n"
        "3. 权重总和必须等于 1.0。\n"
        "4. 每个权重保留 3 位小数以内。\n"
        "5. 只输出 JSON 对象，例如 {\"A\":0.4,\"B\":0.6}。\n\n"
        f"当前指标维度:\n{dimension_lines}"
    )
    return system_prompt, user_prompt


def build_score_prompts(metric: str, sample: SampleRecord, weights: dict[str, float]) -> tuple[str, str]:
    metric_info = METRIC_DEFINITIONS[metric]
    weight_lines = "\n".join(f"- {name}: {value}" for name, value in weights.items())
    dimension_lines = "\n".join(
        f"- {code}: {desc}" for code, desc in metric_info["dimensions"].items()
    )
    system_prompt = (
        "You are a strict evaluation judge.\n"
        "Score only the requested metric.\n"
        "Output exactly one number from 0.0 to 5.0 with one decimal place.\n"
        "Do not output words, JSON, units, reasons, prefixes, suffixes, or line labels."
    )
    user_prompt = (
        f"Metric: {metric_info['name']} ({metric})\n"
        f"Domain: {sample.domain}\n"
        f"Audience: {sample.audience}\n"
        f"Topic: {sample.topic_text}\n"
        f"Prompt Source Field: {sample.source_prompt_field}\n\n"
        "Evaluate the generated article according to the shared protocol.\n"
        "First score each dimension from 0 to 5 internally, then compute the weighted sum.\n"
        "Return only the final metric score.\n\n"
        "Dimension definitions:\n"
        f"{dimension_lines}\n\n"
        "Frozen weights:\n"
        f"{weight_lines}\n\n"
        "Scale guidance:\n"
        "- 0 means the article completely fails this metric.\n"
        "- 5 means the article is excellent for this metric.\n\n"
        "Generated article:\n"
        f"{sample.output}"
    )
    return system_prompt, user_prompt


def load_weight_store(weights_file: Path) -> dict[str, Any]:
    payload = read_json_file(weights_file)
    if not isinstance(payload, dict) or "weights" not in payload:
        raise DatasetError(f"权重文件格式不合法: {weights_file}")
    weights = payload["weights"]
    if not isinstance(weights, dict):
        raise DatasetError(f"权重文件中的 weights 字段必须是对象: {weights_file}")
    return payload
