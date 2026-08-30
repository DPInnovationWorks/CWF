import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Optional


TRAINING_DIR = Path(__file__).resolve().parent
MOE_PEFT_DIR = TRAINING_DIR.parent
if str(MOE_PEFT_DIR) not in sys.path:
    sys.path.insert(0, str(MOE_PEFT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import moe_peft
from moe_peft.common import LLMBatchConfig, LLMModelInput


class TwoMixLoraAlphaNet(nn.Module):
    """Token-level alpha gate that blends two frozen MixLoRA branches."""

    def __init__(self, hidden_dim: int, hidden_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_a: torch.Tensor, hidden_b: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([hidden_a, hidden_b], dim=-1))


class InstructionOutputDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length: int = 1024):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.data = []
        for item in data:
            if not isinstance(item, dict):
                continue
            instruction = item.get("instruction") or item.get("input") or ""
            output = item.get("output") or item.get("answer") or ""
            if instruction or output:
                self.data.append({"instruction": instruction, "output": output})

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        output = item["output"]

        if output:
            full_text = f"指令：{instruction}\n回答：{output}"
            prompt_text = f"指令：{instruction}\n回答："
        else:
            full_text = instruction
            prompt_text = ""

        tokens = self.tokenizer.encode(full_text)[: self.max_length]
        labels = tokens.copy()
        if prompt_text:
            prompt_len = min(len(self.tokenizer.encode(prompt_text)), len(labels))
            labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": tokens, "labels": labels}


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append([1] * len(x["input_ids"]) + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def build_model_input(
    batch: Dict[str, torch.Tensor],
    adapter_name: str,
    gradient_checkpoint: str = "none",
) -> LLMModelInput:
    batch_size = batch["input_ids"].shape[0]
    return LLMModelInput(
        batch_configs_=[
            LLMBatchConfig(
                adapter_name_=adapter_name,
                batch_start_idx_=0,
                batch_end_idx_=batch_size,
            )
        ],
        batch_tokens_=batch["input_ids"],
        batch_labels_=None,
        batch_masks_=batch["attention_mask"],
        output_router_logits_=False,
        gradient_checkpoint_=gradient_checkpoint,
        inference_mode_=False,
    )


def get_adapter_hidden_states(model, input_args: LLMModelInput) -> torch.Tensor:
    input_ids, inputs_embeds, _, causal_mask, cache_position = model._prepare_inputs(
        input_args
    )
    del input_ids
    rotary_emb = model.model_.rotary_embed(inputs_embeds, cache_position.unsqueeze(0))
    hidden_states, _ = model._call_decoder_stack(
        inputs_embeds,
        input_args,
        rotary_emb,
        causal_mask,
        cache_position,
        None,
    )
    return hidden_states


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(logits.device)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


@dataclass
class TrainConfig:
    base_model_path: str
    adapter_a_path: str
    adapter_b_path: str
    train_json: str
    val_json: Optional[str] = None
    output_dir: str = "./alphanet_output"
    adapter_a_name: str = "mixlora_a"
    adapter_b_name: str = "mixlora_b"
    hidden_size: int = 256
    dropout: float = 0.1
    num_epochs: int = 2
    batch_size: int = 1
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    load_16bit: bool = True
    load_8bit: bool = False
    load_4bit: bool = False
    flash_attn: bool = False
    device: str = "cuda"
    adapter_a_device: Optional[str] = None
    adapter_b_device: Optional[str] = None
    alpha_device: Optional[str] = None
    parallel_hidden: bool = True
    alpha_on_b: bool = True


class TwoMixLoraAlphaTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

        self.device = cfg.device or moe_peft.executor.default_device_name()
        self.adapter_a_device = cfg.adapter_a_device or self.device
        self.adapter_b_device = cfg.adapter_b_device or self.device
        self.alpha_device = cfg.alpha_device or self.adapter_a_device
        self.dual_model = self.adapter_a_device != self.adapter_b_device

        load_dtype = torch.bfloat16 if cfg.load_16bit else torch.float32
        bits = 8 if cfg.load_8bit else (4 if cfg.load_4bit else None)

        self.model_a = moe_peft.LLMModel.from_pretrained(
            cfg.base_model_path,
            device=self.adapter_a_device,
            attn_impl="flash_attn" if cfg.flash_attn else "eager",
            bits=bits,
            load_dtype=load_dtype,
        )
        self.tokenizer = moe_peft.Tokenizer(cfg.base_model_path)
        self.model_a.load_adapter(cfg.adapter_a_path, cfg.adapter_a_name)
        self.model_a.requires_grad_(False)
        self.model_a.eval()

        if self.dual_model:
            self.model_b = moe_peft.LLMModel.from_pretrained(
                cfg.base_model_path,
                device=self.adapter_b_device,
                attn_impl="flash_attn" if cfg.flash_attn else "eager",
                bits=bits,
                load_dtype=load_dtype,
            )
            self.model_b.load_adapter(cfg.adapter_b_path, cfg.adapter_b_name)
            self.model_b.requires_grad_(False)
            self.model_b.eval()
        else:
            self.model_b = self.model_a
            self.model_a.load_adapter(cfg.adapter_b_path, cfg.adapter_b_name)

        hidden_dim = self.model_a.config_.dim_
        self.alpha_net = TwoMixLoraAlphaNet(
            hidden_dim=hidden_dim,
            hidden_size=cfg.hidden_size,
            dropout=cfg.dropout,
        ).to(self.alpha_device)

        self.output_layer = self.model_a.output_.layers_[cfg.adapter_a_name]
        if hasattr(self.output_layer, "requires_grad_"):
            self.output_layer.requires_grad_(False)
        elif hasattr(self.output_layer, "lm_head_"):
            self.output_layer.lm_head_.requires_grad_(False)

    def _adapter_hiddens(self, batch: Dict[str, torch.Tensor]):
        input_a = build_model_input(batch, self.cfg.adapter_a_name)
        input_b = build_model_input(batch, self.cfg.adapter_b_name)

        with torch.no_grad():
            if self.dual_model and self.cfg.parallel_hidden:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_a = executor.submit(
                        get_adapter_hidden_states, self.model_a, input_a
                    )
                    future_b = executor.submit(
                        get_adapter_hidden_states, self.model_b, input_b
                    )
                    hidden_a = future_a.result().detach()
                    hidden_b = future_b.result().detach()
            else:
                hidden_a = get_adapter_hidden_states(self.model_a, input_a).detach()
                hidden_b = get_adapter_hidden_states(self.model_b, input_b).detach()

        hidden_a = hidden_a.to(self.alpha_device)
        hidden_b = hidden_b.to(self.alpha_device)
        return hidden_a, hidden_b

    def alpha_for_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden_a, hidden_b = self._adapter_hiddens(batch)
        return self.alpha_net(hidden_a.float(), hidden_b.float())

    def _fused_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden_a, hidden_b = self._adapter_hiddens(batch)

        alpha = self.alpha_net(hidden_a.float(), hidden_b.float())
        if self.cfg.alpha_on_b:
            fused_hidden = (1.0 - alpha) * hidden_a.float() + alpha * hidden_b.float()
        else:
            fused_hidden = alpha * hidden_a.float() + (1.0 - alpha) * hidden_b.float()

        return self.output_layer.forward(fused_hidden.to(hidden_a.dtype))

    def _make_loader(self, json_path: str, shuffle: bool) -> DataLoader:
        dataset = InstructionOutputDataset(
            json_path, self.tokenizer, self.cfg.max_length
        )
        if len(dataset) == 0:
            raise ValueError(f"No usable training samples found in {json_path}")
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            collate_fn=make_collate_fn(self.tokenizer.pad_id_),
        )

    def train(self) -> TwoMixLoraAlphaNet:
        train_loader = self._make_loader(self.cfg.train_json, shuffle=True)
        val_loader = (
            self._make_loader(self.cfg.val_json, shuffle=False)
            if self.cfg.val_json
            else None
        )

        optimizer = optim.AdamW(self.alpha_net.parameters(), lr=self.cfg.learning_rate)
        update_steps = max(
            1,
            (len(train_loader) * self.cfg.num_epochs)
            // self.cfg.gradient_accumulation_steps,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=update_steps,
            eta_min=self.cfg.learning_rate * 0.1,
        )

        best_val_loss = float("inf")
        optimizer.zero_grad()

        for epoch in range(self.cfg.num_epochs):
            self.alpha_net.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.num_epochs}")

            for step, batch in enumerate(pbar):
                logits = self._fused_logits(batch)
                labels = batch["labels"].to(self.alpha_device)
                loss = causal_lm_loss(logits, labels)
                (loss / self.cfg.gradient_accumulation_steps).backward()
                epoch_loss += loss.item()

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.alpha_net.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if len(train_loader) % self.cfg.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.alpha_net.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1} train loss: {avg_loss:.4f}")

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f"Epoch {epoch + 1} val loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save("alphanet_best.pth")

        self.save("alphanet_final.pth")
        return self.alpha_net

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.alpha_net.eval()
        total_loss = 0.0
        for batch in loader:
            logits = self._fused_logits(batch)
            labels = batch["labels"].to(self.alpha_device)
            total_loss += causal_lm_loss(logits, labels).item()
        return total_loss / len(loader)

    def save(self, filename: str):
        save_path = os.path.join(self.cfg.output_dir, filename)
        torch.save(
            {
                "alpha_net": self.alpha_net.state_dict(),
                "config": asdict(self.cfg),
                "hidden_dim": self.model_a.config_.dim_,
            },
            save_path,
        )
        print(f"Saved AlphaNet to {save_path}")


def train_alphanet(**kwargs) -> TwoMixLoraAlphaNet:
    cfg = TrainConfig(**kwargs)
    trainer = TwoMixLoraAlphaTrainer(cfg)
    return trainer.train()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train AlphaNet to blend two frozen MixLoRA adapters."
    )
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_a_path", required=True)
    parser.add_argument("--adapter_b_path", required=True)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--val_json", default=None)
    parser.add_argument("--output_dir", default="./alphanet_output")
    parser.add_argument("--adapter_a_name", default="mixlora_a")
    parser.add_argument("--adapter_b_name", default="mixlora_b")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapter_a_device", default=None)
    parser.add_argument("--adapter_b_device", default=None)
    parser.add_argument("--alpha_device", default=None)
    parser.add_argument("--no_parallel_hidden", dest="parallel_hidden", action="store_false")
    parser.add_argument("--load_16bit", action="store_true", default=True)
    parser.add_argument("--no_load_16bit", dest="load_16bit", action="store_false")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--alpha_on_a", dest="alpha_on_b", action="store_false")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    TwoMixLoraAlphaTrainer(parse_args()).train()
