#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import io
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple, Optional

# ---- hygiene ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
for k in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
    os.environ.pop(k, None)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOGIX_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if LOGIX_ROOT not in sys.path:
    sys.path.insert(0, LOGIX_ROOT)

import logix


try:
    from data_id_injector import make_dataloader_with_data_id as injector_loader
    HAS_INJECTOR = True
except Exception:
    HAS_INJECTOR = False


def load_openelm_and_llama_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
    dtype: Optional[str] = "auto",
    device_map: str = "auto",
):
    # 1) Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    ).eval()

    tok_source = None
    try:
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            use_fast=True,
            add_bos_token=True,
            padding_side="left",     
            truncation_side="left",
            cache_dir=cache_dir,
        )
        tok_source = "meta-llama/Llama-2-7b-hf"
    except Exception as e:
        print(f"[tok][warn] cannot load LLaMA-2 tokenizer ({e}); fallback to hf-internal-testing/llama-tokenizer")
        tok = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            use_fast=True,
            cache_dir=cache_dir,
        )
        # Ensure we have pad & bos
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": "<pad>"})
        setattr(tok, "add_bos_token", True)
        tok_source = "hf-internal-testing/llama-tokenizer"

    print(f"[tok] using tokenizer = {tok_source}")
    print(f"[tok] vocab_size={tok.vocab_size}, bos={tok.bos_token_id}, eos={tok.eos_token_id}, pad={tok.pad_token_id}")
    # smoke test
    sm = tok("hello world", return_tensors="pt", add_special_tokens=True)
    print("[tok] smoke input_ids shape:", tuple(sm["input_ids"].shape), "; ids[:10]=", sm["input_ids"][0, :10])

    return model, tok, tok_source


@dataclass
class BookChunk:
    data_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class BookChunksDataset(Dataset):
    """
    Build chunks from *.txt files under books_dir.
    Each chunk yields one training sample (causal LM).
    """
    def __init__(self, books_dir: str, tokenizer, block_size: int, max_files: Optional[int] = None):
        super().__init__()
        self.samples: List[BookChunk] = []
        files = sorted(glob.glob(os.path.join(books_dir, "**/*.txt"), recursive=True))
        if max_files is not None:
            files = files[:max_files]
        assert len(files) > 0, f"No .txt found under {books_dir}"

        for f in files:
            with io.open(f, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            # tokenize whole file; we will slide chunks by block_size tokens
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
            # naive chunking
            for i in range(0, ids.numel(), block_size):
                chunk_ids = ids[i:i + block_size]
                if chunk_ids.numel() < 2:
                    continue  # need at least 2 tokens for shift
                attn = torch.ones_like(chunk_ids, dtype=torch.long)
                data_id = f"{os.path.basename(f)}#chunk:{i//block_size}"
                self.samples.append(BookChunk(data_id, chunk_ids, attn))

        assert len(self.samples) > 0, "No chunk produced; increase files or reduce block_size"
        print(f"[dataset] files={len(files)}, chunks={len(self.samples)} (block_size={block_size})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "data_id": s.data_id,
            "input_ids": s.input_ids,
            "attention_mask": s.attention_mask,
        }


def collate_chunks(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    # dynamic left padding to max length in the batch
    max_len = max(int(x["input_ids"].numel()) for x in batch)
    input_ids, attention_mask, data_ids = [], [], []
    for x in batch:
        ids = x["input_ids"]
        att = x["attention_mask"]
        pad_len = max_len - int(ids.numel())
        if pad_len > 0:
            ids = torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids], dim=0)
            att = torch.cat([torch.zeros((pad_len,), dtype=att.dtype), att], dim=0)
        input_ids.append(ids.unsqueeze(0))
        attention_mask.append(att.unsqueeze(0))
        data_ids.append(x["data_id"])
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    # labels: copy then mask padding to -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "data_id": data_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    ap = argparse.ArgumentParser("Step-1: Extract logs for OpenELM on books")
    ap.add_argument("--project", type=str, required=True)
    ap.add_argument("--config_path", type=str, default="./config.yaml")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--model_name", type=str, default="apple/OpenELM-270M")

    # data modes
    ap.add_argument("--books_dir", type=str, default=None, help="Mode A: root dir of *.txt books")
    ap.add_argument("--max_files", type=int, default=None, help="Limit #files for quick smoke")
    ap.add_argument("--use_injector", action="store_true", help="Mode B: use your data_id_injector (compat)")

    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)

    # LogIX/LoRA/Hessian
    ap.add_argument("--lora", type=str, default="random", choices=["none", "random", "pca"])
    ap.add_argument("--hessian", type=str, default="raw", choices=["none", "raw", "kfac", "ekfac"])
    ap.add_argument("--save", type=str, default="grad", choices=["none", "grad", "act", "cov"])
    ap.add_argument("--name_filter", nargs="+", default=["attn", "ffn"])
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(args)

    # load model & tokenizer
    model, tok, tok_source = load_openelm_and_llama_tokenizer(
        args.model_name, cache_dir=args.cache_dir, dtype="auto", device_map="auto"
    )

    accelerator = Accelerator()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.use_injector:
        assert HAS_INJECTOR, "data_id_injector not importable; disable --use_injector or fix your PYTHONPATH."
        print("[data] Using injector_loader (compat mode). WARNING: ensure injector uses the SAME tokenizer config for Step-2.")
        data_loader = injector_loader(
            cache_dir=args.cache_dir,
            block_size=args.block_size,
            batch_size=args.batch_size,
            max_sellers=args.max_files if args.max_files is not None else 10**9,
            num_workers=args.num_workers,
        )
        print(f"[data][compat] injector provides tokenized batches; our tokenizer = {tok_source}. "
              f"Run Step-2 with the SAME tokenizer used in Step-1 to avoid mismatch.")
    else:
        assert args.books_dir is not None and os.path.isdir(args.books_dir), \
            "Provide --books_dir pointing to a folder with .txt files, OR use --use_injector."
        ds = BookChunksDataset(args.books_dir, tok, args.block_size, max_files=args.max_files)
        data_loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_chunks(b, pad_id=tok.pad_token_id),
            pin_memory=True,
        )

    from itertools import islice
    first_batch = next(iter(data_loader))
    print("[peek] batch keys:", list(first_batch.keys()))
    print("[peek] batch shapes:",
          {k: tuple(v.shape) for k, v in first_batch.items() if torch.is_tensor(v)})
    print("[peek] first data_id:", first_batch["data_id"][0])

    head_dec = tok.decode(first_batch["input_ids"][0, first_batch["attention_mask"][0].bool()][:32])
    tail_dec = tok.decode(first_batch["input_ids"][0, first_batch["attention_mask"][0].bool()][-32:])
    print("[peek] decode head:", repr(head_dec))
    print("[peek] decode tail:", repr(tail_dec))

    run = logix.init(args.project, config=args.config_path)
    logix.watch(model, name_filter=args.name_filter)
    if args.lora != "none":
        run.add_lora()
    scheduler = logix.LogIXScheduler(run, lora="none", hessian=args.hessian, save=args.save)

    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()

    step_cnt = 0
    total_valid_tokens = 0
    from tqdm import tqdm
    for _ in scheduler:
        for batch in tqdm(data_loader, desc="extract-log (books)"):
            data_ids = batch.pop("data_id")
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            valid = int(attention_mask[:, 1:].sum().item())
            total_valid_tokens += valid
            if step_cnt < 3:  # print first 3 for quick check
                print(f"[dbg] bs={input_ids.shape[0]} seq={input_ids.shape[1]} valid_tokens(after shift)={valid} data_id[0]={data_ids[0]}")

            with run(data_id=data_ids, mask=attention_mask):
                model.zero_grad(set_to_none=True)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits if hasattr(out, "logits") else out
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()  # padding already -100
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                    ignore_index=-100,
                )
                accelerator.backward(loss)
            step_cnt += 1
        logix.finalize()

    print("===================================================")
    print(f"[OK] Project logged under: ./logix_logs/{args.project}/")
    print("  Files like: state/mean_state.pt, state/others.pt, lora/lora_state_dict.pt etc.")
    print(f"  Total batches={step_cnt}, total_valid_tokens(after shift)={total_valid_tokens}")
    print("===================================================")

    # -------------- Re-open and inspect --------------
    print("[post] Re-open logs & build log loader for a quick structural check ...")
    logix.initialize_from_log()  # load what we just wrote
    log_loader = logix.build_log_dataloader()
    print("[post] iterate one log batch ...")
    one = next(iter(log_loader))
    if isinstance(one, (tuple, list)) and len(one) == 2:
        src_ids, payload = one
        print(f"[post] tuple/list batch: len(src_ids)={len(src_ids)}, payload_keys={list(payload.keys())[:8]} ...")
        # typical LoRA-compressed gradient logs are keyed by module names, no 'value' tensor
        print("[post] NOTE: tuple 2-tuple format is expected for compressed per-layer logs.")
    elif isinstance(one, dict):
        print(f"[post] dict batch: keys={list(one.keys())[:8]} ...")
        if "value" in one:
            val = one["value"]
            print(f"[post] has 'value' tensor of shape={tuple(val.shape)}")
        else:
            print("[post] dict has no 'value' (this is fine for LoRA-compressed gradients).")
    else:
        print(f"[post] unexpected batch type: {type(one)}")

    blocks = sum(1 for _ in log_loader)
    print(f"[post] total log blocks={blocks}")
    print("Done.")


if __name__ == "__main__":
    main()
