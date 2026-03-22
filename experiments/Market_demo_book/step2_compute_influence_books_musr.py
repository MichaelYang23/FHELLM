#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step-2: Compute per-book Influence for the Book Data Market using MuSR as eval set.

What this script does:
  1) Read Step-1 logs (training side: seller book chunk gradients, LoRA compressed)
  2) Use MuSR (team_allocation.json) to construct eval texts, compute eval-side gradient g_test
  3) Through LogIX influence engine, compute H^{-1} g_test . g_train (or approximation) with test and training logs
  4) Aggregate per-chunk influence to per-book scores and output ranking

Robustness:
  - If the provided config.yaml contains keys unrecognized by the old version (e.g., log_dir), auto-sanitize then init
  - If still fails, fallback to default init (without reading YAML)
  - Only use dynamic length (no fixed padding) to avoid indexSelectLargeIndex assertion failures
"""

import os
import copy
import sys
import json
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional
try:
    import yaml
except ImportError:
    yaml = None

# Resolve local LogIX source path (repo-local) when package is not installed.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOGIX_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if LOGIX_ROOT not in sys.path:
    sys.path.insert(0, LOGIX_ROOT)

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

import logix
from logix.utils import merge_logs

# -------------- environment optimization --------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


# ---------- OpenELM + Tokenizer ----------
def load_openelm_and_llama_tokenizer(model_name: str,
                                     cache_dir: Optional[str] = None,
                                     device_map: str = "auto"):
    # Model (requires trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device_map,
        cache_dir=cache_dir,
    ).eval()

    # Tokenizer: prefer LLaMA2, fallback to testing if it fails
    tok = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        use_fast=True,
        add_bos_token=True,
        padding_side="left",
        truncation_side="left",
        cache_dir=cache_dir,
    )
    print(f"[tok] vocab={tok.vocab_size}, bos={tok.bos_token_id}, eos={tok.eos_token_id}, pad={tok.pad_token_id}")
    return model, tok


# ---------- MuSR loading and text conversion ----------
def _read_json_flex(path: str) -> List[Dict[str, Any]]:
    """Supports JSON list or JSONL"""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            # May have 'samples' / 'data'
            for k in ("samples", "data", "items"):
                if k in data and isinstance(data[k], list):
                    return data[k]
            # Wrap single item into a list
            return [data]
        elif isinstance(data, list):
            return data
        else:
            return []
    except Exception:
        # JSONL
        out = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out


def musr_item_to_text(d: Dict[str, Any]) -> str:
    """
    Robustly assemble a MuSR item into a "text that can be learned by a causal LM".
    Logic:
      - Take the longest field among question/prompt/instruction/input/text as "stem"
      - If options/choices exist, list them as A/B/C...
      - If answer/label/target exists, append "Answer: ..." at the end (optional)
    """
    keys_q = ["question", "prompt", "instruction", "input", "text"]
    stem = ""
    for k in keys_q:
        v = d.get(k)
        if isinstance(v, str) and len(v) > len(stem):
            stem = v.strip()
    if not stem:
        # Fallback: concatenate all string fields
        stem = " ".join(str(v).strip() for v in d.values() if isinstance(v, str))[:1024]

    opts = d.get("options") or d.get("choices")
    lines = [stem]
    if isinstance(opts, (list, tuple)) and opts:
        abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, opt in enumerate(opts):
            if isinstance(opt, dict) and "text" in opt:
                txt = str(opt["text"])
            else:
                txt = str(opt)
            lines.append(f"{abc[i%26]}. {txt}")

    ans = d.get("answer") or d.get("label") or d.get("target")
    if isinstance(ans, (str, int)):
        lines.append(f"Answer: {ans}")

    return "\n".join(lines)


def load_musr_texts(path: str, max_items: int) -> List[str]:
    items = _read_json_flex(path)
    texts = []
    for d in items:
        t = musr_item_to_text(d)
        if t:
            texts.append(t)
        if len(texts) >= max_items:
            break
    assert len(texts) > 0, f"No eval texts parsed from {path}"
    print(f"[musr] loaded {len(texts)} eval texts. Example head:\n---\n{texts[0][:400]}\n---")
    return texts


# ---------- Old Schema compatibility: config sanitization ----------
def _drop_key_recursive(obj, bad_keys={"log_dir"}):
    if isinstance(obj, dict):
        return {k: _drop_key_recursive(v, bad_keys) for k, v in obj.items() if k not in bad_keys}
    elif isinstance(obj, list):
        return [_drop_key_recursive(x, bad_keys) for x in obj]
    else:
        return obj


def try_logix_init_with_sanitization(project: str,
                                     config_path: Optional[str],
                                     out_dir: str):
    """
    1) If config_path is empty, use default init directly
    2) If read fails with unknown keyword (e.g., log_dir), sanitize YAML then retry
    3) If still fails, use default init
    """
    os.makedirs(out_dir, exist_ok=True)
    if not config_path:
        print("[init] No config_path provided; using default LogIX config.")
        return logix.init(project)
    if yaml is None:
        print("[init] PyYAML not installed; cannot sanitize custom config. "
              "Proceeding with direct init(project, config_path).")
        return logix.init(project, config_path)

    try:
        print(f"[init] Trying config_path={config_path}")
        return logix.init(project, config_path)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword" in msg or "got an unexpected keyword" in msg:
            print(f"[init] Detected schema mismatch ({msg}). Sanitizing YAML keys like 'log_dir' and retrying...")
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            cleaned = _drop_key_recursive(raw, bad_keys={"log_dir"})
            tmp_path = os.path.join(out_dir, "config.sanitized.yaml")
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cleaned, f, sort_keys=False, allow_unicode=True)
            print(f"[init] Retry with sanitized config: {tmp_path}")
            try:
                return logix.init(project, tmp_path)
            except Exception as e2:
                print(f"[init] Sanitized init still failed: {e2}. Fallback to default.")
                return logix.init(project)
        else:
            print(f"[init] TypeError but not schema-related: {e}. Fallback to default.")
            return logix.init(project)
    except Exception as e:
        print(f"[init] Init failed ({e}); fallback to default.")
        return logix.init(project)


# ---------- Build test-side gradient (g_test) ----------
def build_test_log(run, model, tok, texts: List[str], accelerator, max_len: int = 2048):
    """
    Dynamic length, no fixed padding. Forward each text individually, loss uses standard shift-CE (ignore_index=-100).
    Returns: merged test_log, can be used directly for influence computation.
    """
    # Only prepare model; wrap with Accelerator consistent with Step-1, but here only involves model
    model = accelerator.prepare(model)
    model.eval()

    test_logs = []
    for t in texts:
        enc = tok(
            t,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,          # No fixed-length padding
            truncation=True,
            max_length=max_len
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(model.device)

        # labels copy input_ids, set padding positions (if any) to -100
        labels = input_ids.clone()
        if "attention_mask" in enc:
            labels[attention_mask == 0] = -100

        # Feed only 1 item at a time (compatible with Step-1's batch-based log mechanism)
        with run(data_id=[f"musr:{hash(t) % 10**9}"], mask=attention_mask):
            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits if hasattr(out, "logits") else out

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )
            accelerator.backward(loss)

        # Extract and save the test log for this sample
        # test_logs.append(run.get_log().copy())
        test_logs.append(copy.deepcopy(run.get_log()))

    # Merge multiple test logs (convenient merging provided by LogIX)
    merged = merge_logs(test_logs)

    # self-check
    if isinstance(merged, (tuple, list)) and len(merged) == 2:
        src_ids, payload = merged
        print(f"[testlog] merged: tuple, len(src_ids)={len(src_ids)}, payload_keys(sample)={list(payload.keys())[:8]}")
    elif isinstance(merged, dict):
        print(f"[testlog] merged: dict, keys(sample)={list(merged.keys())[:8]}")
    else:
        print(f"[testlog] merged: type={type(merged)} (unexpected?)")


    return merged

# ---------- Seller book aggregation (per-book ranking) ----------
def extract_book_key(src_id: str) -> str:
    """
    In Step-1 our data_id looks like:
      seller:1-2-this-is-only-the-beginning.epub.txt#chunk:auto3
    When aggregating by "book" unit, take the part before # as the book key.
    """
    return src_id.split("#", 1)[0]


def aggregate_per_book(tgt_ids: List[str], scores: torch.Tensor) -> List[Tuple[str, float]]:
    """
    tgt_ids and scores are aligned (scores are IF scores for each training sample/chunk).
    Sum (or average) for the same "book key", return sorted from "more negative -> more useful".
    """
    book2sum: Dict[str, float] = {}
    for sid, sc in zip(tgt_ids, scores.tolist()):
        bkey = extract_book_key(sid)
        book2sum[bkey] = book2sum.get(bkey, 0.0) + sc

    # More negative is better (reduces loss more), sort in ascending order
    ranked = sorted(book2sum.items(), key=lambda x: x[1])
    return ranked


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Step-2: Books Influence (MuSR as eval)")
    ap.add_argument("--project", required=True, type=str)
    ap.add_argument("--config_path", default=None, type=str,
                    help="(Optional) Path to Step-1 config.yaml. If omitted, use default.")
    ap.add_argument("--cache_dir", default=None, type=str)
    ap.add_argument("--model_name", default="apple/OpenELM-270M", type=str)
    ap.add_argument("--musr_json", required=True, type=str)
    ap.add_argument("--max_items", default=8, type=int)
    ap.add_argument("--name_filter", nargs="+", default=["attn", "ffn"])
    ap.add_argument("--damping", default=1e-2, type=float)
    ap.add_argument("--batch_size_src", default=64, type=int, help="batch for log loader")
    ap.add_argument("--max_len", default=2048, type=int)
    ap.add_argument("--out_dir", default="./debug_out", type=str)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Model & tokenizer
    model, tok = load_openelm_and_llama_tokenizer(args.model_name, cache_dir=args.cache_dir)

    # 2) LogIX initialization (with YAML sanitization fallback)
    run = try_logix_init_with_sanitization(args.project, args.config_path, args.out_dir)
    logix.watch(model, name_filter=args.name_filter)

    logix.setup({"grad": ["log"]})
    logix.eval()

    # 3) Load Step-1 logs and build loader (training side)
    logix.initialize_from_log()
    log_loader = logix.build_log_dataloader(batch_size=args.batch_size_src)

    # 4) Construct MuSR test texts -> test_log
    accelerator = Accelerator()
    texts = load_musr_texts(args.musr_json, max_items=args.max_items)
    test_log = build_test_log(run, model, tok, texts, accelerator, max_len=args.max_len)

    # 5) Compute influence

    result = run.influence.compute_influence_all(test_log, log_loader, damping=args.damping, mode="dot")

    # 6) Aggregate by "book" & rank
    infl = result["influence"]           # Tensor [#train_samples]
    if isinstance(infl, torch.Tensor) and infl.ndim == 2:
        infl = infl.sum(dim=0)  # or .mean(dim=0) if you prefer averaging
    tgt_ids = result["tgt_ids"]          # Training-side data_id list (consistent with your Step-1 seller:#chunk format)
    src_ids = result["src_ids"]          # Test-side id (here is musr:xxx), for record keeping

    ranked = aggregate_per_book(tgt_ids, infl)

    # 7) Save results
    import csv
    torch.save(infl, os.path.join(args.out_dir, "per_chunk_scores.pt"))
    with open(os.path.join(args.out_dir, "per_chunk_scores.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["train_id", "IF_score"])
        for sid, sc in zip(tgt_ids, infl.tolist()):
            w.writerow([sid, sc])

    with open(os.path.join(args.out_dir, "per_book_scores.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["book_key", "IF_sum"])
        for bk, sc in ranked:
            w.writerow([bk, sc])

    print("\n[top-10 books (most negative → best)]")
    for i, (bk, sc) in enumerate(ranked[:10], 1):
        print(f"{i:2d}. {bk}   {sc:.6f}")

    print(f"\n[done] wrote:\n  - {os.path.join(args.out_dir, 'per_chunk_scores.pt')}\n"
          f"  - {os.path.join(args.out_dir, 'per_chunk_scores.tsv')}\n"
          f"  - {os.path.join(args.out_dir, 'per_book_scores.tsv')}")
    

if __name__ == "__main__":
    main()
