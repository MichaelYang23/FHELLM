import os
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

OPENELM_MODEL_NAME = "apple/OpenELM-1_1B"

# Tokenizer candidates, in order
TOKENIZER_CANDIDATES = [
    "meta-llama/Llama-2-7b-hf",              # likely 403 in your env
    "hf-internal-testing/llama-tokenizer",   # public fallback
]

HUGE_MAX_LEN = 10**12  # silence tokenizer length warnings safely



def set_seed(seed: int = 0):
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(
    cache_dir: Optional[str] = None,
    add_padding_token: bool = True,
    add_bos_token: bool = True,
):

    last_err = None
    for name in TOKENIZER_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(
                name,
                use_fast=True,
                cache_dir=cache_dir,
            )
            print(f"[utils] Loaded tokenizer: {name}")

            # BOS behavior
            try:
                tok.add_bos_token = bool(add_bos_token)
            except Exception:
                pass

            # ensure pad token
            if add_padding_token and tok.pad_token is None:
                tok.add_special_tokens({"pad_token": "<pad>"})

            # silence warning at tokenization time (we split later anyway)
            try:
                tok.model_max_length = HUGE_MAX_LEN
            except Exception:
                pass

            return tok
        except Exception as e:
            print(f"[utils] Failed to load tokenizer {name}: {e}")
            last_err = e

    raise RuntimeError(
        f"All tokenizer candidates failed. Last error: {last_err}"
    )

def get_model(cache_dir: Optional[str] = None, model_name: Optional[str] = None):
    """
    Load an OpenELM model (defaults to 270M for safety if not specified).
    """
    model_name = model_name or "apple/OpenELM-270M"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    return model

class BookSellerDataset(Dataset):
    """
    A seller-aware dataset for lucadiliello/bookcorpusopen.

    - Each book is one seller.
    - We tokenize the book ONCE, then split into fixed-length blocks.
    - Each returned item is a dict of tensors (input_ids, attention_mask, labels) of length = block_size,
      and a string 'seller_id' (book title).
    """

    def __init__(
        self,
        hf_dataset_name: str = "lucadiliello/bookcorpusopen",
        cache_dir: Optional[str] = None,
        tokenizer=None,
        block_size: int = 512,
        max_sellers: Optional[int] = 1000,
        min_tokens: int = 32,           # skip tiny books/fragments
        use_special_tokens: bool = True # include BOS/EOS where tokenizer supports
    ):
        assert tokenizer is not None, "tokenizer must be provided"
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.samples: List[Dict[str, Any]] = []

        ds = load_dataset(hf_dataset_name, cache_dir=cache_dir)
        train_ds = ds["train"]

        if max_sellers is not None:
            n = min(int(max_sellers), len(train_ds))
            train_ds = train_ds.select(range(n))

        # Build block-level samples
        for row in train_ds:
            title = row.get("title") or "<untitled>"
            text = row.get("text")  or ""
            seller_id = title  # 1 book = 1 seller

            if not text or text.strip() == "":
                continue

            # Whole-book tokenization (NOT fed to model directly)
            enc = self.tokenizer(
                text,
                add_special_tokens=bool(use_special_tokens),
                return_attention_mask=False,
                truncation=False,          # we split manually
                return_tensors=None,
            )
            ids = enc["input_ids"]
            if isinstance(ids[0], list):
                # some tokenizers may return [[...]] for single sequence
                ids = ids[0]

            if len(ids) < max(self.block_size, min_tokens):
                continue

            # Fixed non-overlapping blocks of exactly block_size
            n_full = len(ids) // self.block_size
            for i in range(n_full):
                chunk = ids[i * self.block_size: (i + 1) * self.block_size]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                labels = input_ids.clone()  # we do left-shift in training code

                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "seller_id": seller_id,
                    }
                )

        if len(self.samples) == 0:
            print("[utils][WARN] BookSellerDataset constructed 0 samples. "
                  "Check dataset availability / max_sellers / block_size.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_seller_dataloader(
    cache_dir: Optional[str] = None,
    block_size: int = 512,
    batch_size: int = 1,
    max_sellers: Optional[int] = 1000,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    tokenizer = get_tokenizer(cache_dir=cache_dir, add_padding_token=True, add_bos_token=True)
    try:
        tokenizer.model_max_length = HUGE_MAX_LEN
    except Exception:
        pass

    dataset = BookSellerDataset(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        block_size=block_size,
        max_sellers=max_sellers,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory) and torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return loader
