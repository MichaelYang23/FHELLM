#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

import utils  # Must provide get_seller_dataloader(...)


def _safe_str(x: Any, default: str) -> str:
    if x is None:
        return default
    try:
        s = str(x)
    except Exception:
        s = default
    return s


class _DatasetWithDataID(Dataset):
    """Wraps the original Dataset, injecting a data_id field in __getitem__."""
    def __init__(self, base_ds: Dataset):
        super().__init__()
        self.base = base_ds
        self._auto_ctr = 0  # Used to generate when chunk_id is missing

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = self.base[idx]

        sid = item.get("seller_id", None)
        cid = item.get("chunk_id", None)

        sid_str = _safe_str(sid, "unknown")
        if cid is None:
            cid_str = f"auto{self._auto_ctr}"
            self._auto_ctr += 1
        else:
            cid_str = _safe_str(cid, "auto")

        item["data_id"] = f"seller:{sid_str}#chunk:{cid_str}"
        return item


class _CollatorWithDataID:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # First collect all keys
        keys = set().union(*[b.keys() for b in batch])

        for k in keys:
            vals = [b.get(k, None) for b in batch]
            if k == "data_id":
                # Strictly generate list[str]
                data_ids: List[str] = []
                for v in vals:
                    if isinstance(v, str):
                        data_ids.append(v)
                    elif v is None:
                        data_ids.append("seller:unknown#chunk:auto")
                    else:
                        data_ids.append(str(v))
                out["data_id"] = data_ids
                continue

            if all(isinstance(v, torch.Tensor) for v in vals):
                out[k] = torch.stack(vals, dim=0)
            elif all(isinstance(v, (int, bool)) for v in vals):
                out[k] = torch.tensor(vals, dtype=torch.long)
            else:
                out[k] = vals
        return out


def make_dataloader_with_data_id(
    cache_dir: Optional[str],
    block_size: int,
    batch_size: int,
    max_sellers: int,
    num_workers: int = 0,
) -> DataLoader:
    base_loader: DataLoader = utils.get_seller_dataloader(
        cache_dir=cache_dir,
        block_size=block_size,
        batch_size=batch_size,
        max_sellers=max_sellers,
    )

    base_ds: Dataset = base_loader.dataset
    wrapped = _DatasetWithDataID(base_ds)

    # Reuse the original sampler (if any) to maintain the same order as the original loader
    sampler = getattr(base_loader, "sampler", None)
    batch_sampler = getattr(base_loader, "batch_sampler", None)

    if batch_sampler is not None:
        # When batch_sampler is present, DataLoader ignores batch_size/sampler
        loader = DataLoader(
            wrapped,
            batch_sampler=batch_sampler,
            collate_fn=_CollatorWithDataID(),
            num_workers=num_workers,
            pin_memory=getattr(base_loader, "pin_memory", False),
        )
    else:
        loader = DataLoader(
            wrapped,
            batch_size=getattr(base_loader, "batch_size", batch_size),
            sampler=sampler,
            shuffle=False if sampler is not None else False,
            drop_last=getattr(base_loader, "drop_last", False),
            collate_fn=_CollatorWithDataID(),
            num_workers=num_workers,
            pin_memory=getattr(base_loader, "pin_memory", False),
        )
    return loader


def _dump_idmap_once(
    loader: DataLoader,
    out_csv: str,
    limit: Optional[int] = None,
) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    n = 0
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("data_id,seller_id,chunk_id\n")
        for batch in loader:
            data_ids: List[str] = batch["data_id"]
            sellers = batch.get("seller_id", [None] * len(data_ids))
            chunks  = batch.get("chunk_id",  [None] * len(data_ids))

            # If seller_id / chunk_id is a tensor, convert to list
            if isinstance(sellers, torch.Tensor):
                sellers = sellers.detach().cpu().tolist()
            if isinstance(chunks, torch.Tensor):
                chunks = chunks.detach().cpu().tolist()

            for did, sid, cid in zip(data_ids, sellers, chunks):
                f.write(f"{did},{_safe_str(sid,'unknown')},{_safe_str(cid,'auto')}\n")
                n += 1
                if limit is not None and n >= limit:
                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inject data_id and (optionally) dump an idmap CSV")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_sellers", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="./logix_logs/_idmap_preview.csv")
    parser.add_argument("--limit", type=int, default=1000, help="dump at most N rows (None for all)")
    args = parser.parse_args()

    dl = make_dataloader_with_data_id(
        cache_dir=args.cache_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_sellers=args.max_sellers,
        num_workers=args.num_workers,
    )
    _dump_idmap_once(dl, args.out_csv, None if args.limit <= 0 else args.limit)
    print(f"[ok] wrote: {args.out_csv}")
