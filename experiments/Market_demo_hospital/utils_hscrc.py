# utils_hscrc.py
import os, glob, math, hashlib, random, re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _stable_hash(s: str) -> int:
    # 8-byte blake2b → hex → int
    return int(hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).hexdigest(), 16)

def _to_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default

def _to_float_pos(x, default=0.0):
    try:
        v = float(x)
        return v if v >= 0 else default
    except Exception:
        return default

class HSCRCFeatureEncoder:
    def __init__(
        self,
        apr_drg_buckets: int = 1024,
        apr_mdc_buckets: int = 128,
        icd_buckets: int = 4096,
        year_min: int = 2015,
        year_max: int = 2025,
        use_year_onehot: bool = True,
    ):
        self.apr_drg_buckets = int(apr_drg_buckets)
        self.apr_mdc_buckets = int(apr_mdc_buckets)
        self.icd_buckets = int(icd_buckets)
        self.year_min = int(year_min)
        self.year_max = int(year_max)
        self.use_year_onehot = bool(use_year_onehot)

        self.num_dim   = 2    # log1p(LOS), log1p(TOT_CHG)
        self.sev_dim   = 4
        self.mort_dim  = 4
        self.apr_drg_dim = self.apr_drg_buckets
        self.apr_mdc_dim = self.apr_mdc_buckets
        self.year_dim  = (self.year_max - self.year_min + 1) if use_year_onehot else 8
        self.qtr_dim   = 4
        self.icd_dim   = self.icd_buckets

        self.input_dim = (
            self.num_dim
            + self.sev_dim
            + self.mort_dim
            + self.apr_drg_dim
            + self.apr_mdc_dim
            + self.year_dim
            + self.qtr_dim
            + self.icd_dim
        )

    @staticmethod
    def _one_hot(idx: int, dim: int) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        if 0 <= idx < dim:
            v[idx] = 1.0
        return v

    def _bucket_oh(self, key: str, buckets: int) -> np.ndarray:
        oh = np.zeros(buckets, dtype=np.float32)
        if key is None or key == "" or (isinstance(key, float) and np.isnan(key)):
            return oh
        idx = _stable_hash(key) % buckets
        oh[idx] = 1.0
        return oh

    def encode_row(self, row: pd.Series) -> np.ndarray:
        feats: List[np.ndarray] = []

        # Numeric features
        los = _to_float_pos(row.get("LOS", 0.0))
        tot = _to_float_pos(row.get("TOT_CHG", 0.0))
        feats.append(np.array([math.log1p(los), math.log1p(tot)], dtype=np.float32))

        # Severity levels one-hot
        sev = _to_int(row.get("SEVERITY", 0))
        mort = _to_int(row.get("MORTALIT", 0))
        feats.append(self._one_hot(sev - 1, self.sev_dim))
        feats.append(self._one_hot(mort - 1, self.mort_dim))

        # Category hash
        feats.append(self._bucket_oh(str(_to_int(row.get("APR_DRG", 0))), self.apr_drg_dim))
        feats.append(self._bucket_oh(str(_to_int(row.get("APR_MDC", 0))), self.apr_mdc_dim))

        # Time features
        year = _to_int(row.get("YEAR", 0))
        qtr  = _to_int(row.get("QTR", 0))
        if self.use_year_onehot:
            feats.append(self._one_hot(year - self.year_min, self.year_dim))
        else:
            feats.append(self._bucket_oh(f"YEAR_{year}", self.year_dim))
        feats.append(self._one_hot(qtr - 1, self.qtr_dim))

        # Diagnosis BOW (4096 buckets)
        icd = np.zeros(self.icd_dim, dtype=np.float32)
        codes: List[str] = []
        pdx = row.get("PRINDIAG", "")
        if isinstance(pdx, str) and pdx:
            codes.append(pdx)
        for k in range(1, 30):
            ck = row.get(f"DIAG{k}", "")
            if isinstance(ck, str) and ck:
                codes.append(ck)
        for c in set(codes):  # deduplicate
            icd[_stable_hash(c) % self.icd_dim] = 1.0
        feats.append(icd)

        return np.concatenate(feats, axis=0).astype(np.float32)

    def encode_df(self, df: pd.DataFrame) -> np.ndarray:
        # encode row by row (simple and reliable)
        arr = np.vstack([self.encode_row(r) for _, r in df.iterrows()]).astype(np.float32)
        assert arr.shape[1:] == (self.input_dim,), f"Encoded dim mismatch: got {arr.shape}"
        return arr


def build_file_map(root_dir: str) -> Dict[str, str]:
    """
    Scan directory for '*_labeled.parquet' (or compatible '*.parquet'), return:
      { '210001': '/path/210001_labeled.parquet', ... }
    """
    patt = os.path.join(root_dir, "*_labeled.parquet")
    paths = glob.glob(patt)
    if not paths:
        paths = glob.glob(os.path.join(root_dir, "*.parquet"))

    out = {}
    for p in paths:
        base = os.path.basename(p)
        m = re.match(r"^(\d+)(?:_labeled)?\.parquet$", base)
        if not m:
            continue
        hosp = m.group(1)
        out[hosp] = os.path.abspath(p)
    if not out:
        raise FileNotFoundError(f"No parquet files found under: {root_dir}")
    return dict(sorted(out.items()))

def _load_filtered_df(fp: str) -> pd.DataFrame:
    """
    Read parquet, **keep only LABEL_UP in {0,1}**, and reset_index(drop=True).
    This way position index .iloc aligns with (hosp, row_idx).
    """
    df = pd.read_parquet(fp)
    df = df.loc[df["LABEL_UP"].isin([0, 1, 0.0, 1.0])].reset_index(drop=True)
    return df

def list_index_tuples_for_hospital(file_map: Dict[str, str], hosp: str) -> List[Tuple[str, int]]:
    fp = file_map[hosp]
    df = _load_filtered_df(fp)
    n = len(df)
    return [(str(hosp), int(i)) for i in range(n)]


class ParquetTupleDataset(Dataset):
    def __init__(self, file_map: Dict[str, str], index_tuples: List[Tuple[str, int]], encoder: HSCRCFeatureEncoder):
        self.file_map = dict(file_map)
        self.index = list(index_tuples)
        self.encoder = encoder
        self._df_cache: Dict[str, pd.DataFrame] = {}  # {hospid: filtered_df}

    def __len__(self):
        return len(self.index)

    def _get_df(self, hospid: str) -> pd.DataFrame:
        if hospid in self._df_cache:
            return self._df_cache[hospid]
        fp = self.file_map[hospid]
        df = _load_filtered_df(fp) 
        self._df_cache[hospid] = df
        return df

    def __getitem__(self, i: int):
        hosp, loc = self.index[i]
        hosp = str(hosp); loc = int(loc)
        df = self._get_df(hosp)
        row = df.iloc[loc]                
        x  = self.encoder.encode_row(row)   
        y_val = int(row["LABEL_UP"])
        y = torch.tensor(y_val, dtype=torch.long)
        return x, y

def make_loader_from_tuples(
    file_map: Dict[str, str],
    tuples: List[Tuple[str, int]],
    encoder: HSCRCFeatureEncoder,
    batch_size: int = 512,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    ds = ParquetTupleDataset(file_map, tuples, encoder)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

def construct_mlp_tabular(input_dim: int, num_classes: int = 2, seed: int = 0) -> nn.Module:
    """
    Consistent with our previous MLP style: Linear(bias=False) + ReLU.
    """
    set_seed(seed)
    return nn.Sequential(
        nn.Linear(input_dim, 512, bias=False),
        nn.ReLU(inplace=True),       
        nn.Linear(512, 256, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes, bias=False),
    )
