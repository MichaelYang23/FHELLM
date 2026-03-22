#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import argparse
import numpy as np
import torch

import logix
from logix.utils import get_logger
from logix.analysis import InfluenceFunctionFHE

from utils import construct_mlp, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Manifest / slicing ----------
def load_manifest(logs_dir):
    p = os.path.join(logs_dir, "manifest.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing manifest at {p}. Run step-2 first.")
    with open(p, "r") as f:
        return json.load(f)

def slice_per_seller(vec, manifest):
    """
    vec: np.ndarray of length == manifest["total_logged"]
    returns dict {"sellerA": vec_A, "sellerB": vec_B, "sellerC": vec_C}
    """
    out = {}
    for s in ("sellerA", "sellerB", "sellerC"):
        off = int(manifest[s]["offset"]); size = int(manifest[s]["size"])
        part = vec[off:off+size]
        if part.shape[0] != size:
            raise RuntimeError(f"slice mismatch for {s}")
        out[s] = part
    return out

# ---------- Greedy top-K (optional) ----------
def greedy_topk(scores, k, mode="max"):
    """
    scores: 1D np.array.
    Our sign convention: more positive ⇒ more helpful ⇒ select_mode="max"
    """
    if k <= 0:
        return []
    if mode == "min":
        idx = np.argpartition(scores, k)[:k]
        return idx.tolist()
    else:
        idx = np.argpartition(-scores, k)[:k]
        return idx.tolist()


def main():
    ap = argparse.ArgumentParser("Fast FHE scoring via plaintext-averaged test-grad (single-class eval)")
    ap.add_argument("--project", type=str, default="mnist_market_4class")
    ap.add_argument("--buyer_ckpt", type=str, default="./checkpoints/buyer/mlp4_seed0_epoch10.pt")
    ap.add_argument("--partitions", type=str, default="./partitions")
    ap.add_argument("--root", type=str, default="./data")
    ap.add_argument("--logs_dir", type=str, default="./logs")

    # single-class eval: original MNIST label ∈ {1,2,3,4}
    ap.add_argument("--eval_digit", type=int, default=3, choices=[1,2,3,4],
                    help="Which original digit the eval set contains (from step-1).")

    # preconditioning & loaders
    ap.add_argument("--hessian", type=str, default="kfac", choices=["kfac", "none"],
                    help="Use 'kfac' if step-2 logged covariance and finalized state; otherwise 'none'.")
    ap.add_argument("--batch_log", type=int, default=64,
                    help="Batch size for reading seller logs in flattened form.")

    # selection
    ap.add_argument("--greedy_k", type=int, default=0,
                    help="If >0, output a greedy plan on the score vector.")
    ap.add_argument("--select_mode", type=str, default="max", choices=["min","max"],
                    help="Our convention: IF positive ⇒ helpful ⇒ use 'max'.")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    # # ---- Load manifest (seller offsets/sizes) ----
    # manifest = load_manifest(args.logs_dir)
    # total = int(manifest["total_logged"])

    # # ---- Build model ----
    # model = construct_mlp(num_classes=4, seed=0).to(DEVICE)
    # sd = torch.load(args.buyer_ckpt, map_location="cpu")
    # model.load_state_dict(sd)
    # model.eval()

    # # ---- Initialize LogIX (NO custom args), watch & re-watch after LoRA to match logging-time shapes ----
    # run = logix.init(project=args.project)
    # run.watch(model)
    # run.add_lora()
    # run.watch(model)
    # run.save(False)  # read-only; we won't write new logs here

    # # ---- CRITICAL: Load saved LogIX state (incl. covariance) from step-2 ----
    # # This enables KFAC preconditioning even in a new process.
    # state_root = os.path.join("./logix_logs", args.project)   # load_state expects <root>, not ".../state/"
    # run._state.load_state(state_root)

    # # ---- Build log loader from saved logs (flatten=True) ----
    # log_loader = run.build_log_dataloader(batch_size=args.batch_log, flatten=True)

    # # ---- Load the PLAINTEXT mean test gradient tree produced by step-2 ----
    # mean_pth = os.path.join(args.logs_dir, "buyer", "mean_test_grad.pt")
    # if not os.path.exists(mean_pth):
    #     raise FileNotFoundError(
    #         f"Missing {mean_pth}. Run step-2 (2_log_sellers_and_build_testgrad.py) first."
    #     )
    # pkg = torch.load(mean_pth, map_location="cpu")
    # if "mean_test_tree" not in pkg:
    #     raise KeyError(f"{mean_pth} does not contain key 'mean_test_tree'.")
    # mean_tree = pkg["mean_test_tree"]
    # n_used = int(pkg.get("count", -1))
    # get_logger().info(f"Built plaintext mean test gradient on {n_used if n_used>0 else 'unknown'} samples.")

    # # Our IF API expects a tuple (src_ids, src_dict). IDs are not used for math; name is for bookkeeping.
    # test_log = ([f"mean_eval_digit_{args.eval_digit}"], mean_tree)

    # ---- Load manifest (seller offsets/sizes) ----
    manifest = load_manifest(args.logs_dir)
    total = int(manifest["total_logged"])

    # ---- Build model ----
    model = construct_mlp(num_classes=4, seed=0).to(DEVICE)
    sd = torch.load(args.buyer_ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    # ---- Initialize LogIX (NO custom args) ----
    run = logix.init(project=args.project)

    run.watch(model)
    run.add_lora()
    run.watch(model)
    run.save(False)
    state_root = os.path.join("./logix_logs", args.project)

    # 1) Preferred public API: initialize_from_log
    resumed = False
    if hasattr(run, "initialize_from_log"):
        try:
            run.initialize_from_log(log_dir=state_root)
            resumed = True
        except Exception as e:
            print(f"[WARN] initialize_from_log failed: {e}")

    if not resumed:
        try:
            from logix.state import LogIXState
            st = LogIXState()
            st.load_state(state_root)           # Note: pass root directory (contains state/ subdirectory)
            # Prioritize standard public attributes
            setattr(run, "state", st)           # Attach state to run so analysis modules can read it
            resumed = True
        except Exception as e:
            print(f"[WARN] fallback load_state failed: {e}")

    cov_ok = False
    try:
        if hasattr(run, "state"):
            cov = run.state.get_covariance_state()
            cov_ok = isinstance(cov, dict) and len(cov) > 0 and any(len(v) > 0 for v in cov.values())
    except Exception as e:
        print(f"[WARN] could not inspect covariance_state: {e}")
    print("[RESUME] covariance loaded." if cov_ok else "[RESUME] covariance NOT found -> will skip KFAC.")

    log_loader = run.build_log_dataloader(batch_size=args.batch_log, flatten=True)

    mean_pth = os.path.join(args.logs_dir, "buyer", "mean_test_grad.pt")
    if not os.path.exists(mean_pth):
        raise FileNotFoundError(
            f"Missing {mean_pth}. Run step-2 (2_log_sellers_and_build_testgrad.py) first."
        )
    pkg = torch.load(mean_pth, map_location="cpu")
    if "mean_test_tree" not in pkg:
        raise KeyError(f"{mean_pth} does not contain key 'mean_test_tree'.")
    mean_tree = pkg["mean_test_tree"]
    n_used = int(pkg.get("count", -1))
    get_logger().info(f"Built plaintext mean test gradient on {n_used if n_used>0 else 'unknown'} samples.")

    # IF API expects (src_ids, src_dict). IDs are just for bookkeeping, don't affect computation.
    test_log = ([f"mean_eval_digit_{args.eval_digit}"], mean_tree)


    # ---- ONE FHE pass for the chosen target ----
    run.add_analysis({"influence_fhe": InfluenceFunctionFHE})
    get_logger().info("Compute FHE IF (ONE pass)…")
    out = run.influence_fhe.compute_influence_all_fhe(
        test_log=test_log,
        log_loader=log_loader,
        decrypt_results=True,
        hessian=args.hessian,   # "kfac" uses the covariance we just loaded; "none" is raw dot-product
        damping=0.0
    )
    if not isinstance(out["influence"], torch.Tensor):
        raise RuntimeError("FHE results are not decrypted. Set decrypt_results=True.")
    scores = out["influence"].detach().cpu().numpy().reshape(-1)
    assert scores.size == total, f"score length {scores.size} != total_logged {total}"

    os.makedirs(args.logs_dir, exist_ok=True)
    tag = f"eval{args.eval_digit}"
    out_path = os.path.join(args.logs_dir, f"scores_{tag}_fhe.npy")
    np.save(out_path, scores)
    print(f"Saved FHE scores ({tag}) → {out_path}")

    # ---- Per-seller aggregates on the chosen target ----
    byseller = slice_per_seller(scores, manifest)
    agg_sum  = {k: float(np.sum(v))  for k, v in byseller.items()}
    agg_mean = {k: float(np.mean(v)) for k, v in byseller.items()}
    print("\n=== IF aggregates on eval digit={} (more positive ⇒ predicted helpful) ===".format(args.eval_digit))
    print({k: f"{v:+.6f}" for k, v in agg_sum.items()})
    # You can also print mean:
    # print("mean:", {k: f"{v:+.6f}" for k, v in agg_mean.items()})

    # ---- Optional greedy plan (on the chosen score vector) ----
    if args.greedy_k and args.greedy_k > 0:
        chosen = greedy_topk(scores, args.greedy_k, mode=args.select_mode)
        plan = {
            "target": tag,
            "k": int(args.greedy_k),
            "select_mode": args.select_mode,
            "chosen_global_indices": [int(i) for i in chosen]
        }
        with open(os.path.join(args.logs_dir, "purchase_plan.json"), "w") as f:
            json.dump(plan, f, indent=2)
        print("\nSaved greedy plan → logs/purchase_plan.json")

    print("\n=== Done (single-pass FHE on plaintext-averaged test-grad) ===")


if __name__ == "__main__":
    main()
