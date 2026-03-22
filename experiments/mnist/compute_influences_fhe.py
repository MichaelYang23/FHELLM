import numpy as np
import argparse
import torch
from tqdm import tqdm

import logix
from logix.analysis import InfluenceFunction, InfluenceFunctionFHE # Import FHE version
from logix.utils import DataIDGenerator, get_logger
from examples.mnist.utils import construct_mlp, get_mnist_dataloader # Reuse utils
from logix import LogIX, LogIXScheduler
import torch.nn as nn

parser = argparse.ArgumentParser("MNIST FHE Influence Analysis")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--checkpoint", type=str, default="checkpoints/mnist_0_epoch_9.pt")
parser.add_argument("--eval_idxs", type=int, nargs="+", default=[0])
parser.add_argument("--hessian", type=str, default="none", help="Hessian type ('none', 'kfac', 'raw')")
parser.add_argument("--lora", type=str, default="random", help="LoRA type ('none', 'random', 'pca')")

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = construct_mlp().to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
model.eval()

project_name = "mnist_fhe"
run = logix.init(project=project_name, config=args.config)

get_logger().info("Starting log extraction phase...")

run.watch(model)

if args.lora != 'none':
    get_logger().info(f"Adding LoRA (type: {args.lora}) to the model...")
    run.add_lora()
    run.watch(model) # Watch relevant layers
    get_logger().info("LoRA added.")
else:
    get_logger().info("LoRA not enabled for this run.")


get_logger().info("Starting log extraction phase...")
if args.hessian == 'kfac':
     run.setup({"forward": ["covariance"], 
                "backward": ["covariance"], 
                "grad": ["log"]})
else:
     run.setup({"grad": ["log"]})
run.save(True)


train_loader = get_mnist_dataloader(batch_size=512, split="train", shuffle=False, subsample=True)
id_gen = DataIDGenerator()


for inputs, targets in tqdm(train_loader, desc="Extracting Logs/Stats"):
     with run(data_id=id_gen(inputs)):
         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
         model.zero_grad()
         outs = model(inputs)
         loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
         loss.backward()
run.finalize()



get_logger().info("Log extraction phase complete.")


log_loader = run.build_log_dataloader(batch_size=64,
                                      flatten=True) # Flatten for FHE

from logix.analysis import InfluenceFunctionFHE # Import FHE version
run.add_analysis({"influence_fhe": InfluenceFunctionFHE})


run.eval() # Switch LogIX to eval mode (disables saving, etc.)
run.setup({"grad": ["log"]}) # Need to log test gradient
test_loader = get_mnist_dataloader(batch_size=1, split="valid", shuffle=False, indices=args.eval_idxs)
test_log = None
for test_input, test_target in test_loader: # Usually just one batch
     with run(data_id=id_gen(test_input)):
         test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
         model.zero_grad()
         test_out = model(test_input)
         test_loss = torch.nn.functional.cross_entropy(
             test_out, test_target, reduction="sum"
         )
         test_loss.backward()
     test_log = run.get_log() # Get the plaintext test gradient log
     break # Process only the first batch/sample usually

if test_log is None:
    raise RuntimeError("Failed to get test gradient.")

get_logger().info("Starting FHE influence computation...")
fhe_results = run.influence_fhe.compute_influence_all_fhe(
    test_log=test_log,
    log_loader=log_loader,
    decrypt_results=True, # Request decryption
    hessian=args.hessian, # Ensure preconditioner uses correct method
    damping=run.config.influence.damping # Use damping from config
)
get_logger().info("FHE influence computation finished.")

if isinstance(fhe_results["influence"], torch.Tensor): # Check if decrypted
    if_scores_fhe = fhe_results["influence"].numpy()
    print(f"Decrypted FHE Influence Scores (shape): {if_scores_fhe.shape}")

    _, top_influential_data_fhe = torch.topk(torch.from_numpy(if_scores_fhe), k=10)
    print("Top FHE influential data indices:", top_influential_data_fhe.numpy().tolist())

    torch.save(if_scores_fhe, "if_fhe_mnist.pt")
    print("FHE scores saved to if_fhe_mnist.pt")

    run.add_analysis({"influence_plain": InfluenceFunction})
    plain_results = run.influence_plain.compute_influence_all(
         test_log, log_loader, damping=run.config.influence.damping, hessian=args.hessian
    )
    if_scores_plain = plain_results["influence"].numpy()
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(if_scores_fhe.flatten(), if_scores_plain.flatten())
    print(f"Pearson Correlation between FHE and Plaintext: {correlation:.4f} (p={p_value:.2e})")
    print(f"Max Absolute Difference: {np.max(np.abs(if_scores_fhe - if_scores_plain)):.2e}")


else:
    print("FHE results are still encrypted (PyCtxt objects). Cannot process further.")

print("MNIST FHE Influence Analysis Complete.")