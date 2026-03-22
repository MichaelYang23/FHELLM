import torch
from scipy.stats import pearsonr

logix = torch.load("if_logix.pt")
fhe = torch.load("if_fhe_mnist.pt")

print("[fhe vs LogIX] pearson:", pearsonr(fhe, logix))
