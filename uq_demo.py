import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------
# 0) Repro & device
# -----------------------
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# -----------------------
# 1) Synthetic heteroscedastic data
#    y = sin(x) + eps,  eps ~ N(0, sigma(x)^2)
# -----------------------
def make_data(n=2000):
    x = np.random.uniform(-4.0, 4.0, size=(n, 1)).astype(np.float32)
    true_mu = np.sin(x).astype(np.float32)

    # 噪声随 |x| 增大而增大：sigma(x) in [0.1, 0.6] roughly
    true_sigma = (0.10 + 0.50 * (np.abs(x) / 4.0)).astype(np.float32)

    y = true_mu + np.random.randn(n, 1).astype(np.float32) * true_sigma
    return x, y, true_mu, true_sigma

X, y, true_mu, true_sigma = make_data(n=2500)

# Train/test split
idx = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_idx, test_idx = idx[:train_size], idx[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

# -----------------------
# 2) Model: predict mu(x) and log_var(x)
# -----------------------
class HeteroscedasticMLP(nn.Module):
    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)

        # 数值稳定：限制 log_var 范围，防止 exp 溢出或极端小方差
        log_var = torch.clamp(log_var, min=-12.0, max=8.0)
        return mu, log_var

# -----------------------
# 3) Heteroscedastic Gaussian NLL
#    NLL = 0.5*(log_var + (y-mu)^2/exp(log_var)) + 0.5*log(2*pi)
# -----------------------
LOG_2PI = math.log(2.0 * math.pi)

def gaussian_nll(y, mu, log_var):
    # exp(-log_var) 比 1/exp(log_var) 更稳
    inv_var = torch.exp(-log_var)
    nll = 0.5 * (log_var + (y - mu) ** 2 * inv_var + LOG_2PI)
    return nll.mean()

# -----------------------
# 4) Train
# -----------------------
model = HeteroscedasticMLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

def evaluate(model, loader):
    model.eval()
    total_nll = 0.0
    total_mse = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mu, log_var = model(xb)
            nll = gaussian_nll(yb, mu, log_var)

            total_nll += nll.item() * len(xb)
            total_mse += ((yb - mu) ** 2).sum().item()
            total_n += len(xb)

    rmse = math.sqrt(total_mse / total_n)
    return total_nll / total_n, rmse


def save_calibration_and_plots(tag: str):
    # mean + interval
    model.eval()
    xs = np.linspace(-4.0, 4.0, 400, dtype=np.float32).reshape(-1, 1)
    with torch.no_grad():
        xb = torch.from_numpy(xs).to(device)
        mu, log_var = model(xb)
        mu = mu.cpu().numpy()
        sigma = np.sqrt(np.exp(log_var.cpu().numpy()))

    z = 1.96
    lower = mu - z * sigma
    upper = mu + z * sigma

    plt.figure(figsize=(6,4))
    plt.scatter(X_test[:, 0], y_test[:, 0], s=8, alpha=0.4)
    plt.plot(xs[:, 0], mu[:, 0])
    plt.fill_between(xs[:, 0], lower[:, 0], upper[:, 0], alpha=0.25)
    plt.title(f"Heteroscedastic Gaussian NLL @ {tag}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"demo_outputs/mean_interval_{tag}.png", dpi=150)
    plt.close()

    # coverage
    with torch.no_grad():
        xb = torch.from_numpy(X_test).to(device)
        yb = torch.from_numpy(y_test).to(device)
        mu, log_var = model(xb)
        sigma = torch.sqrt(torch.exp(log_var))
        z = 1.96
        lo = mu - z * sigma
        hi = mu + z * sigma
        coverage = ((yb >= lo) & (yb <= hi)).float().mean().item()
        avg_width = (hi - lo).mean().item()
    print(f"[{tag}] coverage95={coverage:.3f} width={avg_width:.3f}")

    # calibration curve
    with torch.no_grad():
        xb = torch.from_numpy(X_test).to(device)
        yb = torch.from_numpy(y_test).to(device)
        mu, log_var = model(xb)
        var = torch.exp(log_var).cpu().numpy().reshape(-1)
        err2 = ((yb - mu) ** 2).cpu().numpy().reshape(-1)

    bins = 15
    idx = np.argsort(var)
    var = var[idx]
    err2 = err2[idx]
    edges = np.linspace(0, len(var), bins + 1, dtype=int)
    xs_b, ys_b = [], []
    for i in range(bins):
        s, t = edges[i], edges[i + 1]
        if t <= s:
            continue
        xs_b.append(np.mean(var[s:t]))
        ys_b.append(np.mean(err2[s:t]))
    xs_b = np.asarray(xs_b)
    ys_b = np.asarray(ys_b)

    plt.figure(figsize=(5,4))
    mx = max(xs_b.max(), ys_b.max())
    plt.plot(xs_b, ys_b, marker="o", label="empirical")
    plt.plot([0, mx], [0, mx], "--", color="gray", label="ideal")
    plt.xlabel("predicted variance")
    plt.ylabel("empirical MSE")
    plt.title(f"Calibration curve @ {tag}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"demo_outputs/calibration_curve_{tag}.png", dpi=150)
    plt.close()

train_history = []
os.makedirs("demo_outputs", exist_ok=True)

for epoch in range(1, 2001):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mu, log_var = model(xb)
        loss = gaussian_nll(yb, mu, log_var)

        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch % 10 == 0 or epoch == 1:
        test_nll, test_rmse = evaluate(model, test_loader)
        train_history.append((epoch, test_nll, test_rmse))
        print(f"epoch {epoch:4d} | test NLL {test_nll:.4f} | test RMSE {test_rmse:.4f}")

    if epoch % 100 == 0:
        save_calibration_and_plots(f"ep{epoch:04d}")


# final metrics curves
if train_history:
    epochs = [t[0] for t in train_history]
    nlls = [t[1] for t in train_history]
    rmses = [t[2] for t in train_history]
    plt.figure(figsize=(6,4))
    plt.plot(epochs, nlls, label="test NLL")
    plt.plot(epochs, rmses, label="test RMSE")
    plt.xlabel("epoch")
    plt.title("Metrics over training")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("demo_outputs/metrics_curve.png", dpi=150)
    plt.close()
