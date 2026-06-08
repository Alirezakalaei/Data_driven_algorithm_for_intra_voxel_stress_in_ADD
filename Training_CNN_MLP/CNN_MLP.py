
import os
import numpy as np
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import optuna
import functools
import math
import json
import matplotlib
import pickle
from scipy.stats import norm

# Use TKAgg for interactive plotting, or 'Agg' if running on a headless server
try:
    matplotlib.use("TKAgg")
except:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from numba import njit

# --- GLOBAL CONFIGURATION ---
N_ENSEMBLE_MODELS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Physical Constants ---
a = 3.615e-10
b_mag = a / np.sqrt(2)
dz = a / np.sqrt(3)
mu = 48e9

# Burgers vectors for the 3 slip systems (normalized)
b_vecs = np.array([
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3) / 2, 0.0],
    [-0.5, np.sqrt(3) / 2, 0.0]
], dtype=float)


# --- Ranger Optimizer ---
class Ranger(optim.Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold,
                        eps=eps, weight_decay=weight_decay)
        super(Ranger, self).__init__(params, defaults)
        self.N_sma_threshhold = N_sma_threshhold
        self.alpha = alpha
        self.k = k
        self.radam_buffer = [[None, None, None] for ind in range(10)]

    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data.float()
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0: grad.add_(p.data, alpha=group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                p.data.copy_(p_data_fp32)
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    p.data.copy_(slow_p)
        return loss


# --- HELPER FUNCTIONS ---
@njit()
def compute_3sys_distribution(angle_tensor, density_tensor, burgers_ID, nbins):
    angle_mod = np.mod(angle_tensor, 2 * np.pi)
    nx, ny, nd = angle_tensor.shape
    n_burgers = 3
    rho_config = np.zeros((n_burgers, nbins))
    bin_edges = np.linspace(0, 2 * np.pi, nbins + 1)
    flat_angles = angle_mod.reshape(-1, nd)
    flat_densities = density_tensor.reshape(-1, nd)

    for k in range(nd):
        b_id = burgers_ID[k]
        if not (0 <= b_id < n_burgers): continue
        angles_k = flat_angles[:, k]
        densities_k = flat_densities[:, k]
        bin_indices = np.searchsorted(bin_edges, angles_k, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, nbins - 1)
        for i in range(angles_k.size):
            rho = densities_k[i]
            idx = bin_indices[i]
            rho_config[b_id, idx] += rho
    return rho_config


@njit()
def compute_net_character_per_system(angle_tensor, density_tensor, burgers_ID, b_angles):
    nx, ny, nd = angle_tensor.shape
    n_burgers = 3
    net_screw = np.zeros(n_burgers)
    net_edge = np.zeros(n_burgers)
    flat_angles = angle_tensor.reshape(-1, nd)
    flat_densities = density_tensor.reshape(-1, nd)

    for k in range(nd):
        b_id = burgers_ID[k]
        if not (0 <= b_id < n_burgers):
            continue
        b_angle = b_angles[b_id]
        for i in range(flat_angles.shape[0]):
            theta = flat_angles[i, k]
            rho = flat_densities[i, k]
            char_angle = theta - b_angle
            net_screw[b_id] += rho * np.cos(char_angle)
            net_edge[b_id] += rho * np.sin(char_angle)

    return net_screw, net_edge


def calculate_screw_and_edge_density(burgers_vectors, density_distribution, scale):
    if density_distribution.size != 1080: return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    dist_reshaped = np.reshape(density_distribution, [3, 360])
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    total_pos_screw, total_neg_screw = np.zeros(3), np.zeros(3)
    total_pos_edge, total_neg_edge = np.zeros(3), np.zeros(3)

    for nd in range(3):
        b_vec = burgers_vectors[nd]
        b_angle = np.arctan2(b_vec[1], b_vec[0])
        for i in range(360):
            theta, val = angles[i], dist_reshaped[nd, i]
            if val == 0: continue
            screw = val * np.cos(theta - b_angle)
            edge = val * np.sin(theta - b_angle)
            if screw > 0:
                total_pos_screw[nd] += screw
            else:
                total_neg_screw[nd] += screw
            if edge > 0:
                total_pos_edge[nd] += edge
            else:
                total_neg_edge[nd] += edge

    if scale == 0: return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    return total_pos_screw / scale, total_neg_screw / scale, total_pos_edge / scale, total_neg_edge / scale


def tau_computer(density_matrix, tau_map, burgers_id_list):
    if density_matrix.ndim < 3: return np.zeros(3)
    nx, ny, num_dis = density_matrix.shape
    tau_av, density, tau = np.zeros(3), np.zeros((nx, ny, 3)), np.zeros((nx, ny, 3))
    for n in range(num_dis):
        bid = burgers_id_list[n]
        if 0 <= bid < 3:
            density[:, :, bid] += density_matrix[:, :, n]
            tau[:, :, bid] += tau_map[:, :, n]
    density[density < 1e9] = 0
    density[density >= 1e9] = 1
    for i in range(3):
        num_el = np.sum(density[:, :, i])
        tau_av[i] = np.sum(tau[:, :, i] * density[:, :, i]) / num_el if num_el > 0 else 0
    return tau_av


# --- Data Loading ---
def load_data(data_dir, file_prefix="batch_coarse_", num_files=8000):
    tabular_inputs, distributional_inputs, outputs = [], [], []
    epsilon = 1e-15
    initial_samples = 0

    def signed_log1p(x):
        return np.sign(x) * np.log1p(np.abs(x))

    b_angles = np.array([np.arctan2(b[1], b[0]) for b in b_vecs])

    for i in range(8001, 8000 + num_files + 1):
        file_path = os.path.join(data_dir, f"{file_prefix}{i}.npz")
        if os.path.exists(file_path):
            try:
                data = np.load(file_path, allow_pickle=True)
                burger_id_array = data['burger_ID']
                angle_tensor = np.squeeze(data['theta_distribution'])
                density_tensor = np.squeeze(data['dislocation_density'])
                if angle_tensor.ndim < 3 or density_tensor.ndim < 3: continue

                initial_samples += 1

                Lx, Ly, Lz = float(data["Lx"]), float(data["Ly"]), float(data["Lz"])
                burger_id_array_sq = np.squeeze(burger_id_array)

                SSD = data["SSD"].flatten();
                SSD[SSD < 0] = 0
                GND = data["GND"].flatten();
                GND[GND < 0] = 0
                all_rho = data["all_rho"].flatten();
                all_rho[all_rho < 0] = 0
                if len(SSD) != 3: continue

                dist_3sys = compute_3sys_distribution(angle_tensor, density_tensor, burger_id_array_sq, nbins=360)
                nx, ny = density_tensor.shape[0], density_tensor.shape[1]
                dist_3sys = dist_3sys / (nx * ny)
                cnn_input = np.log1p(dist_3sys)

                total_density_config = dist_3sys.flatten()
                tau_map = np.squeeze(data["tau_map"]) * (b_mag / mu)

                # --- STRATEGY: LOWER BOUND CONSTRAINT ON RAW MAP ---
                # Clip raw map to -0.01 before averaging
                tau_map = np.clip(tau_map, -0.01, 0)

                if np.sum(tau_map[:]) == 0: continue

                tau_homo = tau_computer(density_tensor, tau_map, burger_id_array_sq)

                # --- STRATEGY: DATA INTEGRITY & ZEROING POSITIVES ---
                # 1. Any positive stress becomes zero
                tau_homo[tau_homo > 0] = 0

                # 2. Ensure no value is smaller than -0.01
                tau_homo = np.clip(tau_homo, -0.01, 0)

                tau_homo_log = signed_log1p(tau_homo)

                volume = Lx * Ly * Lz
                pos_screw, neg_screw, pos_edge, neg_edge = calculate_screw_and_edge_density(
                    b_vecs, total_density_config, scale=1.0)
                svr = (2 * (Lx * Ly + Lx * Lz + Ly * Lz)) / volume if volume > 0 else 0
                disloc_length = all_rho * volume

                net_screw, net_edge = compute_net_character_per_system(
                    angle_tensor, density_tensor, burger_id_array_sq, b_angles)
                net_screw = net_screw / (nx * ny)
                net_edge = net_edge / (nx * ny)

                screw_asymmetry = net_screw / (np.abs(pos_screw) + np.abs(neg_screw) + epsilon)
                edge_asymmetry = net_edge / (np.abs(pos_edge) + np.abs(neg_edge) + epsilon)

                base_row = np.concatenate([
                    np.log1p(SSD), np.log1p(GND), np.log1p(all_rho), np.array([svr]),
                    np.log1p(pos_screw), signed_log1p(neg_screw), np.log1p(pos_edge), signed_log1p(neg_edge),
                    (np.abs(pos_screw) + np.abs(neg_screw)) / (np.abs(pos_edge) + np.abs(neg_edge) + epsilon),
                    np.log1p(np.array([all_rho[0] * all_rho[1], all_rho[0] * all_rho[2], all_rho[1] * all_rho[2]])),
                    np.log1p(disloc_length), signed_log1p(net_screw), signed_log1p(net_edge),
                    screw_asymmetry, edge_asymmetry,
                ])

                tabular_inputs.append(base_row)
                distributional_inputs.append(cnn_input)
                outputs.append(tau_homo_log.flatten())
            except Exception as e:
                continue

    print(f"\nLoaded and processed {len(tabular_inputs)} samples from {initial_samples} initial files.\n")
    if not tabular_inputs: raise ValueError("No valid data found.")
    return np.array(tabular_inputs, dtype=np.float32), np.array(distributional_inputs, dtype=np.float32), np.array(
        outputs, dtype=np.float32)


def inverse_log_transform(y_log):
    y_abs_clipped = np.clip(np.abs(y_log), a_min=None, a_max=80)
    return np.sign(y_log) * (np.exp(y_abs_clipped) - 1)


def compute_sample_weights(y_orig):
    weights = np.ones(len(y_orig))
    magnitudes = np.abs(y_orig).max(axis=1)
    median_mag = np.median(magnitudes[magnitudes > 1e-9])
    if median_mag > 0:
        mag_weights = 1.0 + np.log1p(magnitudes / median_mag)
        weights *= mag_weights
    return (weights / weights.mean()).astype(np.float32)


# --- MODEL COMPONENTS ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        reduced_channels = max(channels // reduction, 4)
        self.excitation = nn.Sequential(nn.Linear(channels, reduced_channels, bias=False), nn.GELU(),
                                        nn.Linear(reduced_channels, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        is_2d = x.dim() == 2
        if is_2d: x = x.unsqueeze(2)
        y = self.squeeze(x).view(x.size(0), -1);
        y = self.excitation(y).unsqueeze(2)
        out = x * y.expand_as(x)
        if is_2d: out = out.squeeze(2)
        return out


class ExpertBranch(nn.Module):
    def __init__(self, input_dim, n_neurons, n_blocks, dropout_rate):
        super(ExpertBranch, self).__init__()
        self.input_layer = nn.Linear(input_dim, n_neurons)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.Sequential(nn.Linear(n_neurons, n_neurons), nn.BatchNorm1d(n_neurons), nn.GELU(),
                                  nn.Dropout(dropout_rate))
            self.blocks.append(nn.ModuleList([block, SEBlock(n_neurons)]))

    def forward(self, x):
        x = self.input_layer(x)
        for block, se in self.blocks: x = x + se(block(x))
        return x


class NeuralNetwork(nn.Module):
    def __init__(self, output_dim, n_neurons, n_blocks, dropout_rate,
                 cnn_out_channels, cnn_kernel_size,
                 branch1_indices, branch2_indices, branch3_indices):
        super(NeuralNetwork, self).__init__()
        self.act = nn.GELU()
        self.branch1_indices, self.branch2_indices, self.branch3_indices = branch1_indices, branch2_indices, branch3_indices
        self.branch1 = ExpertBranch(len(self.branch1_indices), n_neurons, n_blocks, dropout_rate)
        self.branch2 = ExpertBranch(len(self.branch2_indices), n_neurons, n_blocks, dropout_rate)
        self.branch3 = ExpertBranch(len(self.branch3_indices), n_neurons, n_blocks, dropout_rate)

        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, padding='same'),
            nn.BatchNorm1d(cnn_out_channels), self.act, nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(cnn_out_channels, cnn_out_channels * 2, kernel_size=cnn_kernel_size, padding='same'),
            nn.BatchNorm1d(cnn_out_channels * 2), self.act, nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(cnn_out_channels * 2, cnn_out_channels * 4, kernel_size=cnn_kernel_size, padding='same'),
            nn.BatchNorm1d(cnn_out_channels * 4), self.act, nn.AdaptiveAvgPool1d(1), nn.Flatten())

        cnn_output_dim = cnn_out_channels * 4
        combined_dim = (n_neurons * 3) + cnn_output_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, n_neurons), nn.BatchNorm1d(n_neurons), self.act, nn.Dropout(dropout_rate),
            nn.Linear(n_neurons, n_neurons // 2), nn.BatchNorm1d(n_neurons // 2), self.act, nn.Dropout(dropout_rate),
            nn.Linear(n_neurons // 2, output_dim))

    def forward(self, x_tabular, x_dist):
        x1 = self.branch1(x_tabular[:, self.branch1_indices])
        x2 = self.branch2(x_tabular[:, self.branch2_indices])
        x3 = self.branch3(x_tabular[:, self.branch3_indices])
        x_cnn = self.cnn_branch(x_dist)
        combined = torch.cat([x1, x2, x3, x_cnn], dim=1)
        out = self.head(combined)
        return -F.softplus(out)


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return torch.log(torch.cosh(diff + 1e-12))


class AdvancedCriterion(nn.Module):
    def __init__(self):
        super(AdvancedCriterion, self).__init__()
        self.loss_fn = LogCoshLoss()

    def forward(self, y_pred_scaled, y_true_scaled, sample_weights):
        loss_per_sample = self.loss_fn(y_pred_scaled, y_true_scaled)
        return (loss_per_sample * sample_weights.unsqueeze(1)).mean()


def safe_r2_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    if not np.isfinite(y_pred).all() or not np.isfinite(y_true).all(): return -float('inf')
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0: return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)


def objective_nn(trial, X_train_tab, X_train_dist, y_train, w_train, X_val_tab, X_val_dist, y_val, w_val,
                 input_indices):
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    n_blocks = trial.suggest_int('n_blocks', 2, 4)
    n_neurons = trial.suggest_categorical('n_neurons', [128, 256, 384])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    batch_size = trial.suggest_categorical('batch_size', [128, 256])
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Ranger'])
    cnn_out_channels = trial.suggest_categorical('cnn_out_channels', [16, 32, 48])
    cnn_kernel_size = trial.suggest_categorical('cnn_kernel_size', [5, 9, 13])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    tab_scaler, dist_scaler, target_scaler = RobustScaler().fit(X_train_tab), RobustScaler().fit(
        X_train_dist.reshape(-1, 3 * 360)), MaxAbsScaler().fit(y_train)

    X_t_t_s, X_v_t_s = tab_scaler.transform(X_train_tab), tab_scaler.transform(X_val_tab)
    X_t_d_s = dist_scaler.transform(X_train_dist.reshape(-1, 3 * 360)).reshape(-1, 3, 360)
    X_v_d_s = dist_scaler.transform(X_val_dist.reshape(-1, 3 * 360)).reshape(-1, 3, 360)
    y_t_s, y_v_s = target_scaler.transform(y_train), target_scaler.transform(y_val)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_t_t_s), torch.FloatTensor(X_t_d_s), torch.FloatTensor(y_t_s),
                      torch.FloatTensor(w_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_v_t_s), torch.FloatTensor(X_v_d_s), torch.FloatTensor(y_v_s),
                      torch.FloatTensor(w_val)), batch_size=batch_size)

    model = NeuralNetwork(y_train.shape[1], n_neurons, n_blocks, dropout_rate, cnn_out_channels, cnn_kernel_size,
                          **input_indices).to(DEVICE)
    criterion = AdvancedCriterion()

    optimizer = Ranger(model.parameters(), lr=lr,
                       weight_decay=weight_decay) if optimizer_name == 'Ranger' else optim.AdamW(model.parameters(),
                                                                                                 lr=lr,
                                                                                                 weight_decay=weight_decay)

    best_r2, patience, counter = -np.inf, 40, 0
    for epoch in range(120):
        model.train()
        for x_tab_b, x_dist_b, yb, wb in train_loader:
            x_tab_b, x_dist_b, yb, wb = x_tab_b.to(DEVICE), x_dist_b.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            optimizer.zero_grad();
            preds = model(x_tab_b, x_dist_b);
            loss = criterion(preds, yb, wb)
            if torch.isnan(loss) or torch.isinf(loss): return -float("inf")
            loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0);
            optimizer.step()

        model.eval();
        preds_list, trues_list = [], []
        with torch.no_grad():
            for x_tab_b, x_dist_b, yb, _ in val_loader:
                yp = model(x_tab_b.to(DEVICE), x_dist_b.to(DEVICE))
                preds_list.append(target_scaler.inverse_transform(yp.cpu().numpy()))
                trues_list.append(target_scaler.inverse_transform(yb.cpu().numpy()))

        val_r2 = safe_r2_score(inverse_log_transform(np.concatenate(trues_list)),
                               inverse_log_transform(np.concatenate(preds_list)))
        if val_r2 > best_r2:
            best_r2, counter = val_r2, 0
        else:
            counter += 1
        if counter >= patience: break
        trial.report(val_r2, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
    return best_r2


# --- REVISED PLOTTING FUNCTIONS ---

def plot_accuracy_prediction(y_true, y_pred):
    y_t, y_p = y_true.flatten(), y_pred.flatten()
    finite_mask = np.isfinite(y_t) & np.isfinite(y_p)
    y_t, y_p = y_t[finite_mask], y_p[finite_mask]

    r2 = safe_r2_score(y_t, y_p)
    mae = mean_absolute_error(y_t, y_p)

    fig, ax = plt.subplots(figsize=(7, 6))

    # --- MODIFICATION START ---
    # Changed cmap to 'turbo' for high contrast and added bins='log'
    # so that low-density regions are highly visible.
    hb = ax.hexbin(y_t, y_p, gridsize=50, cmap='turbo', mincnt=1, bins='log')
    # --- MODIFICATION END ---

    min_val = min(y_t.min(), y_p.min())
    max_val = max(y_t.max(), y_p.max())
    lims = [min_val, max_val]

    ax.plot(lims, lims, 'r--', lw=2, label='Ideal Prediction')
    ax.set_title(f"Prediction Accuracy\n$R^2={r2:.4f}$, MAE={mae:.4e}", fontsize=14)
    ax.set_xlabel(r"True Stress ($\tau / \mu$)", fontsize=12)
    ax.set_ylabel(r"Predicted Stress ($\tau / \mu$)", fontsize=12)

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Log10(Count)', fontsize=10)  # Updated label to reflect log scale

    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig


def plot_residual_distribution(y_true, y_pred):
    y_t, y_p = y_true.flatten(), y_pred.flatten()
    finite_mask = np.isfinite(y_t) & np.isfinite(y_p)
    y_t, y_p = y_t[finite_mask], y_p[finite_mask]

    residuals = y_p - y_t
    fig, ax = plt.subplots(figsize=(7, 6))
    n, bins, patches = ax.hist(residuals, bins=50, density=True, alpha=0.6, color='green', edgecolor='black')
    mu_res, std_res = norm.fit(residuals)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu_res, std_res)
    ax.plot(x, p, 'k', linewidth=2, label=f'Fit ($\mu={mu_res:.2e}, \sigma={std_res:.2e}$)')
    ax.set_title("Residual Distribution", fontsize=14)
    ax.set_xlabel(r"Residual ($\tau_{pred} - \tau_{true}$) [$\tau / \mu$]", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig


def plot_cdf_error(y_true, y_pred):
    y_t, y_p = y_true.flatten(), y_pred.flatten()
    finite_mask = np.isfinite(y_t) & np.isfinite(y_p)
    y_t, y_p = y_t[finite_mask], y_p[finite_mask]

    abs_errors = np.abs(y_p - y_t)
    sorted_errors = np.sort(abs_errors)
    p = 1. * np.arange(len(sorted_errors)) / (len(sorted_errors) - 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sorted_errors, p, linewidth=2, color='darkorange')
    ax.set_title("CDF of Absolute Prediction Error", fontsize=14)
    ax.set_xlabel(r"Absolute Error $|\tau_{pred} - \tau_{true}|$ [$\tau / \mu$]", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    idx_90 = int(len(sorted_errors) * 0.9)
    err_90 = sorted_errors[idx_90]
    ax.axvline(err_90, color='red', linestyle=':', label=f'90% of errors < {err_90:.2e}')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_stress_vs_density_final(X_tabular_test, y_true_stress, y_pred_stress):
    log_rho_sys1, log_rho_sys2, log_rho_sys3 = X_tabular_test[:, 6], X_tabular_test[:, 7], X_tabular_test[:, 8]
    total_density = np.expm1(log_rho_sys1) + np.expm1(log_rho_sys2) + np.expm1(log_rho_sys3)
    y_true_flat, y_pred_flat = y_true_stress.flatten(), y_pred_stress.flatten()
    if len(y_true_flat) > len(total_density): total_density = np.repeat(total_density, y_true_stress.shape[1])

    plt.figure(figsize=(10, 6))
    plt.scatter(total_density, y_true_flat, alpha=0.5, c='blue', label='True Stress', s=10)
    plt.scatter(total_density, y_pred_flat, alpha=0.5, c='red', marker='x', label='Predicted Stress', s=10)
    plt.xlabel("Total Dislocation Density ($m^{-2}$)")
    plt.ylabel(r"Stress ($\tau / \mu$)")
    plt.title("Stress vs. Dislocation Density (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig("stress_vs_density.png")
    plt.show()


# --- NEW FEATURE IMPORTANCE FUNCTION ---
def compute_permutation_importance(model, X_tab, X_dist, y_true, feature_names, device):
    """
    Computes permutation importance for tabular features.
    """
    model.eval()

    # Baseline metric
    with torch.no_grad():
        pred_base = model(X_tab.to(device), X_dist.to(device)).cpu().numpy()

    # Since model output is scaled/log-transformed, we compute importance in that space
    # to avoid inverse transform overhead inside the loop.
    baseline_r2 = r2_score(y_true.flatten(), pred_base.flatten())

    importances = {}
    X_tab_np = X_tab.numpy()

    print("\nComputing Permutation Importance...")

    for i, name in enumerate(feature_names):
        # Save original column
        original_col = X_tab_np[:, i].copy()

        # Shuffle column
        np.random.shuffle(X_tab_np[:, i])

        # Predict with shuffled column
        with torch.no_grad():
            X_tab_shuffled = torch.FloatTensor(X_tab_np).to(device)
            pred_shuffled = model(X_tab_shuffled, X_dist.to(device)).cpu().numpy()

        shuffled_r2 = r2_score(y_true.flatten(), pred_shuffled.flatten())

        # Importance is the drop in R2
        importances[name] = baseline_r2 - shuffled_r2

        # Restore column
        X_tab_np[:, i] = original_col

    return importances


def plot_feature_importance(importances):
    names = list(importances.keys())
    values = list(importances.values())

    # Sort by importance
    sorted_idx = np.argsort(values)
    sorted_names = [names[i] for i in sorted_idx]
    sorted_values = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(sorted_names, sorted_values, color='teal')
    ax.set_xlabel("Decrease in $R^2$ Score (Importance)")
    ax.set_title("Feature Importance (Permutation Method)")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    data_dir = "/home/user/PycharmProjects/Intravoxelstress"
    tabular_inputs, distributional_inputs, outputs_log = load_data(data_dir)
    outputs_orig = inverse_log_transform(outputs_log)
    weights = compute_sample_weights(outputs_orig)

    X_train_tab, X_test_tab, X_train_dist, X_test_dist, y_train, y_test, w_train, w_test = train_test_split(
        tabular_inputs, distributional_inputs, outputs_log, weights, test_size=0.15, random_state=42)
    X_opt_tab, X_val_tab, X_opt_dist, X_val_dist, y_opt, y_val, w_opt, w_val = train_test_split(
        X_train_tab, X_train_dist, y_train, w_train, test_size=0.2, random_state=42)

    input_indices = {'branch1_indices': list(range(10)), 'branch2_indices': list(range(10, 31)),
                     'branch3_indices': list(range(31, tabular_inputs.shape[1]))}

    print("\n--- Optimizing Hyperparameters ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(
        functools.partial(objective_nn, X_train_tab=X_opt_tab, X_train_dist=X_opt_dist, y_train=y_opt, w_train=w_opt,
                          X_val_tab=X_val_tab, X_val_dist=X_val_dist, y_val=y_val, w_val=w_val,
                          input_indices=input_indices), n_trials=60)
    bp = study.best_params
    print(f"\nBest Params: {bp}")
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(bp, f, indent=4)

    print("\n--- Training Final Ensemble Models ---")
    tab_scaler, dist_scaler, target_scaler = RobustScaler().fit(X_train_tab), RobustScaler().fit(
        X_train_dist.reshape(-1, 3 * 360)), MaxAbsScaler().fit(y_train)

    with open('scalers.pkl', 'wb') as f:
        pickle.dump({'tab_scaler': tab_scaler, 'dist_scaler': dist_scaler, 'target_scaler': target_scaler}, f)

    X_train_t_s, y_train_s = torch.FloatTensor(tab_scaler.transform(X_train_tab)), torch.FloatTensor(
        target_scaler.transform(y_train))
    X_train_d_s = torch.FloatTensor(dist_scaler.transform(X_train_dist.reshape(-1, 3 * 360)).reshape(-1, 3, 360))
    w_train_s = torch.FloatTensor(w_train)
    X_test_t_s = torch.FloatTensor(tab_scaler.transform(X_test_tab))
    X_test_d_s = torch.FloatTensor(dist_scaler.transform(X_test_dist.reshape(-1, 3 * 360)).reshape(-1, 3, 360))

    train_loader = DataLoader(TensorDataset(X_train_t_s, X_train_d_s, y_train_s, w_train_s),
                              batch_size=bp['batch_size'], shuffle=True)

    for i in range(N_ENSEMBLE_MODELS):
        print(f"\n--- Training Ensemble Model {i + 1}/{N_ENSEMBLE_MODELS} ---")
        model = NeuralNetwork(y_train.shape[1], bp['n_neurons'], bp['n_blocks'], bp['dropout_rate'],
                              bp['cnn_out_channels'], bp['cnn_kernel_size'], **input_indices).to(DEVICE)
        criterion = AdvancedCriterion()
        optimizer = Ranger(model.parameters(), lr=bp['lr'], weight_decay=bp.get('weight_decay', 1e-5)) if bp[
                                                                                                              'optimizer'] == 'Ranger' else optim.AdamW(
            model.parameters(), lr=bp['lr'], weight_decay=bp.get('weight_decay', 1e-5))
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=bp['lr'], steps_per_epoch=len(train_loader),
                                                  epochs=250)

        best_r2, patience, counter, best_model_state = -np.inf, 40, 0, None
        for epoch in range(250):
            model.train()
            for x_tab_b, x_dist_b, yb, wb in train_loader:
                x_tab_b, x_dist_b, yb, wb = x_tab_b.to(DEVICE), x_dist_b.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
                optimizer.zero_grad();
                preds = model(x_tab_b, x_dist_b);
                loss = criterion(preds, yb, wb)
                if torch.isnan(loss) or torch.isinf(loss): print(
                    f"NaN loss detected at epoch {epoch}. Stopping."); break
                loss.backward();
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0);
                optimizer.step();
                scheduler.step()
            if torch.isnan(loss) or torch.isinf(loss): break

            model.eval()
            with torch.no_grad():
                yp_test = model(X_test_t_s.to(DEVICE), X_test_d_s.to(DEVICE))
                yp_test_orig = inverse_log_transform(target_scaler.inverse_transform(yp_test.cpu().numpy()))
                y_test_orig = inverse_log_transform(y_test)
            val_r2 = safe_r2_score(y_test_orig.flatten(), yp_test_orig.flatten())
            sign_acc = (np.sign(y_test_orig) == np.sign(yp_test_orig)).mean() * 100
            if val_r2 > best_r2:
                best_r2, best_model_state, counter = val_r2, model.state_dict(), 0
            else:
                counter += 1
            if (epoch + 1) % 25 == 0: print(
                f"  Epoch {epoch + 1}: R²: {val_r2:.4f}, Sign Acc: {sign_acc:.1f}% (Best R²: {best_r2:.4f})")
            if counter >= patience: print(f"  Early stopping at epoch {epoch + 1}."); break
        if best_model_state: torch.save(
            {'hyperparameters': bp, 'input_indices': input_indices, 'model_state_dict': best_model_state},
            f"packaged_hybrid_model_ensemble_{i}.pth")

    print("\n--- Evaluating Ensemble Performance ---")
    all_preds, y_test_orig = [], inverse_log_transform(y_test)

    # Load first model for feature importance calculation
    first_model = None

    for i in range(N_ENSEMBLE_MODELS):
        package_path = f"packaged_hybrid_model_ensemble_{i}.pth"
        if not os.path.exists(package_path): continue
        checkpoint = torch.load(package_path, map_location=DEVICE)
        hp = checkpoint['hyperparameters']
        model = NeuralNetwork(y_train.shape[1], hp['n_neurons'], hp['n_blocks'], hp['dropout_rate'],
                              hp['cnn_out_channels'], hp['cnn_kernel_size'], **checkpoint['input_indices']).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict']);
        model.eval()

        if i == 0: first_model = model

        with torch.no_grad():
            pred = model(X_test_t_s.to(DEVICE), X_test_d_s.to(DEVICE))
            all_preds.append(inverse_log_transform(target_scaler.inverse_transform(pred.cpu().numpy())))

    if all_preds:
        ensemble_preds = np.mean(all_preds, axis=0)
        final_r2 = safe_r2_score(y_test_orig.flatten(), ensemble_preds.flatten())
        final_sign_acc = (np.sign(y_test_orig) == np.sign(ensemble_preds)).mean() * 100
        print(
            f"\n{'=' * 40}\nFinal Ensemble R²: {final_r2:.4f}\nFinal Ensemble Sign Accuracy: {final_sign_acc:.1f}%\n{'=' * 40}")

        # --- PLOTTING ---
        # 1. Accuracy Prediction
        fig_acc = plot_accuracy_prediction(y_test_orig, ensemble_preds)
        fig_acc.savefig("paper_prediction_accuracy.png", dpi=300)

        # 2. Residual Distribution
        fig_res = plot_residual_distribution(y_test_orig, ensemble_preds)
        fig_res.savefig("paper_residual_distribution.png", dpi=300)

        # 3. CDF of Error
        fig_cdf = plot_cdf_error(y_test_orig, ensemble_preds)
        fig_cdf.savefig("paper_error_cdf.png", dpi=300)

        # 4. Feature Importance
        if first_model:
            # Define feature names based on load_data construction
            feat_names = [
                "SSD_1", "SSD_2", "SSD_3",
                "GND_1", "GND_2", "GND_3",
                "AllRho_1", "AllRho_2", "AllRho_3",
                "SVR",
                "PosScrew_1", "PosScrew_2", "PosScrew_3",
                "NegScrew_1", "NegScrew_2", "NegScrew_3",
                "PosEdge_1", "PosEdge_2", "PosEdge_3",
                "NegEdge_1", "NegEdge_2", "NegEdge_3",
                "ScrewEdgeRatio_1", "ScrewEdgeRatio_2", "ScrewEdgeRatio_3",
                "RhoProd_12", "RhoProd_13", "RhoProd_23",
                "DislocLen_1", "DislocLen_2", "DislocLen_3",
                "NetScrew_1", "NetScrew_2", "NetScrew_3",
                "NetEdge_1", "NetEdge_2", "NetEdge_3",
                "ScrewAsym_1", "ScrewAsym_2", "ScrewAsym_3",
                "EdgeAsym_1", "EdgeAsym_2", "EdgeAsym_3"
            ]

            # Ensure feature names match dimension of X_test_tab
            if len(feat_names) != X_test_tab.shape[1]:
                feat_names = [f"Feat_{i}" for i in range(X_test_tab.shape[1])]

            # Compute importance using the scaled test set and scaled predictions (internal to function)
            # Note: We pass the target_scaler transformed y_test to match the model output space for R2 calc
            y_test_scaled_tensor = torch.FloatTensor(target_scaler.transform(y_test)).numpy()

            importances = compute_permutation_importance(
                first_model, X_test_t_s, X_test_d_s, y_test_scaled_tensor, feat_names, DEVICE
            )
            fig_imp = plot_feature_importance(importances)
            fig_imp.savefig("paper_feature_importance.png", dpi=300)
            print("Feature importance plot saved.")

        # 5. Stress vs Density
        print("\nPlotting Stress vs Dislocation Density...");
        plot_stress_vs_density_final(X_test_tab, y_test_orig, ensemble_preds)
        plt.show()