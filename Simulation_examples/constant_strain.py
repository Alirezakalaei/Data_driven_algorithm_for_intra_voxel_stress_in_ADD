import numpy as np
import matplotlib
from scipy.interpolate import PPoly
from scipy.stats import lognorm
import imageio.v2 as imageio
import nn_predictor  # Import the helper file
import copy  # Added for state snapshotting

# Use non-interactive backend to prevent display errors on servers/clusters
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.path import Path
import functions_rhombus as funcs  # Assuming these exist in your env
import random
import computational_functions as compute  # Assuming these exist in your env
from matplotlib.colors import LogNorm
import os
import shutil
from scipy.ndimage import gaussian_filter
from scipy.io import savemat, loadmat
import warnings

# ==============================================================================
# --- CONFIGURATION & MODEL LOADING ---
# ==============================================================================

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATHS = [
    "packaged_hybrid_model_ensemble_0.pth",
    "packaged_hybrid_model_ensemble_1.pth",
    "packaged_hybrid_model_ensemble_2.pth",
    "packaged_hybrid_model_ensemble_3.pth",
    "packaged_hybrid_model_ensemble_4.pth"
]
HYPERPARAMS_PATH = "best_hyperparameters.json"

print("--- Initializing Neural Network Predictor ---")
try:
    ensemble_models, tab_scaler, dist_scaler, target_scaler = nn_predictor.load_packaged_ensemble(
        MODEL_PATHS, HYPERPARAMS_PATH
    )
    print(f"Successfully loaded {len(ensemble_models)} models.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. {e}")
    exit()

# ==============================================================================
# --- DIRECTORY SETUP ---
# ==============================================================================
main_output_dir = "constant_strain_rate"
frames_dir = os.path.join(main_output_dir, "frames")

os.makedirs(main_output_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

print(f"Output directory set to: {main_output_dir}")

# ==============================================================================
# --- Physical Constants (UPDATED FOR COPPER) ---
# ==============================================================================
u_strain = .00025
std = np.pi / 54
# Copper Constants
mu = 48 * 10 ** 9
nu = 0.34
b = 2.56 * 10 ** -10
V_lim = 2300
n_vec = np.array([0, 0, 1])
b_vecs = np.array(
    [[1, 0, 0], [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3), 0], [np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3), 0]])
cut_off_density = 10 ** 8
dt_global =  2e-10  # Fixed small time step
cell_size = 500 * b
dz = dx = dy = ds = cell_size
mesh_size = 20000 * b
a = 1 * b

# CHANGED: Setup for Cubic Grid (Orthogonal Vectors)
mobility = 1.25 * 10 ** -5
force_direction = -.0001 * mu * np.array([1, 0, 1])/np.sqrt(2)
i_vector = cell_size * np.array([1.0, 0.0])  # Purely in X
Pierls_stress = -3 * 10 ** -5 * mu
j_vector = cell_size * np.array([0.0, 1.0])  # Purely in Y

num_cells_x = int(np.ceil(mesh_size / cell_size))
num_cells_y = int(np.ceil(mesh_size / cell_size))
nuc_size = np.random.uniform(1000 * b, 10000 * b, (num_cells_x, num_cells_y))
Nuc_crit = mu * b / nuc_size
max_nuc = 10 ** 13
pho_nuc = 10 ** 10
num_slip_planes = np.ceil((np.sqrt(3 / 2) * dz) / b)
cut_off_distance = 10000 * b
initial_density = 1*10 ** 12
pad_num = np.ceil(cut_off_distance / ds)
pho_threshold_strain = initial_density * .05

# --- CONTROL PARAMETERS (UPDATED STRATEGY) ---
target_strain_rate = 1000  # Target Strain Rate (1/s)
# Significantly reduced gain to prevent oscillation
k_p = 0.1
max_correction_iterations: int = 30  # Increased iterations to allow slower convergence

# --- Mesh and Initial Dislocation Generation ---
mesh_points = []
for m in range(num_cells_x):
    for n in range(num_cells_y):
        point = m * i_vector + n * j_vector
        mesh_points.append(point)
mesh_points = np.array(mesh_points)
XX = mesh_points[:, 0].reshape(num_cells_y, num_cells_x)
YY = mesh_points[:, 1].reshape(num_cells_y, num_cells_x)

v1 = 0 * i_vector + 0 * j_vector
v4 = num_cells_x * i_vector + 0 * j_vector
v2 = 0 * i_vector + num_cells_y * j_vector
v3 = num_cells_x * i_vector + num_cells_y * j_vector
cubic_vertices = np.array([v1, v4, v3, v2]) # Renamed for clarity

num_dislocations = 300
burger_ids = np.zeros(num_dislocations)
x_centers, y_centers = [], []
while len(x_centers) < num_dislocations:
    x_rand = np.random.uniform(np.min(XX), np.max(XX))
    y_rand = np.random.uniform(np.min(YY), np.max(YY))
    if funcs.is_inside_rhombus(x_rand, y_rand, cubic_vertices):
        x_centers.append(x_rand)
        y_centers.append(y_rand)

loops = []
num_loop_points = 360
dtheta = np.pi / (num_loop_points - 1)
theta = np.linspace(0, 2 * np.pi, num_loop_points)
radii_x = np.random.uniform(2000 * b, 10000 * b, num_dislocations)
radii_y = np.random.uniform(2000 * b, 10000 * b, num_dislocations)
for i in range(num_dislocations):
    x_loop = x_centers[i] + radii_x[i] * np.cos(theta)
    y_loop = y_centers[i] + radii_y[i] * np.sin(theta)
    z_loop = np.ones_like(x_loop) * random.randint(0, num_slip_planes - 1) * b
    burger_ids[i] = random.randint(0, b_vecs.shape[1] - 1)
    loops.append(np.column_stack((x_loop, y_loop, z_loop)))
dislocation_loops = np.array(loops)

# --- Initial Computations ---
nx, ny = XX.shape
rho_loops = funcs.compute_loop_length_density(dislocation_loops, mesh_size, mesh_size, dz, cubic_vertices)
rho_tensor, angle_tensor = funcs.compute_dislocation_density_and_tangent(
    dislocation_loops, 3 * cell_size, nx, ny, XX, YY)
funcs.scale_rho_tensor(rho_tensor, rho_loops, nx, ny)
QQ = funcs.Coarse_grainer(rho_tensor, angle_tensor, theta, std, b_vecs.shape[1], burger_ids)
PP = compute.local_max_python(1, QQ, cut_off_density)
PP = np.array(PP)
for i in range(QQ.shape[0]):
    QQ[i, :, :, :] = QQ[i, :, :, :] * initial_density * nx * ny / np.sum(PP[i, :, :, :])
dV = cell_size * cell_size * dz
epsilon = funcs.levi_civita()
C = funcs.elastic_tensor(mu, nu)
GND, x_i = funcs.compute_gnd_and_xi(PP, theta)

RSS = funcs.compute_rss(n_vec, b_vecs, force_direction)
print(f"Initial Resolved Shear Stresses (RSS): {RSS}")

# --- Setup for Plotting ---
norm = matplotlib.colors.Normalize(vmin=0, vmax=1e13)
# CHANGED: Added 3D projection
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
fig.colorbar(sm, ax=ax, label=r"Dislocation Density ($\mathrm{m}^{-2}$)")

# --- nucleation requirements ---
nucleation_angles = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
num_dist = int(np.ceil(num_loop_points * (5 * std / (2 * np.pi))))
up = np.zeros(num_loop_points)
left = np.zeros(num_loop_points)
down = np.zeros(num_loop_points)
right = np.zeros(num_loop_points)

for rot in range(num_dist):
    up_idx = int(np.floor(num_loop_points * (nucleation_angles[0] / (2 * np.pi))))
    left_idx = int(np.floor(num_loop_points * (nucleation_angles[1] / (2 * np.pi))))
    down_idx = int(np.floor(num_loop_points * (nucleation_angles[2] / (2 * np.pi))))
    right_idx = int(np.floor(num_loop_points * (nucleation_angles[3] / (2 * np.pi))))
    up[(up_idx + rot) % num_loop_points] += np.exp(-0.5 * (3 * rot / num_dist) ** 2)
    left[(left_idx + rot) % num_loop_points] += np.exp(-0.5 * (3 * rot / num_dist) ** 2)
    down[(down_idx + rot) % num_loop_points] += np.exp(-0.5 * (3 * rot / num_dist) ** 2)
    right[(right_idx + rot) % num_loop_points] += np.exp(-0.5 * (3 * rot / num_dist) ** 2)
    if rot != 0:
        up[(up_idx - rot) % num_loop_points] += np.exp(-0.5 * (3 * rot / num_dist) ** 2)
        left[(left_idx - rot) % num_loop_points] += np.exp(-0.5 * (3 * rot / num_dist) ** 2)
        down[(down_idx - rot) % num_loop_points] += np.exp(-0.5 * (3 * rot / num_dist) ** 2)
        right[(right_idx - rot) % num_loop_points] += np.exp(-0.5 * (3 * rot / num_dist) ** 2)
up = up / np.max(up)
down = down / np.max(down)
left = left / np.max(left)
right = right / np.max(right)

print("\nStarting Constant Strain Rate Simulation (Material: Copper)...")

# --- DYNAMICS SIMULATION LOOP ---
cos_thetaV = np.cos(theta - (np.pi / 2))
sin_thetaV = np.sin(theta - (np.pi / 2))

anni_amount_list = [[] for _ in range(b_vecs.shape[1])]
nuc_amount_list = [[] for _ in range(b_vecs.shape[1])]
density_amount_list = [[] for _ in range(b_vecs.shape[1])]
strain_Von_list = []
pp_sum_list = []

# History lists
time_history = []
total_density_history = []
strain_history = []
avg_velocity_history = []
rss_history = []
kp_correction_history = []

strain = np.zeros((3, 3))
V_all = np.zeros((b_vecs.shape[1], nx, ny))
CFL = 0.5
time_step = 0
current_strain_von = 0
current_time = 0

while current_strain_von < u_strain:

    # ==========================================================================
    # --- FEEDBACK CONTROL LOOP START ---
    # ==========================================================================

    state_snapshot = {
        'QQ': QQ.copy(),
        'strain': strain.copy(),
        'anni_amount_list': copy.deepcopy(anni_amount_list),
        'nuc_amount_list': copy.deepcopy(nuc_amount_list),
        'V_all': V_all.copy(),
        'GND': GND.copy()
    }

    correction_iter = 0
    strain_rate_ok = False
    calculated_strain_rate = 0

    # Adaptive Gain Variables
    current_kp = k_p
    last_error_sign = 0  # To detect oscillation

    while not strain_rate_ok and correction_iter < max_correction_iterations:
        correction_iter += 1

        if correction_iter > 1:
            QQ = state_snapshot['QQ'].copy()
            strain = state_snapshot['strain'].copy()
            anni_amount_list = copy.deepcopy(state_snapshot['anni_amount_list'])
            nuc_amount_list = copy.deepcopy(state_snapshot['nuc_amount_list'])
            V_all = state_snapshot['V_all'].copy()
            GND = state_snapshot['GND'].copy()

        # --- STEP 1: Predict Stress ---
        QQ_for_nn = compute.local_max_python(1, QQ, cut_off_density)
        tau_nn = nn_predictor.predict_stress(
            QQ_for_nn, GND, b_vecs, cell_size, dz,
            ensemble_models, tab_scaler, dist_scaler, target_scaler
        )
        tau_nn = -np.abs(tau_nn)
        #for s in range(tau_nn.shape[0]):
         #   tau_nn[s] = gaussian_filter(tau_nn[s], sigma=1.0)

        tau_alpha_int = funcs.compute_stress(XX, YY, GND, x_i, PP, n_vec, mu, nu, a, C, dV, b_vecs, epsilon, np.eye(3),
                                             cut_off_density, int(pad_num), cut_off_distance, i_vector, j_vector) * b

        T_effective_all = (tau_nn * mu) + RSS[:, np.newaxis, np.newaxis] + np.transpose(tau_alpha_int, (2, 0, 1))

        for iii in range(T_effective_all.shape[0]):
            if RSS[iii - 1] == 0:
                T_effective_all[iii - 1, :, :] = np.zeros((num_cells_x, num_cells_y))
        #T_effective_all = np.maximum(T_effective_all, 0)

        # --- STEP 2: Velocity Calculation ---
        V_all_temp = mobility * T_effective_all
        V_all_temp[T_effective_all < np.abs(Pierls_stress)] = 0
        V_all_temp = np.clip(V_all_temp, -V_lim, V_lim)
        V_all_temp = np.clip(V_all_temp, -CFL * dx / dt_global, CFL * dx / dt_global)
        V_strain= V_all_temp
        # --- STEP 3: Physics Update ---
        PP = compute.local_max_python(1, QQ, cut_off_density)
        PP = np.array(PP)

        for slip in range(b_vecs.shape[1]):
            V = V_all_temp[slip, :, :]
            V= gaussian_filter(V, sigma= 1)
            last_anni = anni_amount_list[slip][-1] if anni_amount_list[slip] else 0
            last_nuc = nuc_amount_list[slip][-1] if nuc_amount_list[slip] else 0

            if np.all(V == 0):
                anni_amount_list[slip].append(last_anni)
                nuc_amount_list[slip].append(last_nuc)
                continue
            V_all[slip, :, :] = V
            # Update Density
            QQ[slip, :, :, :] = compute.difff7_central(QQ[slip, :, :, :], dt_global, cos_thetaV, sin_thetaV, V / ds)
            QQ[QQ < 0] = 0

            V_gradient = compute.theta_diff2_python(QQ[slip, :, :, :], V, cut_off_density, theta, ds)
            V_gradient = np.clip(V_gradient, -CFL * dtheta / dt_global, CFL * dtheta / dt_global)
            V_gradient = gaussian_filter(V_gradient, sigma=(1, 1, 0))
            QQ[slip, :, :, :] = compute.thetadiff(QQ[slip, :, :, :], dt_global, V_gradient, dtheta)
            QQ[QQ < 0] = 0

            # CHANGED: Removed np.sqrt(3) scaling on dy for the cubic grid
            new_anni, current_anni_val = compute.annihilation_python(QQ[slip, :, :, :], V, dx, dy,
                                                                     dt_global, b, last_anni)
            anni_amount_list[slip].append(current_anni_val)
            QQ[slip, :, :, :] = new_anni

            new_nuc, current_nuc_val = compute.nucleation_python(QQ[slip, :, :, :], T_effective_all[slip], Nuc_crit,
                                                                 max_nuc, pho_nuc, dt_global, V, ds, 1.2 * nuc_size,
                                                                 PP[slip, :, :, :], last_nuc, up, left, down, right)
            nuc_amount_list[slip].append(current_nuc_val)
            QQ[slip, :, :, :] += new_nuc

        # --- STEP 4: Calculate Strain Rate ---
        PP = compute.local_max_python(1, QQ, cut_off_density)
        PP = np.array(PP)
        GND, x_i = funcs.compute_gnd_and_xi(PP, theta)

        new_strain = compute.strain_finder_python(strain, GND, V_strain, b_vecs, n_vec, dt_global, b, num_slip_planes,
                                                      pho_threshold_strain, target_strain_rate * .1)
        strain_increment_tensor = new_strain - state_snapshot['strain']
        dev_inc = strain_increment_tensor - np.trace(strain_increment_tensor) / 3 * np.eye(3)

            # Calculate standard von Mises magnitude (always positive)
        strain_increment_vm = np.sqrt(3 / 2 * np.sum(dev_inc ** 2))

            # --- NEW FIX: Restore the negative sign if strain is reversing ---
            # Calculate the plastic work direction (Applied Stress * Velocity)
        work_increment = np.sum(RSS[:, np.newaxis, np.newaxis] * V_all)
        if work_increment < 0:
            strain_increment_vm = -strain_increment_vm  # Strain is going backwards!

        calculated_strain_rate = strain_increment_vm / dt_global
        ratio = calculated_strain_rate / target_strain_rate

        # Wider tolerance to prevent fighting over noise
        if (0.85 <= ratio <= 1.15) or (calculated_strain_rate > 0 and correction_iter == max_correction_iterations):
            strain_rate_ok = True
            strain = new_strain
            if correction_iter == max_correction_iterations:
                print(f"Warning: Max correction reached. Ratio: {ratio:.3f}. Proceeding...")
        else:
            # --- DAMPED FEEDBACK CORRECTION ---
            error = target_strain_rate - calculated_strain_rate
            current_error_sign = np.sign(error)

            # If error flips sign (oscillation), cut gain in half immediately
            if last_error_sign != 0 and current_error_sign != last_error_sign:
                current_kp *= 0.5
                print(f"  -> Oscillation detected. Reducing gain to {current_kp:.4f}")

            last_error_sign = current_error_sign

            # Calculate correction factor (P-Control)
            # Correction = 1 + kp * (relative_error)
            correction_factor = 1 + current_kp * (error / target_strain_rate)

            # Hard Clamp to prevent explosions (0.8x to 1.2x max change per iter)
            correction_factor = max(min(correction_factor, 1.1), 0.9)

            print(f"  Iter {correction_iter}: Ratio {ratio:.3f}, Correction {correction_factor:.4f}")

            kp_correction_history.append(correction_factor)


            RSS = RSS * correction_factor

    # ==========================================================================
    # --- END FEEDBACK LOOP ---
    # ==========================================================================

    current_time += dt_global
    current_densities = np.sum(PP, axis=(1, 2, 3))
    for slip in range(b_vecs.shape[1]):
        density_amount_list[slip].append(current_densities[slip])

    current_strain_von = np.sqrt(3 / 2 * np.sum(((strain - np.trace(strain) / 3 * np.eye(3)) ** 2)))
    strain_Von_list.append(current_strain_von)
    pp_sum_list.append(np.sum(PP))

    time_history.append(current_time)
    total_density_history.append(np.sum(current_densities))
    strain_history.append(current_strain_von)
    avg_velocity_history.append(np.mean(np.abs(V_all)))
    rss_history.append(np.mean(RSS[RSS > 0]))

    QQ[QQ > 1e15] = 1e15
    QQ[QQ<0]=0
    time_step += 1

    if time_step % 1 == 0:
        ax.clear()
        slip_to_visualize = 0
        density_map = np.sum(PP[slip_to_visualize, :, :, :], axis=2)

        # CHANGED: Replaced contourf with plot_surface for "shading interp"
        surf = ax.plot_surface(XX, YY, density_map, cmap='viridis', norm=norm,
                               edgecolor='none', antialiased=True, shade=False)

        # CHANGED: Set 2D top-down view
        ax.view_init(elev=90, azim=-90)

        # CHANGED: Hide the Z-axis entirely
        ax.set_zticks([])
        ax.zaxis.set_visible(False)

        ax.set_title(
            f'Dislocation Density (Cu) - Step: {time_step}\nRate: {calculated_strain_rate:.2e} / Target: {target_strain_rate:.2e}')
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")

        frame_filename = os.path.join(frames_dir, f"frame_{time_step:04d}.png")
        fig.savefig(frame_filename, dpi=150)

    print(
        f"Step: {time_step}, Strain: {current_strain_von:.4e}, Rate: {calculated_strain_rate:.2e}, RSS: {np.mean(RSS):.2e}")

# --- Post-processing ---
print("\nSimulation finished. Assembling video...")

output_video_file = os.path.join(main_output_dir, 'dislocation_density_animation_Cu.gif')
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])

if frame_files:
    try:
        with imageio.get_writer(output_video_file, mode='I', duration=0.1) as writer:
            for filename in frame_files:
                image = imageio.imread(filename)
                writer.append_data(image)
        print(f"\nAnimation '{output_video_file}' saved successfully!")
        shutil.rmtree(frames_dir)
    except Exception as e:
        print(f"Video creation failed: {e}")
else:
    print("No frames were generated.")
plt.close(fig)

# --- PLOTTING METRICS ---
print("Generating metrics plots...")
fig_metrics, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].plot(time_history, total_density_history, 'b-', linewidth=2)
axs[0, 0].set_title("Total Dislocation Density vs Time")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel(r"Density ($m^{-2}$)")
axs[0, 0].grid(True)

axs[0, 1].plot(time_history, strain_history, 'r-', linewidth=2)
axs[0, 1].set_title("Strain vs Time")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Strain")
axs[0, 1].grid(True)

axs[1, 0].plot(strain_history, rss_history, 'k-', linewidth=2)
axs[1, 0].set_title("Stress vs Strain (Response)")
axs[1, 0].set_xlabel("Strain")
axs[1, 0].set_ylabel("Applied Stress (Pa)")
axs[1, 0].grid(True)

axs[1, 1].plot(time_history, avg_velocity_history, 'g-', linewidth=2)
axs[1, 1].set_title("Average Dislocation Velocity vs Time")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Velocity (m/s)")
axs[1, 1].grid(True)

plt.tight_layout()
metrics_filename = os.path.join(main_output_dir, "simulation_metrics.png")
plt.savefig(metrics_filename, dpi=300)
print(f"Metrics plot saved to '{metrics_filename}'")

# --- SAVING DATA ---
anni_amount = np.array(anni_amount_list) / (num_cells_x * num_cells_y)
nuc_amount = np.array(nuc_amount_list) / (num_cells_x * num_cells_y)
density_amount = np.array(density_amount_list) / (num_cells_x * num_cells_y)
strain_Von = np.array(strain_Von_list)
pp_sum_history = np.array(pp_sum_list)

variables_to_save = {
    "QQ": QQ, "PP": PP, "V_all": V_all, "strain_all": strain_Von, "RSS": RSS, "GND": GND,
    "XX": XX, "YY": YY, "nuc_amount": nuc_amount, "ani_amount": anni_amount,
    "cell_size": cell_size, "dz": dz, "initial_density": initial_density,
    "num_slip_planes": num_slip_planes, "density_list": density_amount,
    "simulation_size": mesh_size,
    "pp_sum_history": pp_sum_history,
    "time_history": np.array(time_history),
    "total_density_history": np.array(total_density_history),
    "strain_history": np.array(strain_history),
    "avg_velocity_history": np.array(avg_velocity_history),
    "rss_history": np.array(rss_history),
    "kp_correction_history": np.array(kp_correction_history)
}

mat_filename = os.path.join(main_output_dir, "all_variables_Cu.mat")
savemat(mat_filename, variables_to_save)
print(f"Variables saved to {mat_filename}")