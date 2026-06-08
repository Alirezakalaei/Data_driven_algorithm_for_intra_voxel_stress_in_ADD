import numpy as np
import matplotlib
from scipy.interpolate import PPoly
from scipy.stats import lognorm
import imageio.v2 as imageio
import nn_predictor  # Import the helper file

# Use non-interactive backend to prevent display errors on servers/clusters
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.path import Path
import functions_rhombus as funcs  # Assuming these exist in your env
import random
import computational_functions as compute  # Assuming these exist in your env
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # Added for 3D plotting
import os
import shutil  # Added for deleting folders
from scipy.ndimage import gaussian_filter
from scipy.io import savemat, loadmat
import warnings

# ==============================================================================
# --- CONFIGURATION & MODEL LOADING ---
# ==============================================================================

# Suppress specific sklearn version warnings to clean up output
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
    # Load the ensemble using the JSON config for architecture and .pth for weights
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
main_output_dir = "constant_stress"
frames_dir = os.path.join(main_output_dir, "frames")

# Create directories if they don't exist
os.makedirs(main_output_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

print(f"Output directory set to: {main_output_dir}")
print(f"Frames directory set to: {frames_dir}")

# ==============================================================================

# --- Physical Constants (UPDATED FOR COPPER) ---
U_time = 1e-7
std = np.pi / 54

# Copper Constants
mu = 48 * 10 ** 9  # Shear Modulus for Copper (Pa)
nu = 0.34  # Poisson's ratio for Copper
b = 2.56 * 10 ** -10  # Burgers vector for Copper (m)
V_lim = 2300

n_vec = np.array([0, 0, 1])
b_vecs = np.array(
    [[1, 0, 0], [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3), 0], [np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3), 0]])
cut_off_density = 10 ** 8

cell_size = 500 * b
dz = dx = dy = ds = cell_size
mesh_size = 40000 * b
a = 1 * b

mobility = 1.25 * 10 ** -5
force_direction = -.002 * mu * np.array([1, 0, 1]) / np.sqrt(2)

# --- MODIFIED FOR CUBIC MESH ---
i_vector = cell_size * np.array([1.0, 0.0])
j_vector = cell_size * np.array([0.0, 1.0])
# -------------------------------

Pierls_stress = 3 * 10 ** -5 * mu
num_cells_x = int(np.ceil(mesh_size / cell_size))
num_cells_y = int(np.ceil(mesh_size / cell_size))
nuc_size = np.random.uniform(1000 * b, 10000 * b, (num_cells_x, num_cells_y))
Nuc_crit = mu * b / nuc_size
max_nuc = 10 ** 13
pho_nuc = 10 ** 10
num_slip_planes = np.ceil((np.sqrt(3 / 2) * dz) / b)
cut_off_distance = 10000 * b
initial_density = 10 ** 12
pad_num = np.ceil(cut_off_distance / ds)
threshold_s_d = .05 * initial_density

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
rhombus_vertices = np.array([v1, v4, v3, v2])  # Now represents a square/rectangle

num_dislocations = 300
burger_ids = np.zeros(num_dislocations)
x_centers, y_centers = [], []
while len(x_centers) < num_dislocations:
    x_rand = np.random.uniform(np.min(XX), np.max(XX))
    y_rand = np.random.uniform(np.min(YY), np.max(YY))
    # Using the same function, but vertices now form a square
    if funcs.is_inside_rhombus(x_rand, y_rand, rhombus_vertices):
        x_centers.append(x_rand)
        y_centers.append(y_rand)

loops = []
num_loop_points = 360
dtheta = np.pi / (num_loop_points - 1)
theta = np.linspace(0, 2 * np.pi, num_loop_points)

# NEW: Define theta_normal as theta + 90 degrees (pi/2) for the 4th dimension
theta_normal = theta - (np.pi / 2)

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
rho_loops = funcs.compute_loop_length_density(dislocation_loops, mesh_size, mesh_size, dz, rhombus_vertices)
rho_tensor, angle_tensor = funcs.compute_dislocation_density_and_tangent(
    dislocation_loops, 3 * cell_size, nx, ny, XX, YY)
funcs.scale_rho_tensor(rho_tensor, rho_loops, nx, ny)

# CHANGED: Pass theta_normal to Coarse_grainer so QQ's 4th dimension is the normal angle
QQ = funcs.Coarse_grainer(rho_tensor, angle_tensor, theta_normal, std, b_vecs.shape[1], burger_ids)
PP = compute.local_max_python(1, QQ, cut_off_density)
PP = np.array(PP)
for i in range(QQ.shape[0]):
    QQ[i, :, :, :] = QQ[i, :, :, :] * initial_density * nx * ny / np.sum(PP[i, :, :, :])
dV = cell_size * cell_size * dz
epsilon = funcs.levi_civita()
C = funcs.elastic_tensor(mu, nu)

# CHANGED: Pass theta_normal to GND computation
GND, x_i = funcs.compute_gnd_and_xi(PP, theta)
RSS = funcs.compute_rss(n_vec, b_vecs, force_direction)
print(f"Resolved Shear Stresses (RSS): {RSS}")

# --- Setup for Plotting ---
norm = LogNorm(vmin=1e10, vmax=1e13, clip=True)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
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

print("\nStarting dynamics simulation (Material: Copper)...")

# --- DYNAMICS SIMULATION LOOP ---
# CHANGED: Since theta_normal already represents the normal direction, we use it directly for velocity components
cos_thetaV = np.cos(theta_normal)
sin_thetaV = np.sin(theta_normal)

anni_amount_list = [[] for _ in range(b_vecs.shape[1])]
nuc_amount_list = [[] for _ in range(b_vecs.shape[1])]
density_amount_list = [[] for _ in range(b_vecs.shape[1])]
strain_Von_list = []
pp_sum_list = []

# NEW: History lists for plotting
time_history = []
total_density_history = []
strain_history = []
avg_velocity_history = []

strain = np.zeros((3, 3))
cumulative_signed_vm_strain = 0.0  # NEW: Variable to track signed equivalent strain
V_all = np.zeros((b_vecs.shape[1], nx, ny))
CFL = 0.5
time_step = 0
current_strain_von = 0
current_time = 0

while current_time < U_time:

    # ==============================================================================
    # --- STEP 1: Predict Stress using Neural Network ---
    # ==============================================================================

    # Ensure QQ is non-negative before log1p inside predictor
    QQ_for_nn = compute.local_max_python(1, QQ, cut_off_density)

    # 1. Prediction (Normalized)
    tau_nn = nn_predictor.predict_stress(
        QQ_for_nn, GND, b_vecs, cell_size, dz,
        ensemble_models, tab_scaler, dist_scaler, target_scaler
    )

    tau_nn = np.minimum(tau_nn, 0)  # Back stress constraint
    # for s in range(tau_nn.shape[0]):
    # tau_nn[s] = gaussian_filter(tau_nn[s], sigma=1.0)
    # ==============================================================================

    # --- STEP 2: Analytical Stress Calculation (Long-range) ---
    tau_alpha_int = funcs.compute_stress(XX, YY, GND, x_i, PP, n_vec, mu, nu, a, C, dV, b_vecs, epsilon, np.eye(3),
                                         cut_off_density, int(pad_num), cut_off_distance, i_vector, j_vector) * b

    # --- STEP 3: Combine Stresses to get Effective Stress ---
    T_effective_all = (tau_nn * mu) + RSS[:, np.newaxis, np.newaxis] + np.transpose(tau_alpha_int, (2, 0, 1))
    for iii in range(T_effective_all.shape[0]):
        if RSS[iii - 1] == 0:
            T_effective_all[iii - 1, :, :] = np.zeros((num_cells_x, num_cells_y))

    # --- STEP 4: Velocity and Dynamics Update ---
    V_all_temp = mobility * T_effective_all

    V_all_temp[T_effective_all < np.abs(Pierls_stress)] = 0

    # Sanitize Velocity before calculating dt
    V_all_temp = np.clip(V_all_temp, -V_lim, V_lim)
    V_strain = V_all_temp


    max_vel = np.max(np.abs(V_all_temp))

    # Safety check for Infinite velocity or Zero velocity
    if max_vel > 0:
        dt = CFL * ds / max_vel
    else:
        dt = 1e-9

    # Prevent dt from becoming numerically zero
    if dt < 1e-15:
        dt = 1e-15

    current_time += dt

    # Re-calculate local max for nucleation/annihilation logic
    PP = compute.local_max_python(1, QQ, cut_off_density)
    PP = np.array(PP)

    pp_sum_list.append(np.sum(PP))

    for slip in range(b_vecs.shape[1]):
        # IMPORTANT: Use T_effective for everything
        T_effective = T_effective_all[slip, :, :]
        V = V_all_temp[slip, :, :]
        V = gaussian_filter(V, sigma=1)
        last_anni = anni_amount_list[slip][-1] if anni_amount_list[slip] else 0
        last_nuc = nuc_amount_list[slip][-1] if nuc_amount_list[slip] else 0

        if np.all(V == 0):
            anni_amount_list[slip].append(last_anni)
            nuc_amount_list[slip].append(last_nuc)
            continue

        V_all[slip, :, :] = V

        # Update Density (Flux) - Using Upwind Scheme
        QQ[slip, :, :, :] = compute.difff7_central(QQ[slip, :, :, :], dt, cos_thetaV, sin_thetaV, V / ds)

        # Update Density (Rotation)
        V_gradient = compute.theta_diff2_python(QQ[slip, :, :, :], V, cut_off_density, theta, ds)
        V_gradient = gaussian_filter(V_gradient, sigma=(1,1,0))
        V_gradient = np.clip(V_gradient, -CFL * dtheta / dt, CFL * dtheta / dt)
        QQ[slip, :, :, :] = compute.thetadiff(QQ[slip, :, :, :], dt, V_gradient, dtheta)

        # Annihilation (MODIFIED FOR CUBIC MESH: Removed np.sqrt(3) scaling on dy)
        new_anni, current_anni_val = compute.annihilation_python(QQ[slip, :, :, :], V, dx, dy, dt, b,
                                                                 last_anni)
        anni_amount_list[slip].append(current_anni_val)
        QQ[slip, :, :, :] = new_anni

        # Nucleation (Uses T_effective)
        new_nuc, current_nuc_val = compute.nucleation_python(QQ[slip, :, :, :], T_effective, Nuc_crit,
                                                             max_nuc, pho_nuc, dt, V, ds, 1.2 * nuc_size,
                                                             PP[slip, :, :, :], last_nuc, up, left, down, right)
        nuc_amount_list[slip].append(current_nuc_val)
        QQ[slip, :, :, :] += new_nuc

    # Update Strain and Density Metrics
    PP = compute.local_max_python(1, QQ, cut_off_density)
    PP = np.array(PP)
    current_densities = np.sum(PP, axis=(1, 2, 3))
    for slip in range(b_vecs.shape[1]):
        density_amount_list[slip].append(current_densities[slip])

    # CHANGED: Pass theta_normal to GND calculations
    GND, x_i = funcs.compute_gnd_and_xi(PP, theta)

    # --- NEW: Signed Strain Calculation Logic ---
    old_strain = strain.copy()  # Save previous strain state

    strain = compute.strain_finder_python(strain, GND, V_strain, b_vecs, n_vec, dt, b, num_slip_planes, threshold_s_d,
                                          1e15)

    # Calculate the increment of the strain tensor
    strain_increment_tensor = strain - old_strain
    dev_inc = strain_increment_tensor - np.trace(strain_increment_tensor) / 3 * np.eye(3)

    # Calculate standard von Mises magnitude of the increment (always positive)
    strain_increment_vm = np.sqrt(3 / 2 * np.sum(dev_inc ** 2))

    # Check the direction of plastic work
    work_increment = np.sum(RSS[:, np.newaxis, np.newaxis] * V_all)
    if work_increment < 0:
        strain_increment_vm = -strain_increment_vm  # Strain is going backwards!

    # Accumulate the signed strain
    cumulative_signed_vm_strain += strain_increment_vm
    current_strain_von = cumulative_signed_vm_strain

    strain_Von_list.append(current_strain_von)
    # --------------------------------------------

    # --- RECORD METRICS FOR PLOTTING ---
    time_history.append(current_time)
    total_density_history.append(np.sum(current_densities))
    strain_history.append(current_strain_von)
    avg_velocity_history.append(np.mean(np.abs(V_all)))

    # Cap density to prevent numerical explosion
    QQ[QQ > 1e15] = 1e15
    time_step += 1

    # Visualization every step
    if time_step % 1 == 0:
        ax.clear()
        slip_to_visualize = 0
        density_map = np.sum(PP[slip_to_visualize, :, :, :], axis=2)

        # Clip the density map strictly between 10^10 and 10^13 for plotting
        density_map_clipped = np.clip(density_map, 1e10, 1e13)

        # Plot surface instead of contourf
        surf = ax.plot_surface(XX, YY, density_map_clipped, cmap='viridis', norm=norm, edgecolor='none', shade=False)
        ax.view_init(elev=90, azim=-90)
        # CHANGED: Hide the Z-axis entirely
        ax.set_zticks([])
        ax.zaxis.set_visible(False)
        ax.set_title(f'Dislocation Density (Cu) - Time Step: {time_step}')
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.set_zlabel(r"Density ($\mathrm{m}^{-2}$)")

        # Save to the specific frames directory inside constant_stress
        frame_filename = os.path.join(frames_dir, f"frame_{time_step:04d}.png")
        fig.savefig(frame_filename, dpi=150)

    print(f"Time step: {time_step}, Von Mises Strain: {current_strain_von:.4e}, dt: {dt:.4e} s")

# --- Post-processing ---
print("\nSimulation finished. Assembling video...")

# Save GIF in the main directory 'constant_stress'
output_video_file = os.path.join(main_output_dir, 'dislocation_density_animation_Cu.gif')
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])

if frame_files:
    try:
        with imageio.get_writer(output_video_file, mode='I', duration=0.1) as writer:
            for filename in frame_files:
                image = imageio.imread(filename)
                writer.append_data(image)
        print(f"\nAnimation '{output_video_file}' saved successfully!")

        # --- DELETE FRAMES AFTER SUCCESSFUL GIF CREATION ---
        print(f"Cleaning up frames directory: {frames_dir}")
        shutil.rmtree(frames_dir)
        print("Frames directory deleted.")

    except Exception as e:
        print(f"Video creation failed: {e}")
        print(f"Frames are preserved in '{frames_dir}' folder.")
else:
    print("No frames were generated.")
plt.close(fig)

# --- PLOTTING METRICS ---
print("Generating metrics plots...")
fig_metrics, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Total Density vs Time
axs[0].plot(time_history, total_density_history, 'b-', linewidth=2)
axs[0].set_title("Total Dislocation Density vs Time")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel(r"Density ($m^{-2}$)")
axs[0].grid(True)

# 2. Strain vs Time (Creep Curve)
axs[1].plot(time_history, strain_history, 'r-', linewidth=2)
axs[1].set_title("Strain vs Time (Creep)")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Strain")
axs[1].grid(True)

# 3. Average Velocity vs Time
axs[2].plot(time_history, avg_velocity_history, 'g-', linewidth=2)
axs[2].set_title("Average Dislocation Velocity vs Time")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Velocity (m/s)")
axs[2].grid(True)

plt.tight_layout()
# Save metrics plot to main directory
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
    # Saved variables
    "time_history": np.array(time_history),
    "total_density_history": np.array(total_density_history),
    "strain_history": np.array(strain_history),
    "avg_velocity_history": np.array(avg_velocity_history)
}

# Save .mat file to main directory
mat_filename = os.path.join(main_output_dir, "all_variables_Cu.mat")
savemat(mat_filename, variables_to_save)
print(f"Variables saved to {mat_filename}")