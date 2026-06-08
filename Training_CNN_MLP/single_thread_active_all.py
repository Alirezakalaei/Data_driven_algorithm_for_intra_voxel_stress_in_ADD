#!/usr/bin/env python3
# ========================================================================
#    single_thread_active_all.py (Modified: In-Plane Rotated Octagons)
# ========================================================================
import os
import argparse
import time
import sys
import traceback
import numpy as np
import matplotlib
from numba.typed import List

# Set backend to avoid display errors on headless systems
try:
    matplotlib.use('TKAgg')
except:
    pass

# -------------------------------------------------
# CRITICAL: Set Environment Variables for Performance
# -------------------------------------------------
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

# Import library AFTER setting environment variables
try:
    import density_extraction as density_extraction
    from density_extraction import elastic_tensor, compute_internal_tau_2d
except ImportError:
    print("Warning: 'density_extraction' module not found.", file=sys.stderr)
    sys.exit(1)

# -------------------------------------------------
# Material constants (FCC Copper)
# -------------------------------------------------
a = 3.615e-10
b = a / np.sqrt(2)
sin_remov = 3 * b
real_sin_remove = 0
dz111 = a / np.sqrt(3)
dz = dz111

# The 3 Burgers vectors
b_vecs = np.array([
    [1.0, 0.0, 0.0],
    [np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0],
    [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3), 0.0]
], dtype=float)

# Slip plane normal vector (normal to XY plane)
n_vec = np.array([0.0, 0.0, 1.0])
pho_lim = 1e7


# -------------------------------------------------
# Geometric Helper Functions
# -------------------------------------------------

def spherical_to_cartesian(theta, phi):
    sin_phi = np.sin(phi)
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])


def get_perpendicular_vector(vec):
    norm_vec = np.linalg.norm(vec)
    if norm_vec < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    vec = vec / norm_vec
    if abs(np.dot(vec, np.array([1.0, 0.0, 0.0]))) < 0.99:
        other_vec = np.array([1.0, 0.0, 0.0])
    else:
        other_vec = np.array([0.0, 1.0, 0.0])
    perp = np.cross(vec, other_vec)
    return perp / np.linalg.norm(perp)


def interpolate_segment(p_start, p_end, pts_per_edge=40):
    xs = np.linspace(p_start[0], p_end[0], pts_per_edge)
    ys = np.linspace(p_start[1], p_end[1], pts_per_edge)
    zs = np.linspace(p_start[2], p_end[2], pts_per_edge)
    return np.column_stack((xs, ys, zs))


def get_boundary_crossing_segment(center, direction, Lx, Ly, extension_factor=2.0):
    d = direction / np.linalg.norm(direction)
    d[np.abs(d) < 1e-12] = 1e-12
    extension_dist = (Lx + Ly) * extension_factor
    p_start = center - extension_dist * d
    p_end = center + extension_dist * d
    return p_start, p_end


def clip_segment_to_box(p_start, p_end, Lx, Ly):
    t_enter, t_exit = 0.0, 1.0
    direction = p_end - p_start
    for i in range(2):  # 0 for X, 1 for Y
        dim_max = [Lx, Ly][i]
        if np.isclose(direction[i], 0):
            if p_start[i] < 0 or p_start[i] > dim_max:
                return None, None
            continue
        t1 = (0 - p_start[i]) / direction[i]
        t2 = (dim_max - p_start[i]) / direction[i]
        if t1 > t2: t1, t2 = t2, t1
        t_enter = max(t_enter, t1)
        t_exit = min(t_exit, t2)
        if t_enter > t_exit: return None, None
    p_clip_start = p_start + t_enter * direction
    p_clip_end = p_start + t_exit * direction
    return p_clip_start, p_clip_end


def correct_loop_winding_orders(dislocation_loops, verbose=False):
    corrected_loops = []
    for i, loop in enumerate(dislocation_loops):
        is_closed = loop.shape[0] >= 4 and np.allclose(loop[0], loop[-1])
        if not is_closed:
            corrected_loops.append(loop)
            continue
        points_2d = loop[:, :2]
        area = 0.0
        for j in range(len(points_2d) - 1):
            p1 = points_2d[j]
            p2 = points_2d[j + 1]
            area += (p1[0] * p2[1] - p2[0] * p1[1])
        area *= 0.5
        if area < 0:
            corrected_loops.append(np.ascontiguousarray(np.flip(loop, axis=0)))
        else:
            corrected_loops.append(loop)
    return corrected_loops


# -------------------------------------------------
# Core Generation Logic
# -------------------------------------------------

def make_sample(rng: np.random.Generator):
    L = rng.uniform(400, 600) * b
    Lx = Ly = Lz = L
    V = Lx * Ly * Lz

    mu, nu = 48e9, 0.34
    C = elastic_tensor(mu, nu)
    epsilon = np.zeros((3, 3, 3), int)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1
    delta = np.eye(3)

    loops = []
    Z_list = []
    burger_ID_list = []
    discrete_lines_x_list = []
    discrete_lines_y_list = []

    num_slip_systems = b_vecs.shape[0]

    low_density_bound = 10
    high_density_bound = 15
    target_densities = [10 ** rng.uniform(low_density_bound, high_density_bound) for _ in range(num_slip_systems)]

    MAX_FEATURES_PER_SYSTEM = 50

    for system_id in range(num_slip_systems):
        target_rho = target_densities[system_id]
        current_rho = 0.0
        feature_count = 0

        while feature_count < MAX_FEATURES_PER_SYSTEM:
            feature_loops = []
            feature_zs = []
            center = np.array([rng.uniform(-0.2 * Lx, 1.2 * Lx),
                               rng.uniform(-0.2 * Ly, 1.2 * Ly),
                               rng.uniform(0 * Lz, 1 * Lz)])

            num_rings = rng.integers(3, 6)
            min_radius = 80 * b
            max_radius = 200 * b

            # ====================================================================
            # >>>>>>>> MODIFICATION FOR IN-PLANE ROTATED OCTAGONS <<<<<<<<
            # This block replaces the previous 3D orientation logic.
            # ====================================================================

            # 1. Force the loop plane to be parallel to the XY plane.
            #    The normal vector is fixed to point along the Z-axis.
            normal = np.array([0.0, 0.0, 1.0])

            # 2. Choose a random in-plane rotation so loops are not always axis-aligned.
            angle_offset = rng.uniform(0, 2 * np.pi)  # 0 to 360 degrees
            u = np.array([np.cos(angle_offset), np.sin(angle_offset), 0.0])  # Rotated local x-axis
            v = np.array([-np.sin(angle_offset), np.cos(angle_offset), 0.0]) # Rotated local y-axis

            # 3. Generate radii with a constrained aspect ratio to prevent extreme squeezing.
            #    This logic is functionally the same as before but uses a more concise style.
            base_radius_u = rng.uniform(min_radius, max_radius)

            # Define a limit for how elliptical the octagon can be.
            # ecc_limit = 1.0 would mean a regular octagon (circle-like).
            # ecc_limit = 1.3 means one axis can be at most 30% larger than the other.
            eccentricity_limit = 1.5

            # Calculate the allowed range for the second radius based on the first,
            # ensuring it also respects the global min/max radius constraints.
            low_v = max(min_radius, base_radius_u / eccentricity_limit)
            high_v = min(max_radius, base_radius_u * eccentricity_limit)

            # Generate the second radius from the new, constrained range.
            if low_v >= high_v:
                 base_radius_v = base_radius_u
            else:
                 base_radius_v = rng.uniform(low_v, high_v)

            # ====================================================================
            # >>>>>>>> END OF MODIFICATION <<<<<<<<
            # ====================================================================

            ring_distance = 50 * b

            for i_ring in range(num_rings):
                radius_u = base_radius_u + (i_ring * ring_distance)
                radius_v = base_radius_v + (i_ring * ring_distance)

                corners = []
                for k in range(16):
                    angle = k * (2 * np.pi / 16)
                    pt = center + (radius_u * np.cos(angle) * u + radius_v * np.sin(angle) * v)
                    corners.append(pt)

                corners.append(corners[0])
                poly_points_list = []
                for k in range(16):
                    seg = interpolate_segment(corners[k], corners[k + 1], pts_per_edge=10)
                    if k > 0: seg = seg[1:]
                    poly_points_list.append(seg)
                full_loop = np.vstack(poly_points_list)
                feature_loops.append(full_loop)
                feature_zs.append(center[2])

            for loop, z_val in zip(feature_loops, feature_zs):
                inside_mask = (loop[:, 0] >= -1e-9) & (loop[:, 0] <= Lx + 1e-9) & \
                              (loop[:, 1] >= -1e-9) & (loop[:, 1] <= Ly + 1e-9)
                if np.sum(inside_mask) < 2:
                    continue
                diffs = loop[1:] - loop[:-1]
                seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
                valid_len = np.sum(seg_lengths)
                loops.append(loop.astype(np.float64))
                Z_list.append(z_val)
                discrete_lines_x_list.append(loop[:, 0].astype(np.float32))
                discrete_lines_y_list.append(loop[:, 1].astype(np.float32))
                burger_ID_list.append(system_id)
                current_rho += valid_len / V
            feature_count += 1
            if current_rho >= target_rho:
                break

    loops = correct_loop_winding_orders(loops, verbose=False)

    if len(loops) == 0:
        dummy_center = np.array([Lx / 2, Ly / 2, Lz / 2])
        p_start, p_end = get_boundary_crossing_segment(dummy_center, np.array([1, 0, 0]), Lx, Ly)
        p_start_c, p_end_c = clip_segment_to_box(p_start, p_end, Lx, Ly)
        dummy = interpolate_segment(p_start_c, p_end_c, 10)
        loops.append(dummy)
        Z_list.append(Lz / 2)
        burger_ID_list.append(0)
        discrete_lines_x_list.append(dummy[:, 0].astype(np.float32))
        discrete_lines_y_list.append(dummy[:, 1].astype(np.float32))

    burger_ID_np = np.array(burger_ID_list, dtype=int)
    Z_list_np = np.array(Z_list, dtype=np.float64)
    discrete_lines_x_np = np.array(discrete_lines_x_list, dtype=object)
    discrete_lines_y_np = np.array(discrete_lines_y_list, dtype=object)

    core_w = 1 * sin_remov
    nx = int(np.ceil(1 * Lx / sin_remov)) + 1
    ny = int(np.ceil(1 * Ly / sin_remov)) + 1
    X_nodes = np.linspace(0, Lx, nx)
    Y_nodes = np.linspace(0, Ly, ny)
    X_grid, Y_grid = np.meshgrid(X_nodes, Y_nodes, indexing='ij')
    dx_grid, dy_grid = X_nodes[1] - X_nodes[0], Y_nodes[1] - Y_nodes[0]

    loops_typed = List()
    for l in loops:
        loops_typed.append(l)

    # 1. Compute Density (returns density based on dz)
    QQ, theta_tensor, angle_tensor, min_dis_ten = density_extraction.compute_dislocation_density_and_tangent(
        loops_typed, dx_grid, dy_grid, dz, sin_remov, nx, ny)

    # 2. Get Z scaling factor
    nz_scaling = np.round(Lz / dz)

    # 3. Adjust Target Densities
    for system_id in range(num_slip_systems):
        poly_indices = np.where(burger_ID_np == system_id)[0]
        if len(poly_indices) == 0:
            continue

        QQ_subset = QQ[:, :, poly_indices]
        current_len = np.sum(QQ_subset) * (dx_grid * dy_grid * dz)

        if current_len > 1e-20:
            goal_dens = target_densities[system_id] * (Lx * Ly * Lz)
            print(f"{target_densities[system_id]:.2e}")
            factor = goal_dens / current_len
            QQ[:, :, poly_indices] *= factor

    # 4. Compute Stress
    dV_stress = (dx_grid * dy_grid * dz)
    tau_int = compute_internal_tau_2d(
        X_grid, Y_grid, Z_list_np, QQ, n_vec, b,
        theta_tensor, mu, nu, real_sin_remove, C, dV_stress,
        burger_ID_np, b_vecs, epsilon, delta, pho_lim, 30 * b)

    QQ_config, _ = density_extraction.compute_density_distribution_burgers_angle(
        angle_tensor, QQ, burger_ID_np, 360)

    tau_avg, tau_avg_abs, tau_avg_pos, tau_avg_neg = density_extraction.average_tau_by_burgers(
        tau_int, QQ, burger_ID_np)

    # --- Calculation of SSD, GND, etc. ---
    SSD, GND, all_rho, GND_vec = density_extraction.compute_ssd_gnd_by_burgers(
        QQ, theta_tensor, burger_ID_np)
    print(f"nz_scaling value: {nz_scaling}")

    norm_factor = (nx * ny * nz_scaling)
    norm_factor_ssd = norm_factor
    norm_factor_gnd = norm_factor
    norm_factor_rho = norm_factor
    norm_factor_gnd_vec = norm_factor

    if SSD.ndim == 1:
        SSD = SSD / norm_factor_ssd
    else:
        SSD = np.sum(SSD, axis=(0, 1)) / norm_factor_ssd

    if GND.ndim == 1:
        GND = GND / norm_factor_gnd
    else:
        GND = np.sum(GND, axis=(0, 1)) / norm_factor_gnd

    if all_rho.ndim == 1:
        all_rho = all_rho / norm_factor_rho
    else:
        all_rho = np.sum(all_rho, axis=(0, 1)) / norm_factor_rho

    if GND_vec.ndim > 1:
        GND_vec = np.sum(GND_vec, axis=(0, 1)) / norm_factor_gnd_vec
    else:
        GND_vec = GND_vec / norm_factor_gnd_vec

    return dict(
        QQ_config=QQ_config.ravel().astype(np.float32),
        SSD=SSD.astype(np.float32),
        GND=GND.astype(np.float32),
        all_rho=all_rho.astype(np.float32),
        GND_vec=GND_vec.ravel().astype(np.float32),
        tau_avg=(tau_avg * b / mu).astype(np.float32),
        tau_avg_abs=(tau_avg_abs * b / mu).astype(np.float32),
        tau_avg_pos=(tau_avg_pos * b / mu).astype(np.float32),
        tau_avg_neg=(tau_avg_neg * b / mu).astype(np.float32),
        Lx=np.float32(Lx), Ly=np.float32(Ly), Lz=np.float32(Lz),
        Core_width=np.float32(core_w),
        burger_ID=burger_ID_np,
        dislocation_density=QQ,
        tau_map=tau_int,
        X_grid=X_grid, Y_grid=Y_grid, Z_grid=Z_list_np,
        theta_distribution=angle_tensor,
        discrete_lines_x=discrete_lines_x_np,
        discrete_lines_y=discrete_lines_y_np)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('n', type=int, help='number of simulations to run in this process')
    p.add_argument('--prefix', default='batch_coarse', help='output file prefix')
    p.add_argument('--start-index', type=int, default=0, help='starting index for file numbering')
    p.add_argument('-o', '--out', help='Legacy output override')

    a = p.parse_args()
    rng = np.random.default_rng()
    t0 = time.time()

    print(f"Starting batch generation: {a.n} samples, starting at index {a.start_index}")

    for i in range(a.n):
        current_idx = a.start_index + i
        if a.out and a.n == 1:
            filename = a.out
        else:
            filename = f"{a.prefix}_{current_idx}.npz"
        try:
            t_sample_start = time.time()
            sample = make_sample(rng)
            np.savez_compressed(filename, **sample)
            t_sample_end = time.time()
            print(f"[{i + 1}/{a.n}] Saved sample to {filename} (took {t_sample_end - t_sample_start:.1f}s)")
        except Exception as e:
            print(f"Skipping sample {current_idx} due to error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"Finished. Total elapsed time: {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()