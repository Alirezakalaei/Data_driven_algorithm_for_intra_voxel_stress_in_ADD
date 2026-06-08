import numpy as np
from numba import njit, prange, float32, int32
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as color
import matplotlib

# Set backend to avoid display errors on headless systems
try:
    matplotlib.use('TkAgg')
except:
    pass


@njit
def dist_pt_segment_2d(px, py, x1, y1, x2, y2):
    """
    Calculates the minimum 2D distance from a point (px, py) to a line segment.
    """
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx ** 2 + dy ** 2)

    if t < 0:
        closest_x, closest_y = x1, y1
    elif t > 1:
        closest_x, closest_y = x2, y2
    else:
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

    return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def compute_dislocation_density_and_tangent(
        dislocation_loops,
        dx, dy, dz,
        core_radius,
        nx, ny):
    """
    Computes dislocation density, tangent vectors, angles, and minimum distance.
    """
    n_loops = np.int64(len(dislocation_loops))

    vol_voxel = dx * dy * dz
    inv_vol = 1.0 / vol_voxel
    sigma = core_radius * 0.33

    rho_tensor = np.zeros((nx, ny, n_loops), dtype=np.float32)
    min_dist_tensor = np.full((nx, ny, n_loops), np.inf, dtype=np.float32)
    dominant_tangent_tensor = np.zeros((nx, ny, n_loops, 3), dtype=np.float32)

    for d in prange(n_loops):
        loop = dislocation_loops[d]
        num_segments = loop.shape[0] - 1

        for s in range(num_segments):
            p1 = loop[s]
            p2 = loop[s + 1]

            seg_vec = p2 - p1
            seg_len = np.sqrt(seg_vec[0] ** 2 + seg_vec[1] ** 2 + seg_vec[2] ** 2)
            if seg_len < 1e-10:
                continue

            t_hat = seg_vec / seg_len

            search_radius = 3.0 * sigma
            x_min_coord = min(p1[0], p2[0]) - search_radius
            x_max_coord = max(p1[0], p2[0]) + search_radius
            y_min_coord = min(p1[1], p2[1]) - search_radius
            y_max_coord = max(p1[1], p2[1]) + search_radius

            x_min = max(0, int(x_min_coord / dx))
            x_max = min(nx, int(x_max_coord / dx) + 1)
            y_min = max(0, int(y_min_coord / dy))
            y_max = min(ny, int(y_max_coord / dy) + 1)

            L_voxel = seg_len

            for ix in range(x_min, x_max):
                x_node = (ix + 0.5) * dx
                for iy in range(y_min, y_max):
                    y_node = (iy + 0.5) * dy

                    dist = dist_pt_segment_2d(x_node, y_node, p1[0], p1[1], p2[0], p2[1])

                    if dist < min_dist_tensor[ix, iy, d]:
                        min_dist_tensor[ix, iy, d] = dist
                        dominant_tangent_tensor[ix, iy, d, 0] = t_hat[0]
                        dominant_tangent_tensor[ix, iy, d, 1] = t_hat[1]
                        dominant_tangent_tensor[ix, iy, d, 2] = t_hat[2]

                    w = np.exp(-0.5 * (dist / sigma) ** 2)
                    rho_tensor[ix, iy, d] += w * L_voxel * inv_vol

    angle_tensor = np.mod(np.arctan2(dominant_tangent_tensor[:, :, :, 1], dominant_tangent_tensor[:, :, :, 0]),
                          2 * np.pi).astype(np.float32)

    print("making dislocation density is over")

    return rho_tensor, dominant_tangent_tensor, angle_tensor, min_dist_tensor


def elastic_tensor(mu, nu):
    C = np.zeros((3, 3, 3, 3))
    lam = 2 * mu * nu / (1 - 2 * nu)
    delta = np.eye(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = (
                            mu * (delta[i, k] * delta[j, l] + delta[i, l] * delta[j, k]) +
                            lam * delta[i, j] * delta[k, l]
                    )
    return C


@njit()
def mura_g_tensor(r, r_prime, n_alpha, b_alpha, b_prime, n_prime, C, mu, nu, a, epsilon_tens, delta):
    x = np.empty(3)
    for i in range(3):
        x[i] = r[i] - r_prime[i]
    R2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + a ** 2
    R = np.sqrt(R2)
    invR3 = 1.0 / (R ** 3)
    invR5 = 1.0 / (R ** 5)
    prefactor_base = 1.0 / (16 * np.pi * mu * (1 - nu))
    g_h = np.zeros(3)
    for h in range(3):
        total = 0.0
        for p in range(3):
            np_p = n_alpha[p]
            for q in range(3):
                bq = b_alpha[q]
                for r_idx in range(3):
                    for s in range(3):
                        coeff = np_p * bq * C[p, q, r_idx, s] * prefactor_base
                        for j in range(3):
                            eps_jsh = epsilon_tens[j, s, h]
                            if eps_jsh == 0:
                                continue
                            for i in range(3):
                                bi_prime = b_prime[i]
                                for k in range(3):
                                    x_k = x[k]
                                    for l in range(3):
                                        x_l = x[l]
                                        x_r = x[r_idx]
                                        Cijkl = C[i, j, k, l]
                                        delta_rk = delta[r_idx, k]
                                        delta_rl = delta[r_idx, l]
                                        delta_kl = delta[k, l]
                                        term2 = (3.0 - 4.0 * nu) * delta_rk * x_l * invR3
                                        term3 = 3.0 * x_r * x_k * x_l * invR5
                                        term4 = - (delta_rl * x_k + delta_kl * x_r) * invR3
                                        bracket = term2 + term3 + term4
                                        contrib = coeff * eps_jsh * bi_prime * Cijkl * bracket
                                        total += contrib
        g_h[h] = total
    return g_h


@njit(parallel=True)
def compute_internal_tau_2d(X_grid, Y_grid, Z_list, QQ, n_vec, b,
                            xi, mu, nu, a, C, dV, burgers_ID, b_vecs,
                            epsilon, delta, pho_lim, r_lim):
    nx, ny, n_slips = QQ.shape
    tau_alpha_int = np.zeros((nx, ny, n_slips))
    r_lim_sq = r_lim ** 2
    for i in prange(nx):
        for j in range(ny):
            for k1 in range(n_slips):
                total_tau = 0.0
                if QQ[i, j, k1] <= pho_lim:
                    continue
                r = np.zeros(3)
                r[0] = X_grid[i, j]
                r[1] = Y_grid[i, j]
                r[2] = Z_list[k1]
                b_alpha = b_vecs[burgers_ID[k1]]
                n_alpha = n_vec
                for i2 in range(nx):
                    for j2 in range(ny):
                        for k2 in range(n_slips):
                           # if k2 == k1:
                            #    continue
                            rho_G = QQ[i2, j2, k2]
                            if rho_G <= pho_lim:
                                continue
                            r_prime = np.zeros(3)
                            r_prime[2] = Z_list[k2]
                            r_prime[0] = X_grid[i2, j2]
                            r_prime[1] = Y_grid[i2, j2]
                            dist_sq = (r[0] - r_prime[0]) ** 2 + \
                                      (r[1] - r_prime[1]) ** 2 + \
                                      (r[2] - r_prime[2]) ** 2
                            if dist_sq < r_lim_sq:
                                continue
                            b_prime = b_vecs[burgers_ID[k2]]
                            n_prime = n_vec
                            g_h = mura_g_tensor(r, r_prime, n_alpha, b_alpha, b_prime,
                                                n_prime, C, mu, nu, a, epsilon, delta)
                            dot_val = g_h[0] * xi[i2, j2, k2, 0] + \
                                      g_h[1] * xi[i2, j2, k2, 1] + \
                                      g_h[2] * xi[i2, j2, k2, 2]
                            total_tau += dot_val * rho_G * dV
                tau_alpha_int[i, j, k1] = -total_tau
    return tau_alpha_int


def plot_scalar_field_surface(X_grid, Y_grid, Z_grid, scalar_field, ax=None,
                              label='tau_internal (Pa)', title='Surface Plot of tau_internal with Color Scaling',
                              cmap='viridis', percentile=99):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    vmax = np.percentile(scalar_field, percentile)
    vmin = -vmax
    norm = color.Normalize(vmin=vmin, vmax=vmax)
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid,
                           facecolors=plt.cm.get_cmap(cmap)(norm(scalar_field)),
                           antialiased=True, alpha=0.7)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(scalar_field)
    plt.colorbar(mappable, ax=ax, label=label)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)


@njit()
def compute_density_distribution_burgers_angle(angle_tensor, density_tensor, burgers_ID, nbins):
    angle_mod = np.mod(angle_tensor, 2 * np.pi)
    nx, ny, nd = angle_tensor.shape
    n_burgers = 3
    rho_config = np.zeros((n_burgers, nbins))
    bin_edges = np.linspace(0, 2 * np.pi, nbins + 1)

    flat_angles = angle_mod.reshape(-1, nd)
    flat_densities = density_tensor.reshape(-1, nd)

    for k in range(nd):
        b_id = burgers_ID[k]
        angles_k = flat_angles[:, k]
        densities_k = flat_densities[:, k]

        bin_indices = np.searchsorted(bin_edges, angles_k, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, nbins - 1)

        for i in range(angles_k.size):
            rho_config[b_id, bin_indices[i]] += densities_k[i]

    return rho_config, bin_edges


@njit()
def average_tau_by_burgers(tau_internal, dislocation_density_tensor, burgers_ID):
    nx, ny, nd = tau_internal.shape
    n_burgers = 3
    tau_avg = np.zeros(n_burgers, dtype=np.float64)
    tau_avg_abs = np.zeros(n_burgers, dtype=np.float64)
    total_density_abs = np.zeros(n_burgers, dtype=np.float64)
    tau_avg_pos = np.zeros(n_burgers, dtype=np.float64)
    total_density_pos = np.zeros(n_burgers, dtype=np.float64)
    tau_avg_neg = np.zeros(n_burgers, dtype=np.float64)
    total_density_neg = np.zeros(n_burgers, dtype=np.float64)

    for k in range(nd):
        b_id = burgers_ID[k]
        rho = dislocation_density_tensor[:, :, k]
        tau = tau_internal[:, :, k]

        tau_avg[b_id] += np.sum(rho * tau)
        tau_avg_abs[b_id] += np.sum(rho * np.abs(tau))
        total_density_abs[b_id] += np.sum(rho)

        for i in range(nx):
            for j in range(ny):
                current_rho = rho[i, j]
                current_tau = tau[i, j]
                if current_tau > 0:
                    tau_avg_pos[b_id] += current_rho * current_tau
                    total_density_pos[b_id] += current_rho
                elif current_tau < 0:
                    tau_avg_neg[b_id] += current_rho * current_tau
                    total_density_neg[b_id] += current_rho

    for b in range(n_burgers):
        if total_density_abs[b] > 1e-12:
            tau_avg[b] /= total_density_abs[b]
            tau_avg_abs[b] /= total_density_abs[b]
        if total_density_pos[b] > 1e-12:
            tau_avg_pos[b] /= total_density_pos[b]
        if total_density_neg[b] > 1e-12:
            tau_avg_neg[b] /= total_density_neg[b]

    return tau_avg, tau_avg_abs, tau_avg_pos, tau_avg_neg


@njit()
def compute_ssd_gnd_by_burgers(QQ, tangent_tensor, burgers_ID):
    nx, ny, nd = QQ.shape
    n_burgers = 3
    GND_vec = np.zeros((n_burgers, 3))
    total_rho = np.zeros(n_burgers)

    for k in range(nd):
        b_id = burgers_ID[k]
        rho_k = QQ[:, :, k]
        t_k = tangent_tensor[:, :, k, :]

        total_rho[b_id] += np.sum(rho_k)
        for d_dim in range(3):
            GND_vec[b_id, d_dim] += np.sum(rho_k * t_k[:, :, d_dim])

    GND_by_b = np.sqrt(np.sum(GND_vec ** 2, axis=1))
    SSD_by_b = total_rho - GND_by_b

    return SSD_by_b, GND_by_b, total_rho, GND_vec