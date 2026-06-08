import numpy as np
from numba import njit, prange


@njit(parallel=True)
def compute_loop_length_density_cartesian(dislocation_loops, Lx, Ly, dz):
    """
    Computes the initial loop length density for a Cartesian bounding box.
    """
    n_loops = len(dislocation_loops)
    rho_loops = np.zeros(n_loops, dtype=np.float64)
    vol = Lx * Ly * dz

    for d in prange(n_loops):
        loop = dislocation_loops[d, :, :]
        n_pts = loop.shape[0]
        length = 0.0

        for i in range(n_pts):
            j = (i + 1) % n_pts
            p1 = loop[i]
            p2 = loop[j]

            # Check if either point is inside the Cartesian box
            in_1 = (0 <= p1[0] <= Lx) and (0 <= p1[1] <= Ly)
            in_2 = (0 <= p2[0] <= Lx) and (0 <= p2[1] <= Ly)

            if in_1 or in_2:
                seg_len = np.sqrt(np.sum((p2 - p1) ** 2))
                length += seg_len

        rho_loops[d] = length / vol

    return rho_loops


def scale_rho_tensor(rho_tensor, density_expected, nx, ny):
    for d in range(rho_tensor.shape[2]):
        total = np.sum(rho_tensor[:, :, d])
        if total > 0:
            scaling_factor = (nx * ny * density_expected[d]) / total
            rho_tensor[:, :, d] *= scaling_factor


@njit(parallel=True)
def compute_dislocation_density_and_tangent(dislocation_loops, core_radius, nx, ny, XX, YY):
    n_loops = len(dislocation_loops)
    sigma = np.float64(core_radius * 0.5)

    rho_tensor = np.zeros((nx, ny, n_loops), dtype=np.float64)
    angle_tensor = np.zeros((nx, ny, n_loops), dtype=np.float64)

    for d in prange(n_loops):
        loop = dislocation_loops[d]
        n_seg = loop.shape[0]

        for ix in range(nx):
            for iy in range(ny):
                grid_pt = np.array([np.float64(XX[ix, iy]), np.float64(YY[ix, iy]), 0.0])
                local_density = 0.0
                best_distance = 1e6
                best_tangent = np.array([0.0, 0.0, 0.0])

                for s in range(n_seg - 1):
                    p1 = loop[s]
                    p2 = loop[(s + 1) % n_seg]

                    seg_vec = p2 - p1
                    seg_len = np.sqrt(np.sum(seg_vec ** 2))

                    if seg_len == 0:
                        continue

                    diff = grid_pt - p1
                    u = np.dot(diff, seg_vec) / (seg_len * seg_len)
                    if u < 0.0:
                        closest_pt = p1
                    elif u > 1.0:
                        closest_pt = p2
                    else:
                        closest_pt = p1 + u * seg_vec

                    d_seg = np.sqrt(np.sum((grid_pt - closest_pt) ** 2))

                    if d_seg < best_distance:
                        best_distance = d_seg
                        best_tangent = seg_vec / seg_len

                w = np.exp(-0.5 * ((best_distance / sigma) ** 2))
                local_density += w
                rho_tensor[ix, iy, d] = local_density

                norm_t = np.sqrt(np.sum(best_tangent ** 2))
                if norm_t > 0:
                    best_tangent /= norm_t
                angle_tensor[ix, iy, d] = np.mod(np.arctan2(best_tangent[1], best_tangent[0]), 2 * np.pi)

    return rho_tensor, angle_tensor


@njit(parallel=True)
def Coarse_grainer(QQ, thet_tensor, theta, std, n_burgers, b_ids):
    nx, ny, n_loops = QQ.shape
    n_theta = theta.shape[0]
    d_theta = 2 * np.pi / n_theta
    n_dist = int(np.ceil(5 * std / d_theta))
    dist_points = np.arange(-n_dist, n_dist + 1, 1) * d_theta

    distributed_coef = (1 / (np.sqrt(2 * np.pi * std ** 2))) * np.exp(-((dist_points) ** 2) / (2 * std ** 2))
    distributed_coef /= np.sum(distributed_coef)

    QQ_coarse = np.zeros((n_burgers, nx, ny, n_theta))

    for n in prange(n_loops):
        b_id = int(b_ids[n])
        for i in range(nx):
            for j in range(ny):
                theta_ind = int(np.floor(thet_tensor[i, j, n] / d_theta))
                for k in range(len(distributed_coef)):
                    coef = distributed_coef[k]
                    shifted_index = (theta_ind + k - n_dist) % n_theta
                    QQ_coarse[b_id, i, j, shifted_index] += QQ[i, j, n] * coef

    return QQ_coarse


@njit
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

                                        term1 = 6.0 * delta_rk * a ** 2 * (1.0 - nu) * x_l * invR5
                                        term2 = (3.0 - 4.0 * nu) * delta_rk * x_l * invR3
                                        term3 = 3.0 * x_r * x_k * x_l * invR5
                                        term4 = - (delta_rl * x_k + delta_kl * x_r) * invR3

                                        bracket = term1 + term2 + term3 + term4
                                        contrib = coeff * eps_jsh * bi_prime * Cijkl * bracket
                                        total += contrib
        g_h[h] = total
    return g_h


@njit(parallel=True)
def precompute_mura_kernel(nx, ny, pad_num, dx, dy, n_vec, b_vecs, C, mu, nu, a, epsilon, delta):
    """
    Precomputes the Mura Green's function kernel for all possible relative distances on a Cartesian grid.
    Returns a 5D array: G_kernel[alpha, alpha_prime, di_idx, dj_idx, h]
    """
    n_slips = b_vecs.shape[0]
    max_di = nx + pad_num
    max_dj = ny + pad_num
    shape_i = 2 * max_di + 1
    shape_j = 2 * max_dj + 1

    G_kernel = np.zeros((n_slips, n_slips, shape_i, shape_j, 3), dtype=np.float64)

    for k1 in prange(n_slips):
        b_alpha = b_vecs[k1]
        for k2 in range(n_slips):
            b_prime = b_vecs[k2]
            for di_idx in range(shape_i):
                di = di_idx - max_di
                for dj_idx in range(shape_j):
                    dj = dj_idx - max_dj

                    # Relative distance vector
                    r_rel = np.array([di * dx, dj * dy, 0.0])
                    r_prime = np.array([0.0, 0.0, 0.0])

                    g_h = mura_g_tensor(r_rel, r_prime, n_vec, b_alpha, b_prime, n_vec, C, mu, nu, a, epsilon, delta)

                    G_kernel[k1, k2, di_idx, dj_idx, 0] = g_h[0]
                    G_kernel[k1, k2, di_idx, dj_idx, 1] = g_h[1]
                    G_kernel[k1, k2, di_idx, dj_idx, 2] = g_h[2]

    return G_kernel


@njit(parallel=True)
def compute_gnd_and_xi(QQ, theta):
    n_slips, nx, ny, n_thet = QQ.shape
    GND = np.zeros((n_slips, nx, ny))
    x_i = np.zeros((n_slips, nx, ny, 3))

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for n in prange(n_slips):
        for i in range(nx):
            for j in range(ny):
                direction = np.zeros(3)
                for k in range(n_thet):
                    direction[0] += QQ[n, i, j, k] * cos_theta[k]
                    direction[1] += QQ[n, i, j, k] * sin_theta[k]

                GND[n, i, j] = np.sqrt(direction[0] ** 2 + direction[1] ** 2)

                if GND[n, i, j] > 1e-12:
                    x_i[n, i, j, 0] = direction[0] / GND[n, i, j]
                    x_i[n, i, j, 1] = direction[1] / GND[n, i, j]

    return GND, x_i


@njit
def mirror_2d(mat, nx, ny, pad_x, pad_y):
    mat_mirrored = np.zeros((nx + (2 * pad_x), ny + (2 * pad_y)), dtype=mat.dtype)
    mat_mirrored[pad_x:pad_x + nx, pad_y:pad_y + ny] = mat
    mat_mirrored[:pad_x, pad_y:pad_y + ny] = mat[:pad_x, :][::-1, :]
    mat_mirrored[pad_x + nx:, pad_y:pad_y + ny] = mat[-pad_x:, :][::-1, :]
    mat_mirrored[:, :pad_y] = mat_mirrored[:, pad_y:2 * pad_y][:, ::-1]
    mat_mirrored[:, pad_y + ny:] = mat_mirrored[:, ny:ny + pad_y][:, ::-1]
    return mat_mirrored


@njit
def mirror_3d(mat, nx, ny, depth, pad_x, pad_y):
    mat_mirrored = np.zeros((nx + (2 * pad_x), ny + (2 * pad_y), depth), dtype=mat.dtype)
    mat_mirrored[pad_x:pad_x + nx, pad_y:pad_y + ny, :] = mat
    mat_mirrored[:pad_x, pad_y:pad_y + ny, :] = mat[:pad_x, :, :][::-1, :, :]
    mat_mirrored[pad_x + nx:, pad_y:pad_y + ny, :] = mat[-pad_x:, :, :][::-1, :, :]
    mat_mirrored[:, :pad_y, :] = mat_mirrored[:, pad_y:2 * pad_y, :][:, ::-1, :]
    mat_mirrored[:, pad_y + ny:, :] = mat_mirrored[:, ny:ny + pad_y, :][:, ::-1, :]
    return mat_mirrored


@njit(parallel=True)
def compute_stress_fast(GND, x_i, QQ, dV, cut_dis, dx, dy, pho_lim, pad_num, G_kernel):
    """
    Fast computation of Mura stress using the precomputed Green's function kernel.
    """
    n_slips, nx, ny, n_thet = QQ.shape
    QQ_G = np.zeros((n_slips, nx + (2 * pad_num), ny + (2 * pad_num), n_thet))
    GND_G = np.zeros((n_slips, nx + (2 * pad_num), ny + (2 * pad_num)))
    x_i_G = np.zeros((n_slips, nx + (2 * pad_num), ny + (2 * pad_num), 3))

    for k in range(n_slips):
        QQ_G[k, :, :, :] = mirror_3d(QQ[k, :, :, :], nx, ny, n_thet, pad_num, pad_num)
        x_i_G[k, :, :, :] = mirror_3d(x_i[k, :, :, :], nx, ny, 3, pad_num, pad_num)
        GND_G[k, :, :] = mirror_2d(GND[k, :, :], nx, ny, pad_num, pad_num)

    tau_alpha_int = np.zeros((nx, ny, n_slips))
    cut_dis_sq = cut_dis ** 2

    # Offsets to map relative distances to positive indices in the kernel array
    max_di = nx + pad_num
    max_dj = ny + pad_num

    for i in prange(nx):
        for j in range(ny):
            for k1 in range(n_slips):
                total_tau = 0.0

                for i2 in range(nx + 2 * pad_num):
                    # Relative index in x
                    di = i - (i2 - pad_num)
                    di_idx = di + max_di
                    dist_x_sq = (di * dx) ** 2

                    for j2 in range(ny + 2 * pad_num):
                        # Relative index in y
                        dj = j - (j2 - pad_num)
                        dist_sq = dist_x_sq + (dj * dy) ** 2

                        if dist_sq > cut_dis_sq:
                            continue

                        dj_idx = dj + max_dj

                        for k2 in range(n_slips):
                            rho_G = GND_G[k2, i2, j2]
                            if rho_G > pho_lim:
                                # Fetch precomputed kernel values
                                g_h0 = G_kernel[k1, k2, di_idx, dj_idx, 0]
                                g_h1 = G_kernel[k1, k2, di_idx, dj_idx, 1]
                                g_h2 = G_kernel[k1, k2, di_idx, dj_idx, 2]

                                dot_val = g_h0 * x_i_G[k2, i2, j2, 0] + \
                                          g_h1 * x_i_G[k2, i2, j2, 1] + \
                                          g_h2 * x_i_G[k2, i2, j2, 2]

                                total_tau += dot_val * rho_G * dV

                tau_alpha_int[i, j, k1] = -total_tau

    return tau_alpha_int


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


def levi_civita():
    eps = np.zeros((3, 3, 3), dtype=int)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[2, 1, 0] = eps[0, 2, 1] = eps[1, 0, 2] = -1
    return eps


def compute_rss(n_vec, b_vecs, force):
    force_m = np.sqrt(np.sum(force ** 2))
    force_d = force / force_m
    stress_tensor = force_m * np.outer(force_d, force_d)

    stress_tensor = np.asarray(stress_tensor)
    n_vec = np.asarray(n_vec)
    b_vecs = np.asarray(b_vecs)

    traction_vector = np.dot(stress_tensor, n_vec)
    rss_values = np.dot(traction_vector, b_vecs.T)

    rss_max = np.max(np.abs(rss_values))
    if rss_max > 0:
        for num in range(rss_values.shape[0]):
            if rss_values[num] < 0:
                b_vecs[num] *= -1
                rss_values[num] *= -1
            if rss_values[num] / rss_max < 0.2:
                rss_values[num] = 0.0

    return rss_values