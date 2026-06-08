import numpy as np
from numba import njit, prange
import time


@njit(parallel=True)
def apply_gaussian_smoothing_3d(arr, sigma):
    """
    Applies a simple 3x3x1 Gaussian-like smoothing kernel to a 3D array
    in the spatial dimensions (i, j) only.
    """
    rows, cols, depth = arr.shape
    smoothed = np.zeros_like(arr)

    w_center = 0.6
    w_neighbor = 0.1

    for k in prange(depth):
        for i in range(rows):
            for j in range(cols):
                val = arr[i, j, k]
                if val == 0: continue

                smoothed[i, j, k] += val * w_center

                if i > 0: smoothed[i - 1, j, k] += val * w_neighbor
                if i < rows - 1: smoothed[i + 1, j, k] += val * w_neighbor
                if j > 0: smoothed[i, j - 1, k] += val * w_neighbor
                if j < cols - 1: smoothed[i, j + 1, k] += val * w_neighbor

    return smoothed


@njit(parallel=True)
def fdm(G, Vbyds, cos_thetaV, sin_thetaV):
    rows, cols, depth = G.shape
    K = np.zeros_like(G)

    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            for k in range(depth):
                flux_x_right = 0.5 * (G[i + 1, j, k] * Vbyds[i + 1, j] + G[i, j, k] * Vbyds[i, j])
                flux_x_left = 0.5 * (G[i, j, k] * Vbyds[i, j] + G[i - 1, j, k] * Vbyds[i - 1, j])

                flux_y_top = 0.5 * (G[i, j + 1, k] * Vbyds[i, j + 1] + G[i, j, k] * Vbyds[i, j])
                flux_y_bottom = 0.5 * (G[i, j, k] * Vbyds[i, j] + G[i, j - 1, k] * Vbyds[i, j - 1])

                term1 = (flux_x_right - flux_x_left) * (-sin_thetaV[k])
                term2 = (flux_y_top - flux_y_bottom) * (cos_thetaV[k])
                K[i, j, k] = -(term1 + term2)

    # Boundaries
    i = 0
    for j in prange(1, cols - 1):
        for k in range(depth):
            flux_x_right = 0.5 * (G[i + 1, j, k] * Vbyds[i + 1, j] + G[i, j, k] * Vbyds[i, j])
            flux_x_left = 0.0
            flux_y_top = 0.5 * (G[i, j + 1, k] * Vbyds[i, j + 1] + G[i, j, k] * Vbyds[i, j])
            flux_y_bottom = 0.5 * (G[i, j, k] * Vbyds[i, j] + G[i, j - 1, k] * Vbyds[i, j - 1])
            K[i, j, k] = -((flux_x_right - flux_x_left) * (-sin_thetaV[k]) + (flux_y_top - flux_y_bottom) * (
            cos_thetaV[k]))

    i = rows - 1
    for j in prange(1, cols - 1):
        for k in range(depth):
            flux_x_right = 0.0
            flux_x_left = 0.5 * (G[i, j, k] * Vbyds[i, j] + G[i - 1, j, k] * Vbyds[i - 1, j])
            flux_y_top = 0.5 * (G[i, j + 1, k] * Vbyds[i, j + 1] + G[i, j, k] * Vbyds[i, j])
            flux_y_bottom = 0.5 * (G[i, j, k] * Vbyds[i, j] + G[i, j - 1, k] * Vbyds[i, j - 1])
            K[i, j, k] = -((flux_x_right - flux_x_left) * (-sin_thetaV[k]) + (flux_y_top - flux_y_bottom) * (
            cos_thetaV[k]))

    j = 0
    for i in prange(1, rows - 1):
        for k in range(depth):
            flux_x_right = 0.5 * (G[i + 1, j, k] * Vbyds[i + 1, j] + G[i, j, k] * Vbyds[i, j])
            flux_x_left = 0.5 * (G[i, j, k] * Vbyds[i, j] + G[i - 1, j, k] * Vbyds[i - 1, j])
            flux_y_top = 0.5 * (G[i, j + 1, k] * Vbyds[i, j + 1] + G[i, j, k] * Vbyds[i, j])
            flux_y_bottom = 0.0
            K[i, j, k] = -((flux_x_right - flux_x_left) * (-sin_thetaV[k]) + (flux_y_top - flux_y_bottom) * (
            cos_thetaV[k]))

    j = cols - 1
    for i in prange(1, rows - 1):
        for k in range(depth):
            flux_x_right = 0.5 * (G[i + 1, j, k] * Vbyds[i + 1, j] + G[i, j, k] * Vbyds[i, j])
            flux_x_left = 0.5 * (G[i, j, k] * Vbyds[i, j] + G[i - 1, j, k] * Vbyds[i - 1, j])
            flux_y_top = 0.0
            flux_y_bottom = 0.5 * (G[i, j, k] * Vbyds[i, j] + G[i, j - 1, k] * Vbyds[i, j - 1])
            K[i, j, k] = -((flux_x_right - flux_x_left) * (-sin_thetaV[k]) + (flux_y_top - flux_y_bottom) * (
            cos_thetaV[k]))

    return K


@njit
def difff7(QQ, dt, cos_thetaV, sin_thetaV, Vbyds):
    K1 = fdm(QQ, Vbyds, cos_thetaV, sin_thetaV)
    G = QQ + K1 * (dt / 2.0)
    G = np.maximum(G, 0)
    K2 = fdm(G, Vbyds, cos_thetaV, sin_thetaV)
    G = QQ + K2 * (dt / 2.0)
    G = np.maximum(G, 0)
    K3 = fdm(G, Vbyds, cos_thetaV, sin_thetaV)
    G = QQ + K3 * dt
    G = np.maximum(G, 0)
    K4 = fdm(G, Vbyds, cos_thetaV, sin_thetaV)
    F = QQ + (K1 + 2.0 * K2 + 2.0 * K3 + K4) * (dt / 6.0)
    return F


@njit
def mirror_2d(mat, nx, ny, pad_x, pad_y):
    mat_mirrored = np.zeros((nx + (2 * pad_x), ny + (2 * pad_y)), dtype=mat.dtype)
    mat_mirrored[pad_x:pad_x + nx, pad_y:pad_y + ny] = mat
    mat_mirrored[:pad_x, pad_y:pad_y + ny] = mat[:pad_x, :][::-1, :]
    mat_mirrored[pad_x + nx:, pad_y:pad_y + ny] = mat[-pad_x:, :][::-1, :]
    mat_mirrored[:, :pad_y] = mat_mirrored[:, pad_y:2 * pad_y][:, ::-1]
    mat_mirrored[:, pad_y + ny:] = mat_mirrored[:, ny:ny + pad_y][:, ::-1]
    return mat_mirrored


@njit(parallel=True)
def theta_diff2_python(Q, V, rho_min, theta, ds):
    if ds == 0:
        raise ValueError("Grid spacing 'ds' cannot be zero.")

    rows, cols = V.shape
    depth = Q.shape[2]
    dVbydxi = np.zeros_like(Q)

    tangent_cos = np.cos(theta)
    tangent_sin = np.sin(theta)

    pad_size = 1
    GV = mirror_2d(V, rows, cols, pad_size, pad_size)

    rho_stiff_threshold = rho_min * 1000.0

    # Precompute sums to avoid doing it inside the loop
    Q_sum = np.zeros((rows, cols))
    for i in prange(rows):
        for j in range(cols):
            sum_val = 0.0
            for k in range(depth):
                sum_val += Q[i, j, k]
            Q_sum[i, j] = sum_val

    for i in prange(rows):
        for j in range(cols):
            rho_local = Q_sum[i, j]
            if rho_local <= rho_min:
                continue

            dv_dy_fwd = (GV[pad_size + i, pad_size + j + 1] - GV[pad_size + i, pad_size + j]) / ds
            dv_dy_bwd = (GV[pad_size + i, pad_size + j] - GV[pad_size + i, pad_size + j - 1]) / ds

            if dv_dy_fwd * dv_dy_bwd > 0:
                dv_dy = dv_dy_fwd if abs(dv_dy_fwd) < abs(dv_dy_bwd) else dv_dy_bwd
            else:
                dv_dy = 0.0

            dv_dx_fwd = (GV[pad_size + i + 1, pad_size + j] - GV[pad_size + i, pad_size + j]) / ds
            dv_dx_bwd = (GV[pad_size + i, pad_size + j] - GV[pad_size + i - 1, pad_size + j]) / ds

            if dv_dx_fwd * dv_dx_bwd > 0:
                dv_dx = dv_dx_fwd if abs(dv_dx_fwd) < abs(dv_dx_bwd) else dv_dx_bwd
            else:
                dv_dx = 0.0

            stiffness_factor = 1.0
            if rho_local > rho_stiff_threshold:
                stiffness_factor = 1.0 / (1.0 + (rho_local / rho_stiff_threshold) ** 2)

            v_local_abs = abs(V[i, j])
            kinematic_max_rotation = v_local_abs / ds

            for k in range(depth):
                raw_rotation = dv_dx * tangent_cos[k] + dv_dy * tangent_sin[k]
                dtheta_dt = raw_rotation * stiffness_factor

                if dtheta_dt > kinematic_max_rotation:
                    dtheta_dt = kinematic_max_rotation
                elif dtheta_dt < -kinematic_max_rotation:
                    dtheta_dt = -kinematic_max_rotation

                dVbydxi[i, j, k] = dtheta_dt

    return dVbydxi


@njit
def periodic_1d(mat, pz):
    rows, cols, depth = mat.shape
    if pz > depth:
        raise ValueError("Padding width `pz` cannot be greater than the array's third dimension.")
    if pz == 0:
        return mat

    padded_shape = (rows, cols, depth + 2 * pz)
    mat_p = np.zeros(padded_shape, dtype=mat.dtype)
    mat_p[:, :, pz: depth + pz] = mat
    mat_p[:, :, 0:pz] = mat[:, :, -pz:]
    mat_p[:, :, -pz:] = mat[:, :, 0:pz]
    return mat_p


@njit(parallel=True)
def fvm_angular_advection(G, omega, dtheta):
    rows, cols, depth = G.shape
    dG_dt = np.zeros_like(G)
    inv_dtheta = 1.0 / dtheta

    G_padded = periodic_1d(G, 1)
    omega_padded = periodic_1d(omega, 1)

    for i in prange(rows):
        for j in range(cols):
            for k in range(depth):
                k_p = k + 1

                omega_face_right = 0.5 * (omega_padded[i, j, k_p] + omega_padded[i, j, k_p + 1])
                if omega_face_right >= 0:
                    G_face_right = G_padded[i, j, k_p]
                else:
                    G_face_right = G_padded[i, j, k_p + 1]
                flux_right = omega_face_right * G_face_right

                omega_face_left = 0.5 * (omega_padded[i, j, k_p - 1] + omega_padded[i, j, k_p])
                if omega_face_left >= 0:
                    G_face_left = G_padded[i, j, k_p - 1]
                else:
                    G_face_left = G_padded[i, j, k_p]
                flux_left = omega_face_left * G_face_left

                dG_dt[i, j, k] = -(flux_right - flux_left) * inv_dtheta

    return dG_dt


@njit(parallel=True)
def fdm_rotation(G, dVbydxi, dtheta):
    rows, cols, depth = G.shape
    K = np.zeros_like(G)
    padding_width = 1
    G_b = periodic_1d(G, padding_width)
    dVbydxi_b = periodic_1d(dVbydxi, padding_width)
    inv_2dtheta = 1.0 / (2.0 * dtheta)

    for i in prange(rows):
        for j in range(cols):
            for k in range(depth):
                k_padded = k + padding_width
                flux_A = max(0, G_b[i, j, k_padded - 1]) * dVbydxi_b[i, j, k_padded - 1]
                flux_B = max(0, G_b[i, j, k_padded + 1]) * dVbydxi_b[i, j, k_padded + 1]
                flux_diff = flux_B - flux_A
                K[i, j, k] = -flux_diff * inv_2dtheta
                if K[i, j, k] < 0:
                    K[i, j, k] = max(K[i, j, k], -G[i, j, k])
    return K


@njit
def thetadiff(Q, dt, omega, dtheta):
    K1 = fvm_angular_advection(Q, omega, dtheta)
    G_intermediate = Q + K1 * (dt / 2.0)
    G_intermediate = np.maximum(G_intermediate, 0.0)
    K2 = fvm_angular_advection(G_intermediate, omega, dtheta)
    G_intermediate = Q + K2 * (dt / 2.0)
    G_intermediate = np.maximum(G_intermediate, 0.0)
    K3 = fvm_angular_advection(G_intermediate, omega, dtheta)
    G_intermediate = Q + K3 * dt
    G_intermediate = np.maximum(G_intermediate, 0.0)
    K4 = fvm_angular_advection(G_intermediate, omega, dtheta)
    Q_final = Q + (K1 + 2.0 * K2 + 2.0 * K3 + K4) * (dt / 6.0)
    Q_final = np.maximum(Q_final, 0.0)
    return Q_final


@njit
def thetadiff2(Q, dt, dVbydxi, dtheta):
    rows, cols, depth = Q.shape
    K1 = fdm_rotation(Q, dVbydxi, dtheta)
    G = Q + K1 * (dt / 2.0)
    G = np.maximum(G, 0)
    K2 = fdm_rotation(G, dVbydxi, dtheta)
    G = Q + K2 * (dt / 2.0)
    G = np.maximum(G, 0)
    K3 = fdm_rotation(G, dVbydxi, dtheta)
    G = Q + K3 * dt
    G = np.maximum(G, 0)
    K4 = fdm_rotation(G, dVbydxi, dtheta)
    Q_final = Q + (K1 + 2.0 * K2 + 2.0 * K3 + K4) * (dt / 6.0)
    for i in range(rows):
        for j in range(cols):
            for k in range(depth):
                if Q_final[i, j, k] < 0: Q_final[i, j, k] = 0
    return Q_final


@njit
def annihilation_python(G, V, dx, dy, dt, b, anni_amount):
    rows, cols, depth = G.shape

    for i in range(rows):
        for j in range(cols):
            if V[i, j] < 0: V[i, j] = 0

    half = depth // 2
    extra = depth // 10

    annihilation_term_denominator = 0.0
    area_product = dx * dy
    if area_product > 0:
        annihilation_term_denominator = np.sqrt(area_product)

    slope = 1.0 / (extra * extra)
    int_func = np.zeros(2 * extra + 1, dtype=G.dtype)
    for i in range(extra):
        int_func[i + 1] = int_func[i] + slope
    for i in range(extra, 2 * extra):
        int_func[i + 1] = int_func[i] - slope

    G_p = periodic_1d(G, extra)
    keeper = G.copy()

    for k_orig in range(half):
        k_padded = k_orig + extra
        back_window = G_p[:, :, k_padded - extra: k_padded + extra + 1]
        back = np.sum(back_window * int_func, axis=2)
        ahead_window = G_p[:, :, k_padded + half - extra: k_padded + half + extra + 1]
        ahead = np.sum(ahead_window * int_func, axis=2)
        min_density = np.minimum(back, ahead)

        if annihilation_term_denominator > 0:
            anni_rate1 = (min_density * V * dt / annihilation_term_denominator)
        else:
            anni_rate1 = np.zeros_like(min_density)

        anni_rate2 = min_density * (50 * b / dx) if dx != 0 else np.zeros_like(min_density)
        anni_rate = np.maximum(anni_rate1, anni_rate2)

        anni_amount += np.sum(anni_rate)
        keeper[:, :, k_orig] -= anni_rate
        keeper[:, :, k_orig + half] -= anni_rate

    G = keeper
    return G, anni_amount


@njit
def _gaussian_normalized_pdf(num_points, mu, sigma):
    if sigma == 0:
        pdf = np.zeros(num_points, dtype=np.float64)
        idx = int(round(mu))
        if 0 <= idx < num_points:
            pdf[idx] = 1.0
        return pdf
    x_vals = np.arange(num_points)
    pdf = np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
    max_val = np.max(pdf)
    if max_val > 1e-9:
        return pdf / max_val
    else:
        return pdf


@njit(parallel=True)
def nucleation_python(QQ_sl1, stress1, crit_nuc, max_nuc, pho_nuc, dt, V, dx, f_l, PP, nuc_amount, up, left, down,
                      right):
    rows, cols, depth = QQ_sl1.shape
    QQ_nuc = np.zeros_like(QQ_sl1)
    QQ_nuc_raw = np.zeros_like(QQ_sl1)

    for i in prange(rows):
        for j in range(cols):
            if (abs(stress1[i, j]) > crit_nuc[i, j]) and \
                    (np.sum(QQ_sl1[i, j, :]) < max_nuc) and \
                    (abs(V[i, j]) > 1e-15):

                if f_l[i, j] == 0:
                    continue

                term_exp = np.exp(-np.sum(PP[i, j, :]))
                v_eff = min(abs(V[i, j]), 2000.0)

                QQ_nucc = (((pho_nuc + (0.1 * term_exp)) * dt * v_eff) / f_l[i, j]) * np.exp(
                    -np.sum(PP[i, j, :]) / max_nuc)
                QQ_nucc = max(0.0, min(QQ_nucc, max_nuc - np.sum(QQ_sl1[i, j, :])))

                spread_radius = min(int(np.ceil(f_l[i, j] / (2 * dx))), 3)

                if i + spread_radius < rows:
                    QQ_nuc_raw[i + spread_radius, j, :] += QQ_nucc * right
                if i - spread_radius > -1:
                    QQ_nuc_raw[i - spread_radius, j, :] += QQ_nucc * left
                if j - spread_radius > -1:
                    QQ_nuc_raw[i, j - spread_radius, :] += QQ_nucc * down
                if j + spread_radius < cols:
                    QQ_nuc_raw[i, j + spread_radius, :] += QQ_nucc * up

    QQ_nuc = apply_gaussian_smoothing_3d(QQ_nuc_raw, 1.0)
    nuc_amount += np.sum(QQ_nuc)
    return QQ_nuc, nuc_amount


@njit
def numba_find_peaks(arr, prominence, distance):
    """
    A fast, Numba-compatible 1D peak finder to replace scipy.signal.find_peaks.
    """
    n = len(arr)
    peak_indices = []
    prominences = []

    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            # Simple prominence check: height must be greater than prominence
            if arr[i] > prominence:
                # Distance check
                if len(peak_indices) == 0 or (i - peak_indices[-1]) >= distance:
                    peak_indices.append(i)
                    prominences.append(arr[i])

    return peak_indices, prominences


@njit(parallel=True)
def local_max_python(converter, QQ, rho_min):
    """
    Fully JIT-compiled and parallelized local_max_python.
    Replaces the Scipy find_peaks bottleneck.
    """
    iteration = len(QQ)
    # We can't easily append to lists inside parallel njit loops if they hold arrays,
    # so we pre-allocate the output array.
    # Assuming QQ has shape (iteration, rows, cols, depth)
    rows = QQ[0].shape[0]
    cols = QQ[0].shape[1]
    depth = QQ[0].shape[2]

    QS = np.zeros((iteration, rows, cols, depth), dtype=np.float64)

    for ii in prange(iteration):
        current_qq = QQ[ii] * converter
        padded_qq = periodic_1d(current_qq, depth)

        for i in range(rows):
            for j in range(cols):
                slice_1d = padded_qq[i, j, :]

                # Using our custom Numba peak finder
                peak_indices, prominences = numba_find_peaks(slice_1d, prominence=1e-4, distance=8)

                for p_idx in range(len(peak_indices)):
                    idx = peak_indices[p_idx]
                    if idx >= depth and idx < 2 * depth:
                        original_idx = idx - depth
                        QS[ii, i, j, original_idx] = prominences[p_idx]


    return QS


@njit(parallel=True)
def strain_finder_python(strain, GND, V, b_vec, n_vec, dt, b, N_Z, pho_thres, strain_thresh):
    n_slips, rows, cols = GND.shape
    n_site = float(rows * cols)
    strain_rate_factor = b / n_site
    max_flux_cap = 1e12 * n_site

    sum_v = 0.0
    for slip in range(n_slips):
        for i in range(rows):
            for j in range(cols):
                sum_v += abs(V[slip, i, j])

    avg_v = sum_v / (n_slips * rows * cols)
    v_cap = 5 * avg_v

    for slip in prange(n_slips):
        b_diad_n = np.empty((3, 3), dtype=np.float64)
        for r in range(3):
            for c in range(3):
                b_diad_n[r, c] = (b_vec[slip, r] * n_vec[c] + n_vec[r] * b_vec[slip, c]) / 2.0

        for i in range(rows):
            for j in range(cols):
                if GND[slip, i, j] < pho_thres:
                    continue

                v_local = V[slip, i, j]
                if v_local > v_cap:
                    v_local = v_cap
                elif v_local < -v_cap:
                    v_local = -v_cap

                raw_flux = GND[slip, i, j] * v_local

                if raw_flux > max_flux_cap:
                    flux = max_flux_cap
                elif raw_flux < -max_flux_cap:
                    flux = -max_flux_cap
                else:
                    flux = raw_flux

                density_flux = flux * strain_rate_factor

                if density_flux > 0.2 * strain_thresh:
                    density_flux = 0.2 * strain_thresh
                elif density_flux < -0.2 * strain_thresh:
                    density_flux = -0.2 * strain_thresh

                for r in range(3):
                    for c in range(3):
                        strain[r, c] += b_diad_n[r, c] * density_flux * dt

    return strain


@njit
def pad_grid_zero_gradient(arr, pad_width):
    if arr.ndim == 3:
        rows, cols, depth = arr.shape
        padded_arr = np.empty((rows + 2 * pad_width, cols + 2 * pad_width, depth))
        padded_arr[pad_width:-pad_width, pad_width:-pad_width, :] = arr
    elif arr.ndim == 2:
        rows, cols = arr.shape
        padded_arr = np.empty((rows + 2 * pad_width, cols + 2 * pad_width))
        padded_arr[pad_width:-pad_width, pad_width:-pad_width] = arr
    else:
        return arr

    padded_arr[0:pad_width, :, ...] = padded_arr[pad_width:pad_width + 1, :, ...]
    padded_arr[-pad_width:, :, ...] = padded_arr[-pad_width - 1:-pad_width, :, ...]
    padded_arr[:, 0:pad_width, ...] = padded_arr[:, pad_width:pad_width + 1, ...]
    padded_arr[:, -pad_width:, ...] = padded_arr[:, -pad_width - 1:-pad_width, ...]
    return padded_arr


@njit(parallel=True)
def fdm_central_3d(QQ, Vbyds, cos_thetaV, sin_thetaV):
    nx, ny, nz = QQ.shape
    dQdt = np.zeros((nx, ny, nz))

    # Parallelize over the depth dimension (nz)
    for k in prange(nz):
        Q_k = QQ[:, :, k]

        Vx = Vbyds * cos_thetaV[k]
        Vy = Vbyds * sin_thetaV[k]

        QV_x = Q_k * Vx
        QV_y = Q_k * Vy

        Fx = np.zeros((nx + 1, ny))
        Fy = np.zeros((nx, ny + 1))

        for i in range(1, nx):
            for j in range(ny):
                Fx[i, j] = 0.5 * (QV_x[i, j] + QV_x[i - 1, j])

        for i in range(nx):
            for j in range(1, ny):
                Fy[i, j] = 0.5 * (QV_y[i, j] + QV_y[i, j - 1])

        for j in range(ny):
            Fx[0, j] = 0.0
            Fx[nx, j] = 0.0

        for i in range(nx):
            Fy[i, 0] = 0.0
            Fy[i, ny] = 0.0

        for i in range(nx):
            for j in range(ny):
                flux_x_in = Fx[i, j]
                flux_x_out = Fx[i + 1, j]

                flux_y_in = Fy[i, j]
                flux_y_out = Fy[i, j + 1]

                dQdt[i, j, k] = (flux_x_in - flux_x_out) + (flux_y_in - flux_y_out)

    return dQdt


@njit
def difff7_central(QQ, dt, cos_thetaV, sin_thetaV, Vbyds):
    K1 = fdm_central_3d(QQ, Vbyds, cos_thetaV, sin_thetaV)
    G = QQ + K1 * (dt / 2.0)
    K2 = fdm_central_3d(G, Vbyds, cos_thetaV, sin_thetaV)
    G = QQ + K2 * (dt / 2.0)
    K3 = fdm_central_3d(G, Vbyds, cos_thetaV, sin_thetaV)
    G = QQ + K3 * dt
    K4 = fdm_central_3d(G, Vbyds, cos_thetaV, sin_thetaV)
    F = QQ + (K1 + 2.0 * K2 + 2.0 * K3 + K4) * (dt / 6.0)
    F = np.maximum(F, 0.0)
    return F


@njit(parallel=True)
def Taylor_strress(QQ, alpha, mu, b):
    T_stress = np.zeros((QQ.shape[0], QQ.shape[1], QQ.shape[2]))
    density = np.sum(QQ, 3)
    for i in prange(QQ.shape[0]):
        sum_dens = np.sum(density, 0) - density[i, :, :]
        T_stress[i, :, :] = mu * b * alpha * np.sqrt(sum_dens)
    return T_stress


@njit
def superbee(r):
    return max(0.0, min(1.0, 2.0 * r), min(2.0, r))


@njit(parallel=True)
def fdm_muscl_3d(QQ, Vbyds, cos_thetaV, sin_thetaV):
    nx, ny, nz = QQ.shape
    dQdt = np.zeros((nx, ny, nz))

    for k in prange(nz):
        Vx = Vbyds * (cos_thetaV[k])
        Vy = Vbyds * sin_thetaV[k]

        Fx = np.zeros((nx + 1, ny))
        Fy = np.zeros((nx, ny + 1))

        for i in range(1, nx):
            for j in range(ny):
                if Vx > 0:
                    dq_down = QQ[i, j, k] - QQ[i - 1, j, k]
                    dq_up = QQ[i - 1, j, k] - QQ[i - 2, j, k] if i >= 2 else 0.0
                    r = dq_up / (dq_down + 1e-12)
                    Q_face = QQ[i - 1, j, k] + 0.5 * superbee(r) * dq_down
                    Fx[i, j] = Vx * Q_face
                else:
                    dq_down = QQ[i - 1, j, k] - QQ[i, j, k]
                    dq_up = QQ[i, j, k] - QQ[i + 1, j, k] if i <= nx - 2 else 0.0
                    r = dq_up / (dq_down + 1e-12)
                    Q_face = QQ[i, j, k] - 0.5 * superbee(r) * dq_down
                    Fx[i, j] = Vx * Q_face

        for j in range(ny):
            Fx[0, j] = 0.0
            Fx[nx, j] = 0.0

        for i in range(nx):
            for j in range(1, ny):
                if Vy > 0:
                    dq_down = QQ[i, j, k] - QQ[i, j - 1, k]
                    dq_up = QQ[i, j - 1, k] - QQ[i, j - 2, k] if j >= 2 else 0.0
                    r = dq_up / (dq_down + 1e-12)
                    Q_face = QQ[i, j - 1, k] + 0.5 * superbee(r) * dq_down
                    Fy[i, j] = Vy * Q_face
                else:
                    dq_down = QQ[i, j - 1, k] - QQ[i, j, k]
                    dq_up = QQ[i, j, k] - QQ[i, j + 1, k] if j <= ny - 2 else 0.0
                    r = dq_up / (dq_down + 1e-12)
                    Q_face = QQ[i, j, k] - 0.5 * superbee(r) * dq_down
                    Fy[i, j] = Vy * Q_face

        for i in range(nx):
            Fy[i, 0] = 0.0
            Fy[i, ny] = 0.0

        for i in range(nx):
            for j in range(ny):
                flux_x_in = Fx[i, j]
                flux_x_out = Fx[i + 1, j]

                flux_y_in = Fy[i, j]
                flux_y_out = Fy[i, j + 1]

                dQdt[i, j, k] = (flux_x_in - flux_x_out) + (flux_y_in - flux_y_out)

    return dQdt


@njit
def difff7_muscl(QQ, dt, cos_thetaV, sin_thetaV, Vbyds):
    K1 = fdm_muscl_3d(QQ, Vbyds, cos_thetaV, sin_thetaV)
    G = QQ + K1 * (dt / 2.0)
    K2 = fdm_muscl_3d(G, Vbyds, cos_thetaV, sin_thetaV)
    G = QQ + K2 * (dt / 2.0)
    K3 = fdm_muscl_3d(G, Vbyds, cos_thetaV, sin_thetaV)
    G = QQ + K3 * dt
    K4 = fdm_muscl_3d(G, Vbyds, cos_thetaV, sin_thetaV)
    F = QQ + (K1 + 2.0 * K2 + 2.0 * K3 + K4) * (dt / 6.0)
    F = np.maximum(F, 0.0)
    return F