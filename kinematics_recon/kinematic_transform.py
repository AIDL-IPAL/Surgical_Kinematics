import math
import numpy as np

# ---------- helpers ----------

def _nz(x, eps=1e-9):
    n = np.linalg.norm(x)
    return x / (n + eps), n

def _orthonormal_from_xy(x, y):
    # Gram–Schmidt → x̂, ŷ, ẑ (right-handed)
    xh, _ = _nz(x)
    y = y - np.dot(y, xh) * xh
    yh, _ = _nz(y)
    zh = np.cross(xh, yh)
    zh, _ = _nz(zh)
    # Re-orthogonalize y to ensure numerical stability
    yh = np.cross(zh, xh)
    return np.stack([xh, yh, zh], axis=1)  # 3x3

def _r6d_from_R(R):
    # first two columns
    return np.concatenate([R[:,0], R[:,1]], axis=0)

def _R_from_r6d(r6d):
    a = r6d[:3]; b = r6d[3:]
    xh, _ = _nz(a)
    b = b - np.dot(b, xh) * xh
    yh, _ = _nz(b)
    zh = np.cross(xh, yh); zh, _ = _nz(zh)
    return np.stack([xh, yh, zh], axis=1)

def _exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-8:
        return np.eye(3)
    k = w / th
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + math.sin(th)*K + (1-math.cos(th))*(K@K)

def _backproject(u, v, Z, K):  # K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=float)

def _ndc(u, v, W, H):
    return np.array([2.0*u/W - 1.0, 2.0*v/H - 1.0], dtype=float)

# ---------- jaw angle from calibration radius ----------
def _theta_from_chord(chord_len, hinge_r_mm):
    # θ = 2 arcsin( c / (2 r) )
    x = max(-1.0, min(1.0, 0.5*chord_len/hinge_r_mm))
    return 2.0 * math.asin(x)

# ---------- main entry ----------
def transform_kps_to_state(
    kps,                       # dict name -> {u,v,conf[,Z]}
    cfg_inst,                  # one INSTRUMENTS[...] dict
    image_size,                # (H, W)
    intrinsics=None,           # 3x3 K or None
    mode="auto",               # "2d"|"3d"|"auto"
    prev_state=None,           # optional previous output dict (for deltas)
    dt=None                    # seconds; uses cfg default if None
):
    H, W = image_size
    Tcfg = {**cfg_inst.get("transform", {})}
    dt = float(dt or Tcfg.get("dt_default", 0.0333))
    emit_r6d = bool(Tcfg.get("emit_r6d", True))
    emit_rel = bool(Tcfg.get("emit_relative", True))
    pixel_to_ndc = Tcfg.get("normalization", {}).get("pixel_to_ndc", True)

    # --- pick role names
    roles = cfg_inst.get("roles", {})
    has_jaw = bool(cfg_inst.get("has_jaw", False))
    ou = roles.get("jaw_upper")
    ol = roles.get("jaw_lower")
    w1 = roles.get("wrist-1"); w2 = roles.get("wrist-2")
    sh = roles.get("shaft")
    tip_role = roles.get("tooltip_role")  # may be None

    # --- extract points (2D + optional Z)
    def _get(name):
        return kps.get(name) if name in kps else None

    JU = _get(ou) if ou else None
    JL = _get(ol) if ol else None
    W1 = _get(w1) if w1 else None
    W2 = _get(w2) if w2 else None
    S  = _get(sh) if sh else None
    TP = _get(tip_role) if tip_role else None

    # --- compute tip & wrist midpoints in pixels
    def _uv(p): return np.array([p["u"], p["v"]], float)
    tip_px = _uv(TP) if TP is not None else (
        0.5*(_uv(JU)+_uv(JL)) if (JU is not None and JL is not None) else None
    )
    wrist_px = 0.5*(_uv(W1)+_uv(W2)) if (W1 is not None and W2 is not None) else (
        _uv(W1) if W1 is not None else (_uv(W2) if W2 is not None else None)
    )
    shaft_px = _uv(S) if S is not None else None

    # local scale for normalization
    s_loc = None
    if tip_px is not None and wrist_px is not None:
        _, s_loc = _nz(wrist_px - tip_px)

    # --- 2D orientation axes in image plane
    # forward axis ~ wrist->tip, jaw axis ~ JU->JL (if exists) else perp
    e_fwd2 = None; e_jaw2 = None
    if tip_px is not None and wrist_px is not None:
        e_fwd2, _ = _nz(tip_px - wrist_px)
    if has_jaw and (JU is not None and JL is not None):
        e_jaw2, _ = _nz(_uv(JU) - _uv(JL))
    if e_jaw2 is None and e_fwd2 is not None:
        # 90° rotation in image plane
        e_jaw2 = np.array([-e_fwd2[1], e_fwd2[0]])

    # --- 3D lifting (if requested and possible)
    use3d = (mode == "3d") or (mode == "auto" and intrinsics is not None and
                               Tcfg.get("lifting_3d", {}).get("require_intrinsics", True))
    T3 = W3 = S3 = JU3 = JL3 = None
    if use3d and intrinsics is not None:
        def _Z(p): return p.get("Z", None) if p else None
        def _bp(p):
            return _backproject(p["u"], p["v"], float(p["Z"]), intrinsics) if (p and ("Z" in p)) else None
        T3 = _bp(TP) if TP is not None else (_bp(JU)+_bp(JL))/2 if (JU and JL and _Z(JU) and _Z(JL)) else None
        W3 = (_bp(W1)+_bp(W2))/2 if (W1 and W2 and _Z(W1) and _Z(W2)) else (_bp(W1) or _bp(W2))
        S3 = _bp(S)
        JU3 = _bp(JU) if JU else None
        JL3 = _bp(JL) if JL else None

    # --- rotation construction
    R = None
    if T3 is not None and W3 is not None:
        z = T3 - W3
        if JU3 is not None and JL3 is not None:
            x = JU3 - JL3
        else:
            # project 2D jaw axis into 3D plane orthogonal to z
            if e_jaw2 is not None:
                # lift e_jaw2 by assuming small out-of-plane, then project
                # pick camera rays at tip for two slightly offset pixels
                u, v = tip_px if tip_px is not None else (W/2, H/2)
                du = np.array([e_jaw2[0], e_jaw2[1]]) * 5.0  # small pixel offset
                pA = _backproject(u+du[0], v+du[1], float(T3[2]), intrinsics)
                x = pA - T3
            else:
                # fallback: x from shaft orthogonalization
                x = (W3 - (S3 if S3 is not None else W3+np.array([1,0,0])))
        R = _orthonormal_from_xy(x, z)  # columns: x̂, ŷ, ẑ
    else:
        # 2D orientation as 3x3 embedded in camera Z
        if (e_fwd2 is not None) and (e_jaw2 is not None):
            x = np.array([e_jaw2[0], e_jaw2[1], 0.0])
            z = np.array([e_fwd2[0], e_fwd2[1], 0.0])
            R = _orthonormal_from_xy(x, z)  # lies in XY plane

    # --- tooltip position
    tooltip = None
    if T3 is not None:
        tooltip = T3
    elif tip_px is not None:
        tooltip = _ndc(tip_px[0], tip_px[1], W, H) if pixel_to_ndc else tip_px

    # --- opening measurement
    opening_cfg = cfg_inst.get("opening", {})
    opening_mode = opening_cfg.get("mode", None)
    hinge_r = opening_cfg.get("hinge_radius_mm", None)
    theta_deg = None
    spread_phi = None

    if has_jaw and (JU is not None and JL is not None):
        # chord length: 3D preferred, else pixels normalized
        if (JU3 is not None and JL3 is not None) and opening_mode == "angle" and hinge_r:
            c = np.linalg.norm(JU3 - JL3)
            theta = _theta_from_chord(c, float(hinge_r))         # radians
            theta_deg = math.degrees(theta)
        else:
            cpx = np.linalg.norm(_uv(JU) - _uv(JL))
            if s_loc and s_loc > 1e-6:
                spread_phi = float(cpx / s_loc)
            if opening_mode == "angle" and hinge_r and T3 is not None:
                # Mixed case: have radius, but only 2D chord → not scale-invariant;
                # prefer learned θ(ϕ); here we still emit ϕ and let caller map.
                pass

    # clip θ to bounds if requested
    if theta_deg is not None and opening_cfg.get("clip_to_bounds", False):
        lo, hi = opening_cfg.get("theta_bounds_deg", [0.0, 90.0])
        theta_deg = max(lo, min(hi, theta_deg))

    # --- pack state (2D or 3D)
    mode_out = "3d" if T3 is not None and R is not None else "2d"
    if mode == "2d": mode_out = "2d"
    if mode == "3d" and (T3 is not None and R is not None): mode_out = "3d"

    state = {}
    if mode_out == "3d":
        p = tooltip
        r6d = _r6d_from_R(R) if emit_r6d else None
        state_vec = np.concatenate([
            p, (r6d if emit_r6d else R.reshape(-1)),
            np.array([theta_deg if theta_deg is not None else (spread_phi if spread_phi is not None else 0.0)]),
            np.zeros(3),  # p_dot
            np.zeros(3),  # omega
            np.zeros(1)   # theta_dot
        ])
        F = np.eye(state_vec.size)
        # simple constant-velocity for p only (extend if you model ω)
        for i in range(3):
            F[i, 3 + (6 if emit_r6d else 9) + 1 + i] = dt
    else:
        # 2D tooltip
        t2 = tooltip if tooltip is not None else np.zeros(2)
        # embed 2D orientation as 4 numbers (two axes)
        e1 = e_fwd2 if e_fwd2 is not None else np.array([1.0,0.0])
        e2 = e_jaw2 if e_jaw2 is not None else np.array([0.0,1.0])
        R2 = np.array([e1[0],e1[1],e2[0],e2[1]])
        open_scalar = theta_deg if theta_deg is not None else (spread_phi if spread_phi is not None else 0.0)
        state_vec = np.concatenate([t2, R2, np.array([open_scalar]), np.zeros(2), np.zeros(1), np.zeros(1)])
        F = np.eye(state_vec.size)
        # p2 const-vel
        F[0, 2+4+1+0] = dt
        F[1, 2+4+1+1] = dt

    # --- deltas (hybrid-relative)
    deltas = None
    if emit_rel and prev_state is not None:
        if mode_out == "3d" and "R" in prev_state and "tooltip" in prev_state:
            δp = tooltip - prev_state["tooltip"]
            δR = prev_state["R"].T @ R
            # axis-angle magnitude as scalar (or return r6d)
            aa = np.array([δR[2,1]-δR[1,2], δR[0,2]-δR[2,0], δR[1,0]-δR[0,1]]) * 0.5
            deltas = {"dp": δp, "dR": δR, "aa_approx": aa}
        elif mode_out == "2d" and "tooltip" in prev_state:
            deltas = {"dp": (tooltip - prev_state["tooltip"])}

    out = {
        "mode": mode_out,
        "tooltip": tooltip,
        "R": R,
        "r6d": (_r6d_from_R(R) if (emit_r6d and R is not None) else None),
        "theta_deg": theta_deg,
        "spread_phi": spread_phi,
        "state_vec": state_vec,
        "F": F,
        "deltas": deltas
    }
    return out
