# DS Flash batch 4 (2026-11-01): Studio pre-development — keyframe track
# sampling; written by DeepSeek Flash from echo_super/workorders/
# WO-sample-track.md; verified by Muse (echo_super/test_ds_batch4.py);
# registered via adapters.register_ds_batch4.


# ---- WO-sample-track ----
def sample_track(keys, t):
    if not keys:
        raise ValueError("keys must contain at least one keyframe")
    if len(keys) == 1:
        return keys[0][1]
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]

    def cubic_bezier(x1, y1, x2, y2, u):
        # solve s in [0,1] such that x(s) = u via 40 bisection iterations
        lo, hi = 0.0, 1.0
        s = u
        for _ in range(40):
            s = (lo + hi) * 0.5
            inv = 1.0 - s
            x = 3.0 * inv * inv * s * x1 + 3.0 * inv * s * s * x2 + s * s * s
            if x < u:
                lo = s
            else:
                hi = s
        s = (lo + hi) * 0.5
        inv = 1.0 - s
        y = 3.0 * inv * inv * s * y1 + 3.0 * inv * s * s * y2 + s * s * s
        return y

    def apply_easing(name, u):
        if name == 'linear':
            return u
        if name == 'hold':
            return 0.0
        if name == 'ease_in':
            return cubic_bezier(0.42, 0.0, 1.0, 1.0, u)
        if name == 'ease_out':
            return cubic_bezier(0.0, 0.0, 0.58, 1.0, u)
        if name == 'ease_in_out':
            return cubic_bezier(0.42, 0.0, 0.58, 1.0, u)
        raise ValueError("unknown easing: %r" % (name,))

    for i in range(len(keys) - 1):
        t_i, v_i, easing = keys[i]
        t_next, v_next, _ = keys[i + 1]
        if t_i <= t < t_next:
            if t_next == t_i:
                return v_next
            u = (t - t_i) / (t_next - t_i)
            eased = apply_easing(easing, u)
            return v_i + (v_next - v_i) * eased
    return keys[-1][1]
