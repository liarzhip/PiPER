import numpy as np

def median_depth_mm(depth_mm, u, v, radius=5, min_depth_mm=150.0, max_depth_mm=2000.0):
    h, w = depth_mm.shape[:2]
    u, v = int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))
    roi = depth_mm[max(0,v-radius):min(h,v+radius+1),
                   max(0,u-radius):min(w,u+radius+1)]
    valid = roi[np.isfinite(roi) & (roi >= min_depth_mm) & (roi <= max_depth_mm)]
    return None if valid.size == 0 else float(np.median(valid))

def deproject(u, v, z_mm, intr):
    z = float(z_mm) / 1000.0
    x = (float(u) - float(intr.cx)) * z / float(intr.fx)
    y = (float(v) - float(intr.cy)) * z / float(intr.fy)
    return np.array([x, y, z], dtype=np.float64)
