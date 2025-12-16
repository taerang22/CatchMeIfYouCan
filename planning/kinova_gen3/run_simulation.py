#!/usr/bin/env python3
"""
Run Kinova + ball scene, throw the ball, and compute intersection with a front plane.
Prints debug info every 1.0 second of simulation time.
"""

import os
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer


def intersect_ballistic_with_plane(p, v_world, a_world, p0, n, eps=1e-10):
    """Compute intersection of ballistic trajectory with plane."""
    A = 0.5 * np.dot(n, a_world)
    B = np.dot(n, v_world)
    C = np.dot(n, (p - p0))

    if abs(A) < eps:
        if abs(B) < eps:
            return None, None
        t = -C / B
        return (p + v_world * t + 0.5 * a_world * t * t, t) if t > 0 else (None, None)

    disc = B * B - 4 * A * C
    if disc < 0:
        return None, None

    sqrt_disc = np.sqrt(disc)
    ts = [t for t in [(-B - sqrt_disc)/(2*A), (-B + sqrt_disc)/(2*A)] if t > 0]
    if not ts:
        return None, None

    t_hit = min(ts)
    x_hit = p + v_world * t_hit + 0.5 * a_world * t_hit * t_hit
    return x_hit, t_hit


def main():
    CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(CURRENT_DIR)
    model = mujoco.MjModel.from_xml_path(str(CURRENT_DIR / "scene.xml"))
    data = mujoco.MjData(model)
    print(f"✅ Loaded scene: nq={model.nq}, nv={model.nv}, nbody={model.nbody}")

    # IDs
    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "ball")
    ball_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    hit_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "hit_marker")
    mocap_id = model.body_mocapid[hit_bid] if hit_bid >= 0 else -1

    # Gravity enable
    try:
        if model.opt.disableflags & mujoco.mjtDisableBit.mjDISABLE_GRAVITY:
            model.opt.disableflags &= ~mujoco.mjtDisableBit.mjDISABLE_GRAVITY
            print("ℹ️ Gravity disable flag cleared.")
    except Exception:
        pass
    model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])
    print("• gravity:", np.array(model.opt.gravity))
    print("• ball mass:", model.body_mass[ball_bid])

    # Addresses
    qpos_adr = model.jnt_qposadr[ball_jid]
    dof_adr  = model.jnt_dofadr[ball_jid]
    print(f"• ball qpos_adr={qpos_adr}, dof_adr={dof_adr}")

    # Initial pose
    data.qpos[qpos_adr:qpos_adr+7] = np.array([1,0,0,0, 1.5,0,1.0])

    # Desired world velocity
    v_world_des = np.array([0.0, -3.0, 1.5])
    w_world_des = np.array([0.0, 0.0, 0.0])

    # Convert to body frame for qvel
    quat_wxyz = data.qpos[qpos_adr:qpos_adr+4].copy()
    mat9 = np.empty(9)
    mujoco.mju_quat2Mat(mat9, quat_wxyz)
    R_wb = mat9.reshape(3,3)
    v_body_des = R_wb.T @ v_world_des
    w_body_des = R_wb.T @ w_world_des

    data.qvel[dof_adr+0:dof_adr+3] = w_body_des
    data.qvel[dof_adr+3:dof_adr+6] = v_body_des
    mujoco.mj_forward(model, data)

    # Verify initial velocity in world frame
    v_body = data.cvel[ball_bid,3:].copy()
    quat_body = data.xquat[ball_bid].copy()
    mujoco.mju_quat2Mat(mat9, quat_body)
    R_wb_now = mat9.reshape(3,3)
    v_world_now = R_wb_now @ v_body
    print("• initial v_world (requested) =", v_world_des)
    print("• initial v_world (achieved)  =", v_world_now)

    # Plane definition (catch board)
    p0 = np.array([0.80, 0.0, 0.90])
    n  = np.array([1.0, 0.0, 0.0])
    a_world = np.array(model.opt.gravity)

    print("== Starting viewer (logs every 1.0s) ==")
    next_log_time = 0.0  # next time to print
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            # Current position/velocity
            p = data.xpos[ball_bid].copy()
            v_body = data.cvel[ball_bid,3:].copy()
            quat_b = data.xquat[ball_bid].copy()
            mujoco.mju_quat2Mat(mat9, quat_b)
            R_wb_t = mat9.reshape(3,3)
            v_world = R_wb_t @ v_body

            # Print once every 1.0s
            if data.time >= next_log_time:
                print(f"t={data.time:6.3f}  p={p}  v_world={v_world}")
                next_log_time += 1.0

            # Intersection
            x_hit, t_hit = intersect_ballistic_with_plane(p, v_world, a_world, p0, n)
            if x_hit is not None and mocap_id >= 0:
                data.mocap_pos[mocap_id]  = x_hit
                data.mocap_quat[mocap_id] = np.array([1,0,0,0])
            elif x_hit is not None and abs(data.time % 1.0) < 1e-3:
                print(f"  ↳ Hit point {x_hit}, Δt={t_hit:5.3f}s")

            viewer.sync()


if __name__ == "__main__":
    main()
