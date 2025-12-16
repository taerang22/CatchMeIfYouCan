# mpc/mpc.py

import numpy as np
from pydrake.systems.controllers import LinearQuadraticRegulator

class CartesianCatchMPC:
    """
    Simple 3D Cartesian MPC using Drake:

        x_k = [p_k, v_k] ∈ R^6
        u_k = a_k ∈ R^3

        p_{k+1} = p_k + dt * v_k + 0.5 * dt^2 * a_k
        v_{k+1} = v_k + dt * a_k

    Cost:
        - terminal: ||p_N - p_hit||^2 (catch point)
        - stage:    small * ||u_k||^2 (smooth accel)

    Constraints:
        - |a_k|_∞ ≤ a_max

    solve(...) returns:
        a_cmd : first-step acceleration (3,)
        traj  : (N+1, 6) array of predicted [p, v] along horizon
    """

    def __init__(
        self,
        dt_sim: float,
        N_max: int = 40,
        a_max: float = 3.0,
        w_terminal: float = 200.0,
        w_control: float = 1.0,
        dt_nominal: float = 0.05,
        min_T: float = 0.25,
        max_T: float = 1.0,
        w_pos_base: float = 1.0,
        w_vel_base: float = 0.1,
    ):
        self.dt_sim = dt_sim        # 시뮬레이션 dt (MuJoCo)
        self.N_max = N_max
        self.a_max = float(a_max)

        self.w_terminal = float(w_terminal)
        self.w_control = float(w_control)

        self.dt_nominal = float(dt_nominal)
        self.min_T = float(min_T)
        self.max_T = float(max_T)

        self.w_pos_base = float(w_pos_base)
        self.w_vel_base = float(w_vel_base)

    # === 내부 유틸: 선형 시스템 행렬 ===
    @staticmethod
    def _AB(dt: float):
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        A = np.block([
            [I3, dt * I3],
            [Z3, I3],
        ])
        B = np.vstack([
            0.5 * (dt ** 2) * I3,
            dt * I3,
        ])
        return A, B

    # === 메인 MPC solve ===
    def solve(
        self,
        p_hit: np.ndarray,
        p_eef: np.ndarray,
        v_eef: np.ndarray,
        p_ref: np.ndarray | None = None,
    ):
        """
        Drake LinearQuadraticRegulator를 사용한 상태 기반 LQR.
        
        t_hit을 사용하지 않고, 목표 위치(p_hit)에만 집중하여 제어한다.
        시간 의존성 제거 → 거리/속도에만 기반한 순수 위치 제어

        Args:
            p_hit : 목표 공 위치 (3,)
            p_eef : 현재 EEF 위치 (3,)
            v_eef : 현재 EEF 속도 (3,)
            p_ref : (옵션) 미사용

        Returns:
            a_cmd : 가속도 명령 (3,)
            traj  : None
        """
        p_hit = np.asarray(p_hit, dtype=float).reshape(3,)
        p_eef = np.asarray(p_eef, dtype=float).reshape(3,)
        v_eef = np.asarray(v_eef, dtype=float).reshape(3,)

        if not (np.all(np.isfinite(p_hit)) and
                np.all(np.isfinite(p_eef)) and
                np.all(np.isfinite(v_eef))):
            print("[MPC-LQR] Non-finite input; returning zero accel.")
            return np.zeros(3, dtype=float), None

        # --- 1) 거리 기반 적응형 가중치 ---
        # 목표까지의 거리가 가까워질수록 더 정밀한 제어
        dist_to_target = np.linalg.norm(p_hit - p_eef)
        
        # 거리에 따른 스케일 팩터 계산 (0.05m ~ 1.0m 범위)
        dist_scale = np.clip(1.0 / (dist_to_target + 0.01), 0.5, 10.0)

        # --- 2) 이산 시스템 행렬 A, B ---
        dt = self.dt_nominal
        A, B = self._AB(dt)

        # --- 3) 상태/레퍼런스 정의 ---
        # x = [p, v],  x_ref = [p_hit, 0]
        x = np.concatenate([p_eef, v_eef])
        x_ref = np.concatenate([p_hit, np.zeros(3)])
        x_err = x - x_ref

        # --- 4) 거리 기반 적응형 가중치 Q, R ---
        # 목표에 가까워질수록 위치/속도 가중치 증가
        w_pos = self.w_pos_base * (dist_scale ** 2)
        w_vel = self.w_vel_base * dist_scale

        Q = np.diag(
            np.concatenate([
                w_pos * np.ones(3),   # p
                w_vel * np.ones(3),   # v
            ])
        )

        R = self.w_control * np.eye(3)

        # --- 5) Drake LQR 호출 ---
        K, S = LinearQuadraticRegulator(A, B, Q, R)

        # --- 6) 제어 law 적용: u = -K x_err ---
        a_cmd = -K @ x_err

        # --- 7) saturation & 안정성 체크 ---
        a_cmd = np.asarray(a_cmd, dtype=float).reshape(3,)
        if not np.all(np.isfinite(a_cmd)):
            print("[MPC-LQR] a_cmd non-finite; zeroing.")
            a_cmd[:] = 0.0

        a_norm_inf = np.max(np.abs(a_cmd))
        if a_norm_inf > self.a_max:
            a_cmd *= (self.a_max / (a_norm_inf + 1e-8))

        return a_cmd, None
