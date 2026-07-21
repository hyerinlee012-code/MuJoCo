# envs/grasp_env_0422.py
#
# 핵심 변경:
#   force 보상을 flat(+5.0)에서 Gaussian 형태로 변경
#   f_min(물리적 최소력) 바로 위가 최고 보상 (+10.0)
#   f_max에 가까울수록 보상 감소 (+5.0)
#   → 에이전트가 "딱 필요한 만큼만 잡는" 전략 학습

import gymnasium as gym
import mujoco
import numpy as np
import os
from envs.material_configs import MATERIAL_CONFIGS, sample_material_config


class GraspFeedbackEnv(gym.Env):

    BASE_FORCE = {
        "can":     5.0,
        "paper":   1.0,
        "plastic": 3.0,
    }

    GRASP_CTRL = np.array([
         0.247,   # fr3_joint1
        -0.196,   # fr3_joint2
        -0.174,   # fr3_joint3
        -1.710,   # fr3_joint4
        -0.140,   # fr3_joint5
         1.720,   # fr3_joint6
         1.120,   # fr3_joint7
    ])

    OBJ_INIT_POS = np.array([0.5290, 0.0050, 0.5746])

    ARM_JOINT_NAMES = [
        "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
        "fr3_joint5", "fr3_joint6", "fr3_joint7",
    ]

    def __init__(self, material="can", xml_path=None):
        super().__init__()

        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "xmls", "grasp_scene.xml")

        self.material   = material
        self.config     = MATERIAL_CONFIGS[material]
        self.base_force = self.BASE_FORCE[material]

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self._apply_material_config()

        # ID 캐싱
        def jnt_qpos(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_qposadr[jid]

        def body_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        def act_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        self.left_qpos_id   = jnt_qpos("finger_joint1")
        self.right_qpos_id  = jnt_qpos("finger_joint2")
        self.obj_qpos_id    = jnt_qpos("object_free")
        self.left_body_id   = body_id("left_finger")
        self.right_body_id  = body_id("right_finger")
        self.gripper_act_id = act_id("gripper_actuator")
        self.obj_body_id    = body_id("object")
        self.obj_geom_id    = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom"
        )

        # 행동 공간: delta force (-3 ~ +3 N)
        self.action_space = gym.spaces.Box(
            low=np.array([-3.0]),
            high=np.array([3.0]),
            dtype=np.float32
        )

        # 관찰 공간 (21차원) — mass/friction 숨김
        # 에이전트는 slip 신호(obj_linvel)로만 파지력 결정
        # [0:6]   left/right contact force
        # [6:9]   obj_pos
        # [9:12]  obj_linvel       ← slip 신호
        # [12:15] obj_angvel       ← 회전 slip 신호
        # [15:17] finger pos
        # [17:19] touch sensor
        # [19]    base_force
        # [20]    total_drift
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(21,),
            dtype=np.float32
        )

        self.max_steps         = 300
        self.step_count        = 0
        self.current_force     = self.base_force
        self.init_obj_pos      = self.OBJ_INIT_POS.copy()
        self.prev_total_force  = 0.0
        self._debug_count      = 0

        # 물리 기반 최소 파지력 (reset마다 갱신)
        self.min_force_physics = self.config["force_range"][0]
        self.current_mass      = self.config["mass"]
        self.current_friction  = self.config["friction"][0]

    # ── 재질 기본값 적용 (최초 1회) ──────────────────────────────────────────
    def _apply_material_config(self):
        cfg     = self.config
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")
        self.model.geom_friction[geom_id] = cfg["friction"]
        self.model.geom_solref[geom_id]   = cfg["solref"]
        self.model.geom_solimp[geom_id]   = cfg["solimp"]
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.model.body_mass[body_id]     = cfg["mass"]

    # ── 매 에피소드마다 랜덤화 + 물리 기반 최소 파지력 계산 ──────────────────
    def _randomize_material(self):
        cfg = sample_material_config(self.material)

        self.current_mass     = cfg["mass"]
        self.current_friction = cfg["friction"][0]

        # MuJoCo에 적용 (3가지 마찰 전부)
        self.model.body_mass[self.obj_body_id]         = self.current_mass
        self.model.geom_friction[self.obj_geom_id][:] = cfg["friction"]

        # ✅ 물리 공식으로 최소 파지력 계산
        # 미끄러지지 않는 조건: F_grasp ≥ (m × g) / (2 × μ)
        # 양손으로 잡으므로 2로 나눔
        raw_min = (self.current_mass * 9.81) / (2.0 * self.current_friction)

        # 재질별 force_range 안에서 클리핑
        f_min_cfg, f_max_cfg = self.config["force_range"]
        self.min_force_physics = float(np.clip(raw_min, f_min_cfg, f_max_cfg))

        print(f"  랜덤화: mass={self.current_mass:.3f}kg  "
              f"friction={self.current_friction:.3f}  "
              f"→ 최소 파지력={self.min_force_physics:.2f}N "
              f"(raw={raw_min:.2f}N)")

    # ── 손가락별 contact force 계산 ──────────────────────────────────────────
    def _get_contact_forces(self):
        left_force  = np.zeros(3)
        right_force = np.zeros(3)
        for i in range(self.data.ncon):
            con   = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            g1 = self.model.geom_bodyid[con.geom1]
            g2 = self.model.geom_bodyid[con.geom2]
            if g1 == self.left_body_id or g2 == self.left_body_id:
                left_force  += force[:3]
            if g1 == self.right_body_id or g2 == self.right_body_id:
                right_force += force[:3]
        return left_force, right_force

    # ── 관찰값 생성 ───────────────────────────────────────────────────────────
    def _get_obs(self):
        left_touch  = np.array([self.data.sensordata[0]])
        right_touch = np.array([self.data.sensordata[1]])
        obj_pos     = self.data.sensordata[2:5].copy()
        obj_linvel  = self.data.sensordata[5:8].copy()
        obj_angvel  = self.data.sensordata[8:11].copy()

        l_pos = np.array([self.data.qpos[self.left_qpos_id]])
        r_pos = np.array([self.data.qpos[self.right_qpos_id]])

        left_force, right_force = self._get_contact_forces()
        base_f = np.array([self.base_force])

        current_pos = self.data.qpos[self.obj_qpos_id:self.obj_qpos_id+3]
        xyz_diff    = current_pos - self.init_obj_pos
        total_drift = np.array([np.linalg.norm(xyz_diff)])

        return np.concatenate([
            left_force, right_force,          # [0:6]
            obj_pos, obj_linvel, obj_angvel,  # [6:15]
            l_pos, r_pos,                     # [15:17]
            left_touch, right_touch,          # [17:19]
            base_f,                           # [19]
            total_drift,                      # [20]
        ]).astype(np.float32)

    # ── 보상 함수 ─────────────────────────────────────────────────────────────
    def _compute_reward(self, obs, delta):
        left_force  = obs[0:3]
        right_force = obs[3:6]
        obj_linvel  = obs[9:12]
        obj_angvel  = obs[12:15]
        total_drift = float(obs[20])

        slip_vel = np.linalg.norm(obj_linvel)
        rot_vel  = np.linalg.norm(obj_angvel)

        force_mag = self.current_force

        # 물리 기반 f_min (에피소드마다 다름)
        f_min = self.min_force_physics
        f_max = self.config["force_range"][1]

        slip_r     = 0.0
        rot_r      = 0.0
        force_r    = 0.0
        delta_r    = 0.0
        bonus_r    = 0.0
        drift_r    = 0.0
        collapse_r = 0.0

        # 1) slip 패널티
        if slip_vel > self.config["slip_threshold"]:
            slip_r = -15.0 * slip_vel

        # 2) 회전 패널티
        if rot_vel > self.config["slip_threshold"]:
            rot_r = -10.0 * rot_vel

        # 3) ✅ 파지력 보상 (Gaussian 형태)
        #
        # 보상 구조:
        #   force < f_min  → 물리적으로 부족 → 강한 패널티 (-5.0×부족량)
        #   force = f_min  → 최적! → +10.0 (최고 보상)
        #   force 올라갈수록 → 보상 감소 (+5.0까지)
        #   force > f_max  → 너무 강함 → 패널티 (-3.0×초과량)
        #
        # 비유:
        #   음식을 딱 필요한 만큼만 먹는 게 최고
        #   너무 조금 먹으면 배고픔 (패널티)
        #   너무 많이 먹어도 감점
        if force_mag < f_min:
            # 물리적으로 부족 → 강한 패널티
            force_r = -5.0 * (f_min - force_mag)

        elif force_mag <= f_max:
            # 범위 안 → f_min에 가까울수록 높은 보상
            range_size = f_max - f_min
            if range_size > 0:
                margin     = force_mag - f_min        # f_min 초과분
                efficiency = 1.0 - (margin / range_size)  # 1.0(f_min) → 0.0(f_max)
                force_r    = 5.0 + 5.0 * efficiency   # +10.0(f_min) → +5.0(f_max)
            else:
                force_r = +5.0

        else:
            # f_max 초과 → 찌그러짐 위험
            force_r = -3.0 * (force_mag - f_max)

        # 4) delta 크기 패널티 (불필요한 힘 변화 억제)
        delta_r = -0.5 * np.abs(delta[0])

        # 5) 완벽 파지 보너스
        actual_contact = (np.linalg.norm(left_force) +
                          np.linalg.norm(right_force)) / 2.0
        has_contact = actual_contact > 1.0
        if (slip_vel < self.config["slip_threshold"] and
                f_min <= force_mag <= f_max and
                has_contact):
            bonus_r = +10.0

        # 6) total_drift 패널티
        if total_drift > 0.001:
            drift_r = -50.0 * total_drift

        # 7) force 붕괴 패널티 (squeeze-out 감지)
        if (self.prev_total_force > 5.0 and
                force_mag < self.prev_total_force * 0.5):
            collapse_r = -20.0
        self.prev_total_force = force_mag

        # 디버그 출력 (200스텝마다)
        self._debug_count += 1
        if self._debug_count % 200 == 0:
            total = slip_r+rot_r+force_r+delta_r+bonus_r+drift_r+collapse_r
            efficiency_val = 1.0 - ((force_mag - f_min) / max(f_max - f_min, 1e-6))
            efficiency_val = max(0.0, min(1.0, efficiency_val))
            print(f"  [reward] slip={slip_r:+.2f} rot={rot_r:+.2f} "
                  f"force={force_r:+.2f} delta={delta_r:+.2f} "
                  f"bonus={bonus_r:+.2f} drift={drift_r:+.2f} "
                  f"collapse={collapse_r:+.2f} | total={total:+.2f} "
                  f"| f_min={f_min:.2f}N force={force_mag:.2f}N "
                  f"efficiency={efficiency_val:.2f}")

        return float(slip_r + rot_r + force_r + delta_r +
                     bonus_r + drift_r + collapse_r)

    # ── 1 스텝 진행 ───────────────────────────────────────────────────────────
    def step(self, action):
        delta = np.clip(action[0], -3.0, 3.0)

        self.current_force = np.clip(
            self.base_force + delta,
            self.config["force_range"][0],
            self.config["force_range"][1]
        )

        grip_ctrl = (1.0 - self.current_force / 10.0) * 255.0
        self.data.ctrl[self.gripper_act_id] = np.clip(grip_ctrl, 0, 255)

        for i in range(7):
            self.data.ctrl[i] = self.GRASP_CTRL[i]

        mujoco.mj_step(self.model, self.data)

        obs    = self._get_obs()
        reward = self._compute_reward(obs, action)

        obj_z        = self.data.qpos[self.obj_qpos_id + 2]
        total_drift  = float(obs[20])
        dropped      = bool(obj_z < 0.4)
        squeezed_out = bool(total_drift > 0.015)

        if squeezed_out:
            reward -= 30.0

        self.step_count += 1
        done = bool(dropped or squeezed_out or self.step_count >= self.max_steps)

        info = {
            "material":          self.material,
            "base_force":        self.base_force,
            "delta":             float(delta),
            "final_force":       float(self.current_force),
            "slip_velocity":     float(np.linalg.norm(obs[9:12])),
            "total_drift":       float(total_drift),
            "current_mass":      self.current_mass,
            "current_friction":  self.current_friction,
            "min_force_physics": self.min_force_physics,
            "dropped":           dropped,
            "squeezed_out":      squeezed_out,
        }
        return obs, reward, done, False, info

    # ── 에피소드 초기화 ───────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        for i, name in enumerate(self.ARM_JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = self.GRASP_CTRL[i]

        # 랜덤화 + 물리 기반 최소 파지력 계산
        self._randomize_material()

        self.model.opt.gravity[2] = 0.0

        self.data.qpos[self.left_qpos_id]  = 0.04
        self.data.qpos[self.right_qpos_id] = 0.04
        self.data.qvel[:] = 0.0

        self.data.qpos[self.obj_qpos_id:self.obj_qpos_id+3]   = self.OBJ_INIT_POS
        self.data.qpos[self.obj_qpos_id+3:self.obj_qpos_id+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

        for _ in range(500):
            for i in range(7):
                self.data.ctrl[i] = self.GRASP_CTRL[i]
            self.data.ctrl[self.gripper_act_id] = 0
            mujoco.mj_step(self.model, self.data)

        self.model.opt.gravity[2] = -9.81
        for _ in range(500):
            for i in range(7):
                self.data.ctrl[i] = self.GRASP_CTRL[i]
            self.data.ctrl[self.gripper_act_id] = 0
            mujoco.mj_step(self.model, self.data)

        self.init_obj_pos     = self.data.qpos[self.obj_qpos_id:self.obj_qpos_id+3].copy()
        self.prev_total_force = 0.0
        self.current_force    = self.base_force
        self.step_count       = 0
        self._debug_count     = 0

        mujoco.mj_forward(self.model, self.data)
        obj_z  = self.data.qpos[self.obj_qpos_id + 2]
        status = "✅ 잡힘" if obj_z > 0.4 else "❌ 떨어짐"
        print(f"reset | z={obj_z:.4f} | {status} | "
              f"재질={self.material} | "
              f"mass={self.current_mass:.3f}kg | "
              f"friction={self.current_friction:.3f} | "
              f"min_force={self.min_force_physics:.2f}N")

        return self._get_obs(), {}

    def render(self):
        pass

    def close(self):
        pass
