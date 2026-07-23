# envs/grasp_env.py
import gymnasium as gym
import mujoco
import numpy as np
import os
from envs.material_configs import MATERIAL_CONFIGS

class GraspFeedbackEnv(gym.Env):
    BASE_FORCE = {
        "can":     5.0,
        "paper":   1.0,
        "plastic": 3.0
    }

    def __init__(self, material="can", xml_path=None):
        super().__init__()
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "xmls", "grasp_scene.xml")

        self.material   = material
        self.config     = MATERIAL_CONFIGS[material]
        self.base_force = self.BASE_FORCE[material]
        self.xml_path   = xml_path

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self._apply_material_config()

        # ── joint ID → qpos 주소로 저장 ──────────────────────────────────────
        left_jnt_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        right_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")

        # ✅ qpos 접근용 주소 (버그 수정 핵심)
        self.left_qpos_id  = self.model.jnt_qposadr[left_jnt_id]
        self.right_qpos_id = self.model.jnt_qposadr[right_jnt_id]

        # body ID (contact force 계산용)
        self.left_body_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        self.gripper_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"
        )

        obj_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.obj_qpos_id = self.model.jnt_qposadr[obj_jnt_id]

        self.action_space = gym.spaces.Box(
            low=np.array([-3.0]),
            high=np.array([3.0]),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20,), dtype=np.float32
        )

        self.max_steps     = 300
        self.step_count    = 0
        self.current_force = self.base_force
        self.home_ctrl     = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

    def _apply_material_config(self):
        cfg     = self.config
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")
        self.model.geom_friction[geom_id] = cfg["friction"]
        self.model.geom_solref[geom_id]   = cfg["solref"]
        self.model.geom_solimp[geom_id]   = cfg["solimp"]

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.model.body_mass[body_id] = cfg["mass"]

    def _get_contact_forces(self):
        left_force  = np.zeros(3)
        right_force = np.zeros(3)

        for i in range(self.data.ncon):
            con   = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            geom1_body = self.model.geom_bodyid[con.geom1]
            geom2_body = self.model.geom_bodyid[con.geom2]

            if geom1_body == self.left_body_id or geom2_body == self.left_body_id:
                left_force  += force[:3]
            if geom1_body == self.right_body_id or geom2_body == self.right_body_id:
                right_force += force[:3]

        return left_force, right_force

    def _get_obs(self):
        left_touch  = np.array([self.data.sensordata[0]])
        right_touch = np.array([self.data.sensordata[1]])
        obj_pos     = self.data.sensordata[2:5].copy()
        obj_linvel  = self.data.sensordata[5:8].copy()
        obj_angvel  = self.data.sensordata[8:11].copy()

        # ✅ qpos 주소로 접근 (버그 수정)
        l_pos = np.array([self.data.qpos[self.left_qpos_id]])
        r_pos = np.array([self.data.qpos[self.right_qpos_id]])

        left_force, right_force = self._get_contact_forces()
        base_f = np.array([self.base_force])

        return np.concatenate([
            left_force, right_force,
            obj_pos, obj_linvel, obj_angvel,
            l_pos, r_pos,
            left_touch, right_touch,
            base_f
        ]).astype(np.float32)

    def _compute_reward(self, obs, delta):
        left_force  = obs[0:3]
        right_force = obs[3:6]
        obj_linvel  = obs[9:12]
        obj_angvel  = obs[12:15]

        slip_vel  = np.linalg.norm(obj_linvel)
        rot_vel   = np.linalg.norm(obj_angvel)
        force_mag = (np.linalg.norm(left_force) + np.linalg.norm(right_force)) / 2.0

        f_min, f_max = self.config["force_range"]
        reward = 0.0

        if slip_vel > self.config["slip_threshold"]:
            reward -= 15.0 * slip_vel
        if rot_vel > self.config["slip_threshold"]:
            reward -= 10.0 * rot_vel

        deform = self._estimate_deformation()
        if deform > self.config["max_deform_threshold"]:
            reward -= 8.0 * (deform - self.config["max_deform_threshold"])

        if f_min <= force_mag <= f_max:
            reward += 5.0
        elif force_mag < f_min:
            reward -= 3.0 * (f_min - force_mag)
        else:
            reward -= 3.0 * (force_mag - f_max)

        reward -= 0.5 * np.abs(delta[0])

        if slip_vel < 0.001 and f_min <= force_mag <= f_max:
            reward += 10.0

        return float(reward)

    def _estimate_deformation(self):
        total = 0.0
        for i in range(self.data.ncon):
            total += abs(self.data.contact[i].dist)
        k = self.config["solimp"][0]
        return total * (1.0 - k) * 0.01

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
            self.data.ctrl[i] = self.home_ctrl[i]

        mujoco.mj_step(self.model, self.data)

        obs    = self._get_obs()
        reward = self._compute_reward(obs, action)

        obj_z   = self.data.qpos[self.obj_qpos_id + 2]
        dropped = bool(obj_z < 0.4)

        self.step_count += 1
        done = bool(dropped or (self.step_count >= self.max_steps))

        info = {
            "material":      self.material,
            "base_force":    self.base_force,
            "delta":         float(delta),
            "final_force":   float(self.current_force),
            "slip_velocity": float(np.linalg.norm(obs[9:12])),
            "dropped":       dropped
        }
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.home_ctrl = self.data.ctrl[:7].copy()

        # 1단계: 중력 끄기
        self.model.opt.gravity[2] = 0

        # 2단계: 손가락 열기 ✅ qpos 주소로 접근 (버그 수정)
        self.data.qpos[self.left_qpos_id]  = 0.04
        self.data.qpos[self.right_qpos_id] = 0.04
        self.data.qvel[:] = 0

        # 3단계: 물체 배치
        self.data.qpos[self.obj_qpos_id:self.obj_qpos_id+3]   = [0.5553, -0.0001, 0.52]
        self.data.qpos[self.obj_qpos_id+3:self.obj_qpos_id+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

        # 4단계: 중력 없는 상태에서 그리퍼 닫기
        for _ in range(500):
            for i in range(7):
                self.data.ctrl[i] = self.home_ctrl[i]
            self.data.ctrl[self.gripper_act_id] = 0
            mujoco.mj_step(self.model, self.data)

        # 5단계: 중력 다시 켜고 안정화
        self.model.opt.gravity[2] = -9.81
        for _ in range(500):
            for i in range(7):
                self.data.ctrl[i] = self.home_ctrl[i]
            self.data.ctrl[self.gripper_act_id] = 0
            mujoco.mj_step(self.model, self.data)

        self.current_force = self.base_force
        self.step_count    = 0
        mujoco.mj_forward(self.model, self.data)

        obj_z = self.data.qpos[self.obj_qpos_id + 2]
        print(f"reset 후 물체 z: {obj_z:.4f} | {'✅ 잡힘' if obj_z > 0.4 else '❌ 떨어짐'}")

        return self._get_obs(), {}

    def render(self):
        pass

    def close(self):
        pass
