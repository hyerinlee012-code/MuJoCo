import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


# ─────────────────────────────────────────────
#  캔 도메인 랜덤화 범위
# ─────────────────────────────────────────────
CAN_RADIUS_RANGE  = (0.030, 0.055)   # 반지름 (m)
CAN_HEIGHT_RANGE  = (0.050, 0.130)   # 높이 절반값 (m)
CAN_FRICTION_RANGE = (0.3, 1.5)      # 마찰력
CAN_MASS_RANGE    = (0.10, 0.50)     # 질량 (kg)
CAN_RGBA_RANGE    = (0.2, 1.0)       # 색상 (반사율 포함)

# 그리퍼 기본 힘
BASE_FORCE = 5.0   # can 재질 기준
FORCE_DELTA = 3.0  # 액션 범위 ±3N


XML_TEMPLATE = """
<mujoco model="fr3_grasp_dr">
  <compiler meshdir="/home/cdsl/mujoco_menagerie/franka_fr3/assets" autolimits="true"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <mesh name="link0"  file="link0.stl"/>
    <mesh name="link1"  file="link1.stl"/>
    <mesh name="link2"  file="link2.stl"/>
    <mesh name="link3"  file="link3.stl"/>
    <mesh name="link4"  file="link4.stl"/>
    <mesh name="link5"  file="link5.stl"/>
    <mesh name="link6"  file="link6.stl"/>
    <mesh name="link7"  file="link7.stl"/>
    <mesh name="hand"   file="hand.stl"/>
    <mesh name="finger" file="finger.stl"/>

    <material name="can_material"
              rgba="{can_r} {can_g} {can_b} 1.0"
              shininess="{can_shine}"
              specular="{can_specular}"/>
  </asset>

  <worldbody>
    <!-- 바닥 -->
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"
          friction="0.5 0.005 0.0001"/>

    <!-- 조명 (랜덤 강도) -->
    <light name="main_light" pos="0 0 2" dir="0 0 -1"
           diffuse="{light_d} {light_d} {light_d}"
           specular="{light_s} {light_s} {light_s}"/>
    <light name="side_light" pos="1 1 1.5" dir="-1 -1 -1"
           diffuse="{side_d} {side_d} {side_d}"
           specular="0.1 0.1 0.1"/>

    <!-- 탁자 -->
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" size="0.6 0.6 0.02"
            pos="0.5 0 0.47" rgba="0.6 0.4 0.2 1" friction="0.8 0.005 0.0001"/>
    </body>

    <!-- 캔 (크기/마찰/질량/색상 랜덤) -->
    <body name="can" pos="{can_x} {can_y} {can_z}">
      <freejoint name="can_joint"/>
      <geom name="can_geom" type="cylinder"
            size="{can_radius} {can_half_height}"
            mass="{can_mass}"
            friction="{can_friction} 0.005 0.0001"
            material="can_material"
            solimp="0.9 0.95 0.001 0.5 2"
            solref="0.02 1"/>
    </body>

    <!-- FR3 로봇 -->
    <body name="fr3_base" pos="0 0 0.47">
      <include file="/home/cdsl/mujoco_menagerie/franka_fr3/fr3_panda_gripper.xml"/>
    </body>
  </worldbody>

  <sensor>
    <!-- 그리퍼 접촉력 -->
    <touch name="left_finger_touch"  site="left_finger_site"/>
    <touch name="right_finger_touch" site="right_finger_site"/>
    <!-- 캔 위치/속도 -->
    <framepos   name="can_pos"     objtype="body" objname="can"/>
    <framelinvel name="can_linvel" objtype="body" objname="can"/>
    <!-- 그리퍼 위치 -->
    <framepos   name="gripper_pos" objtype="body" objname="hand"/>
  </sensor>
</mujoco>
"""


class FR3GraspDREnv(gym.Env):
    """
    FR3 로봇팔 + Franka Hand 파지 환경 with Domain Randomization

    매 reset()마다 캔의 크기 / 마찰력 / 질량 / 색상 / 조명을 랜덤화합니다.
    SubprocVecEnv로 N개를 병렬 실행하면 N종의 캔을 동시에 학습합니다.

    Observation (20-dim):
        0-2   : 그리퍼 위치 (x, y, z)
        3-5   : 캔 위치 (x, y, z)
        6-8   : 캔 속도 (vx, vy, vz)
        9-10  : 손가락 접촉력 (left, right)
        11-12 : 그리퍼 조인트 각도 (finger1, finger2)
        13    : 캔 반지름 (랜덤화 파라미터 → obs에 포함해 정책이 인식하도록)
        14    : 캔 높이
        15    : 캔 마찰력
        16    : 캔 질량
        17-19 : 목표 위치까지 오차 (dx, dy, dz)

    Action (1-dim):
        그리퍼 힘 델타 [-3N, +3N]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, seed=None):
        super().__init__()
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        # 현재 에피소드의 캔 파라미터 (reset에서 채워짐)
        self.can_params = {}

        # 액션/관측 공간
        self.action_space = spaces.Box(
            low=-FORCE_DELTA, high=FORCE_DELTA, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        self._model = None
        self._data  = None
        self._viewer = None
        self._current_force = BASE_FORCE
        self._step_count = 0
        self.max_steps = 500

    # ──────────────────────────────────────────
    #  Domain Randomization: 매 reset마다 호출
    # ──────────────────────────────────────────
    def _sample_can_params(self):
        r = self._rng
        return {
            "radius"      : r.uniform(*CAN_RADIUS_RANGE),
            "half_height" : r.uniform(*CAN_HEIGHT_RANGE),
            "friction"    : r.uniform(*CAN_FRICTION_RANGE),
            "mass"        : r.uniform(*CAN_MASS_RANGE),
            "color_r"     : r.uniform(*CAN_RGBA_RANGE),
            "color_g"     : r.uniform(*CAN_RGBA_RANGE),
            "color_b"     : r.uniform(*CAN_RGBA_RANGE),
            "shininess"   : r.uniform(0.1, 1.0),
            "specular"    : r.uniform(0.1, 0.8),
            # 조명
            "light_main"  : r.uniform(0.4, 1.0),
            "light_side"  : r.uniform(0.0, 0.4),
            "light_spec"  : r.uniform(0.1, 0.5),
        }

    def _build_xml(self, p):
        """캔 파라미터로 XML 문자열 생성"""
        can_z = 0.49 + p["half_height"]  # 탁자 위에 올려놓기
        return XML_TEMPLATE.format(
            can_radius      = p["radius"],
            can_half_height = p["half_height"],
            can_friction    = p["friction"],
            can_mass        = p["mass"],
            can_r           = p["color_r"],
            can_g           = p["color_g"],
            can_b           = p["color_b"],
            can_shine       = p["shininess"],
            can_specular    = p["specular"],
            light_d         = p["light_main"],
            light_s         = p["light_spec"],
            side_d          = p["light_side"],
            can_x           = 0.5553,
            can_y           = -0.0001,
            can_z           = round(can_z, 4),
        )

    # ──────────────────────────────────────────
    #  Gymnasium API
    # ──────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # 새 캔 파라미터 샘플링 → XML 재생성 → 모델 재로드
        self.can_params = self._sample_can_params()
        xml_str = self._build_xml(self.can_params)

        self._model = mujoco.MjModel.from_xml_string(xml_str)
        self._data  = mujoco.MjData(self._model)

        # 그리퍼 열린 상태로 초기화
        gripper_act_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"
        )
        if gripper_act_id >= 0:
            self._data.ctrl[gripper_act_id] = 255  # 열림

        mujoco.mj_forward(self._model, self._data)

        self._current_force = BASE_FORCE
        self._step_count = 0

        if self.render_mode == "human":
            self._init_viewer()

        return self._get_obs(), {}

    def step(self, action):
        # 힘 업데이트
        self._current_force = np.clip(
            self._current_force + float(action[0]),
            BASE_FORCE - FORCE_DELTA,
            BASE_FORCE + FORCE_DELTA,
        )

        # 그리퍼 힘 → ctrl 변환 (0=닫힘, 255=열림, 반비례)
        gripper_ctrl = np.clip(255 - (self._current_force / 8.0) * 255, 0, 255)
        gripper_act_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"
        )
        if gripper_act_id >= 0:
            self._data.ctrl[gripper_act_id] = gripper_ctrl

        # 시뮬레이션 스텝
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        obs     = self._get_obs()
        reward  = self._compute_reward()
        terminated = self._is_terminated()
        truncated  = self._step_count >= self.max_steps

        if self.render_mode == "human" and self._viewer:
            self._viewer.sync()

        return obs, reward, terminated, truncated, {}

    # ──────────────────────────────────────────
    #  관측값 구성
    # ──────────────────────────────────────────
    def _get_obs(self):
        d, m = self._data, self._model

        def sensor(name):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                return np.zeros(3)
            adr = m.sensor_adr[sid]
            dim = m.sensor_dim[sid]
            return d.sensordata[adr:adr+dim].copy()

        gripper_pos = sensor("gripper_pos")   # 3
        can_pos     = sensor("can_pos")        # 3
        can_vel     = sensor("can_linvel")     # 3
        lf_touch    = sensor("left_finger_touch")   # 1
        rf_touch    = sensor("right_finger_touch")  # 1

        # 그리퍼 조인트 각도
        f1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        f2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        fj1 = d.qpos[m.jnt_qposadr[f1_id]] if f1_id >= 0 else 0.0
        fj2 = d.qpos[m.jnt_qposadr[f2_id]] if f2_id >= 0 else 0.0

        # 캔 파라미터 (정책이 캔 종류를 인식하도록)
        p = self.can_params
        can_info = np.array([
            p.get("radius", 0.04),
            p.get("half_height", 0.06),
            p.get("friction", 0.8),
            p.get("mass", 0.3),
        ], dtype=np.float32)

        # 목표까지 오차 (캔 위 10cm가 목표 파지 위치)
        target = can_pos + np.array([0, 0, p.get("half_height", 0.06)])
        error  = target - gripper_pos

        obs = np.concatenate([
            gripper_pos,          # 0-2
            can_pos,              # 3-5
            can_vel,              # 6-8
            lf_touch[:1],         # 9
            rf_touch[:1],         # 10
            [fj1, fj2],           # 11-12
            can_info,             # 13-16
            error,                # 17-19
        ]).astype(np.float32)

        return obs

    # ──────────────────────────────────────────
    #  보상 함수
    # ──────────────────────────────────────────
    def _compute_reward(self):
        obs = self._get_obs()
        gripper_pos = obs[0:3]
        can_pos     = obs[3:6]
        can_vel     = obs[6:9]
        lf_touch    = obs[9]
        rf_touch    = obs[10]
        error       = obs[17:20]

        p = self.can_params
        friction = p.get("friction", 0.8)
        mass     = p.get("mass", 0.3)

        # 1. 거리 보상: 그리퍼가 캔에 가까울수록
        dist = np.linalg.norm(error)
        r_approach = -dist * 2.0

        # 2. 접촉 보상: 양쪽 손가락이 캔을 잡을수록
        contact_total = lf_touch + rf_touch
        r_contact = contact_total * 0.01

        # 3. 적정 파지력 보상
        #    - 너무 약하면 슬립 → 음수 보상
        #    - 너무 강하면 캔 찌그러짐 → 음수 보상 (캔 질량/마찰 기반 적정치 계산)
        optimal_force = mass * 9.81 / max(friction, 0.1) * 1.5
        force_error = abs(self._current_force - optimal_force)
        r_force = -force_error * 0.1

        # 4. 캔이 움직이지 않아야 함 (파지 성공 시)
        can_speed = np.linalg.norm(can_vel)
        r_stable = -can_speed * 0.5 if contact_total > 0.1 else 0.0

        # 5. 캔이 들어올려졌으면 큰 보상
        lift_height = can_pos[2] - (0.49 + p.get("half_height", 0.06))
        r_lift = lift_height * 10.0 if lift_height > 0.02 else 0.0

        return r_approach + r_contact + r_force + r_stable + r_lift

    def _is_terminated(self):
        obs = self._get_obs()
        can_pos = obs[3:6]
        # 캔이 탁자 아래로 떨어지면 실패
        if can_pos[2] < 0.3:
            return True
        # 충분히 들어올리면 성공
        p = self.can_params
        lift = can_pos[2] - (0.49 + p.get("half_height", 0.06))
        if lift > 0.15:
            return True
        return False

    # ──────────────────────────────────────────
    #  렌더링
    # ──────────────────────────────────────────
    def _init_viewer(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def render(self):
        if self.render_mode == "human" and self._viewer:
            self._viewer.sync()

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
