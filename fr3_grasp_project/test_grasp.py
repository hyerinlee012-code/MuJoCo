import mujoco
import mujoco.viewer
import numpy as np
import os
import time

base_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(base_dir, "xmls", "grasp_scene.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

obj_jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
obj_qpos_id = model.jnt_qposadr[obj_jnt_id]

# ── viewer에서 찾은 포즈 ────────────────────────────────────────────────────────
# Joint 패널에서 직접 잡은 값 → ctrl로 그대로 사용 (position-controlled actuator)
GRASP_CTRL = np.array([
    0.247,   # fr3_joint1
   -0.196,   # fr3_joint2
   -0.174,   # fr3_joint3
   -1.71,    # fr3_joint4
   -0.14,    # fr3_joint5
    1.72,    # fr3_joint6
    1.12,    # fr3_joint7
    1000,      # gripper (0=닫힘, spring-loaded)
])

OBJ_POS = [0.5553, -0.0001, 0.5592]  # 검증된 contact 위치

# ── contact force 출력 ─────────────────────────────────────────────────────────
def print_contact_status(step):
    contact_count = 0
    total_force = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        magnitude = np.linalg.norm(force[:3])
        if magnitude > 0.01:
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom{contact.geom1}"
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom{contact.geom2}"
            print(f"  Contact [{g1}] ↔ [{g2}]: {magnitude:.3f} N")
            total_force += magnitude
            contact_count += 1

    obj_z = data.qpos[obj_qpos_id + 2]
    status = "✅ 잡힘" if obj_z > 0.45 else "❌ 떨어짐"
    force_status = f"총 {total_force:.2f} N ({contact_count}개)" if contact_count > 0 else "⚠️  contact 없음"
    print(f"[step {step:4d}] 물체 z={obj_z:.4f} {status} | force: {force_status}")

# ── 초기화 ─────────────────────────────────────────────────────────────────────
def init_scene():
    # home 키프레임 로드
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # 물체 배치
    data.qpos[obj_qpos_id:obj_qpos_id+3]   = OBJ_POS
    data.qpos[obj_qpos_id+3:obj_qpos_id+7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # 찾은 포즈로 이동 (500 step)
    print("포즈 이동 중...")
    for _ in range(500):
        data.ctrl[:] = GRASP_CTRL
        mujoco.mj_step(model, data)
        viewer.sync()

    # 추가 안정화 (300 step)
    print("안정화 중...")
    for _ in range(300):
        data.ctrl[:] = GRASP_CTRL
        mujoco.mj_step(model, data)
        viewer.sync()

    print("준비 완료!")

# ── 실행 ───────────────────────────────────────────────────────────────────────
print("🚀 Grasp pose 테스트 시작!")
print(f"   물체 위치: {OBJ_POS}\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance  = 1.5
    viewer.cam.azimuth   = 120
    viewer.cam.elevation = -20

    init_scene()

    step = 0
    while viewer.is_running():
        data.ctrl[:] = GRASP_CTRL
        mujoco.mj_step(model, data)
        viewer.sync()

        obj_z = data.qpos[obj_qpos_id + 2]

        if step % 100 == 0:
            print_contact_status(step)

        # 물체 떨어지면 재초기화
        if obj_z < 0.45:
            print(f"\n❌ step {step} | 물체 떨어짐 → 재초기화")
            init_scene()
            step = 0
            continue

        step += 1
        time.sleep(0.002)
