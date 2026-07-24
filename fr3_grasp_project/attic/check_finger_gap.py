import mujoco
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(base_dir, "xmls", "grasp_scene.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

print("=== 열린 상태 (home) ===")
print(f"left_finger  pos: {data.xpos[left_id]}")
print(f"right_finger pos: {data.xpos[right_id]}")
gap_open = np.linalg.norm(data.xpos[left_id] - data.xpos[right_id])
print(f"간격: {gap_open*1000:.1f}mm")

left_jnt  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
right_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
data.qpos[left_jnt]  = 0.0
data.qpos[right_jnt] = 0.0
mujoco.mj_forward(model, data)

print("\n=== 닫힌 상태 (joint=0) ===")
print(f"left_finger  pos: {data.xpos[left_id]}")
print(f"right_finger pos: {data.xpos[right_id]}")
gap_closed = np.linalg.norm(data.xpos[left_id] - data.xpos[right_id])
print(f"간격: {gap_closed*1000:.1f}mm")
print(f"\n물체 지름: {0.033*2*1000:.1f}mm")
print(f"열린 간격: {gap_open*1000:.1f}mm")
print(f"닫힌 간격: {gap_closed*1000:.1f}mm")
