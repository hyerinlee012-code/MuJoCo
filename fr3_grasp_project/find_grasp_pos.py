# find_grasp_pos.py
import mujoco
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(base_dir, "xmls", "grasp_scene.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)

# 그리퍼 열린 상태로 안정화
for _ in range(500):
    data.ctrl[0] = 0
    data.ctrl[1] = 0
    data.ctrl[2] = 0
    data.ctrl[3] = -1.57079
    data.ctrl[4] = 0
    data.ctrl[5] = 1.57079
    data.ctrl[6] = -0.7853
    data.ctrl[7] = 255  # 완전히 열림
    mujoco.mj_step(model, data)

left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

left_pos  = data.xpos[left_id]
right_pos = data.xpos[right_id]
mid_pos   = (left_pos + right_pos) / 2

print(f"열린 상태 left_finger:  {left_pos}")
print(f"열린 상태 right_finger: {right_pos}")
print(f"두 손가락 중간:         {mid_pos}")
print(f"\n물체 배치 좌표: [{mid_pos[0]:.4f}, {mid_pos[1]:.4f}, {mid_pos[2]:.4f}]")

