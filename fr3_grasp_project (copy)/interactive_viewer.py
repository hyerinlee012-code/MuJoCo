import mujoco
import mujoco.viewer
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(base_dir, "xmls", "grasp_scene.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)

left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
data.ctrl[7]=0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

        # 실시간으로 손가락 위치 출력
        left  = data.xpos[left_id]
        right = data.xpos[right_id]
        mid   = (left + right) / 2

        print(f"\r손가락 중간: [{mid[0]:.4f}, {mid[1]:.4f}, {mid[2]:.4f}]", end="")
