import mujoco
import mujoco.viewer
import numpy as np
import os
import time

base_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(base_dir, "xmls", "grasp_scene.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)

obj_jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
obj_qpos_id = model.jnt_qposadr[obj_jnt_id]
data.qpos[obj_qpos_id:obj_qpos_id+3]   = [0.554, 0.0, 0.54]
data.qpos[obj_qpos_id+3:obj_qpos_id+7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

left_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
right_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        # 그리퍼 천천히 닫기
        grip = min(255, step * 0.5)
        data.ctrl[7] = grip
        data.ctrl[0] = 0
        data.ctrl[1] = 0
        data.ctrl[2] = 0
        data.ctrl[3] = -1.57079
        data.ctrl[4] = 0
        data.ctrl[5] = 1.57079
        data.ctrl[6] = -0.7853

        mujoco.mj_step(model, data)
        viewer.sync()

        # 접촉력 실시간 출력
        left_force  = np.zeros(3)
        right_force = np.zeros(3)

        for i in range(data.ncon):
            con   = data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            geom1_body = model.geom_bodyid[con.geom1]
            geom2_body = model.geom_bodyid[con.geom2]
            if geom1_body == left_body_id or geom2_body == left_body_id:
                left_force  += force[:3]
            if geom1_body == right_body_id or geom2_body == right_body_id:
                right_force += force[:3]

        if step % 50 == 0:
            print(f"step {step:4d} | "
                  f"grip_ctrl: {grip:6.1f} | "
                  f"ncon: {data.ncon} | "
                  f"left_force: {np.linalg.norm(left_force):.3f}N | "
                  f"right_force: {np.linalg.norm(right_force):.3f}N")

        step += 1
        time.sleep(0.002)
