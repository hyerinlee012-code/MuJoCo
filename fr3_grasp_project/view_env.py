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

print("뷰어 실행 중...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance  = 1.5
    viewer.cam.azimuth   = 120
    viewer.cam.elevation = -20

    def init_scene():
        # home 포즈로 초기화
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # 물체를 손가락 사이에 배치
        data.qpos[obj_qpos_id:obj_qpos_id+3]   = [0.5553, -0.0001, 0.54]
        data.qpos[obj_qpos_id+3:obj_qpos_id+7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        # 그리퍼 닫으면서 뷰어도 같이 업데이트
        print("그리퍼 닫는 중...")
        for _ in range(500):
            data.ctrl[0] = 0
            data.ctrl[1] = 0
            data.ctrl[2] = 0
            data.ctrl[3] = -1.57079
            data.ctrl[4] = 0
            data.ctrl[5] = 1.57079
            data.ctrl[6] = -0.7853
           # data.ctrl[7] = 128 
            mujoco.mj_step(model, data)
            viewer.sync()  
        print("초기화 완료")

    init_scene()

    step = 0
    while viewer.is_running():
        data.ctrl[0] = 0
        data.ctrl[1] = 0
        data.ctrl[2] = 0
        data.ctrl[3] = -1.57079
        data.ctrl[4] = 0
        data.ctrl[5] = 1.57079
        data.ctrl[6] = -0.7853
        data.ctrl[7] = 128

        mujoco.mj_step(model, data)
        viewer.sync()

        obj_z = data.qpos[obj_qpos_id + 2]

        # 물체 떨어지면 자동 재초기화
        if obj_z < 0.4:
            print(f"step {step} | 물체 떨어짐 → 재초기화 중...")
            init_scene()

        if step % 100 == 0:
            print(f"step {step:5d} | 물체 z: {obj_z:.4f} | "
                  f"{'잡힘' if obj_z > 0.4 else ' 떨어짐'}")

        step += 1
        time.sleep(0.002)
