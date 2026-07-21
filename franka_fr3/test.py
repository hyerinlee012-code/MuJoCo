import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("fr3_panda_gripper.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[0] = 0
        data.ctrl[1] = 0
        data.ctrl[2] = 0
        data.ctrl[3] = -1.57079
        data.ctrl[4] = 0
        data.ctrl[5] = 1.57079
        data.ctrl[6] = -0.7853

        data.ctrl[7] = 255   # gripper open
        # data.ctrl[7] = 0   # gripper close

        mujoco.mj_step(model, data)
        viewer.sync()
