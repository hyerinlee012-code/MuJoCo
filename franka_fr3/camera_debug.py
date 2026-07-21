import mujoco
import mujoco.viewer

XML_PATH = "/home/cdsl/mujoco_menagerie/franka_fr3/fr3_panda_gripper.xml"
CAMERA_NAME = "d435i_rgb"   # 네 XML 안 camera name으로 바꿔

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = model.camera(CAMERA_NAME).id

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
