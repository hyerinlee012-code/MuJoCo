import mujoco
import numpy as np
from PIL import Image
import os

# 1. 폴더 생성
os.makedirs('yolo_dataset/images', exist_ok=True)
os.makedirs('yolo_dataset/labels', exist_ok=True)

# 2. 모델 로드 (여기서 'scene.xml' 파일을 불러옵니다)
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=640, width=640)

# 3. 설정 (0: 유리컵)
CLASS_ID = 0 
target_body_name = 'cup_glass'
target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)

def get_yolo_coords(segmentation_img, target_id):
    rows, cols = np.where(segmentation_img[:, :, 0] == target_id)
    if len(rows) == 0: return None
    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)
    dw, dh = 1./640, 1./640
    x_center = (x_min + x_max) / 2.0 * dw
    y_center = (y_min + y_max) / 2.0 * dh
    w = (x_max - x_min) * dw
    h = (y_max - y_min) * dh
    return x_center, y_center, w, h

# 4. 루프 시작
print("데이터 생성을 시작합니다...")
for i in range(500):
    data.qpos[0:2] = [np.random.uniform(0.4, 0.7), np.random.uniform(-0.2, 0.2)]
    mujoco.mj_forward(model, data)
    
    renderer.update_scene(data, camera="material_cam")
    rgb_img = renderer.render()
    Image.fromarray(rgb_img).save(f'yolo_dataset/images/glass_{i:04d}.png')
    
    renderer.enable_segmentation_rendering()
    seg_img = renderer.render()
    coords = get_yolo_coords(seg_img, target_id)
    renderer.disable_segmentation_rendering()
    
    if coords:
        with open(f'yolo_dataset/labels/glass_{i:04d}.txt', 'w') as f:
            f.write(f"{CLASS_ID} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")

    if i % 50 == 0: print(f"{i}/500 완료")

print("완성!")
