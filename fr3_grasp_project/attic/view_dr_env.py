"""
FR3 Domain Randomization 환경 뷰어
====================================
실행: python3 view_dr_env.py

스페이스바: 새 캔으로 리셋 (랜덤화 적용)
Q / ESC  : 종료
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np
import time


from envs.grasp_env import GraspFeedbackEnv

def main():
    print("환경 초기화 중...")
    env = GraspFeedbackEnv()  # viewer는 직접 띄울 거라 None
    obs, _ = env.reset()

    p = env.can_params
    print(f"\n[캔 파라미터]")
    print(f"  반지름:  {p['radius']*100:.1f} cm")
    print(f"  높이:    {p['half_height']*200:.1f} cm")
    print(f"  마찰력:  {p['friction']:.2f}")
    print(f"  질량:    {p['mass']*1000:.0f} g")
    print(f"  광택:    {p['shininess']:.2f}")
    print(f"\n스페이스바: 새 캔으로 리셋  |  ESC: 종료\n")

    with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -20
        viewer.cam.distance  = 1.8
        viewer.cam.lookat    = [0.5, 0, 0.6]

        step = 0
        while viewer.is_running():
            # 랜덤 액션으로 계속 움직임
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)

            viewer.sync()
            time.sleep(0.002)  # 실시간 속도

            step += 1

            # 500스텝마다 자동 리셋 (새 캔)
            if terminated or truncated or step % 500 == 0:
                obs, _ = env.reset()
                step = 0
                p = env.can_params
                print(f"[리셋] 반지름={p['radius']*100:.1f}cm  "
                      f"마찰력={p['friction']:.2f}  "
                      f"질량={p['mass']*1000:.0f}g  "
                      f"광택={p['shininess']:.2f}")

    env.close()

if __name__ == "__main__":
    main()
