import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from envs.grasp_env_0422 import GraspFeedbackEnv
import numpy as np


# ── 콜백 1: 보상 원인 분석 ────────────────────────────────────────────────────
class DebugCallback(BaseCallback):
    """
    200스텝마다 slip, force, drift 통계 출력
    → 어떤 패널티가 주로 발생하는지 파악
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_infos = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "slip_velocity" in info:
                self.ep_infos.append(info)

        if self.num_timesteps % 200 == 0 and self.ep_infos:
            slips    = [i["slip_velocity"]           for i in self.ep_infos]
            forces   = [i["final_force"]             for i in self.ep_infos]
            drifts   = [i.get("total_drift", 0)      for i in self.ep_infos]
            masses   = [i.get("current_mass", 0)     for i in self.ep_infos]
            frics    = [i.get("current_friction", 0) for i in self.ep_infos]
            dropped  = [i["dropped"]                 for i in self.ep_infos]
            squeezed = [i.get("squeezed_out", False) for i in self.ep_infos]

            print(f"\n[step {self.num_timesteps}] 보상 원인 분석:")
            print(f"  slip_vel  평균: {np.mean(slips):.4f}  최대: {np.max(slips):.4f}")
            print(f"  force     평균: {np.mean(forces):.2f}N  범위: {np.min(forces):.2f}~{np.max(forces):.2f}N")
            print(f"  drift     평균: {np.mean(drifts)*1000:.2f}mm  최대: {np.max(drifts)*1000:.2f}mm")
            print(f"  mass      평균: {np.mean(masses):.3f}kg  범위: {np.min(masses):.3f}~{np.max(masses):.3f}kg")
            print(f"  friction  평균: {np.mean(frics):.3f}  범위: {np.min(frics):.3f}~{np.max(frics):.3f}")
            print(f"  dropped:  {sum(dropped)}/{len(dropped)}")
            print(f"  squeezed: {sum(squeezed)}/{len(squeezed)}")
            self.ep_infos = []

        return True


# ── 콜백 2: MuJoCo 뷰어로 실시간 시연 ────────────────────────────────────────
class RenderCallback(BaseCallback):
    """
    render_freq 스텝마다 MuJoCo 뷰어를 열어서
    현재 학습된 정책이 얼마나 잘 잡는지 시각적으로 확인
    → 뷰어가 닫히면 자동으로 학습 재개
    """
    def __init__(self, render_env, render_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.render_env  = render_env
        self.render_freq = render_freq

    def _on_step(self):
        if self.num_timesteps % self.render_freq == 0:
            print(f"\n{'='*50}")
            print(f"[시연] step {self.num_timesteps} 현재 정책으로 파지 시연 중...")
            print(f"  뷰어를 닫으면 학습 재개됩니다.")
            print(f"{'='*50}")

            import mujoco.viewer
            obs, _ = self.render_env.reset()

            with mujoco.viewer.launch_passive(
                self.render_env.model,
                self.render_env.data
            ) as viewer:
                # 카메라 설정
                viewer.cam.distance  = 1.2
                viewer.cam.azimuth   = 120
                viewer.cam.elevation = -20

                for step in range(300):
                    if not viewer.is_running():
                        break

                    # 현재 학습된 정책으로 action 결정
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = self.render_env.step(action)
                    viewer.sync()
                    time.sleep(0.01)  # 실시간 속도

                    if step % 50 == 0:
                        print(f"  step {step:>3} | "
                              f"force={info['final_force']:.2f}N | "
                              f"drift={info['total_drift']*1000:.1f}mm | "
                              f"slip={info['slip_velocity']:.4f} | "
                              f"mass={info['current_mass']:.3f}kg | "
                              f"fric={info['current_friction']:.3f}")

                    if done:
                        status = "✅ 성공" if not info['dropped'] and not info['squeezed_out'] else "❌ 실패"
                        print(f"  에피소드 종료! step={step} {status}")
                        break

        return True


# ── 재질별 순서대로 학습 ──────────────────────────────────────────────────────
# can → paper → plastic (쉬운 것부터 = 커리큘럼 학습)
MATERIALS = ["can", "paper", "plastic"]

for material in MATERIALS:
    print(f"\n{'='*40}")
    print(f"  학습 시작: {material.upper()}")
    print(f"  기본 파지력: {GraspFeedbackEnv.BASE_FORCE[material]}N")
    print(f"{'='*40}")

    # 환경 생성
    env        = GraspFeedbackEnv(material=material)  # 학습용
    eval_env   = GraspFeedbackEnv(material=material)  # 평가용
    render_env = GraspFeedbackEnv(material=material)  # 시연용
    check_env(env, warn=True)

    # 저장 폴더 생성
    os.makedirs(f"./models/{material}", exist_ok=True)
    os.makedirs(f"./logs/{material}",   exist_ok=True)

    # SAC 모델
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        verbose=1,
        tensorboard_log=f"./logs/{material}/"
    )

    # 콜백: 1만 스텝마다 체크포인트 저장
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=f"./models/{material}/",
        name_prefix=f"sac_{material}"
    )

    # 콜백: 5000스텝마다 성능 평가 → 역대 최고면 best_model 저장
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{material}/best/",
        log_path=f"./logs/{material}/eval/",
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )

    # 콜백: 1만 스텝마다 뷰어 시연
    render_cb = RenderCallback(
        render_env,
        render_freq=10_000   # 1만 스텝마다 시연 (너무 자주하면 학습 느려짐)
    )

    # 학습 시작
    # TensorBoard: 다른 터미널에서 아래 명령어 실행
    #   cd /home/cdsl/fr3_grasp_project
    #   tensorboard --logdir ./logs/
    #   브라우저: http://localhost:6006
    model.learn(
        total_timesteps=300_000,
        callback=[checkpoint_cb, eval_cb, DebugCallback(), render_cb],
        progress_bar=True
    )

    # 최종 저장
    model.save(f"./models/grasp_{material}_final")
    print(f"✅ {material} 학습 완료! → models/grasp_{material}_final.zip")

print("\n🎉 모든 재질 학습 완료!")
