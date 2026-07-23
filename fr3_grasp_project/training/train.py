import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from envs.grasp_env_0422 import GraspFeedbackEnv
import numpy as np


# ── 보상 원인 분석 콜백 ────────────────────────────────────────────────────────
class DebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_infos = []

    def _on_step(self):
        # 매 스텝마다 env의 info 수집
        for info in self.locals.get("infos", []):
            if "slip_velocity" in info:
                self.ep_infos.append(info)

        # 200스텝마다 통계 출력
        if self.num_timesteps % 200 == 0 and self.ep_infos:
            slips    = [i["slip_velocity"]           for i in self.ep_infos]
            forces   = [i["final_force"]             for i in self.ep_infos]
            drifts   = [i.get("total_drift", 0)      for i in self.ep_infos]  # ✅ z_drift → total_drift
            dropped  = [i["dropped"]                 for i in self.ep_infos]
            squeezed = [i.get("squeezed_out", False) for i in self.ep_infos]

            print(f"\n[step {self.num_timesteps}] 보상 원인 분석:")
            print(f"  slip_vel  평균: {np.mean(slips):.4f}  최대: {np.max(slips):.4f}")
            print(f"  force     평균: {np.mean(forces):.2f}N  범위: {np.min(forces):.2f}~{np.max(forces):.2f}N")
            print(f"  drift     평균: {np.mean(drifts)*1000:.2f}mm  최대: {np.max(drifts)*1000:.2f}mm")  # ✅ mm 단위로 출력
            print(f"  dropped:  {sum(dropped)}/{len(dropped)}")
            print(f"  squeezed: {sum(squeezed)}/{len(squeezed)}")
            self.ep_infos = []

        return True


# ── 재질별 순서대로 학습 ──────────────────────────────────────────────────────
# can → paper → plastic 순서 (쉬운 것부터 = 커리큘럼 학습)
MATERIALS = ["plastic"] #원래는 ["can", "paper", "plastic"]

for material in MATERIALS:
    print(f"\n{'='*40}")
    print(f"  학습 시작: {material.upper()}")
    print(f"  기본 파지력: {GraspFeedbackEnv.BASE_FORCE[material]}N")
    print(f"{'='*40}")

    # 학습용 / 평가용 환경 분리
    env      = GraspFeedbackEnv(material=material)  # 학습용 (막 탐색)
    eval_env = GraspFeedbackEnv(material=material)  # 평가용 (실력 측정)
    check_env(env, warn=True)

    # 저장 폴더 생성
    os.makedirs(f"./models/{material}", exist_ok=True)
    os.makedirs(f"./logs/{material}",   exist_ok=True)

    # SAC 모델 생성
    model = SAC(
        "MlpPolicy",          # MLP 신경망: obs(21) → [256] → [256] → [128] → action(1)
        env,
        learning_rate=3e-4,   # 한 번에 얼마나 크게 배울지
        buffer_size=100_000,  # 과거 경험 저장소 (10만 개)
        batch_size=256,       # 한 번 학습할 때 꺼내는 경험 수
        gamma=0.99,           # 미래 보상 중요도 (1에 가까울수록 미래 중시)
        tau=0.005,            # 타겟 신경망 업데이트 속도
        ent_coef="auto",      # 탐색/활용 균형 자동 조절
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        verbose=1,
        tensorboard_log=f"./logs/{material}/"
    )

    # 콜백 1: 1만 스텝마다 모델 저장 (게임 세이브 포인트)
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=f"./models/{material}/",
        name_prefix=f"sac_{material}"
    )

    # 콜백 2: 5000스텝마다 성능 평가 → 역대 최고면 best_model 저장
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{material}/best/",
        log_path=f"./logs/{material}/eval/",
        eval_freq=5_000,
        n_eval_episodes=10,   # 10번 시험봐서 평균
        deterministic=True,   # 평가 시엔 랜덤 없이 최선만
        verbose=1
    )

    # 학습 시작
    model.learn(
        total_timesteps=300_000,
        callback=[checkpoint_cb, eval_cb, DebugCallback()],
        progress_bar=True
    )

    # 최종 모델 저장
    model.save(f"./models/grasp_{material}_final")
    print(f"✅ {material} 학습 완료! → models/grasp_{material}_final.zip")

print("\n🎉 모든 재질 학습 완료!")
