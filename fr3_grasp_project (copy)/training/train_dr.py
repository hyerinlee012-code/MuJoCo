"""
FR3 파지 강화학습 - Domain Randomization 병렬 학습
=====================================================
실행: python3 train_dr.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from envs.grasp_env import GraspFeedbackEnv

# ──────────────────────────────────────────
#  설정
# ──────────────────────────────────────────
N_ENVS        = 8          # 병렬 환경 수 (CPU 코어 수에 맞게 조절)
TOTAL_STEPS   = 2_000_000  # 전체 학습 스텝
SAVE_INTERVAL = 50_000     # 체크포인트 저장 주기
LOG_DIR       = "./logs_dr/"
SAVE_DIR      = "./models_dr/"

os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# ──────────────────────────────────────────
#  환경 생성 함수
# ──────────────────────────────────────────
def make_env(rank: int, seed: int = 0):
    """각 worker 프로세스용 환경 생성기"""
    def _init():
        env = GraspFeedbackEnv()
        return env
    set_random_seed(seed + rank)
    return _init


def main():
    print(f"병렬 환경 {N_ENVS}개 생성 중...")
    print("각 환경마다 캔의 크기/마찰력/질량/색상/조명이 다릅니다.\n")

    # N_ENVS개의 환경을 별도 프로세스로 실행
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecMonitor(env, LOG_DIR)

    # 평가용 단일 환경 (학습 중 성능 체크)
    eval_env = SubprocVecEnv([make_env(99)])
    eval_env = VecMonitor(eval_env)

    # ──────────────────────────────────────
    #  SAC 모델
    # ──────────────────────────────────────
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate   = 3e-4,
        buffer_size     = 1_000_000,
        learning_starts = 10_000,
        batch_size      = 256,
        tau             = 0.005,
        gamma           = 0.99,
        train_freq      = 1,
        gradient_steps  = 1,
        policy_kwargs   = dict(net_arch=[256, 256, 256]),  # 3층 MLP
        verbose         = 1,
        tensorboard_log = LOG_DIR,
        device          = "auto",
    )

    # ──────────────────────────────────────
    #  콜백
    # ──────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = SAVE_INTERVAL // N_ENVS,
        save_path   = SAVE_DIR,
        name_prefix = "fr3_dr",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = SAVE_DIR + "best/",
        log_path             = LOG_DIR,
        eval_freq            = SAVE_INTERVAL // N_ENVS,
        n_eval_episodes      = 10,
        deterministic        = True,
    )

    # ──────────────────────────────────────
    #  학습 시작
    # ──────────────────────────────────────
    print("=" * 50)
    print("Domain Randomization 학습 시작")
    print(f"  병렬 환경: {N_ENVS}개")
    print(f"  총 스텝:   {TOTAL_STEPS:,}")
    print(f"  로그:      {LOG_DIR}")
    print(f"  모델저장:  {SAVE_DIR}")
    print("=" * 50)
    print("\nTensorBoard 모니터링:")
    print(f"  tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = [checkpoint_cb, eval_cb],
        progress_bar    = True,
    )

    model.save(SAVE_DIR + "fr3_dr_final")
    print("\n학습 완료! 최종 모델 저장됨.")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
