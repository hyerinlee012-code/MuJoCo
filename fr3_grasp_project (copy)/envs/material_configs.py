# envs/material_configs.py
#
# friction 3가지가 물리적으로 연동:
#   torsional = 0.01  × sliding  (비틀림 마찰)
#   rolling   = 0.0005 × sliding  (구름 마찰)
#
# friction_range 설계 근거 (미끄러지지 않는 조건):
#   friction ≥ (mass × 9.81) / (2 × grasp_force)
#
#   낮은 friction + 무거운 mass → 최대 파지력으로 간신히 가능 (어려운 케이스)
#   높은 friction + 가벼운 mass → 최소 파지력으로도 충분    (쉬운 케이스)

import numpy as np  # random 대신 numpy 사용 (시드 제어, 재현성 보장)


MATERIAL_CONFIGS = {
    "can": {
        "mass":           0.30,
        "friction":       [0.45, 0.0045, 0.000225],  # torsional/rolling 연동
        "solref":         [0.01, 1.0],
        "solimp":         [0.99, 0.999, 0.001, 0.5, 2.0],
        "slip_threshold": 0.015,
        "force_range":    [2.0, 8.0],
        "color":          [0.7, 0.7, 0.7, 1.0],
        "description":    "알루미늄 캔: 딱딱하고 비교적 미끄러움",
        # 랜덤화 범위
        # friction=0.30, mass=0.40 → 최소력 6.54N (f_max=8N으로 간신히)
        # friction=0.60, mass=0.20 → 최소력 1.63N (f_min=2N으로 충분)
        "mass_range":     [0.02, 0.30],
        "friction_range": [0.2, 0.5],
    },
    "paper": {
        "mass":           0.05,
        "friction":       [0.90, 0.009, 0.00045],
        "solref":         [0.02, 0.5],
        "solimp":         [0.7, 0.95, 0.01, 0.5, 2.0],
        "slip_threshold": 0.010,
        "force_range":    [0.5, 2.0],
        "color":          [0.9, 0.85, 0.7, 1.0],
        "description":    "종이컵: 가볍고 마찰이 크지만 쉽게 변형됨",
        # friction=0.50, mass=0.08 → 최소력 0.78N (f_max=2N으로 충분)
        # friction=1.10, mass=0.03 → 최소력 0.13N (f_min=0.5N으로 충분)
        "mass_range":     [0.01, 0.18],
        "friction_range": [0.50, 1.10],
    },
    "plastic": {
        "mass":           0.15,
        "friction":       [0.65, 0.0065, 0.000325],
        "solref":         [0.015, 0.8],
        "solimp":         [0.85, 0.98, 0.005, 0.5, 2.0],
        "slip_threshold": 0.012,
        "force_range":    [1.5, 5.0],
        "color":          [0.4, 0.7, 1.0, 1.0],
        "description":    "플라스틱 컵: 캔보다 덜 미끄럽고 종이컵보다 마찰 작음",
        # friction=0.45, mass=0.25 → 최소력 2.73N (f_max=5N으로 충분)
        # friction=0.85, mass=0.10 → 최소력 0.58N (f_min=1.5N으로 충분)
        "mass_range":     [0.015, 0.4],
        "friction_range": [0.45, 0.85],
    }
}


def sample_material_config(material_name: str) -> dict:
    """
    매 에피소드마다 호출 → mass, friction 랜덤 샘플링
    torsional, rolling은 sliding에 비례해서 자동 계산 (물리적으로 일관성 있음)
    numpy.random 사용 → SAC 시드 제어와 연동됨
    """
    base    = MATERIAL_CONFIGS[material_name]
    mass    = float(np.random.uniform(*base["mass_range"]))
    sliding = float(np.random.uniform(*base["friction_range"]))

    # 물리적으로 연동된 마찰 계수
    torsional = 0.01   * sliding   # 비틀림 마찰 = sliding의 1%
    rolling   = 0.0005 * sliding   # 구름 마찰   = sliding의 0.05%

    cfg = dict(base)               # 기본값 복사
    cfg["mass"]     = mass
    cfg["friction"] = [sliding, torsional, rolling]

    return cfg
