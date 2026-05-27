"""
Synapse.py — 시냅스 모델 및 STDP 학습 규칙
============================================
뉴런 간 연결을 나타내는 시냅스 클래스.
Spike-Timing-Dependent Plasticity(STDP) 규칙으로 가중치를 자동 업데이트한다.

STDP 수식 (F-05):
    delta_t = t_post - t_pre

    delta_t > 0  →  LTP (Long-Term Potentiation, 가중치 강화)
        Δw =  A₊ · exp(-delta_t / τ_stdp)

    delta_t < 0  →  LTD (Long-Term Depression, 가중치 약화)
        Δw = -A₋ · exp( delta_t / τ_stdp)

    최종 업데이트: w ← clip(w + Δw × learning_rate,  0, 1)
"""

import numpy as np

# ══════════════════════════════════════════════════════════════
#  STDP 기본 파라미터 상수 — 코드 상단에서 쉽게 조정 가능 (F-09)
# ══════════════════════════════════════════════════════════════
DEFAULT_INITIAL_WEIGHT = 0.5    # 시냅스 초기 가중치
DEFAULT_LEARNING_RATE  = 0.01   # STDP 학습률 — Δw 에 곱해지는 전체 배율
DEFAULT_TAU_STDP       = 20.0   # STDP 시상수 (time steps) — 클수록 더 먼 시차도 학습
DEFAULT_A_PLUS         = 0.01   # LTP 진폭 — 가중치 강화의 최대 크기
DEFAULT_A_MINUS        = 0.01   # LTD 진폭 — 가중치 약화의 최대 크기
WEIGHT_MIN             = 0.0    # 가중치 하한 (음의 가중치 금지)
WEIGHT_MAX             = 1.0    # 가중치 상한 (포화 방지)


class Synapse:
    """
    시냅스 모델 — 뉴런 간 연결 및 STDP 기반 가소성 (F-04, F-05)

    두 뉴런(pre → post)을 연결하며, 사전·사후 뉴런의 스파이크 타이밍 차이에
    따라 가중치를 자동으로 강화(LTP) 또는 약화(LTD)시킨다.

    Attributes:
        pre_idx (int)          : 사전 뉴런(pre-synaptic neuron) 인덱스
        post_idx (int)         : 사후 뉴런(post-synaptic neuron) 인덱스
        weight (float)         : 현재 시냅스 가중치 [0, 1]
        learning_rate (float)  : STDP 학습률
        tau_stdp (float)       : STDP 시상수
        pre_spike_time (int)   : 사전 뉴런의 가장 최근 스파이크 타임스텝
        post_spike_time (int)  : 사후 뉴런의 가장 최근 스파이크 타임스텝
        weight_history (list)  : 가중치 변화 이력 (시각화용)
    """

    def __init__(
        self,
        pre_neuron_idx,
        post_neuron_idx,
        initial_weight = DEFAULT_INITIAL_WEIGHT,
        learning_rate  = DEFAULT_LEARNING_RATE,
        tau_stdp       = DEFAULT_TAU_STDP,
    ):
        """
        시냅스 초기화

        Args:
            pre_neuron_idx (int)   : 신호를 보내는 사전 뉴런 인덱스
            post_neuron_idx (int)  : 신호를 받는 사후 뉴런 인덱스
            initial_weight (float) : 초기 가중치 [0, 1], 기본값 0.5
            learning_rate (float)  : STDP 학습률, 기본값 0.01
            tau_stdp (float)       : STDP 시상수 (time steps), 기본값 20.0

        Raises:
            ValueError: 인덱스가 동일하거나 파라미터가 유효하지 않을 때 (F-10)
        """
        # ── 입력 유효성 검사 (F-10) ──────────────────────────────
        if pre_neuron_idx == post_neuron_idx:
            raise ValueError(
                f"[Synapse] pre_idx와 post_idx가 같을 수 없습니다 "
                f"(자기 연결 금지). 입력값: {pre_neuron_idx}"
            )
        if not (WEIGHT_MIN <= initial_weight <= WEIGHT_MAX):
            raise ValueError(
                f"[Synapse] initial_weight는 [{WEIGHT_MIN}, {WEIGHT_MAX}] "
                f"범위여야 합니다. 입력값: {initial_weight}"
            )
        if learning_rate <= 0:
            raise ValueError(
                f"[Synapse] learning_rate는 양수여야 합니다. "
                f"입력값: {learning_rate}"
            )
        if tau_stdp <= 0:
            raise ValueError(
                f"[Synapse] tau_stdp는 양수여야 합니다. "
                f"입력값: {tau_stdp}"
            )

        # ── 연결 정보 ─────────────────────────────────────────────
        self.pre_idx  = pre_neuron_idx    # 신호를 보내는 뉴런
        self.post_idx = post_neuron_idx   # 신호를 받는 뉴런

        # ── 가중치 및 학습 파라미터 ──────────────────────────────
        self.weight        = initial_weight   # 현재 시냅스 강도
        self.learning_rate = learning_rate    # 학습 속도 배율
        self.tau_stdp      = tau_stdp         # STDP 시상수: 클수록 더 먼 타이밍도 학습

        # ── 스파이크 타이밍 추적 (STDP 계산용) ──────────────────
        self.pre_spike_time  = None   # 사전 뉴런의 가장 최근 스파이크 시각
        self.post_spike_time = None   # 사후 뉴런의 가장 최근 스파이크 시각

        # ── 가중치 이력 — 시각화·분석용 ──────────────────────────
        self.weight_history = [initial_weight]   # 업데이트마다 현재값 기록

    # ────────────────────────────────────────────────────────────
    def stdp_rule(self, delta_t):
        """
        STDP 학습 규칙: 타이밍 차이 → 가중치 변화량 계산 (F-05)

        delta_t = t_post - t_pre 로 정의:
          - delta_t > 0 : 사전 뉴런이 먼저 발화 → LTP (인과적 연결 강화)
          - delta_t < 0 : 사후 뉴런이 먼저 발화 → LTD (비인과적 연결 약화)
          - delta_t = 0 : 동시 발화 → 변화 없음

        수식:
            Δw =  A₊ · exp(-delta_t / τ)   if delta_t > 0   (LTP)
            Δw = -A₋ · exp( delta_t / τ)   if delta_t < 0   (LTD)

        Args:
            delta_t (int): 사후 스파이크 시각 - 사전 스파이크 시각

        Returns:
            float: 학습률 적용 전 가중치 변화량 Δw
        """
        if delta_t == 0:
            return 0.0   # 동시 발화: 변화 없음

        if delta_t > 0:
            # LTP — 사전이 먼저, 사후가 나중: 인과 관계 → 연결 강화
            # delta_t가 클수록 지수 감쇠로 Δw가 작아짐
            delta_w = DEFAULT_A_PLUS * np.exp(-delta_t / self.tau_stdp)
        else:
            # LTD — 사후가 먼저, 사전이 나중: 비인과 관계 → 연결 약화
            # delta_t가 음수이므로 exp(delta_t/tau)는 0~1 범위
            delta_w = -DEFAULT_A_MINUS * np.exp(delta_t / self.tau_stdp)

        # 학습률을 곱해 실제 업데이트 크기 조절
        return delta_w * self.learning_rate

    # ────────────────────────────────────────────────────────────
    def update_weight(self, pre_spike_time, post_spike_time):
        """
        스파이크 타이밍에 따라 가중치 업데이트

        STDP 규칙을 적용하고 가중치를 [WEIGHT_MIN, WEIGHT_MAX] 범위로 클리핑한다.
        가중치가 실제로 변화한 경우에만 이력에 기록한다.

        Args:
            pre_spike_time (int)  : 사전 뉴런의 스파이크 타임스텝
            post_spike_time (int) : 사후 뉴런의 스파이크 타임스텝
        """
        # 두 뉴런 모두 스파이크 기록이 있어야 STDP 계산 가능
        if pre_spike_time is None or post_spike_time is None:
            return

        # 타이밍 차이 계산 및 STDP 가중치 변화량 산출
        delta_t = post_spike_time - pre_spike_time
        delta_w = self.stdp_rule(delta_t)

        if delta_w == 0.0:
            return   # 변화량이 0이면 이력 기록 불필요

        # 가중치 업데이트 후 [0, 1] 범위로 강제 제한
        self.weight = float(
            np.clip(self.weight + delta_w, WEIGHT_MIN, WEIGHT_MAX)
        )

        # 변화된 가중치를 이력에 추가 (시각화용)
        self.weight_history.append(self.weight)

    # ────────────────────────────────────────────────────────────
    def get_output(self, pre_spike):
        """
        사전 뉴런의 스파이크를 가중치로 변조해 출력

        스파이크가 있으면 가중치(w) 크기의 전류를,
        스파이크가 없으면 0을 사후 뉴런으로 전달한다.

        Args:
            pre_spike (bool): 사전 뉴런의 이번 스텝 발화 여부

        Returns:
            float: 사후 뉴런으로 전달되는 시냅스 전류 (0 또는 weight)
        """
        return self.weight if pre_spike else 0.0

    # ────────────────────────────────────────────────────────────
    def get_connection_info(self):
        """
        시냅스 연결 정보를 딕셔너리로 반환

        Returns:
            dict: pre_idx, post_idx, weight 를 담은 딕셔너리
        """
        return {
            'pre_idx' : self.pre_idx,
            'post_idx': self.post_idx,
            'weight'  : self.weight,
        }
