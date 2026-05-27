"""
Neuron.py — LIF(Leaky Integrate-and-Fire) 뉴런 모델
======================================================
뉴로모픽 컴퓨팅의 핵심 단위인 스파이킹 뉴런을 정규화된 단위로 구현한다.

LIF 미분방정식 (정규화 형태):
    τ_m · dV/dt = -V + I(t)

수치 해법 — 지수 감쇠 근사 (오일러보다 안정적):
    V(t+dt) = exp(-dt/τ_m) · V(t) + (1 - exp(-dt/τ_m)) · I(t)

참고:
    이 구현은 생물학적 단위(-70 mV, -55 mV 등) 대신
    정규화된 무단위(threshold=1.0, reset=0.0) 형태를 사용한다.
    요구사항 명세서(F-01)의 V_rest 기반 공식과 대응 관계:
        V_rest   ↔  0.0  (reset_potential)
        V_thresh ↔  1.0  (threshold)
        V_reset  ↔  0.0  (reset_potential)
"""

import numpy as np

# ══════════════════════════════════════════════════════════════
#  기본 파라미터 상수 — 코드 상단에서 쉽게 조정 가능 (F-09)
# ══════════════════════════════════════════════════════════════
DEFAULT_TAU_M            = 20.0   # 막시정수 (time steps) — 클수록 전압이 천천히 감쇠
DEFAULT_THRESHOLD        = 1.0    # 스파이크 역치 (정규화 단위, 생물학적 ≈ -55 mV)
DEFAULT_RESET_POTENTIAL  = 0.0    # 스파이크 후 리셋 전위 (생물학적 ≈ -75 mV)
DEFAULT_REFRACTORY       = 5      # 불응기 길이 (time steps, 생물학적 ≈ 2~5 ms)


class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) 뉴런 모델 (F-01, F-02, F-03)

    생물학적 뉴런의 발화(firing) 동작을 수식으로 단순화한 모델.
    외부 전류가 입력되면 막전위가 누적되고, 역치를 초과하면
    스파이크(action potential)를 발생시킨 뒤 전위를 리셋한다.
    스파이크 직후에는 불응기(refractory period) 동안 추가 발화가 억제된다.

    Attributes:
        tau_m (float)             : 막시정수 — 전기 신호 감쇠 속도 결정
        threshold (float)         : 스파이크 발생 임계값
        reset_potential (float)   : 스파이크 이후 막전위가 돌아오는 값
        refractory_period (int)   : 불응기 길이 (time steps)
        membrane_potential (float): 현재 막전위
        spike (bool)              : 이번 타임스텝 발화 여부
        refractory_count (int)    : 남은 불응기 카운터 (0이면 불응기 없음)
        potential_history (list)  : 매 스텝 막전위 이력 — 시각화용
        spike_times (list)        : 스파이크 발생 타임스텝 목록
    """

    def __init__(
        self,
        tau_m            = DEFAULT_TAU_M,
        threshold        = DEFAULT_THRESHOLD,
        reset_potential  = DEFAULT_RESET_POTENTIAL,
        refractory_period= DEFAULT_REFRACTORY,
    ):
        """
        LIF 뉴런 초기화

        Args:
            tau_m (float)            : 막시정수 (time steps), 기본값 20.0
            threshold (float)        : 스파이크 발생 임계값, 기본값 1.0
            reset_potential (float)  : 리셋 후 막전위, 기본값 0.0
            refractory_period (int)  : 불응기 길이 (time steps), 기본값 5

        Raises:
            ValueError: 파라미터가 물리적으로 유효하지 않을 때 (F-10)
        """
        # ── 입력 유효성 검사 (F-10) ─────────────────────────────────
        if tau_m <= 0:
            raise ValueError(
                f"[LIFNeuron] tau_m은 양수여야 합니다. 입력값: {tau_m}"
            )
        if threshold <= reset_potential:
            raise ValueError(
                f"[LIFNeuron] threshold({threshold})는 "
                f"reset_potential({reset_potential})보다 커야 합니다."
            )
        if refractory_period < 0:
            raise ValueError(
                f"[LIFNeuron] refractory_period는 0 이상이어야 합니다. "
                f"입력값: {refractory_period}"
            )

        # ── 모델 파라미터 저장 ─────────────────────────────────────
        self.tau_m             = tau_m             # 막시정수
        self.threshold         = threshold         # 발화 역치
        self.reset_potential   = reset_potential   # 리셋 전위
        self.refractory_period = refractory_period # 불응기 길이

        # ── 실시간 상태 변수 ───────────────────────────────────────
        self.membrane_potential = reset_potential  # 초기 막전위 = 리셋값
        self.spike              = False            # 현재 스텝 발화 여부
        self.refractory_count   = 0               # 남은 불응기 카운터

        # ── 이력 기록 — 시각화·분석용 ─────────────────────────────
        self.potential_history = []   # 매 스텝 막전위 값 저장
        self.spike_times       = []   # 스파이크 발생 타임스텝 저장

    # ──────────────────────────────────────────────────────────────
    def update(self, input_current, dt=1.0, time_step=0):
        """
        한 타임스텝 뉴런 상태 업데이트 (F-01, F-02, F-03)

        처리 순서:
          1) 불응기 중이면 막전위를 리셋값으로 고정하고 카운터 감소
          2) 평상시 — LIF 지수 감쇠 방정식으로 막전위 갱신
          3) 역치 초과 시 스파이크 발생 + 막전위 리셋 + 불응기 시작
          4) 막전위 이력에 현재값 기록

        수식 (지수 감쇠 근사):
            decay = exp(-dt / τ_m)
            V(t+dt) = decay · V(t) + (1 - decay) · I(t)

        Args:
            input_current (float): 이번 스텝 총 입력 전류 (외부 + 시냅스 합산)
            dt (float)           : 시간 간격, 기본값 1.0 time step
            time_step (int)      : 현재 절대 타임스텝 번호 (스파이크 기록용)
        """
        # 이번 스텝 스파이크 플래그 초기화 (매 스텝마다 새로 판단)
        self.spike = False

        if self.refractory_count > 0:
            # ── 불응기 처리 (F-03) ───────────────────────────────
            # 불응기 동안에는 아무리 강한 자극도 추가 발화 불가
            self.refractory_count   -= 1               # 남은 불응기 감소
            self.membrane_potential  = self.reset_potential  # 전위 강제 고정
        else:
            # ── LIF 방정식 수치 적분 (F-01) ─────────────────────
            # 입력이 없으면 V가 0(리셋값)으로 자연 감쇠
            decay = np.exp(-dt / self.tau_m)
            self.membrane_potential = (
                decay * self.membrane_potential       # 이전 전압의 지수 감쇠 성분
                + (1.0 - decay) * input_current       # 입력 전류의 누적 충전 성분
            )

            # ── 역치 초과 → 스파이크 발생 (F-02) ────────────────
            if self.membrane_potential >= self.threshold:
                self.spike = True                          # 발화 플래그 ON
                self.spike_times.append(time_step)         # 발화 시각 기록
                self.membrane_potential = self.reset_potential  # 막전위 리셋
                self.refractory_count   = self.refractory_period  # 불응기 시작

        # ── 막전위 이력 기록 (시각화용) ──────────────────────────
        self.potential_history.append(self.membrane_potential)

    # ──────────────────────────────────────────────────────────────
    def get_spike(self):
        """
        이번 타임스텝 스파이크 발생 여부 반환

        Returns:
            bool: 발화했으면 True, 아니면 False
        """
        return self.spike

    # ──────────────────────────────────────────────────────────────
    def reset(self):
        """
        뉴런 상태 완전 초기화 — 새 시뮬레이션 시작 전 사용

        파라미터(tau_m, threshold 등)는 유지하고,
        막전위·스파이크 이력 등 실행 상태만 초기화한다.
        """
        self.membrane_potential = self.reset_potential  # 막전위를 리셋값으로 복원
        self.spike              = False                 # 발화 플래그 해제
        self.refractory_count   = 0                    # 불응기 카운터 초기화
        self.potential_history  = []                   # 막전위 이력 비우기
        self.spike_times        = []                   # 스파이크 기록 비우기
