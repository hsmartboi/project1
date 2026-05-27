"""
neuromorphic_simulator.py
==========================================================
뉴로모픽 컴퓨팅 시뮬레이터 (단일 파일 통합본)
일운농업고등학교 21013 이은수

구성:
  1부. LIFNeuron  클래스  — LIF 뉴런 모델          (Neuron.py)
  2부. Synapse    클래스  — 시냅스 + STDP 학습     (Synapse.py)
  3부. SNNSimulator 클래스 — SNN 시뮬레이션 엔진   (Simulator.py)
  4부. 시뮬레이션 3종 + 시각화                     (main.py)

실행 방법:
    python neuromorphic_simulator.py

참고 문헌:
  - ETRI 전자통신동향분석 35권 3호, 뉴로모픽 반도체 기술 동향 (2020)
  - IBM Korea, 뉴로모픽 컴퓨팅이란 무엇인가? (2025)
  - 박성수·오성록, STDP 알고리즘이 SNN 학습에 미치는 영향 분석 (2021)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ===========================================================
#  1부. LIF 뉴런 모델 (F-01, F-02, F-03)
# ===========================================================
#
#  LIF 미분방정식 (정규화 형태):
#      tau_m * dV/dt = -V + I(t)
#
#  수치 해법 - 지수 감쇠 근사:
#      V(t+dt) = exp(-dt/tau_m) * V(t) + (1 - exp(-dt/tau_m)) * I(t)
#
#  참고: 생물학적 단위(-70 mV, -55 mV 등) 대신
#        정규화된 무단위(threshold=1.0, reset=0.0) 형태를 사용한다.
# ===========================================================

# ── 뉴런 기본 파라미터 상수 (F-09) ──────────────────────────
DEFAULT_TAU_M            = 20.0  # 막시정수 (time steps)
DEFAULT_THRESHOLD        = 1.0   # 스파이크 역치 (정규화 단위)
DEFAULT_RESET_POTENTIAL  = 0.0   # 리셋 전위
DEFAULT_REFRACTORY       = 5     # 불응기 길이 (time steps)


class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) 뉴런 모델 (F-01, F-02, F-03)

    외부 전류가 입력되면 막전위가 누적되고, 역치를 초과하면
    스파이크를 발생시킨 뒤 전위를 리셋한다.
    스파이크 직후에는 불응기(refractory period) 동안 추가 발화가 억제된다.
    """

    def __init__(
        self,
        tau_m             = DEFAULT_TAU_M,
        threshold         = DEFAULT_THRESHOLD,
        reset_potential   = DEFAULT_RESET_POTENTIAL,
        refractory_period = DEFAULT_REFRACTORY,
    ):
        """
        LIF 뉴런 초기화

        Args:
            tau_m (float)            : 막시정수, 기본값 20.0
            threshold (float)        : 스파이크 발생 임계값, 기본값 1.0
            reset_potential (float)  : 리셋 후 막전위, 기본값 0.0
            refractory_period (int)  : 불응기 길이 (time steps), 기본값 5

        Raises:
            ValueError: 파라미터가 물리적으로 유효하지 않을 때 (F-10)
        """
        # ── 입력 유효성 검사 (F-10) ──────────────────────────
        if tau_m <= 0:
            raise ValueError(f"[LIFNeuron] tau_m은 양수여야 합니다. 입력값: {tau_m}")
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

        # ── 모델 파라미터 ─────────────────────────────────────
        self.tau_m             = tau_m             # 막시정수
        self.threshold         = threshold         # 발화 역치
        self.reset_potential   = reset_potential   # 리셋 전위
        self.refractory_period = refractory_period # 불응기 길이

        # ── 실시간 상태 변수 ──────────────────────────────────
        self.membrane_potential = reset_potential  # 현재 막전위
        self.spike              = False            # 이번 스텝 발화 여부
        self.refractory_count   = 0               # 남은 불응기 카운터

        # ── 이력 기록 (시각화용) ──────────────────────────────
        self.potential_history = []   # 매 스텝 막전위 저장
        self.spike_times       = []   # 스파이크 발생 타임스텝 목록

    def update(self, input_current, dt=1.0, time_step=0):
        """
        한 타임스텝 뉴런 상태 업데이트 (F-01, F-02, F-03)

        처리 순서:
          1) 불응기 중이면 막전위를 리셋값으로 고정하고 카운터 감소
          2) LIF 지수 감쇠 방정식으로 막전위 갱신
          3) 역치 초과 시 스파이크 발생 + 막전위 리셋 + 불응기 시작
          4) 막전위 이력 기록

        수식 (지수 감쇠 근사):
            decay = exp(-dt / tau_m)
            V(t+dt) = decay * V(t) + (1 - decay) * I(t)

        Args:
            input_current (float): 이번 스텝 총 입력 전류 (외부 + 시냅스)
            dt (float)           : 시간 간격, 기본값 1.0
            time_step (int)      : 현재 타임스텝 번호 (스파이크 기록용)
        """
        self.spike = False   # 이번 스텝 스파이크 플래그 초기화

        if self.refractory_count > 0:
            # ── 불응기 처리 (F-03) ────────────────────────────
            self.refractory_count   -= 1
            self.membrane_potential  = self.reset_potential  # 전위 강제 고정
        else:
            # ── LIF 방정식 수치 적분 (F-01) ──────────────────
            decay = np.exp(-dt / self.tau_m)
            self.membrane_potential = (
                decay * self.membrane_potential       # 이전 전압 지수 감쇠
                + (1.0 - decay) * input_current       # 입력 전류 누적 충전
            )

            # ── 역치 초과 → 스파이크 발생 (F-02) ─────────────
            if self.membrane_potential >= self.threshold:
                self.spike = True                             # 발화 플래그 ON
                self.spike_times.append(time_step)            # 발화 시각 기록
                self.membrane_potential = self.reset_potential # 막전위 리셋
                self.refractory_count   = self.refractory_period # 불응기 시작

        # ── 막전위 이력 기록 ──────────────────────────────────
        self.potential_history.append(self.membrane_potential)

    def get_spike(self):
        """이번 타임스텝 스파이크 여부 반환 (bool)"""
        return self.spike

    def reset(self):
        """뉴런 상태 초기화 (파라미터 유지, 실행 상태만 초기화)"""
        self.membrane_potential = self.reset_potential
        self.spike              = False
        self.refractory_count   = 0
        self.potential_history  = []
        self.spike_times        = []


# ===========================================================
#  2부. 시냅스 모델 + STDP 학습 규칙 (F-04, F-05)
# ===========================================================
#
#  STDP 수식 (delta_t = t_post - t_pre):
#
#    delta_t > 0  ->  LTP  Dw =  A+ * exp(-delta_t / tau_stdp)
#    delta_t < 0  ->  LTD  Dw = -A- * exp( delta_t / tau_stdp)
#
#  가중치 범위: 0 <= w <= 1 (np.clip 으로 유지)
# ===========================================================

# ── STDP 기본 파라미터 상수 (F-09) ──────────────────────────
DEFAULT_INITIAL_WEIGHT = 0.5    # 시냅스 초기 가중치
DEFAULT_LEARNING_RATE  = 0.01   # STDP 학습률
DEFAULT_TAU_STDP       = 20.0   # STDP 시상수 (time steps)
DEFAULT_A_PLUS         = 0.01   # LTP 진폭
DEFAULT_A_MINUS        = 0.01   # LTD 진폭
WEIGHT_MIN             = 0.0    # 가중치 하한
WEIGHT_MAX             = 1.0    # 가중치 상한


class Synapse:
    """
    시냅스 모델 - 뉴런 간 연결 및 STDP 기반 가소성 (F-04, F-05)

    두 뉴런(pre -> post)을 연결하며, 스파이크 타이밍 차이에 따라
    가중치를 자동으로 강화(LTP) 또는 약화(LTD)시킨다.
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
            pre_neuron_idx (int)   : 사전 뉴런 인덱스
            post_neuron_idx (int)  : 사후 뉴런 인덱스
            initial_weight (float) : 초기 가중치 [0, 1]
            learning_rate (float)  : STDP 학습률
            tau_stdp (float)       : STDP 시상수

        Raises:
            ValueError: 자기 연결이거나 파라미터가 유효하지 않을 때 (F-10)
        """
        # ── 입력 유효성 검사 (F-10) ──────────────────────────
        if pre_neuron_idx == post_neuron_idx:
            raise ValueError(
                f"[Synapse] pre_idx와 post_idx가 같을 수 없습니다 (자기 연결 금지)."
            )
        if not (WEIGHT_MIN <= initial_weight <= WEIGHT_MAX):
            raise ValueError(
                f"[Synapse] initial_weight는 [{WEIGHT_MIN}, {WEIGHT_MAX}] "
                f"범위여야 합니다. 입력값: {initial_weight}"
            )
        if learning_rate <= 0:
            raise ValueError(f"[Synapse] learning_rate는 양수여야 합니다.")
        if tau_stdp <= 0:
            raise ValueError(f"[Synapse] tau_stdp는 양수여야 합니다.")

        # ── 연결 정보 ─────────────────────────────────────────
        self.pre_idx  = pre_neuron_idx    # 신호를 보내는 뉴런
        self.post_idx = post_neuron_idx   # 신호를 받는 뉴런

        # ── 가중치 및 학습 파라미터 ──────────────────────────
        self.weight        = initial_weight
        self.learning_rate = learning_rate
        self.tau_stdp      = tau_stdp

        # ── 스파이크 타이밍 추적 (STDP 계산용) ──────────────
        self.pre_spike_time  = None   # 사전 뉴런 최근 스파이크 시각
        self.post_spike_time = None   # 사후 뉴런 최근 스파이크 시각

        # ── 가중치 이력 (시각화용) ────────────────────────────
        self.weight_history = [initial_weight]

    def stdp_rule(self, delta_t):
        """
        STDP 학습 규칙 계산 (F-05)

        delta_t = t_post - t_pre:
          delta_t > 0 : 사전이 먼저 발화 -> LTP (가중치 강화)
          delta_t < 0 : 사후가 먼저 발화 -> LTD (가중치 약화)
          delta_t = 0 : 동시 발화 -> 변화 없음

        Args:
            delta_t (int): 사후 스파이크 시각 - 사전 스파이크 시각

        Returns:
            float: 학습률 적용 전 가중치 변화량 Dw
        """
        if delta_t == 0:
            return 0.0   # 동시 발화: 변화 없음

        if delta_t > 0:
            # LTP - 사전 먼저 발화: 인과 관계 -> 연결 강화
            delta_w = DEFAULT_A_PLUS * np.exp(-delta_t / self.tau_stdp)
        else:
            # LTD - 사후 먼저 발화: 비인과 관계 -> 연결 약화
            delta_w = -DEFAULT_A_MINUS * np.exp(delta_t / self.tau_stdp)

        return delta_w * self.learning_rate  # 학습률 반영

    def update_weight(self, pre_spike_time, post_spike_time):
        """
        스파이크 타이밍에 따라 가중치 업데이트

        STDP 규칙을 적용하고 [0, 1] 범위로 클리핑한다.

        Args:
            pre_spike_time (int)  : 사전 뉴런 스파이크 타임스텝
            post_spike_time (int) : 사후 뉴런 스파이크 타임스텝
        """
        if pre_spike_time is None or post_spike_time is None:
            return

        delta_t = post_spike_time - pre_spike_time  # 타이밍 차이
        delta_w = self.stdp_rule(delta_t)            # 가중치 변화량

        if delta_w == 0.0:
            return

        # 가중치 업데이트 및 클리핑
        self.weight = float(np.clip(self.weight + delta_w, WEIGHT_MIN, WEIGHT_MAX))
        self.weight_history.append(self.weight)  # 이력 기록

    def get_output(self, pre_spike):
        """
        사전 뉴런의 스파이크를 가중치로 변조해 반환

        Args:
            pre_spike (bool): 이번 스텝 사전 뉴런 발화 여부

        Returns:
            float: 사후 뉴런으로 전달되는 시냅스 전류
        """
        return self.weight if pre_spike else 0.0

    def get_connection_info(self):
        """시냅스 연결 정보를 딕셔너리로 반환"""
        return {'pre_idx': self.pre_idx, 'post_idx': self.post_idx, 'weight': self.weight}


# ===========================================================
#  3부. SNN 시뮬레이션 엔진 (F-04)
# ===========================================================
#
#  한 타임스텝(step()) 처리 순서:
#    1) 시냅스 신호 계산  - 이전 스텝 발화 뉴런의 출력 합산
#    2) 뉴런 업데이트    - 외부 + 시냅스 전류로 LIF 방정식 적분
#    3) STDP 적용       - LTP(사후 발화) + LTD(사전 발화) 모두 처리
#
#  완전한 STDP 구현:
#    - 스파이크 시각 갱신은 LTP/LTD 계산 완료 후 수행
#      (같은 스텝에서 발화한 두 뉴런의 잘못된 자기 참조 방지)
# ===========================================================

STDP_TIME_WINDOW = 100   # STDP 계산에 사용하는 최대 타이밍 차이 (time steps)


class SNNSimulator:
    """
    Spiking Neural Network (SNN) 시뮬레이터 (F-04)

    여러 LIF 뉴런과 STDP 시냅스를 관리하며,
    주어진 입력 시퀀스에 따라 신경망을 시간 순서로 실행한다.
    층(layer) 구조는 add_synapse() 연결 패턴으로 표현한다.
    """

    def __init__(self, num_neurons):
        """
        시뮬레이터 초기화

        Args:
            num_neurons (int): 전체 뉴런 수

        Raises:
            ValueError: num_neurons < 1 일 때 (F-10)
        """
        if num_neurons < 1:
            raise ValueError(
                f"[SNNSimulator] num_neurons는 1 이상이어야 합니다. 입력값: {num_neurons}"
            )

        # ── 기본 구조 ──────────────────────────────────────────
        self.num_neurons = num_neurons
        self.neurons     = [LIFNeuron() for _ in range(num_neurons)]  # 뉴런 생성
        self.synapses    = []   # 시냅스 목록
        self.time_step   = 0    # 전역 타임스텝

        # ── 로그 (시각화용) ────────────────────────────────────
        self.spike_log     = {i: [] for i in range(num_neurons)}
        self.potential_log = {i: [] for i in range(num_neurons)}

    def add_synapse(self, pre_idx, post_idx, initial_weight=0.5, learning_rate=0.01):
        """
        두 뉴런 사이에 시냅스 연결 추가 (F-04)

        Args:
            pre_idx (int)          : 사전 뉴런 인덱스
            post_idx (int)         : 사후 뉴런 인덱스
            initial_weight (float) : 초기 가중치
            learning_rate (float)  : STDP 학습률

        Raises:
            ValueError: 인덱스가 범위를 벗어날 때 (F-10)
        """
        valid = range(self.num_neurons)
        if pre_idx not in valid:
            raise ValueError(
                f"[SNNSimulator] pre_idx({pre_idx}) 범위 초과. "
                f"유효: 0~{self.num_neurons - 1}"
            )
        if post_idx not in valid:
            raise ValueError(
                f"[SNNSimulator] post_idx({post_idx}) 범위 초과. "
                f"유효: 0~{self.num_neurons - 1}"
            )
        self.synapses.append(Synapse(pre_idx, post_idx, initial_weight, learning_rate))

    def encode_input(self, data, input_neuron_indices=None):
        """
        입력 데이터를 각 뉴런의 전류 배열로 변환 (Rate Coding)

        Args:
            data (list)                 : 입력 전류값 목록
            input_neuron_indices (list) : 입력 뉴런 인덱스 (None이면 0번부터 순서대로)

        Returns:
            numpy.ndarray: 크기 num_neurons 의 입력 전류 배열
        """
        if input_neuron_indices is None:
            input_neuron_indices = list(range(len(data)))

        input_currents = np.zeros(self.num_neurons)
        for i, idx in enumerate(input_neuron_indices):
            if i < len(data):
                input_currents[idx] = data[i]
        return input_currents

    def step(self, input_currents):
        """
        한 타임스텝 신경망 전체 실행 (핵심 루프)

        Args:
            input_currents (numpy.ndarray): 외부 입력 전류 배열
        """
        # ── 1단계: 시냅스 신호 계산 ──────────────────────────
        # 이전 스텝에서 스파이크를 낸 뉴런의 출력을 사후 뉴런에 합산
        synaptic_inputs = np.zeros(self.num_neurons)
        for synapse in self.synapses:
            if self.neurons[synapse.pre_idx].get_spike():
                synaptic_inputs[synapse.post_idx] += synapse.get_output(True)

        # ── 2단계: 뉴런 업데이트 ─────────────────────────────
        for i, neuron in enumerate(self.neurons):
            total_current = input_currents[i] + synaptic_inputs[i]
            neuron.update(total_current, time_step=self.time_step)
            # 로그 기록
            self.spike_log[i].append(1 if neuron.get_spike() else 0)
            self.potential_log[i].append(neuron.membrane_potential)

        # ── 3단계: 완전한 STDP 가중치 업데이트 ──────────────
        # 스파이크 시각 갱신은 LTP/LTD 계산 완료 후 수행
        for synapse in self.synapses:
            pre_neuron  = self.neurons[synapse.pre_idx]
            post_neuron = self.neurons[synapse.post_idx]
            pre_fired   = pre_neuron.get_spike()
            post_fired  = post_neuron.get_spike()

            # LTP: 사후 발화, 최근 사전 발화 확인 (delta_t > 0)
            if post_fired and synapse.pre_spike_time is not None:
                delta_t = self.time_step - synapse.pre_spike_time
                if 0 < delta_t <= STDP_TIME_WINDOW:
                    synapse.update_weight(synapse.pre_spike_time, self.time_step)

            # LTD: 사전 발화, 최근 사후 발화 확인 (delta_t < 0)
            if pre_fired and synapse.post_spike_time is not None:
                delta_t = synapse.post_spike_time - self.time_step
                if -STDP_TIME_WINDOW <= delta_t < 0:
                    synapse.update_weight(self.time_step, synapse.post_spike_time)

            # 스파이크 시각 갱신 (STDP 계산 완료 후)
            if pre_fired:
                synapse.pre_spike_time = self.time_step
            if post_fired:
                synapse.post_spike_time = self.time_step

        self.time_step += 1

    def run(self, input_sequence, num_steps=None):
        """
        시뮬레이션 전체 실행

        Args:
            input_sequence (list): 타임스텝별 입력 리스트
            num_steps (int)      : 총 스텝 수 (None이면 입력 시퀀스 길이)
        """
        if num_steps is None:
            num_steps = len(input_sequence)
        for idx in range(num_steps):
            currents = (self.encode_input(input_sequence[idx])
                        if idx < len(input_sequence)
                        else np.zeros(self.num_neurons))
            self.step(currents)

    def get_spike_log(self):
        """스파이크 이력 반환 {뉴런인덱스: [0/1 리스트]}"""
        return self.spike_log

    def get_potential_log(self):
        """막전위 이력 반환 {뉴런인덱스: [막전위 리스트]}"""
        return self.potential_log

    def get_synapse_weights(self):
        """현재 시냅스 가중치 정보 반환"""
        return {f"synapse_{i}": s.get_connection_info() for i, s in enumerate(self.synapses)}

    def reset(self):
        """시뮬레이터 상태 초기화 (구조 유지, 상태만 초기화)"""
        for neuron in self.neurons:
            neuron.reset()
        self.time_step     = 0
        self.spike_log     = {i: [] for i in range(self.num_neurons)}
        self.potential_log = {i: [] for i in range(self.num_neurons)}
        for synapse in self.synapses:
            synapse.pre_spike_time  = None
            synapse.post_spike_time = None


# ===========================================================
#  4부. 시뮬레이션 3종 + 시각화
# ===========================================================

# ── 공통 파라미터 상수 (F-09) ────────────────────────────────

# 시각화 설정
VIS_DPI     = 150          # 저장 해상도 (NF-06: 150 DPI 이상 필수)
VIS_FIGSIZE = (12, 10)     # figure 크기 (인치)
OUTPUT_DIR  = "output"     # 그래프 저장 폴더

# 시뮬레이션 1: 기본 피드포워드 신경망
SIM1_N_INPUT   = 2      # 입력층 뉴런 수
SIM1_N_HIDDEN  = 3      # 은닉층 뉴런 수
SIM1_N_OUTPUT  = 1      # 출력층 뉴런 수
SIM1_STEPS     = 100    # 총 타임스텝
SIM1_WEIGHT    = 0.5    # 시냅스 초기 가중치
SIM1_LR        = 0.01   # STDP 학습률
# 역치=1.0 이므로 발화시키려면 입력 전류 > 1.0 필요
SIM1_PATTERN_A = [1.5, 0.3]  # 전반 50스텝: 입력1 발화 유발, 입력2 무발화
SIM1_PATTERN_B = [0.3, 1.5]  # 후반 50스텝: 입력1 무발화, 입력2 발화 유발
SIM1_SWITCH    = 50     # 패턴 전환 타임스텝

# 시뮬레이션 2: STDP 학습 시연
SIM2_N_NEURONS   = 3      # 총 뉴런 수
SIM2_STEPS       = 100    # 총 타임스텝
SIM2_LR          = 0.05   # STDP 학습률
# 단일 타임스텝 펄스로 즉시 발화: I > 1.0 / (1 - exp(-1/20)) = 약 20.4 필요
SIM2_CURRENT     = 25.0   # 즉시 발화 유발 전류
SIM2_PRE_TIMES   = [10, 40, 70]   # 뉴런 0 발화 시각 (사전 뉴런)
SIM2_POST1_TIMES = [15, 45, 65]   # 뉴런 1 발화 시각 (LTP 대상: 사전보다 나중)
SIM2_POST2_TIMES = [5,  35, 75]   # 뉴런 2 발화 시각 (LTD 대상: 사전보다 먼저)

# 시뮬레이션 3: 무작위 신경망
SIM3_N_NEURONS  = 5      # 총 뉴런 수
SIM3_STEPS      = 150    # 총 타임스텝
SIM3_CONN_PROB  = 0.4    # 뉴런 간 연결 확률 (40%)
SIM3_WEIGHT_MIN = 0.3    # 무작위 초기 가중치 하한
SIM3_WEIGHT_MAX = 0.7    # 무작위 초기 가중치 상한
SIM3_LR         = 0.02   # STDP 학습률
SIM3_FIRE_RATE  = 0.3    # 포아송 입력 발화율 (30%)
SIM3_INPUT_I    = 25.0   # 포아송 펄스 전류 (즉시 발화 유발)
SIM3_SEED       = 42     # 난수 시드 - 재현성 보장 (NF-04)
SIM3_N_INPUT    = 2      # 외부 입력 뉴런 수 (0, 1번)


# ── 공통 유틸리티 ─────────────────────────────────────────────

def _ensure_output_dir():
    """그래프 저장 폴더가 없으면 생성한다."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"  [폴더 생성] {OUTPUT_DIR}/")


def plot_simulation_results(simulator, title="Neuromorphic Simulation", filename="simulation"):
    """
    시뮬레이션 결과를 3개 서브플롯으로 시각화하고 파일 저장 (F-06, F-07, F-08)

    서브플롯 구성:
      1) Spike Raster Plot  - 뉴런별 발화 시점 (F-07)
      2) Membrane Potential - 막전위 시계열    (F-06)
      3) Synaptic Weights   - 가중치 변화 이력 (F-08)

    저장: OUTPUT_DIR/<filename>.png  (VIS_DPI 해상도, NF-06)

    Args:
        simulator (SNNSimulator): 완료된 시뮬레이터 객체
        title (str)             : figure 제목
        filename (str)          : PNG 파일명 (확장자 제외)
    """
    spike_log     = simulator.get_spike_log()
    potential_log = simulator.get_potential_log()

    fig, axes = plt.subplots(3, 1, figsize=VIS_FIGSIZE)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── 1) Spike Raster Plot (F-07) ──────────────────────────
    ax1 = axes[0]
    for nidx in range(simulator.num_neurons):
        # 스파이크(=1)인 타임스텝만 추출해 수직선으로 표시
        times = [t for t, s in enumerate(spike_log[nidx]) if s == 1]
        ax1.vlines(times, nidx - 0.4, nidx + 0.4, colors='red', linewidth=2)
    ax1.set_xlim(0, len(spike_log[0]))
    ax1.set_ylim(-0.5, simulator.num_neurons - 0.5)
    ax1.set_xlabel('Time Step (타임스텝)', fontsize=10)
    ax1.set_ylabel('Neuron Index (뉴런 번호)', fontsize=10)
    ax1.set_title('Spike Raster Plot - 뉴런별 발화 시점', fontsize=11)
    ax1.set_yticks(range(simulator.num_neurons))
    ax1.grid(True, alpha=0.3)

    # ── 2) Membrane Potential (F-06) ─────────────────────────
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, simulator.num_neurons))
    for nidx in range(simulator.num_neurons):
        ax2.plot(potential_log[nidx], label=f'Neuron {nidx}',
                 color=colors[nidx], linewidth=1.5)
    # 역치선 표시
    ax2.axhline(y=DEFAULT_THRESHOLD, color='gray', linestyle='--',
                linewidth=1, alpha=0.6, label=f'Threshold={DEFAULT_THRESHOLD}')
    ax2.set_xlabel('Time Step (타임스텝)', fontsize=10)
    ax2.set_ylabel('Membrane Potential (막전위)', fontsize=10)
    ax2.set_title('Membrane Potential - 막전위 시계열', fontsize=11)
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── 3) Synaptic Weights (F-08) ───────────────────────────
    ax3 = axes[2]
    if simulator.synapses:
        for synapse in simulator.synapses:
            ax3.plot(synapse.weight_history,
                     label=f'W({synapse.pre_idx}->{synapse.post_idx})',
                     linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel('STDP Update Step (가중치 업데이트 횟수)', fontsize=10)
        ax3.set_ylabel('Synaptic Weight (가중치)', fontsize=10)
        ax3.set_title('Synaptic Weights - STDP 학습에 따른 가중치 변화', fontsize=11)
        ax3.legend(loc='best', fontsize=8)
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No synapses defined', ha='center', va='center',
                 fontsize=12, color='gray')
        ax3.set_title('Synaptic Weights', fontsize=11)

    plt.tight_layout()

    # ── 파일 저장 (NF-06: 150 DPI 이상) ─────────────────────
    _ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    fig.savefig(save_path, dpi=VIS_DPI, bbox_inches='tight')
    print(f"  [저장 완료] {save_path}  (저장 DPI={VIS_DPI})")

    plt.show()
    plt.close(fig)   # 메모리 해제


# ── 시뮬레이션 1: 기본 피드포워드 신경망 ─────────────────────

def simulation_1_basic():
    """
    기본 피드포워드 신경망 시뮬레이션 (F-04)

    구조: 입력층(2) -> 은닉층(3) -> 출력층(1) = 총 6개 뉴런
    특징: 50스텝마다 입력 패턴을 전환하여 네트워크 반응 변화 관찰
    """
    print("\n" + "=" * 60)
    print("  시뮬레이션 1: 기본 피드포워드 신경망")
    print("=" * 60)
    print(f"  구조: 입력층({SIM1_N_INPUT}) -> 은닉층({SIM1_N_HIDDEN}) -> 출력층({SIM1_N_OUTPUT})")

    # ── 시뮬레이터 생성 ────────────────────────────────────────
    total_neurons = SIM1_N_INPUT + SIM1_N_HIDDEN + SIM1_N_OUTPUT
    simulator     = SNNSimulator(num_neurons=total_neurons)

    # 뉴런 인덱스 매핑: 입력(0,1) -> 은닉(2,3,4) -> 출력(5)
    hidden_start = SIM1_N_INPUT
    output_start = SIM1_N_INPUT + SIM1_N_HIDDEN

    # ── 입력층 -> 은닉층 시냅스 연결 ─────────────────────────
    for i in range(SIM1_N_INPUT):
        for j in range(SIM1_N_HIDDEN):
            simulator.add_synapse(i, hidden_start + j,
                                  initial_weight=SIM1_WEIGHT, learning_rate=SIM1_LR)

    # ── 은닉층 -> 출력층 시냅스 연결 ─────────────────────────
    for i in range(SIM1_N_HIDDEN):
        simulator.add_synapse(hidden_start + i, output_start,
                              initial_weight=SIM1_WEIGHT, learning_rate=SIM1_LR)

    # ── 입력 시퀀스: 전반/후반 패턴 전환 ─────────────────────
    input_sequence = [
        SIM1_PATTERN_A if t < SIM1_SWITCH else SIM1_PATTERN_B
        for t in range(SIM1_STEPS)
    ]

    print(f"\n  >> 실행 중... ({SIM1_STEPS} 타임스텝)")
    simulator.run(input_sequence, num_steps=SIM1_STEPS)
    print("  [완료] 시뮬레이션 완료")

    # ── 결과 분석 ─────────────────────────────────────────────
    print("\n  결과 - 뉴런별 스파이크 발생 횟수:")
    for i in range(total_neurons):
        count = sum(simulator.get_spike_log()[i])
        layer = ("입력층" if i < hidden_start
                 else "은닉층" if i < output_start
                 else "출력층")
        print(f"    뉴런 {i:2d} ({layer}): {count:3d}회")

    plot_simulation_results(simulator,
                            title="시뮬레이션 1: 기본 피드포워드 신경망",
                            filename="sim1_feedforward")


# ── 시뮬레이션 2: STDP 학습 시연 ─────────────────────────────

def simulation_2_stdp_learning():
    """
    STDP 학습 시연 시뮬레이션 (F-05)

    구조: 뉴런 0(사전) -> 뉴런 1(LTP 대상), 뉴런 2(LTD 대상)
    - 뉴런 1: 뉴런 0보다 나중에 발화 -> delta_t > 0 -> LTP -> 가중치 증가
    - 뉴런 2: 뉴런 0보다 먼저 발화  -> delta_t < 0 -> LTD -> 가중치 감소
    """
    print("\n" + "=" * 60)
    print("  시뮬레이션 2: STDP 학습 - 스파이크 타이밍 의존성")
    print("=" * 60)
    print(f"  뉴런 0 발화 시각 (사전): {SIM2_PRE_TIMES}")
    print(f"  뉴런 1 발화 시각 (사후): {SIM2_POST1_TIMES}  -> delta_t > 0 -> LTP 예상")
    print(f"  뉴런 2 발화 시각 (사후): {SIM2_POST2_TIMES}  -> delta_t < 0 -> LTD 예상")

    # ── 시뮬레이터 생성 ────────────────────────────────────────
    simulator = SNNSimulator(num_neurons=SIM2_N_NEURONS)
    simulator.add_synapse(0, 1, initial_weight=0.5, learning_rate=SIM2_LR)  # LTP 관찰
    simulator.add_synapse(0, 2, initial_weight=0.5, learning_rate=SIM2_LR)  # LTD 관찰

    # ── 입력 시퀀스 구성 ──────────────────────────────────────
    input_sequence = [[0.0] * SIM2_N_NEURONS for _ in range(SIM2_STEPS)]
    for t in SIM2_PRE_TIMES:   input_sequence[t][0] = SIM2_CURRENT   # 사전 뉴런
    for t in SIM2_POST1_TIMES: input_sequence[t][1] = SIM2_CURRENT   # LTP 유발
    for t in SIM2_POST2_TIMES: input_sequence[t][2] = SIM2_CURRENT   # LTD 유발

    print(f"\n  >> 실행 중... ({SIM2_STEPS} 타임스텝)")
    simulator.run(input_sequence, num_steps=SIM2_STEPS)
    print("  [완료] 시뮬레이션 완료")

    # ── 결과 분석 ─────────────────────────────────────────────
    print("\n  결과 - 학습 후 최종 시냅스 가중치:")
    for synapse in simulator.synapses:
        expected = "[위 LTP 예상]" if synapse.post_idx == 1 else "[아래 LTD 예상]"
        print(f"    W({synapse.pre_idx}->{synapse.post_idx}): "
              f"{synapse.weight:.4f}  {expected}")

    plot_simulation_results(simulator,
                            title="시뮬레이션 2: STDP 학습 - LTP / LTD 비교",
                            filename="sim2_stdp")


# ── 시뮬레이션 3: 무작위 신경망 ──────────────────────────────

def simulation_3_random_network():
    """
    무작위 신경망 시뮬레이션 (NF-04: 재현성)

    구조: 5개 뉴런, 40% 확률로 무작위 연결
    입력: 포아송 과정으로 생성한 확률적 스파이크 입력
    재현성: np.random.seed(SIM3_SEED) 로 항상 동일한 결과 보장
    """
    print("\n" + "=" * 60)
    print("  시뮬레이션 3: 무작위 신경망")
    print("=" * 60)
    print(f"  구조: {SIM3_N_NEURONS}개 뉴런, 연결 확률 {SIM3_CONN_PROB*100:.0f}%")
    print(f"  난수 시드: {SIM3_SEED}  (NF-04 재현성 보장)")

    # ── 난수 시드 고정 (NF-04) ────────────────────────────────
    np.random.seed(SIM3_SEED)

    simulator = SNNSimulator(num_neurons=SIM3_N_NEURONS)

    # ── 무작위 연결 생성 ──────────────────────────────────────
    print("\n  연결 구조 (무작위 생성):")
    count = 0
    for pre in range(SIM3_N_NEURONS):
        for post in range(SIM3_N_NEURONS):
            if pre != post and np.random.random() < SIM3_CONN_PROB:
                w = np.random.uniform(SIM3_WEIGHT_MIN, SIM3_WEIGHT_MAX)
                simulator.add_synapse(pre, post, initial_weight=w, learning_rate=SIM3_LR)
                print(f"    뉴런 {pre} -> {post}  (초기 가중치: {w:.3f})")
                count += 1
    print(f"  총 연결 수: {count}개")

    # ── 포아송 입력 시퀀스 생성 ───────────────────────────────
    # 매 스텝마다 SIM3_FIRE_RATE 확률로 SIM3_INPUT_I 전류 인가
    # 단일 스텝 펄스이므로 즉시 발화하려면 SIM3_INPUT_I > 20.4 필요
    input_sequence = []
    for _ in range(SIM3_STEPS):
        currents = [
            SIM3_INPUT_I if np.random.random() < SIM3_FIRE_RATE else 0.0
            for _ in range(SIM3_N_INPUT)
        ]
        currents += [0.0] * (SIM3_N_NEURONS - SIM3_N_INPUT)  # 나머지 뉴런은 0
        input_sequence.append(currents)

    print(f"\n  >> 실행 중... ({SIM3_STEPS} 타임스텝, 발화율={SIM3_FIRE_RATE*100:.0f}%)")
    simulator.run(input_sequence, num_steps=SIM3_STEPS)
    print("  [완료] 시뮬레이션 완료")

    # ── 결과 분석 ─────────────────────────────────────────────
    print(f"\n  결과 - 뉴런별 총 스파이크 수 (/ {SIM3_STEPS} 스텝):")
    for i in range(SIM3_N_NEURONS):
        cnt  = sum(simulator.get_spike_log()[i])
        rate = cnt / SIM3_STEPS * 100
        print(f"    뉴런 {i}: {cnt:3d}회  (발화율 {rate:.1f}%)")

    plot_simulation_results(simulator,
                            title=f"시뮬레이션 3: 무작위 신경망 (seed={SIM3_SEED})",
                            filename="sim3_random")


# ── 메인 메뉴 ─────────────────────────────────────────────────

def print_menu():
    """인터랙티브 선택 메뉴 출력"""
    print("\n" + "=" * 60)
    print("  뉴로모픽 컴퓨팅 시뮬레이션 프로젝트")
    print("  SNN (Spiking Neural Network) 시뮬레이터")
    print("=" * 60)
    print("  실행할 시뮬레이션을 선택하세요:")
    print("    1.  기본 피드포워드 신경망")
    print("    2.  STDP 학습 - 스파이크 타이밍 의존성")
    print("    3.  무작위 신경망 시뮬레이션")
    print("    4.  모두 실행")
    print("    0.  종료")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#  뉴로모픽 컴퓨팅 시뮬레이션")
    print("#  일운농업고등학교  21013 이은수")
    print("#  그래프는 output/ 폴더에 PNG로 저장됩니다.")
    print("#" * 60)

    while True:
        print_menu()
        choice = input("\n  선택 (0~4): ").strip()

        if choice == "1":
            simulation_1_basic()
        elif choice == "2":
            simulation_2_stdp_learning()
        elif choice == "3":
            simulation_3_random_network()
        elif choice == "4":
            simulation_1_basic()
            simulation_2_stdp_learning()
            simulation_3_random_network()
        elif choice == "0":
            print("\n  프로그램을 종료합니다.")
            break
        else:
            print("\n  [경고] 잘못된 입력입니다. 0~4 사이 숫자를 입력하세요.")
