"""
Simulator.py — SNN(Spiking Neural Network) 시뮬레이션 엔진
============================================================
LIFNeuron 과 Synapse 를 조합해 전체 신경망을 실행하는 컨트롤러.

한 타임스텝(step()) 처리 순서:
  1) 시냅스 신호 계산  — 이전 스텝에서 발화한 뉴런의 출력 전류 합산
  2) 뉴런 업데이트    — (외부 전류 + 시냅스 전류)를 LIF 방정식에 입력
  3) STDP 적용       — 완전한 LTP / LTD 를 올바른 순서로 적용

완전한 STDP 구현 원칙:
  - LTP  : 사후 뉴런이 발화할 때 최근 사전 뉴런 발화 시각 조회
           delta_t = t_post - t_pre > 0  →  가중치 증가
  - LTD  : 사전 뉴런이 발화할 때 최근 사후 뉴런 발화 시각 조회
           delta_t = t_post - t_pre < 0  →  가중치 감소
  - 스파이크 시각 갱신은 STDP 계산 완료 후 수행 (같은 스텝 중복 계산 방지)
"""

import numpy as np
from Neuron  import LIFNeuron
from Synapse import Synapse

# ══════════════════════════════════════════════════════════════
#  시뮬레이터 파라미터 상수 (F-09)
# ══════════════════════════════════════════════════════════════
STDP_TIME_WINDOW = 100   # STDP 계산에 사용하는 최대 타이밍 차이 (time steps)
                         # 이 범위를 벗어난 스파이크 쌍은 STDP 에서 무시


class SNNSimulator:
    """
    Spiking Neural Network (SNN) 시뮬레이터 (F-04)

    여러 LIF 뉴런과 STDP 시냅스를 관리하며,
    주어진 입력 시퀀스에 따라 신경망을 타임스텝 순서로 실행한다.

    층(layer) 구조는 시뮬레이터 내부에 없으며,
    add_synapse() 로 연결 패턴을 정의함으로써 입력층-은닉층-출력층을 표현한다.

    Attributes:
        num_neurons (int)     : 전체 뉴런 수
        neurons (list)        : LIFNeuron 객체 리스트 (인덱스 = 뉴런 번호)
        synapses (list)       : Synapse 객체 리스트
        time_step (int)       : 현재 절대 타임스텝 카운터
        spike_log (dict)      : {뉴런인덱스: [스텝별 스파이크 값(0/1)]}
        potential_log (dict)  : {뉴런인덱스: [스텝별 막전위 값]}
    """

    def __init__(self, num_neurons):
        """
        시뮬레이터 초기화

        Args:
            num_neurons (int): 신경망을 구성할 전체 뉴런 수

        Raises:
            ValueError: num_neurons 가 1 미만일 때 (F-10)
        """
        # ── 입력 유효성 검사 (F-10) ──────────────────────────────
        if num_neurons < 1:
            raise ValueError(
                f"[SNNSimulator] num_neurons 는 1 이상이어야 합니다. "
                f"입력값: {num_neurons}"
            )

        # ── 기본 구조 초기화 ─────────────────────────────────────
        self.num_neurons = num_neurons
        self.neurons     = [LIFNeuron() for _ in range(num_neurons)]  # 뉴런 생성
        self.synapses    = []   # 시냅스 목록 (add_synapse 로 추가)
        self.time_step   = 0    # 전역 타임스텝 카운터

        # ── 로그 초기화 — 시각화·분석용 ─────────────────────────
        self.spike_log     = {i: [] for i in range(num_neurons)}  # 발화 이력
        self.potential_log = {i: [] for i in range(num_neurons)}  # 막전위 이력

    # ────────────────────────────────────────────────────────────
    def add_synapse(self, pre_idx, post_idx, initial_weight=0.5, learning_rate=0.01):
        """
        두 뉴런 사이에 시냅스 연결 추가 (F-04)

        이 메서드를 반복 호출하는 패턴으로
        입력층 → 은닉층 → 출력층 구조를 표현한다.

        Args:
            pre_idx (int)          : 신호를 보내는 사전 뉴런 인덱스
            post_idx (int)         : 신호를 받는 사후 뉴런 인덱스
            initial_weight (float) : 초기 시냅스 가중치 [0, 1]
            learning_rate (float)  : STDP 학습률

        Raises:
            ValueError: 인덱스가 범위를 벗어날 때 (F-10)
        """
        # ── 인덱스 범위 검사 (F-10) ──────────────────────────────
        valid_range = range(self.num_neurons)
        if pre_idx not in valid_range:
            raise ValueError(
                f"[SNNSimulator] pre_idx({pre_idx}) 가 범위를 벗어났습니다. "
                f"유효 범위: 0 ~ {self.num_neurons - 1}"
            )
        if post_idx not in valid_range:
            raise ValueError(
                f"[SNNSimulator] post_idx({post_idx}) 가 범위를 벗어났습니다. "
                f"유효 범위: 0 ~ {self.num_neurons - 1}"
            )

        # Synapse 객체 생성 후 목록에 추가
        synapse = Synapse(pre_idx, post_idx, initial_weight, learning_rate)
        self.synapses.append(synapse)

    # ────────────────────────────────────────────────────────────
    def encode_input(self, data, input_neuron_indices=None):
        """
        외부 입력 데이터를 각 뉴런의 전류 배열로 변환 (Rate Coding)

        입력값이 클수록 해당 뉴런에 더 큰 전류가 흘러
        스파이크 발생 빈도가 높아진다 (Rate Coding 방식).

        Args:
            data (list or array)        : 각 입력 뉴런에 전달할 전류값 목록
            input_neuron_indices (list) : 입력을 받는 뉴런 인덱스 목록.
                                          None 이면 0번부터 순서대로 매핑

        Returns:
            numpy.ndarray: 크기 num_neurons 의 입력 전류 배열
        """
        # 기본값: 0번 뉴런부터 데이터 길이만큼 순서대로 매핑
        if input_neuron_indices is None:
            input_neuron_indices = list(range(len(data)))

        # 입력이 없는 뉴런은 전류 0 으로 초기화
        input_currents = np.zeros(self.num_neurons)

        # 각 입력값을 해당 뉴런 인덱스에 할당
        for i, neuron_idx in enumerate(input_neuron_indices):
            if i < len(data):
                input_currents[neuron_idx] = data[i]

        return input_currents

    # ────────────────────────────────────────────────────────────
    def step(self, input_currents):
        """
        한 타임스텝 신경망 전체 실행 — 핵심 실행 루프

        처리 순서 (순서 변경 금지 — 인과성 유지 필수):
          1) 시냅스 신호 계산 : 이전 스텝 발화 뉴런의 출력을 사후 뉴런에 합산
          2) 뉴런 업데이트   : 외부 + 시냅스 전류를 합쳐 LIF 방정식 적용
          3) STDP 업데이트   : LTP(사후 발화) + LTD(사전 발화) 모두 처리,
                               스파이크 시각 갱신은 STDP 계산 완료 후 수행

        Args:
            input_currents (numpy.ndarray): 크기 num_neurons 의 외부 입력 전류 배열
        """
        # ── 1단계: 시냅스 신호 계산 ──────────────────────────────
        # 이전 타임스텝에서 스파이크를 낸 뉴런들의 출력 전류를 합산
        synaptic_inputs = np.zeros(self.num_neurons)

        for synapse in self.synapses:
            # 사전 뉴런이 발화했으면 가중치 크기의 전류를 사후 뉴런에 추가
            if self.neurons[synapse.pre_idx].get_spike():
                synaptic_inputs[synapse.post_idx] += synapse.get_output(True)

        # ── 2단계: 뉴런 업데이트 ─────────────────────────────────
        for i, neuron in enumerate(self.neurons):
            # 외부 입력 전류 + 시냅스 전류를 합산해 LIF 모델에 전달
            total_current = input_currents[i] + synaptic_inputs[i]
            neuron.update(total_current, time_step=self.time_step)

            # 이번 스텝 결과를 로그에 기록
            self.spike_log[i].append(1 if neuron.get_spike() else 0)  # 발화: 1/0
            self.potential_log[i].append(neuron.membrane_potential)    # 막전위 기록

        # ── 3단계: 완전한 STDP 가중치 업데이트 ──────────────────
        # 스파이크 시각 갱신은 LTP/LTD 계산 완료 후 수행해야
        # 같은 타임스텝에서 발화한 두 뉴런이 서로를 잘못 참조하는 것을 방지
        for synapse in self.synapses:
            pre_neuron  = self.neurons[synapse.pre_idx]
            post_neuron = self.neurons[synapse.post_idx]

            pre_fired  = pre_neuron.get_spike()   # 이번 스텝 사전 뉴런 발화 여부
            post_fired = post_neuron.get_spike()  # 이번 스텝 사후 뉴런 발화 여부

            # LTP — 사후 뉴런 발화 시: 최근 사전 뉴런 발화와의 시차 확인
            if post_fired and synapse.pre_spike_time is not None:
                delta_t = self.time_step - synapse.pre_spike_time  # 양수 → LTP
                if 0 < delta_t <= STDP_TIME_WINDOW:
                    # 사전이 먼저, 사후가 나중 → 인과 연결 강화
                    synapse.update_weight(synapse.pre_spike_time, self.time_step)

            # LTD — 사전 뉴런 발화 시: 최근 사후 뉴런 발화와의 시차 확인
            if pre_fired and synapse.post_spike_time is not None:
                delta_t = synapse.post_spike_time - self.time_step  # 음수 → LTD
                if -STDP_TIME_WINDOW <= delta_t < 0:
                    # 사후가 먼저, 사전이 나중 → 비인과 연결 약화
                    synapse.update_weight(self.time_step, synapse.post_spike_time)

            # 스파이크 시각 갱신 (STDP 계산 완료 후)
            if pre_fired:
                synapse.pre_spike_time = self.time_step
            if post_fired:
                synapse.post_spike_time = self.time_step

        # 전역 타임스텝 카운터 증가
        self.time_step += 1

    # ────────────────────────────────────────────────────────────
    def run(self, input_sequence, num_steps=None):
        """
        시뮬레이션 전체 실행

        input_sequence 의 각 항목을 한 타임스텝씩 step() 으로 처리한다.
        num_steps 가 input_sequence 보다 길면 나머지는 0 전류로 채운다.

        Args:
            input_sequence (list): 각 타임스텝의 입력 리스트
                                    [[뉴런0전류, 뉴런1전류, ...], ...]
            num_steps (int)      : 총 타임스텝 수.
                                    None 이면 input_sequence 길이를 사용
        """
        # 총 스텝 수 결정
        if num_steps is None:
            num_steps = len(input_sequence)

        for step_idx in range(num_steps):
            # 입력 시퀀스가 남아 있으면 사용, 소진되면 0 전류 입력
            if step_idx < len(input_sequence):
                input_currents = self.encode_input(input_sequence[step_idx])
            else:
                input_currents = np.zeros(self.num_neurons)

            self.step(input_currents)

    # ────────────────────────────────────────────────────────────
    #  결과 조회 메서드
    # ────────────────────────────────────────────────────────────

    def get_spike_log(self):
        """
        전체 스파이크 이력 반환

        Returns:
            dict: {뉴런인덱스: [타임스텝별 0/1 값]}
        """
        return self.spike_log

    def get_potential_log(self):
        """
        전체 막전위 이력 반환

        Returns:
            dict: {뉴런인덱스: [타임스텝별 막전위 값]}
        """
        return self.potential_log

    def get_synapse_weights(self):
        """
        현재 모든 시냅스의 가중치 정보 반환

        Returns:
            dict: {'synapse_0': {'pre_idx', 'post_idx', 'weight'}, ...}
        """
        return {
            f"synapse_{i}": synapse.get_connection_info()
            for i, synapse in enumerate(self.synapses)
        }

    # ────────────────────────────────────────────────────────────
    def reset(self):
        """
        시뮬레이터 상태 완전 초기화 — 새 시뮬레이션 시작 전 사용

        파라미터(뉴런 수, 시냅스 구조)는 유지하고
        모든 뉴런 상태, 로그, 타임스텝을 초기화한다.
        """
        # 모든 뉴런 상태 초기화
        for neuron in self.neurons:
            neuron.reset()

        # 타임스텝 카운터 및 로그 초기화
        self.time_step     = 0
        self.spike_log     = {i: [] for i in range(self.num_neurons)}
        self.potential_log = {i: [] for i in range(self.num_neurons)}

        # 시냅스 스파이크 추적 타이머 초기화
        for synapse in self.synapses:
            synapse.pre_spike_time  = None
            synapse.post_spike_time = None
