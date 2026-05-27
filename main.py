"""
main.py - 뉴로모픽 컴퓨팅 시뮬레이션 진입점
==============================================
3가지 시뮬레이션 시나리오를 실행하고 결과를 시각화한다.

시뮬레이션 종류:
  1) 기본 피드포워드 신경망  - 입력(2) → 은닉(3) → 출력(1)
  2) STDP 학습 시연         - LTP / LTD 가중치 변화 비교
  3) 무작위 신경망          - 5뉴런, 포아송 입력, 재현성 시드 고정

실행 방법 (반드시 hsm-main/ 디렉터리에서):
    python main.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from Simulator import SNNSimulator

# ══════════════════════════════════════════════════════════════
#  공통 파라미터 상수 (F-09) - 이 블록에서 모든 수치를 조정
# ══════════════════════════════════════════════════════════════

# ── 시각화 설정 ────────────────────────────────────────────────
VIS_DPI      = 150          # 저장 해상도 (NF-06: 150 DPI 이상 필수)
VIS_FIGSIZE  = (12, 10)     # figure 가로·세로 크기 (인치)
OUTPUT_DIR   = "output"     # 그래프 파일을 저장할 폴더명

# ── 시뮬레이션 1: 기본 피드포워드 신경망 ──────────────────────
SIM1_N_INPUT   = 2      # 입력층 뉴런 수
SIM1_N_HIDDEN  = 3      # 은닉층 뉴런 수
SIM1_N_OUTPUT  = 1      # 출력층 뉴런 수
SIM1_STEPS     = 100    # 총 타임스텝 수
SIM1_WEIGHT    = 0.5    # 시냅스 초기 가중치
SIM1_LR        = 0.01   # STDP 학습률
# 역치=1.0 이므로 발화시키려면 입력 전류 > 1.0 필요
# 1.5 → 막전위 수렴값 1.5 > 역치 → 약 22스텝 후 발화
# 0.3 → 막전위 수렴값 0.3 < 역치 → 발화 없음
SIM1_PATTERN_A = [1.5, 0.3]   # 전반부 50스텝: 입력1 발화 유발, 입력2 무발화
SIM1_PATTERN_B = [0.3, 1.5]   # 후반부 50스텝: 입력1 무발화, 입력2 발화 유발
SIM1_SWITCH    = 50     # 패턴 전환 타임스텝

# ── 시뮬레이션 2: STDP 학습 시연 ──────────────────────────────
SIM2_N_NEURONS  = 3      # 총 뉴런 수
SIM2_STEPS      = 100    # 총 타임스텝 수
SIM2_LR         = 0.05   # STDP 학습률 (효과를 잘 보기 위해 높게 설정)
# 단일 타임스텝 펄스로 즉시 발화시키려면 I > 1.0/(1-exp(-1/20)) ≈ 20.4 필요
SIM2_CURRENT    = 25.0   # 단일 스텝 펄스로 즉시 스파이크 유발
# 뉴런 0 발화 시각 (사전 뉴런)
SIM2_PRE_TIMES  = [10, 40, 70]
# 뉴런 1 발화 시각 - 사전보다 나중 → LTP 유발
SIM2_POST1_TIMES = [15, 45, 65]
# 뉴런 2 발화 시각 - 사전보다 먼저 → LTD 유발
SIM2_POST2_TIMES = [5, 35, 75]

# ── 시뮬레이션 3: 무작위 신경망 ───────────────────────────────
SIM3_N_NEURONS   = 5      # 총 뉴런 수
SIM3_STEPS       = 150    # 총 타임스텝 수
SIM3_CONN_PROB   = 0.4    # 뉴런 간 연결 확률 (40%)
SIM3_WEIGHT_MIN  = 0.3    # 무작위 초기 가중치 하한
SIM3_WEIGHT_MAX  = 0.7    # 무작위 초기 가중치 상한
SIM3_LR          = 0.02   # STDP 학습률
SIM3_FIRE_RATE   = 0.3    # 포아송 입력 발화율 (30%)
SIM3_INPUT_I     = 25.0   # 단일 스텝 포아송 펄스 전류 (즉시 발화 유발)
SIM3_SEED        = 42     # 난수 시드 - 재현성 보장 (NF-04)
SIM3_N_INPUT     = 2      # 외부 입력을 받는 뉴런 수 (0, 1번)


# ══════════════════════════════════════════════════════════════
#  공통 유틸리티
# ══════════════════════════════════════════════════════════════

def _ensure_output_dir():
    """
    그래프 저장 폴더(OUTPUT_DIR)가 없으면 생성한다.
    이미 존재하면 아무것도 하지 않는다.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"  [폴더 생성] {OUTPUT_DIR}/")


def plot_simulation_results(simulator, title="Neuromorphic Computing Simulation",
                            filename="simulation"):
    """
    시뮬레이션 결과를 3개 서브플롯으로 시각화하고 파일로 저장한다 (F-06, F-07, F-08).

    서브플롯 구성:
      1) Spike Raster Plot    - 각 뉴런의 발화 시점 (F-07)
      2) Membrane Potential   - 선택 뉴런의 막전위 시계열 (F-06)
      3) Synaptic Weights     - STDP 학습에 따른 가중치 이력 (F-08)

    저장 경로: OUTPUT_DIR/<filename>.png  (VIS_DPI 해상도, NF-06)

    Args:
        simulator (SNNSimulator): 시뮬레이션이 완료된 시뮬레이터 객체
        title (str)             : figure 상단에 표시할 제목
        filename (str)          : 저장할 PNG 파일명 (확장자 제외)
    """
    # 결과 데이터 가져오기
    spike_log     = simulator.get_spike_log()
    potential_log = simulator.get_potential_log()

    # 3행 1열 서브플롯 생성
    fig, axes = plt.subplots(3, 1, figsize=VIS_FIGSIZE)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── 1) Spike Raster Plot (F-07) ──────────────────────────
    ax1 = axes[0]
    for neuron_idx in range(simulator.num_neurons):
        # 스파이크가 1인 타임스텝만 추출해 수직선으로 표시
        spike_times = [
            t for t, s in enumerate(spike_log[neuron_idx]) if s == 1
        ]
        ax1.vlines(spike_times,
                   neuron_idx - 0.4, neuron_idx + 0.4,
                   colors='red', linewidth=2)

    ax1.set_xlim(0, len(spike_log[0]))          # x축: 타임스텝 범위
    ax1.set_ylim(-0.5, simulator.num_neurons - 0.5)
    ax1.set_xlabel('Time Step (타임스텝)', fontsize=10)
    ax1.set_ylabel('Neuron Index (뉴런 번호)', fontsize=10)
    ax1.set_title('Spike Raster Plot - 뉴런별 발화 시점', fontsize=11)
    ax1.set_yticks(range(simulator.num_neurons))
    ax1.grid(True, alpha=0.3)

    # ── 2) Membrane Potential (F-06) ─────────────────────────
    ax2 = axes[1]
    # 뉴런 수에 따라 색상 팔레트 자동 생성
    colors = plt.cm.viridis(np.linspace(0, 1, simulator.num_neurons))
    for neuron_idx in range(simulator.num_neurons):
        ax2.plot(
            potential_log[neuron_idx],
            label=f'뉴런 {neuron_idx}',
            color=colors[neuron_idx],
            linewidth=1.5,
        )

    ax2.axhline(y=1.0, color='gray', linestyle='--',
                linewidth=1, alpha=0.6, label='역치 (threshold=1.0)')  # 역치선 표시
    ax2.set_xlabel('Time Step (타임스텝)', fontsize=10)
    ax2.set_ylabel('Membrane Potential (막전위)', fontsize=10)
    ax2.set_title('Membrane Potential - 막전위 시계열', fontsize=11)
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── 3) Synaptic Weights (F-08) ───────────────────────────
    ax3 = axes[2]
    if simulator.synapses:
        for synapse in simulator.synapses:
            label = f'W({synapse.pre_idx}→{synapse.post_idx})'
            ax3.plot(synapse.weight_history,
                     label=label, linewidth=2,
                     marker='o', markersize=4)

        ax3.set_xlabel('STDP Update Step (가중치 업데이트 횟수)', fontsize=10)
        ax3.set_ylabel('Synaptic Weight (가중치)', fontsize=10)
        ax3.set_title('Synaptic Weights - STDP 학습에 따른 가중치 변화', fontsize=11)
        ax3.legend(loc='best', fontsize=8)
        ax3.set_ylim(-0.05, 1.05)   # 가중치 범위 [0, 1] 에 여백 추가
        ax3.grid(True, alpha=0.3)
    else:
        # 시냅스가 없는 경우 안내 메시지 표시
        ax3.text(0.5, 0.5, '시냅스 없음 (No synapses defined)',
                 ha='center', va='center', fontsize=12, color='gray')
        ax3.set_title('Synaptic Weights', fontsize=11)

    plt.tight_layout()

    # ── 파일 저장 (NF-06: 150 DPI 이상) ────────────────────
    _ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    fig.savefig(save_path, dpi=VIS_DPI, bbox_inches='tight')
    print(f"  [저장 완료] {save_path}  (저장 DPI={VIS_DPI})")

    plt.show()
    plt.close(fig)   # 메모리 해제


# ══════════════════════════════════════════════════════════════
#  시뮬레이션 1: 기본 피드포워드 신경망
# ══════════════════════════════════════════════════════════════

def simulation_1_basic():
    """
    기본 피드포워드 신경망 시뮬레이션 (F-04)

    구조: 입력층(2개) → 은닉층(3개) → 출력층(1개) = 총 6개 뉴런
    특징: 50스텝마다 입력 패턴을 전환하여 네트워크 반응 변화를 관찰
    """
    print("\n" + "=" * 60)
    print("  시뮬레이션 1: 기본 피드포워드 신경망")
    print("=" * 60)
    print(f"  구조: 입력층({SIM1_N_INPUT}) → 은닉층({SIM1_N_HIDDEN}) → 출력층({SIM1_N_OUTPUT})")
    print(f"  총 타임스텝: {SIM1_STEPS}")

    # ── 시뮬레이터 생성 (총 뉴런 수 = 입력+은닉+출력) ─────
    total_neurons = SIM1_N_INPUT + SIM1_N_HIDDEN + SIM1_N_OUTPUT
    simulator = SNNSimulator(num_neurons=total_neurons)

    # 뉴런 인덱스 매핑
    # 입력층: 0, 1
    # 은닉층: 2, 3, 4
    # 출력층: 5
    input_start  = 0
    hidden_start = SIM1_N_INPUT
    output_start = SIM1_N_INPUT + SIM1_N_HIDDEN

    # ── 입력층 → 은닉층 시냅스 연결 (F-04) ───────────────
    for i in range(SIM1_N_INPUT):
        for j in range(SIM1_N_HIDDEN):
            simulator.add_synapse(
                pre_idx        = input_start + i,
                post_idx       = hidden_start + j,
                initial_weight = SIM1_WEIGHT,
                learning_rate  = SIM1_LR,
            )

    # ── 은닉층 → 출력층 시냅스 연결 (F-04) ───────────────
    for i in range(SIM1_N_HIDDEN):
        simulator.add_synapse(
            pre_idx        = hidden_start + i,
            post_idx       = output_start,
            initial_weight = SIM1_WEIGHT,
            learning_rate  = SIM1_LR,
        )

    # ── 입력 시퀀스 생성 (전반/후반 패턴 전환) ─────────────
    input_sequence = []
    for t in range(SIM1_STEPS):
        if t < SIM1_SWITCH:
            input_sequence.append(SIM1_PATTERN_A)   # 전반: 입력1 강, 입력2 약
        else:
            input_sequence.append(SIM1_PATTERN_B)   # 후반: 입력1 약, 입력2 강

    print(f"\n  >> 실행 중... ({SIM1_STEPS} 타임스텝)")
    simulator.run(input_sequence, num_steps=SIM1_STEPS)
    print("  [완료] 시뮬레이션 완료")

    # ── 결과 분석: 뉴런별 총 스파이크 수 출력 ─────────────
    print("\n  결과 - 뉴런별 스파이크 발생 횟수:")
    for i in range(total_neurons):
        count = sum(simulator.get_spike_log()[i])
        layer = ("입력층" if i < hidden_start
                 else "은닉층" if i < output_start
                 else "출력층")
        print(f"    뉴런 {i:2d} ({layer}): {count:3d}회")

    # ── 결과 시각화 및 저장 ────────────────────────────────
    plot_simulation_results(
        simulator,
        title    = "시뮬레이션 1: 기본 피드포워드 신경망",
        filename = "sim1_feedforward",
    )


# ══════════════════════════════════════════════════════════════
#  시뮬레이션 2: STDP 학습 시연
# ══════════════════════════════════════════════════════════════

def simulation_2_stdp_learning():
    """
    STDP 학습 시연 시뮬레이션 (F-05)

    구조: 뉴런 0(사전) → 뉴런 1(LTP 대상), 뉴런 2(LTD 대상)

    타이밍 설계:
      - 뉴런 1: 뉴런 0보다 나중에 발화 → delta_t > 0 → LTP → 가중치 증가
      - 뉴런 2: 뉴런 0보다 먼저 발화  → delta_t < 0 → LTD → 가중치 감소
    """
    print("\n" + "=" * 60)
    print("  시뮬레이션 2: STDP 학습 - 스파이크 타이밍 의존성")
    print("=" * 60)
    print("  목표: Pre-Post 타이밍에 따른 LTP / LTD 가중치 변화 관찰")
    print(f"  뉴런 0 발화 시각 (사전): {SIM2_PRE_TIMES}")
    print(f"  뉴런 1 발화 시각 (사후): {SIM2_POST1_TIMES}  → delta_t > 0 → LTP 예상")
    print(f"  뉴런 2 발화 시각 (사후): {SIM2_POST2_TIMES}  → delta_t < 0 → LTD 예상")

    # ── 시뮬레이터 생성 ────────────────────────────────────
    simulator = SNNSimulator(num_neurons=SIM2_N_NEURONS)

    # 뉴런 0 → 뉴런 1 (LTP 관찰용 시냅스)
    simulator.add_synapse(0, 1, initial_weight=0.5, learning_rate=SIM2_LR)
    # 뉴런 0 → 뉴런 2 (LTD 관찰용 시냅스)
    simulator.add_synapse(0, 2, initial_weight=0.5, learning_rate=SIM2_LR)

    # ── 입력 시퀀스 구성 (기본값: 0 전류) ─────────────────
    input_sequence = [[0.0] * SIM2_N_NEURONS for _ in range(SIM2_STEPS)]

    # 뉴런 0 발화 시각에 강한 전류 주입 (사전 뉴런)
    for t in SIM2_PRE_TIMES:
        input_sequence[t][0] = SIM2_CURRENT

    # 뉴런 1 발화 시각에 강한 전류 주입 (사전보다 나중 → LTP)
    for t in SIM2_POST1_TIMES:
        input_sequence[t][1] = SIM2_CURRENT

    # 뉴런 2 발화 시각에 강한 전류 주입 (사전보다 먼저 → LTD)
    for t in SIM2_POST2_TIMES:
        input_sequence[t][2] = SIM2_CURRENT

    print(f"\n  >> 실행 중... ({SIM2_STEPS} 타임스텝)")
    simulator.run(input_sequence, num_steps=SIM2_STEPS)
    print("  [완료] 시뮬레이션 완료")

    # ── 결과 분석: 최종 가중치 출력 ────────────────────────
    print("\n  결과 - 학습 후 최종 시냅스 가중치:")
    for synapse in simulator.synapses:
        expected = "↑ LTP 예상" if synapse.post_idx == 1 else "↓ LTD 예상"
        print(f"    W({synapse.pre_idx}→{synapse.post_idx}): "
              f"{synapse.weight:.4f}  ({expected})")

    # ── 결과 시각화 및 저장 ────────────────────────────────
    plot_simulation_results(
        simulator,
        title    = "시뮬레이션 2: STDP 학습 - LTP / LTD 비교",
        filename = "sim2_stdp",
    )


# ══════════════════════════════════════════════════════════════
#  시뮬레이션 3: 무작위 신경망
# ══════════════════════════════════════════════════════════════

def simulation_3_random_network():
    """
    무작위 신경망 시뮬레이션 (NF-04: 재현성)

    구조: 5개 뉴런, 40% 확률로 무작위 연결
    입력: 포아송 과정으로 생성한 확률적 스파이크 입력
    재현성: np.random.seed(SIM3_SEED) 로 매번 동일한 결과 보장
    """
    print("\n" + "=" * 60)
    print("  시뮬레이션 3: 무작위 신경망")
    print("=" * 60)
    print(f"  구조: {SIM3_N_NEURONS}개 뉴런, 연결 확률 {SIM3_CONN_PROB*100:.0f}%")
    print(f"  난수 시드: {SIM3_SEED}  (NF-04 재현성 보장)")

    # ── 난수 시드 고정 (재현성, NF-04) ────────────────────
    np.random.seed(SIM3_SEED)

    # ── 시뮬레이터 생성 ────────────────────────────────────
    simulator = SNNSimulator(num_neurons=SIM3_N_NEURONS)

    # ── 무작위 연결 생성 ───────────────────────────────────
    print("\n  연결 구조 (무작위 생성):")
    connection_count = 0
    for pre in range(SIM3_N_NEURONS):
        for post in range(SIM3_N_NEURONS):
            if pre != post and np.random.random() < SIM3_CONN_PROB:
                # 가중치도 지정 범위 내에서 무작위 설정
                w = np.random.uniform(SIM3_WEIGHT_MIN, SIM3_WEIGHT_MAX)
                simulator.add_synapse(pre, post,
                                      initial_weight=w,
                                      learning_rate=SIM3_LR)
                print(f"    뉴런 {pre} → {post}  (초기 가중치: {w:.3f})")
                connection_count += 1

    print(f"  총 연결 수: {connection_count}개")

    # ── 포아송 입력 시퀀스 생성 ────────────────────────────
    # 매 타임스텝마다 각 입력 뉴런이 SIM3_FIRE_RATE 확률로 독립 발화
    input_sequence = []
    for _ in range(SIM3_STEPS):
        # 처음 SIM3_N_INPUT 개 뉴런만 외부 입력을 받음
        # 포아송 과정: SIM3_FIRE_RATE 확률로 SIM3_INPUT_I 전류 인가
        # 단일 스텝 펄스이므로 즉시 발화하려면 SIM3_INPUT_I > 20.4 필요
        currents = [
            SIM3_INPUT_I if np.random.random() < SIM3_FIRE_RATE else 0.0
            for _ in range(SIM3_N_INPUT)
        ]
        # 나머지 뉴런은 0 입력 (시냅스를 통해서만 자극받음)
        currents += [0.0] * (SIM3_N_NEURONS - SIM3_N_INPUT)
        input_sequence.append(currents)

    print(f"\n  >> 실행 중... ({SIM3_STEPS} 타임스텝, 발화율={SIM3_FIRE_RATE*100:.0f}%)")
    simulator.run(input_sequence, num_steps=SIM3_STEPS)
    print("  [완료] 시뮬레이션 완료")

    # ── 결과 분석: 뉴런별 총 스파이크 수 및 발화율 ────────
    print(f"\n  결과 - 뉴런별 총 스파이크 수 (/ {SIM3_STEPS} 스텝):")
    spike_log = simulator.get_spike_log()
    for i in range(SIM3_N_NEURONS):
        count       = sum(spike_log[i])
        firing_rate = count / SIM3_STEPS * 100
        print(f"    뉴런 {i}: {count:3d}회  (발화율 {firing_rate:.1f}%)")

    # ── 결과 시각화 및 저장 ────────────────────────────────
    plot_simulation_results(
        simulator,
        title    = "시뮬레이션 3: 무작위 신경망 (seed=42)",
        filename = "sim3_random",
    )


# ══════════════════════════════════════════════════════════════
#  메인 메뉴
# ══════════════════════════════════════════════════════════════

def print_menu():
    """인터랙티브 선택 메뉴를 출력한다."""
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
    # 프로그램 시작 배너 출력
    print("\n" + "#" * 60)
    print("#  뉴로모픽 컴퓨팅 시뮬레이션")
    print("#  일운농업고등학교  21013 이은수")
    print("#  그래프는 output/ 폴더에 PNG로 저장됩니다.")
    print("#" * 60)

    # 인터랙티브 메뉴 루프
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
            # 3가지 시뮬레이션 전부 순서대로 실행
            simulation_1_basic()
            simulation_2_stdp_learning()
            simulation_3_random_network()
        elif choice == "0":
            print("\n  프로그램을 종료합니다.")
            break
        else:
            print("\n  [경고] 잘못된 입력입니다. 0~4 사이 숫자를 입력하세요.")
