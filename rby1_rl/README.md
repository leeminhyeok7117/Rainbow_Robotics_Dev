# RB-Y1 강화학습 (MuJoCo, CPU 전용)

RB-Y1 오른팔 7 DoF 리칭 태스크를 PPO로 학습한다. GPU 없이 노트북 CPU에서 동작.

- conda 환경: `rby1_dev` (Python 3.10)
- 주요 패키지: mujoco 3.13.0 / gymnasium 1.3.0 / stable-baselines3 2.9.0 / torch 2.14.0+cpu
- 벤치마크 머신: i5-1340P (16스레드), GPU 없음

## 태스크 정의

| 항목 | 내용 |
|---|---|
| 제어 대상 | 오른팔 7관절 (`right_arm_0` ~ `right_arm_6`) |
| 고정 | 토르소/왼팔/머리는 position 액추에이터로 `READY_*` 자세 유지 |
| 제어 주기 | 50 Hz (timestep 0.002 x frame_skip 10) |
| 에피소드 | 150 스텝 = 3초 |
| 관측 (23) | 팔 q(7) + 팔 q̇(7) + TCP(3) + 목표(3) + (TCP-목표)(3) |
| 액션 (7) | `[-1,1]` → position: 목표각 증분(±0.05 rad) / torque: 관절 토크 |
| 보상 | `-거리 + 도달보너스 - 0.01*a² - 0.005*Δa² - 0.0005*q̇²` |
| 성공 | 최종 거리 < 5 cm |

## 파일

| 파일 | 역할 |
|---|---|
| `build_model.py` | SDK MuJoCo 모델 → RL용 XML 2종 생성. **최초 1회 + 모델 수정 시 실행** |
| `rby1_reach_env.py` | Gymnasium 환경. 관측/액션/보상 정의. **모든 것의 중심** |
| `train_reach.py` | 단순 학습 (최종 결과만 저장) |
| `train_ladder.py` | 체크포인트를 주기적으로 저장하며 학습 |
| `eval_reach.py` | 성공률 측정 + mp4 녹화 |
| `eval_ladder.py` | 전 체크포인트 성공률 표 출력 |
| `play_reach.py` | 대화형 뷰어에서 실시간(50Hz) 재생 |
| `*.bak` | 수정 전 백업 |

## 생성물

| 경로 | 내용 |
|---|---|
| `models/rby1_reach.xml` | position 액추에이터 (기본) |
| `models/rby1_reach_torque.xml` | 오른팔만 motor(토크), ctrlrange ±[150,150,100,100,50,50,50] |
| `runs/reach/` | 최초 학습 결과 (`ppo_reach.zip` + `vecnorm.pkl`) |
| `runs/ladder/` | position, 25k 간격 체크포인트 12개 |
| `runs/torque/` | torque, 25k 간격 체크포인트 12개 |

정책(`.zip`)과 관측 정규화 통계(`.pkl`)는 **반드시 짝을 맞춰** 로드해야 한다.

## 명령어

```bash
conda activate rby1_dev && cd ~/rb/rby1_rl

# 모델 생성 (최초 1회)
python build_model.py

# 모델 눈으로 확인
python -m mujoco.viewer --mjcf=models/rby1_reach.xml

# 학습
python train_reach.py  --steps 300000 --n-envs 8              # 단순
python train_ladder.py --steps 300000 --n-envs 8 --out runs/ladder          # 체크포인트
python train_ladder.py --steps 300000 --n-envs 8 --out runs/torque --torque # 토크 제어

# 평가
python eval_reach.py                                   # 성공률 + reach.mp4
python eval_ladder.py --dir runs/ladder                # 학습 사다리 표
python eval_ladder.py --dir runs/torque --torque

# 실시간 재생
python play_reach.py --out runs/ladder --list          # 체크포인트 목록
python play_reach.py --out runs/ladder --ckpt 75000 --speed 0.5
python play_reach.py --out runs/ladder --ckpt 300000 --seed 42   # 정책 간 공정 비교

# 학습 곡선
tensorboard --logdir runs
```

## 실측 결과

### 학습 사다리 (40 에피소드씩)

| 스텝 | position 성공률 | torque 성공률 |
|---:|---:|---:|
| 25k | 0% | 0% |
| 50k | 18% | 0% |
| 75k | 38% | 0% |
| 100k | 82% | 18% |
| 125k | 98% | 38% |
| 150k | **100%** | 68% |
| 250k | 98% | **100%** |
| 300k | 100% (13mm) | 100% (12mm) |

토크 제어는 같은 성능에 약 **2배의 샘플**이 필요.

### 학습 여부 검증 (position 300k)

| 조건 | 성공률 | 평균 오차 |
|---|---:|---:|
| 랜덤 액션 | 0% | 310mm |
| 0 액션 | 0% | 241mm |
| 학습 전 신경망 | 0% | 216mm |
| **학습된 정책** | **100%** | **14mm** |
| 학습 범위 밖 목표 | 58% | 146mm |

마지막 행이 핵심: IK 솔버였다면 범위 밖에서도 100%여야 한다. 58%로 무너지는 것이 학습된 함수라는 증거.

## 함정 (직접 겪은 것)

1. **SDK 모델이 최신 MuJoCo에서 로드 실패** — `rby1_v1.x.xml`이 같은 `<include>`를 중복 사용(바퀴 2, 손가락 4). `build_model.py`가 평탄화로 해결.
2. **접촉 계산이 속도를 18배 깎음** — contact ON 3,508 steps/s vs OFF 62,709 steps/s. 리칭은 반드시 OFF.
3. **`SubprocVecEnv`는 `if __name__ == "__main__":` 가드 필수** — 없으면 forkserver가 터진다.
4. **서브프로세스에 변수 전달은 lambda 기본인자로** — `global`은 재임포트 시 초기화된다.
5. **`Pendulum-v1` + PPO 기본값은 학습 안 됨** — `gamma=0.99`가 부적합. `gamma=0.9`, `lr=1e-3`, `use_sde=True`, 100k 스텝 필요.
6. **`save_freq`는 환경 1개당 스텝 기준** — `n_envs`로 나눠야 한다. `save_vecnormalize=True`도 필수.
7. **모델의 `damping=50`, `armature=10`이 비정상적으로 큼** — 중력 토크(최대 28Nm)보다 댐핑이 크다. 이 마찰이 PD 제어기 역할을 일부 대신하므로 토크 제어가 실제보다 쉬워 보인다.

## 다음 단계 후보

| 태스크 | 수정할 것 | 난이도 |
|---|---|---|
| 정밀도 상향 | `success_radius` 0.05 → 0.01 | ⭐⭐ |
| 진짜 토크 제어 | `dof_damping` 50→2, `dof_armature` 10→0.5 | ⭐⭐⭐⭐ |
| 희소 보상 | `-d` 제거, 도달 시 +1만 | ⭐⭐⭐⭐ |
| 토르소+팔 (13 DoF) | `CTRL_JOINTS = TORSO + ARM`, 자세 페널티 추가 | ⭐⭐⭐ |
| 6D 포즈 리칭 | 관측에 회전오차, 보상에 각도항 | ⭐⭐⭐ |
| 접촉 있는 태스크 | `disable_contact=False` + 테이블/장애물 | ⭐⭐⭐⭐ |

## 실물 이전 시 주의

- 액션 변화율 페널티(`0.005`)를 절대 제거하지 말 것 — 없으면 실물에서 고주파 진동
- 도메인 랜덤화 필요: 링크 질량 ±10%, 마찰, kp, 관측 노이즈, 액션 지연 1~2스텝
- 실물 전 반드시 도커 시뮬(`~/rb/rby1_sim/docker-compose.sim.yaml`)에 `rby1_sdk`로 붙여 리허설
- 정책 출력(관절 목표각)이 SDK `JointPositionCommand`와 1:1 대응 → `rby1-sdk/examples/python/32_command_stream.py` 참고
