# 실험 (experiments)

정적 그래프의 `alpha` 값 문제를 재검증하기 위한 폴더다.

정적 `graph_constructor` 가 `alpha=propalpha`(0.05) 로 만들어지고 있었는데 원본 MTGNN 은
`alpha=tanhalpha`(3) 를 쓴다. 이 값은 baseline 과 Dynamic 양쪽에 모두 영향을 주므로,
**논문 Table 의 수치를 그대로 믿기 전에 다시 재보는 것**이 이 폴더의 목적이다.

처음 보는 사람은 [실험이해하기.md](실험이해하기.md) 를 먼저 읽으면 된다.
(무엇을 학습하는지, 왜 오래 걸리는지, 자원이 어디에 쓰이는지 설명)

## 빠른 시작

```bash
# 1. 환경 + 데이터 (한 번만)
python3 experiments/setup.py

# 2. 설정 목록 보기
./venv/bin/python experiments/run.py --list

# 3. 짧게 돌려서 파이프라인 확인 (약 4분)
#    주의: 1 epoch 결과의 horizon 3/6/12 수치는 의미 없다(아래 "epoch 은 함부로 줄이면 안 된다")
./venv/bin/python experiments/run.py --configs original fixed_base --epochs 1 --runs 1

# 4. 결과 표
./venv/bin/python experiments/summarize.py --dataset METR-LA
```

## 6가지 설정

| 설정 | 코드 | `new_graph_learning` | `tanhalpha` | 의미 |
|---|---|---|---|---|
| `original` | 원본 MTGNN 저장소 | — | 3 | 원본 논문 코드 그대로. 논문 수치가 재현되는지 |
| `thesis_base` | 이 저장소 | False | 0.05 | 석사논문이 실제로 돌린 baseline |
| `thesis_dyn` | 이 저장소 | True | 0.05 | 석사논문이 실제로 돌린 Dynamic |
| `fixed_base` | 이 저장소 | False | 3 | alpha 수정 후 baseline |
| `fixed_dyn` | 이 저장소 | True | 3 | alpha 수정 후 Dynamic (기여의 진짜 성능) |
| `fixed_base_big` | 이 저장소 | False | 3 | **용량 맞춘 baseline** — 채널 37/74/148 로 키워 사용 파라미터 483,197개 (`fixed_dyn` 실효치 486,011 대비 -0.6%) |

`original` 은 `../MTGNN_original` 체크아웃을 그대로 실행한다. 원본 소스는 전혀 건드리지
않고, 데이터/저장 경로만 절대경로로 넘긴다. 없으면 이렇게 받는다:

```bash
git clone https://github.com/nnzhan/MTGNN.git ../MTGNN_original
```

`tanhalpha` 는 `net.py` 딱 한 곳(정적 graph_constructor)에서만 쓰이기 때문에,
**소스를 고치지 않고** CLI 값만 바꿔서 수정 전(0.05)/후(3) 를 모두 재현할 수 있다.

## 이 조합으로 알 수 있는 것

읽는 순서가 곧 검증 순서다.

1. **`original` vs 원본 논문 수치** — 비슷하면 "내 맥 + MPS + 이 데이터" 조합이 정상이라는 뜻.
   여기가 어긋나면 아래 비교는 전부 의미가 없으므로 가장 먼저 확인한다.
2. **`fixed_base` vs `original`** — 비슷해야 정상. 기여 코드의 baseline 경로가 원본에
   충실하다는 확인. (완전히 같은 값은 안 나온다. `new_gc` 모듈이 쓰이지 않아도 파라미터
   초기화 단계에서 난수를 소비하므로 초기 가중치가 달라진다.)
3. **`thesis_base` vs `fixed_base`** — alpha 버그가 baseline 을 실제로 얼마나 망가뜨렸는지.
   `thesis_base` 가 나쁘다면 논문의 Dynamic 우위가 과대평가된 것이다.
4. **`fixed_dyn` vs `fixed_base`** — 양쪽 모두 올바른 alpha 를 쓴 상태에서의 순수 기여 효과.
   **논문의 주장이 살아남는지가 여기서 갈린다.**
5. **`fixed_dyn` vs `fixed_base_big`** — `fixed_dyn` 이 4번에서 이겼을 때, 그 우위가
   "동적 그래프 덕"인지 "파라미터가 많아서"인지 가려낸다. 사용 파라미터를 맞췄으므로
   여기서도 이겨야 동적 그래프 자체의 효과라고 주장할 수 있다.

   참고: `new_gc` 모듈은 `new_graph_learning=False` 여도 항상 생성되므로 **총** 파라미터는
   모든 thesis 설정에서 764,219개로 같다. 위 표의 숫자는 forward 에서 실제로 쓰이는
   파라미터 기준이다 (`fixed_dyn` 은 한 번도 호출되지 않는 `new_gc.norm` 278,208개를 제외한
   실효치).

추가로, `fixed_dyn` 의 학습이 끝나면 **동적 그래프가 정말 동적으로 동작하는지** 확인한다:

```bash
./venv/bin/python experiments/dynamics_check.py \
    --checkpoint save/METR-LA/fixed_dyn/PEMS-BAY_exp1_0.pth
```

출근 시간대와 심야 입력에 대한 인접행렬 A 를 뽑아 시간대 간 차이를 측정한다.
A 가 입력과 무관한 상수로 붕괴했다면(sigmoid 출력이라 흔한 붕괴 모드) "동적 그래프"
주장 자체가 성립하지 않으므로, 성능 비교와 별개로 반드시 확인해야 한다.
(체크포인트 파일명의 `PEMS-BAY` 는 `train_multi_step.py` 의 하드코딩이며 내용은 METR-LA 다.)

참고로 석사논문 Table 3 의 MTGNN 행은 원본 논문 값과 다르다(직접 재실행한 값으로 보임):

| METR-LA, Horizon 3 | MAE | RMSE | MAPE |
|---|---|---|---|
| 원본 논문 MTGNN | 2.69 | 5.18 | 6.86% |
| 석사논문 MTGNN | 2.60 | 4.89 | 6.86% |

PEMS-BAY 는 차이가 더 크다(원본 1.32/2.79 vs 석사논문 1.20/2.35). 그래서 `original` 을
직접 돌려 기준선을 잡는 것이 이번 재실험의 출발점이다.

## 결과 위치

```
experiments/results/<데이터셋>/<설정>/
    train.log      # 전체 학습 로그 (요약 스크립트가 이 파일을 읽는다)
    *.csv          # 이 저장소 설정만 생성 (원본 코드는 CSV 를 안 만든다)
save/<데이터셋>/<설정>/    # 체크포인트(.pth)
```

`experiments/results/`, `save/`, `data/`, `venv/` 는 모두 `.gitignore` 에 들어 있다.

## 실행 시간

M3 Pro(11코어/18GB), `--device mps`, METR-LA 실측:

| 설정 종류 | 파라미터 | 1 epoch | 100 epoch |
|---|---|---|---|
| baseline 계열 (`original`, `*_base`) | 405,452 | **67.3초** | 약 1.9시간 |
| 동적 계열 (`*_dyn`) | 764,219 | **194.3초** | 약 5.4시간 |

동적 그래프를 켜면 **2.9배 느려진다**(`new_gc` 가 사실상 TC+GC 스택 하나를 더 얹는다).

| 범위 | METR-LA |
|---|---|
| 5설정 × 100 epoch × 1 run | **약 17~18시간** |
| 5설정 × 100 epoch × 6 run | 약 4.5일 |

PEMS-BAY 는 노드가 207→325 로 늘어 그래프 연산이 제곱으로 커지고 배치 수도 1.5배라
대략 3배 정도로 **추정**한다(직접 재보지 않았다).

### epoch 은 함부로 줄이면 안 된다

커리큘럼 러닝 때문에 학습 대상 horizon 이 2500 반복마다 하나씩 늘어난다. METR-LA 는
**약 73 epoch 이 지나야 horizon 12 까지 전부 학습**된다(PEMS-BAY 는 약 48 epoch).
30 epoch 만 돌리면 horizon 12 는 한 번도 학습되지 않은 채 평가되어 수치가 무의미하다.

실제로 1~2 epoch 짜리 실행에서는 `original` 과 `fixed_dyn` 의 horizon 3/6/12 값이
소수점 4자리까지 동일하게 나온다(둘 다 학습 안 된 구간에서 상수를 뱉기 때문).
자세한 설명은 [실험이해하기.md](실험이해하기.md) 4.1절.

따라서 시간을 줄여야 한다면 **epoch 이 아니라 `--runs` 를 줄이거나 설정 개수를 줄인다.**

- 최소 구성: `--configs original fixed_base fixed_dyn --runs 1` (약 9시간)
- CUDA 장비가 있으면 `--device cuda`

## 파일

| 파일 | 역할 |
|---|---|
| `setup.py` | venv·패키지·데이터 다운로드·무결성 검증·npz 생성 |
| `run.py` | 설정별 학습/평가 실행, 로그와 CSV 수집 |
| `summarize.py` | 결과를 원본 논문 / 석사논문 수치와 나란히 출력 |
| `실험이해하기.md` | 실험 원리·학습 과정·시간이 걸리는 이유 설명 |
