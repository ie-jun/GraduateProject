#!/usr/bin/env python3
"""참 그래프가 시간에 따라 전환되는 합성 다변량 시계열을 만든다.

두 가지 전환 방식:

  --switching clock (기본, SYN-SWITCH)
      낮(06~18시)/밤 2개 레짐. 레짐이 시계의 함수라서, 모델이 time-of-day 입력
      채널만 보고 feature 경로로 우회할 수 있음이 실험으로 확인됐다(동적 A 불필요).
      -> 게이트/붕괴 진단용 무대로는 유효하나 "성능 기여" 증명 무대로는 부적합.

  --switching semimarkov (SYN-REGIME)
      K개 레짐이 반마르코프로 전환(체류시간 ~ Uniform[dwell_min, dwell_max] 스텝).
      레짐이 시계와 무관 -> 시계 채널이 무력화되고, 레짐은 창(12스텝)의 노드 간
      패턴에서만 추론 가능. oracle 사전검증: 창 기반 레짐 식별 87.9%, 동적 정보의
      성능 가치 H3 +33.9% / H6 +14.2% (정적 평균 W 대비). 동적 그래프가 성능에
      기여할 수 있는 무대다.

생성 모델(공통):
    x_t = mu + W_{r(t)} (x_{t-1} - mu) + eps,   eps ~ N(0, sigma^2)
    W_k 는 노드당 parents 개 부모의 희소 방향 그래프, 스펙트럼 반경 radius 안정화.
    W[i, j] != 0  <=>  edge j -> i  (행 = 타깃 노드. mixprop 의 A[v,w] 방향과 동일)

출력 (기존 파이프라인에 그대로 꽂힌다):
    data/<name>.h5                       # pandas DataFrame, 5분 간격
    data/sensor_graph/adj_<name>.pkl     # util.load_adj 형식 (참 그래프 합집합)
    data/<NAME>/ground_truth.npz         # W 목록, 레짐 시퀀스 등 평가용 참값

사용:
    ./venv/bin/python experiments/make_switching_data.py                        # SYN-SWITCH
    ./venv/bin/python experiments/make_switching_data.py --switching semimarkov # SYN-REGIME
    ./venv/bin/python generate_training_data.py \
        --output_dir=data/SYN-REGIME --traffic_df_filename=data/syn-regime.h5
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def sparse_stable_W(rng, n, k, radius):
    """노드당 부모 k 개(자기 자신 제외)의 희소 W. 스펙트럼 반경을 radius 로 맞춘다."""
    W = np.zeros((n, n))
    for i in range(n):
        parents = rng.choice([j for j in range(n) if j != i], size=k, replace=False)
        # 부호 섞인 계수 — 순양수만 쓰면 모든 노드가 금방 동조화되어 그래프가 안 보인다
        W[i, parents] = rng.uniform(0.4, 1.0, size=k) * rng.choice([-1, 1], size=k)
        W[i, i] = rng.uniform(0.3, 0.6)          # 자기회귀(관성)
    rad = max(abs(np.linalg.eigvals(W)))
    return W * (radius / rad)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--switching", choices=["clock", "semimarkov"], default="clock")
    ap.add_argument("--name", default=None,
                    help="출력 이름 (기본: clock=syn-switch, semimarkov=syn-regime)")
    ap.add_argument("--nodes", type=int, default=30)
    ap.add_argument("--parents", type=int, default=3, help="노드당 부모 수")
    ap.add_argument("--regimes", type=int, default=6,
                    help="레짐 수 (semimarkov 전용; clock 은 항상 2)")
    ap.add_argument("--dwell-min", type=int, default=24,
                    help="레짐 최소 체류 스텝 (semimarkov 전용; 24스텝=2시간)")
    ap.add_argument("--dwell-max", type=int, default=72,
                    help="레짐 최대 체류 스텝 (semimarkov 전용)")
    ap.add_argument("--radius", type=float, default=None,
                    help="스펙트럼 반경 (기본: clock 0.95 유지, semimarkov 0.95)")
    ap.add_argument("--steps", type=int, default=34272,
                    help="시점 수 (기본: METR-LA 와 동일 -> 커리큘럼 타임라인 동일)")
    ap.add_argument("--seed", type=int, default=None,
                    help="기본: clock 42 (기존 SYN-SWITCH 재현), semimarkov 7")
    ap.add_argument("--sigma", type=float, default=1.0, help="노이즈 표준편차")
    ap.add_argument("--mu", type=float, default=50.0,
                    help="평균 수준 (양수 유지 -> MAPE 계산 안전)")
    args = ap.parse_args()

    clock = args.switching == "clock"
    name = args.name or ("syn-switch" if clock else "syn-regime")
    seed = args.seed if args.seed is not None else (42 if clock else 7)
    radius = args.radius if args.radius is not None else 0.95
    rng = np.random.default_rng(seed)
    n = args.nodes

    index = pd.date_range("2012-03-01", periods=args.steps, freq="5min")

    if clock:
        # ── SYN-SWITCH: 기존과 동일한 생성 순서 유지 (seed 42 재현성 보존) ──
        Ws = [sparse_stable_W(rng, n, args.parents, radius),   # W_day
              sparse_stable_W(rng, n, args.parents, radius)]   # W_night
        hours = index.hour + index.minute / 60
        regimes = np.where((hours >= 6) & (hours < 18), 0, 1).astype(int)
        burn_W = Ws[1]                                         # night 레짐으로 번인
    else:
        # ── SYN-REGIME: K개 레짐, 반마르코프 체류 ──
        Ws = [sparse_stable_W(rng, n, args.parents, radius) for _ in range(args.regimes)]
        regimes = np.zeros(args.steps, dtype=int)
        t, r = 0, 0
        while t < args.steps:
            dur = int(rng.integers(args.dwell_min, args.dwell_max + 1))
            regimes[t:t + dur] = r
            t += dur
            r = int((r + rng.integers(1, args.regimes)) % args.regimes)  # 자기 제외
        burn_W = Ws[0]

    # 번인 후 생성
    x = np.zeros((args.steps, n))
    state = rng.normal(0, args.sigma, size=n)
    for _ in range(200):
        state = burn_W @ state + rng.normal(0, args.sigma, size=n)
    for t in range(args.steps):
        state = Ws[regimes[t]] @ state + rng.normal(0, args.sigma, size=n)
        x[t] = state
    x = x + args.mu

    # 1) h5 (기존 generate_training_data.py 가 읽는 형식)
    df = pd.DataFrame(x, index=index, columns=[f"n{i:03d}" for i in range(n)])
    h5_path = ROOT / "data" / f"{name}.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(h5_path, key="df")

    # 2) predefined adj pkl (util.load_adj 형식: ids, id_to_ind, adj)
    union = np.zeros((n, n), dtype=bool)
    for W in Ws:
        union |= (W != 0)
    ids = list(df.columns)
    pkl_path = ROOT / "data" / "sensor_graph" / f"adj_{name.replace('-', '_')}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump((ids, {s: i for i, s in enumerate(ids)}, union.astype(np.float32)), f)

    # 3) 평가용 참값 (W 스택 + 레짐 시퀀스)
    gt_dir = ROOT / "data" / name.upper()
    gt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(gt_dir / "ground_truth.npz",
             Ws=np.stack(Ws), regimes=regimes, mu=args.mu, sigma=args.sigma,
             switching=args.switching, seed=seed,
             dwell_min=args.dwell_min, dwell_max=args.dwell_max)

    switches = int((regimes[1:] != regimes[:-1]).sum())
    print(f"[{name}] nodes={n}, steps={args.steps}, 레짐 {len(Ws)}개, "
          f"전환 {switches}회 (평균 체류 {args.steps / max(1, switches):.0f}스텝)")
    print(f"값 범위 [{x.min():.1f}, {x.max():.1f}] (mu={args.mu})")
    print(f"저장: {h5_path}")
    print(f"      {pkl_path}")
    print(f"      {gt_dir / 'ground_truth.npz'}")
    print(f"다음:  ./venv/bin/python generate_training_data.py "
          f"--output_dir=data/{name.upper()} --traffic_df_filename=data/{name}.h5")


if __name__ == "__main__":
    main()
