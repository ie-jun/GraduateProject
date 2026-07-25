#!/usr/bin/env python3
"""참 그래프가 시간대에 따라 전환되는 합성 다변량 시계열을 만든다.

동기: METR-LA 의 참 그래프(도로망)는 사실상 정적이라 동적 그래프의 가치를 보여줄
무대가 아니다. 여기서는 의존 그래프 자체가 낮(06~18시)과 밤에 서로 다른 VAR(1)
데이터를 만들어, (1) 게이트가 열리는지 (2) 학습된 A 가 참 그래프를 복원하는지를
참값 대조로 정량 평가할 수 있게 한다.

생성 모델:
    x_t = mu + W_r(t) (x_{t-1} - mu) + eps,   eps ~ N(0, sigma^2)
    r(t) = day  (06:00 <= 시각 < 18:00)  -> W_day
         = night (그 외)                  -> W_night
    W_r 은 노드당 k 개 부모를 갖는 희소 방향 그래프, 스펙트럼 반경 0.95 로 안정화.
    W[i, j] != 0  <=>  edge j -> i  (행 = 타깃 노드. mixprop 의 A[v,w] 방향과 동일)

출력 (기존 파이프라인에 그대로 꽂힌다):
    data/syn-switch.h5              # pandas DataFrame, 5분 간격 datetime index
    data/sensor_graph/adj_syn_switch.pkl   # util.load_adj 형식 (참 그래프 합집합)
    data/SYN-SWITCH/ground_truth.npz       # W_day, W_night, mu 등 평가용 참값

사용:
    ./venv/bin/python experiments/make_switching_data.py
    ./venv/bin/python generate_training_data.py \
        --output_dir=data/SYN-SWITCH --traffic_df_filename=data/syn-switch.h5
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def sparse_stable_W(rng, n, k, radius=0.95):
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
    ap.add_argument("--nodes", type=int, default=30)
    ap.add_argument("--parents", type=int, default=3, help="노드당 부모 수")
    ap.add_argument("--steps", type=int, default=34272,
                    help="시점 수 (기본: METR-LA 와 동일 -> 커리큘럼 타임라인 동일)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=1.0, help="노이즈 표준편차")
    ap.add_argument("--mu", type=float, default=50.0,
                    help="평균 수준 (양수 유지 -> MAPE 계산 안전)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.nodes

    W_day = sparse_stable_W(rng, n, args.parents)
    W_night = sparse_stable_W(rng, n, args.parents)

    # 5분 간격, METR-LA 와 같은 시작점
    index = pd.date_range("2012-03-01", periods=args.steps, freq="5min")
    hours = index.hour + index.minute / 60
    is_day = (hours >= 6) & (hours < 18)

    # 번인 후 생성
    x = np.zeros((args.steps, n))
    state = rng.normal(0, args.sigma, size=n)
    for _ in range(200):                                   # burn-in (night 레짐)
        state = W_night @ state + rng.normal(0, args.sigma, size=n)
    for t in range(args.steps):
        W = W_day if is_day[t] else W_night
        state = W @ state + rng.normal(0, args.sigma, size=n)
        x[t] = state
    x = x + args.mu

    # 1) h5 (기존 generate_training_data.py 가 읽는 형식)
    df = pd.DataFrame(x, index=index, columns=[f"n{i:03d}" for i in range(n)])
    h5_path = ROOT / "data" / "syn-switch.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(h5_path, key="df")

    # 2) predefined adj pkl (util.load_adj 형식: ids, id_to_ind, adj)
    union = ((W_day != 0) | (W_night != 0)).astype(np.float32)
    ids = list(df.columns)
    pkl_path = ROOT / "data" / "sensor_graph" / "adj_syn_switch.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump((ids, {s: i for i, s in enumerate(ids)}, union), f)

    # 3) 평가용 참값
    gt_dir = ROOT / "data" / "SYN-SWITCH"
    gt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(gt_dir / "ground_truth.npz",
             W_day=W_day, W_night=W_night, mu=args.mu, sigma=args.sigma,
             day_start=6.0, day_end=18.0, seed=args.seed)

    overlap = ((W_day != 0) & (W_night != 0) & ~np.eye(n, dtype=bool)).sum()
    edges = (args.parents * n)
    print(f"nodes={n}, steps={args.steps}, 레짐별 엣지 {edges}개 "
          f"(자기루프 제외 겹침 {overlap}개 = {overlap/edges:.0%})")
    print(f"값 범위 [{x.min():.1f}, {x.max():.1f}] (mu={args.mu})")
    print(f"저장: {h5_path}")
    print(f"      {pkl_path}")
    print(f"      {gt_dir / 'ground_truth.npz'}")
    print("다음:  ./venv/bin/python generate_training_data.py "
          "--output_dir=data/SYN-SWITCH --traffic_df_filename=data/syn-switch.h5")


if __name__ == "__main__":
    main()
