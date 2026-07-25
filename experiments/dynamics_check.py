"""학습된 fixed_dyn 모델의 동적 인접행렬 A 가 입력에 따라 실제로 달라지는지 측정한다.

동기: new_graph_constructor 는 sigmoid 로만 A 를 만들기 때문에, 학습이 잘못되면
입력과 무관한 거의 상수 행렬로 붕괴할 수 있다. 그 경우 "동적 그래프"는 이름뿐이다.
이 스크립트는 출근 시간대(07:30~09:30)와 심야(01:00~03:00) 입력에 대한 A 를 뽑아,

  - 그룹 내 변동: 같은 시간대 입력들 사이에서 A 항목별 표준편차의 평균
  - 그룹 간 차이: 시간대별 평균 A 의 항목별 |차이| 의 평균

을 비교한다. 그룹 간 차이가 그룹 내 변동보다 뚜렷하면(비율 >> 1)
A 가 입력의 시간대에 실제로 반응한다는 증거다. 비율이 0 에 가깝고 전체 변동도
미미하면 사실상 정적 그래프다.

사용법 (본 실험이 끝나 체크포인트가 생긴 뒤):
    ./venv/bin/python experiments/dynamics_check.py \
        --checkpoint save/METR-LA/fixed_dyn/PEMS-BAY_exp1_0.pth
    (체크포인트 파일명의 PEMS-BAY 는 train_multi_step.py 의 하드코딩 때문이며
     실제 내용은 --data 로 학습한 데이터셋이다)

동작만 확인하려면(체크포인트 없이 무작위 초기화로 파이프라인 검증):
    ./venv/bin/python experiments/dynamics_check.py --smoke

주의: time-in-day 채널만으로 시간대를 고르므로 주말 샘플도 섞인다.
주말의 "출근 시간"은 평일과 달라 노이즈로 작용하지만, 비율의 방향은 바꾸지 않는다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from net import gtnet          # noqa: E402
from util import load_dataset, load_adj  # noqa: E402


def build_model(num_nodes, predefined_A, device):
    """train_multi_step.py 의 fixed_dyn 설정과 동일한 모델을 만든다."""
    return gtnet(gcn_true=True, buildA_true=True, hidden_channels=32,
                 seq_length=12, gcn_depth=2, layer_depth=3,
                 num_nodes=num_nodes, device=device,
                 new_graph_learning=True, new_graph_only_TC=False,
                 predefined_A=predefined_A, dropout=0.3, subgraph_size=20,
                 node_dim=40, dilation_exponential=1,
                 conv_channels=32, residual_channels=32,
                 skip_channels=64, end_channels=128,
                 in_dim=2, out_dim=12, layers=3,
                 propalpha=0.05, tanhalpha=3.0)


def dynamic_adjs(model, x, static_A, batch, device):
    """x: (n, 12, N, 2) numpy → A: (n, N, N) numpy"""
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).transpose(1, 3).to(device)
            outs.append(model.new_gc(xb, static_A).cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", help="fixed_dyn 학습 결과 .pth (state_dict)")
    ap.add_argument("--data", default=str(ROOT / "data" / "METR-LA"))
    ap.add_argument("--adj_data", default=str(ROOT / "data" / "sensor_graph" / "adj_mx.pkl"))
    ap.add_argument("--num_nodes", type=int, default=207)
    ap.add_argument("--device", default="cpu", help="본 실험이 mps 를 쓰는 동안은 cpu 권장")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=512, help="그룹당 최대 샘플 수")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "dynamics_check.npz"))
    ap.add_argument("--smoke", action="store_true",
                    help="체크포인트 없이 무작위 가중치로 파이프라인만 검증")
    args = ap.parse_args()

    if not args.smoke and not args.checkpoint:
        ap.error("--checkpoint 가 필요합니다 (동작 확인만 하려면 --smoke)")

    device = torch.device(args.device)
    predefined_A = torch.tensor(load_adj(args.adj_data)) - torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    model = build_model(args.num_nodes, predefined_A, device).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    data = load_dataset(args.data, args.batch, args.batch, args.batch)
    x_test = data["x_test"]                      # (n, 12, N, 2), 채널1 = time-in-day
    tod = x_test[:, -1, 0, 1]                    # 입력 마지막 시점의 하루 중 시각 (0~1)

    if args.smoke:
        n = min(8, len(x_test))
        rush, night = x_test[:n], x_test[-n:]
    else:
        rush = x_test[(tod >= 7.5 / 24) & (tod < 9.5 / 24)][:args.max_samples]
        night = x_test[(tod >= 1.0 / 24) & (tod < 3.0 / 24)][:args.max_samples]
    assert len(rush) and len(night), "시간대 그룹이 비어 있습니다"

    with torch.no_grad():
        static_A = model.gc(model.idx)
    A_rush = dynamic_adjs(model, rush, static_A, args.batch, device)
    A_night = dynamic_adjs(model, night, static_A, args.batch, device)
    assert A_rush.shape[1:] == (args.num_nodes, args.num_nodes)

    mean_rush, mean_night = A_rush.mean(0), A_night.mean(0)
    within = 0.5 * (A_rush.std(0).mean() + A_night.std(0).mean())
    between = np.abs(mean_rush - mean_night)
    overall_std = np.concatenate([A_rush, A_night]).std(0).mean()

    print(f"샘플 수: 출근 {len(rush)}, 심야 {len(night)}")
    print(f"A 항목 평균값           : {np.concatenate([A_rush, A_night]).mean():.4f}")
    print(f"그룹 내 변동 (std 평균)  : {within:.6f}")
    print(f"그룹 간 차이 (|Δ| 평균)  : {between.mean():.6f}  (최대 {between.max():.4f})")
    print(f"전체 입력 민감도 (std)   : {overall_std:.6f}")
    ratio = between.mean() / within if within > 0 else float("inf")
    print(f"그룹간/그룹내 비율       : {ratio:.2f}")
    print("해석: 비율이 1 보다 충분히 크면 A 가 시간대에 반응(동적), "
          "차이·변동이 모두 ~0 이면 상수로 붕괴(사실상 정적).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, static_A=static_A.cpu().numpy(),
                        mean_rush=mean_rush, mean_night=mean_night,
                        std_rush=A_rush.std(0), std_night=A_night.std(0))
    print(f"행렬 저장: {out}")


if __name__ == "__main__":
    main()
