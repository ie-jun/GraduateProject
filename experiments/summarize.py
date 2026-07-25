#!/usr/bin/env python3
"""실험 결과를 논문 수치와 나란히 출력한다.

    ./venv/bin/python experiments/summarize.py --dataset METR-LA

experiments/results/<dataset>/<config>/train.log 를 읽는다. 원본 저장소와 이 저장소
모두 마지막에 같은 형식의 표를 찍기 때문에 로그 파싱 하나로 둘 다 처리된다.

    test|horizon	MAE-mean	RMSE-mean	MAPE-mean	MAE-std	RMSE-std	MAPE-std
    3	2.6912	5.1783	0.0686	...
"""
import argparse
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
HORIZONS = (3, 6, 12)

# 비교 기준값. (MAE, RMSE, MAPE%)
# - 원본 논문  : Wu et al., KDD 2020, Table 3
# - 석사논문   : 석사졸업논문_황예준 Table 3
REFERENCE = {
    "METR-LA": {
        "원본논문 MTGNN": {3: (2.69, 5.18, 6.86), 6: (3.05, 6.17, 8.19), 12: (3.49, 7.23, 9.87)},
        "석사논문 MTGNN": {3: (2.60, 4.89, 6.86), 6: (3.10, 6.27, 8.80), 12: (3.47, 7.18, 10.10)},
        "석사논문 Dynamic": {3: (2.57, 4.85, 6.51), 6: (3.03, 6.12, 8.15), 12: (3.47, 7.01, 9.40)},
    },
    "PEMS-BAY": {
        "원본논문 MTGNN": {3: (1.32, 2.79, 2.77), 6: (1.65, 3.74, 3.69), 12: (1.94, 4.49, 4.53)},
        "석사논문 MTGNN": {3: (1.20, 2.35, 2.71), 6: (1.60, 3.53, 3.71), 12: (1.86, 4.25, 4.40)},
        "석사논문 Dynamic": {3: (1.12, 2.24, 2.30), 6: (1.58, 3.53, 3.53), 12: (1.64, 4.24, 4.32)},
    },
}

ROW = re.compile(r"^(\d+)\t([\d.]+)\t([\d.]+)\t([\d.]+)")


def parse_log(log_path):
    """로그 마지막의 test|horizon 표에서 {horizon: (MAE, RMSE, MAPE%)} 를 뽑는다."""
    if not log_path.exists():
        return None
    lines = log_path.read_text(errors="replace").splitlines()
    try:  # 마지막 표만 사용 (run 이 여러 번이어도 최종 집계가 맨 뒤에 온다)
        start = len(lines) - 1 - next(
            i for i, l in enumerate(reversed(lines)) if l.startswith("test|horizon"))
    except StopIteration:
        return None

    out = {}
    for line in lines[start + 1:]:
        m = ROW.match(line)
        if not m:
            break
        h, mae, rmse, mape = int(m[1]), float(m[2]), float(m[3]), float(m[4])
        if h in HORIZONS:
            out[h] = (mae, rmse, mape * 100)  # MAPE 는 분수로 저장돼 있다
    return out or None


def fmt(vals):
    return "     -       -        -   " if vals is None else \
        f"{vals[0]:6.2f} {vals[1]:7.2f} {vals[2]:7.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="METR-LA", choices=list(REFERENCE))
    args = ap.parse_args()

    width = 20 + 24 * len(HORIZONS)
    print(f"\n=== {args.dataset} ===")
    print(" " * 20 + "".join(f"      Horizon {h:<2d}     " for h in HORIZONS))
    print(" " * 20 + "   MAE    RMSE    MAPE  " * len(HORIZONS))
    print("-" * width)

    for label, per_h in REFERENCE[args.dataset].items():
        print(f"{label:20s}" + "".join(fmt(per_h.get(h)) for h in HORIZONS))

    print("-" * width)
    root = RESULTS / args.dataset
    if not root.is_dir():
        print(f"(아직 결과 없음: {root})")
    else:
        found = False
        for cfg_dir in sorted(root.iterdir()):
            if not cfg_dir.is_dir():
                continue
            got = parse_log(cfg_dir / "train.log")
            print(f"{cfg_dir.name:20s}" + "".join(fmt(got.get(h) if got else None)
                                                  for h in HORIZONS))
            found = True
        if not found:
            print(f"(아직 결과 없음: {root})")
    print()
    print("해석 요령:")
    print("  original 이 '원본논문 MTGNN' 과 비슷해야 → 재현 환경이 정상이라는 뜻")
    print("  fixed_base 가 original 과 비슷해야   → 기여 코드의 baseline 경로가 원본에 충실")
    print("  fixed_dyn 과 fixed_base 의 차이      → alpha 를 바로잡은 뒤 기여의 순수 효과")
    print()


if __name__ == "__main__":
    main()
