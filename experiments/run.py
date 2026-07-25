#!/usr/bin/env python3
"""실험 러너 — 원본 MTGNN과 이 저장소(기여 버전)를 같은 조건으로 돌린다.

    ./venv/bin/python experiments/run.py --list
    ./venv/bin/python experiments/run.py --configs original fixed_base --epochs 2 --runs 1
    ./venv/bin/python experiments/run.py --dataset METR-LA --epochs 100 --runs 1

배경:
net.py 가 정적 graph_constructor 를 alpha=propalpha(0.05) 로 만들고 있었다. 원본
MTGNN 은 alpha=tanhalpha(3) 을 쓴다. tanhalpha 는 net.py 단 한 곳에서만 참조되므로
소스를 고치지 않고 --tanhalpha 값만 바꿔서 수정 전/후를 모두 재현할 수 있다.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # GraduateProject/
ORIGINAL = ROOT.parent / "MTGNN_original"               # 원본 MTGNN 체크아웃
VENV_PY = ROOT / "venv" / "bin" / "python"
RESULTS = ROOT / "experiments" / "results"

# train_multi_step.py 가 CSV 를 떨구는 경로. 모듈 상단 상수(my_result_path)라 CLI 로
# 못 바꾸므로, 실행 후 이번 실행이 만든 파일만 골라 가져온다.
THESIS_CSV_DIR = ROOT / "save" / "result" / "PEMS-BAY"

DATASETS = {
    "METR-LA": dict(num_nodes=207, adj="adj_mx.pkl"),
    "PEMS-BAY": dict(num_nodes=325, adj="adj_mx_bay.pkl"),
}

# repo: "original" = 원본 MTGNN 저장소, "thesis" = 이 저장소(기여 버전)
CONFIGS = {
    "original": dict(
        repo="original", tanhalpha=3.0,
        desc="원본 MTGNN 코드 그대로 — 논문 수치가 재현되는지 확인"),
    "thesis_base": dict(
        repo="thesis", ngl=False, tanhalpha=0.05,
        desc="기여 버전 baseline, alpha 수정 전 — 논문 Table 의 'MTGNN' 행"),
    "thesis_dyn": dict(
        repo="thesis", ngl=True, tanhalpha=0.05,
        desc="기여 버전 Dynamic, alpha 수정 전 — 논문 Table 의 'Dynamic MTGNN' 행"),
    "fixed_base": dict(
        repo="thesis", ngl=False, tanhalpha=3.0,
        desc="기여 버전 baseline, alpha 수정 후 — original 과 같아야 정상"),
    "fixed_dyn": dict(
        repo="thesis", ngl=True, tanhalpha=3.0,
        desc="기여 버전 Dynamic, alpha 수정 후 — 기여의 진짜 성능"),
}


def build_command(cfg_name, cfg, ds_name, ds, args, save_dir):
    """설정에 맞는 실행 커맨드와 작업 디렉터리를 만든다."""
    data_dir = ROOT / "data" / ds_name
    adj_path = ROOT / "data" / "sensor_graph" / ds["adj"]

    common = [
        # -u : 자식 프로세스 출력 버퍼링을 끈다. 없으면 학습이 끝날 때까지 로그가 비어 있어
        #      장시간 실행 중에 진행 상황을 볼 수 없다.
        str(VENV_PY), "-u", "train_multi_step.py",
        "--device", args.device,
        "--data", str(data_dir),
        "--adj_data", str(adj_path),
        "--num_nodes", str(ds["num_nodes"]),
        "--epochs", str(args.epochs),
        "--runs", str(args.runs),
        "--tanhalpha", str(cfg["tanhalpha"]),
    ]

    if cfg["repo"] == "original":
        # 원본은 save 경로를 문자열로 이어붙이므로(args.save + "exp...") 끝에 구분자가 필요하고,
        # 디렉터리를 스스로 만들지 않으므로 미리 만들어 둔다.
        return common + ["--save", str(save_dir) + os.sep], ORIGINAL

    return common + [
        "--train", "True",
        "--new_graph_learning", str(cfg["ngl"]),
        "--save", str(save_dir),
    ], ROOT


def stream(cmd, cwd, log_path):
    """커맨드를 실행하며 출력을 화면과 로그 파일에 동시에 흘린다."""
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return proc.wait()


def collect_csvs(out_dir, since):
    """이번 실행이 만든 CSV 만 골라 결과 폴더로 복사한다(기존 파일은 건드리지 않음)."""
    if not THESIS_CSV_DIR.is_dir():
        return
    for f in THESIS_CSV_DIR.glob("*.csv"):
        if f.stat().st_mtime >= since:
            shutil.copy2(f, out_dir / f.name)


def main():
    ap = argparse.ArgumentParser(
        description="원본 MTGNN / 기여 버전 실험 러너",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="METR-LA", choices=list(DATASETS))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    help=f"실행할 설정 (기본: 전부). 선택지: {', '.join(CONFIGS)}")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--device", default="mps", help="mps | cpu | cuda")
    ap.add_argument("--list", action="store_true", help="설정 목록만 출력")
    ap.add_argument("--dry-run", action="store_true", help="실행 없이 커맨드만 출력")
    args = ap.parse_args()

    if args.list:
        print("\n사용 가능한 설정:\n")
        for name, cfg in CONFIGS.items():
            repo = "원본 저장소" if cfg["repo"] == "original" else "이 저장소"
            ngl = "" if cfg["repo"] == "original" else f", new_graph_learning={cfg['ngl']}"
            print(f"  {name:12s} [{repo}] tanhalpha={cfg['tanhalpha']}{ngl}")
            print(f"  {'':12s} {cfg['desc']}\n")
        return 0

    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"알 수 없는 설정: {unknown}\n선택지: {list(CONFIGS)}")

    ds = DATASETS[args.dataset]
    failures = []

    for name in args.configs:
        cfg = CONFIGS[name]

        if cfg["repo"] == "original" and not ORIGINAL.exists():
            print(f"[건너뜀] {name}: 원본 체크아웃이 없습니다 ({ORIGINAL})")
            print(f"         git clone https://github.com/nnzhan/MTGNN.git {ORIGINAL}")
            failures.append(name)
            continue

        out_dir = RESULTS / args.dataset / name
        save_dir = ROOT / "save" / args.dataset / name
        cmd, cwd = build_command(name, cfg, args.dataset, ds, args, save_dir)

        print("=" * 70)
        print(f"  {args.dataset} / {name}")
        print(f"  {cfg['desc']}")
        print(f"  epochs={args.epochs} runs={args.runs} device={args.device}")
        print("=" * 70)

        if args.dry_run:
            print(f"  (cwd={cwd})\n  {' '.join(cmd)}\n")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        THESIS_CSV_DIR.mkdir(parents=True, exist_ok=True)

        started = time.time()
        code = stream(cmd, cwd, out_dir / "train.log")
        if cfg["repo"] == "thesis":
            collect_csvs(out_dir, started)

        elapsed = (time.time() - started) / 60
        if code == 0:
            print(f"--> 완료 ({elapsed:.1f}분). 결과: {out_dir}")
        else:
            print(f"--> 실패 (exit {code}, {elapsed:.1f}분). 로그: {out_dir / 'train.log'}")
            failures.append(name)

    if not args.dry_run:
        print("\n요약 보기:  ./venv/bin/python experiments/summarize.py "
              f"--dataset {args.dataset}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
