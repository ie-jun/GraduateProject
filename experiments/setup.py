#!/usr/bin/env python3
"""실험 환경 세팅 (macOS / Apple Silicon 기준).

    python3 experiments/setup.py

하는 일:
  1. venv 생성 + 패키지 설치
  2. 인접행렬(adj_mx) 확보
  3. 원본 교통 데이터(.h5) 다운로드 + 무결성 검증
  4. 학습용 윈도우(.npz) 생성

이 스크립트는 venv 바깥(시스템 python3)에서 실행되므로 표준 라이브러리만 사용한다.
"""
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # GraduateProject/
ORIGINAL = ROOT.parent / "MTGNN_original"              # 원본 MTGNN 체크아웃(있으면 사용)
VENV_PY = ROOT / "venv" / "bin" / "python"
DATA = ROOT / "data"

MIRROR = "https://raw.githubusercontent.com/deepkashiwa20/DL-Traff-Graph/main"

# 원본 MTGNN 논문에 실린 표본 수 / 노드 수. 미러에서 받은 파일이 정품인지 확인하는 용도.
# (석사논문 Table 1은 pems-bay를 52,166으로 적었으나 실제·원본 논문 모두 52,116이다)
EXPECTED = {"metr-la": (34272, 207), "pems-bay": (52116, 325)}


def run(cmd, echo=True, **kw):
    if echo:
        print("    $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def step_venv():
    print("==> [1/4] venv + 패키지")
    if not VENV_PY.exists():
        run([sys.executable, "-m", "venv", str(ROOT / "venv")])
    run([str(VENV_PY), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    # requirements.txt 는 2019년 버전(torch 1.2.0 / numpy 1.17 / pandas 0.25)을 고정하고 있어
    # Python 3.11 + arm64 에서 빌드되지 않는다. 최신 대응 버전으로 설치한다.
    # pandas 3.x 는 이 구형 HDF5 파일을 못 읽으므로 2.x 로 묶는다.
    run([str(VENV_PY), "-m", "pip", "install", "--quiet",
         "torch", "numpy", "pandas<3", "scipy", "scikit-learn", "tables", "matplotlib"])


def step_adj():
    print("==> [2/4] 인접행렬")
    (DATA / "sensor_graph").mkdir(parents=True, exist_ok=True)
    for name in ("adj_mx.pkl", "adj_mx_bay.pkl"):
        dst = DATA / "sensor_graph" / name
        if dst.exists():
            continue
        src = ORIGINAL / "data" / "sensor_graph" / name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"    {name} <- {src}")
        else:
            print(f"    [없음] {dst} — https://github.com/nnzhan/MTGNN 에서 받아 넣으세요")


def step_download():
    print("==> [3/4] 원본 교통 데이터(.h5)")
    DATA.mkdir(parents=True, exist_ok=True)

    if not (DATA / "metr-la.h5").exists():
        print("    metr-la.h5 내려받는 중...")
        urllib.request.urlretrieve(f"{MIRROR}/METRLA/metr-la.h5", DATA / "metr-la.h5")

    if not (DATA / "pems-bay.h5").exists():
        print("    pems-bay.zip 내려받는 중...")
        zip_path = DATA / "pems-bay.zip"
        urllib.request.urlretrieve(f"{MIRROR}/PEMSBAY/pems-bay.zip", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA)
        zip_path.unlink()

    # 미러에서 받았으므로 논문에 실린 크기와 대조해 검증한다.
    check = "\n".join(
        f"df = pd.read_hdf(r'{DATA / (n + '.h5')}')\n"
        f"assert df.shape == {exp}, ('{n} 크기 불일치', df.shape, {exp})\n"
        f"print('    {n:9s}', df.shape, 'OK')"
        for n, exp in EXPECTED.items()
    )
    run([str(VENV_PY), "-c", "import pandas as pd\n" + check], echo=False)


def step_npz():
    print("==> [4/4] 학습용 윈도우(.npz) 생성")
    for out, h5 in (("METR-LA", "metr-la.h5"), ("PEMS-BAY", "pems-bay.h5")):
        (DATA / out).mkdir(parents=True, exist_ok=True)
        if (DATA / out / "train.npz").exists():
            print(f"    {out} 이미 있음, 건너뜀")
            continue
        run([str(VENV_PY), "generate_training_data.py",
             f"--output_dir=data/{out}", f"--traffic_df_filename=data/{h5}"], cwd=ROOT)


def main():
    # 파이프로 넘길 때 자식 프로세스 출력과 순서가 뒤섞이지 않도록 줄 단위로 흘린다.
    sys.stdout.reconfigure(line_buffering=True)
    step_venv()
    step_adj()
    step_download()
    step_npz()
    if not ORIGINAL.exists():
        print(f"\n[안내] 원본 MTGNN 체크아웃이 {ORIGINAL} 에 없습니다.")
        print("       original 설정을 돌리려면:")
        print(f"       git clone https://github.com/nnzhan/MTGNN.git {ORIGINAL}")
    print("\n완료. 다음 단계:  ./venv/bin/python experiments/run.py --list")


if __name__ == "__main__":
    main()
