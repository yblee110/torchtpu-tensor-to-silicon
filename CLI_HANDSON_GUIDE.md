# TorchTPU v0.2.0 (TPU v6e) 복사·붙여넣기 실습 가이드

이 가이드는 처음부터 끝까지 **터미널에 순서대로 복사·붙여넣기**만 하면 **TPU v6e 생성 → 환경 구축 → Lab 01~03 벤치마크 → XProf 프로파일링 → 결과 회수 및 삭제**까지 완료할 수 있도록 구성되어 있습니다.

> **터미널 구분 표기**
> - **`💻 [로컬 PC 터미널]`**: 지금 작업 중인 내 리눅스 PC 터미널에서 실행
> - **`🚀 [TPU VM SSH 터미널]`**: `gcloud compute ssh`로 접속한 클라우드 TPU v6e 내부에서 실행

---

## Step 0. 환경변수 지정 & GCP 사전 점검 (`Pre-flight`)

**`💻 [로컬 PC 터미널]`** — 본인 GCP 프로젝트 ID만 첫 줄에 넣어 실행하세요.

```bash
export PROJECT_ID="여기에_GCP_프로젝트_ID_입력"
export ZONE="us-east5-a"
export VM_NAME="torchtpu-sprint-vm"
export MACHINE_TYPE="ct6e-standard-1t"

gcloud auth login
gcloud config set project "$PROJECT_ID"
gcloud services enable compute.googleapis.com tpu.googleapis.com
gcloud components install alpha --quiet
```

**(권장) 쿼터 및 `GPUS_ALL_REGIONS` 제한 여부 사전 진단**

**`💻 [로컬 PC 터미널]`**
```bash
curl -sSO https://gist.githubusercontent.com/RobMulla/ee1a530f9ff0bdb9aa5b493c7faf9aa2/raw/tpu_diagnostic.py
python3 tpu_diagnostic.py
rm -f tpu_diagnostic.py
```
* 출력된 리포트에서 `GPUS_ALL_REGIONS`가 `0`이 아닌지 확인하세요. (만약 `us-east5-a` 재고가 부족하거나 4칩 쿼터만 열려 있다면 `export MACHINE_TYPE="ct6e-standard-4t"` 또는 `export ZONE="us-central1-a"`로 바꾼 뒤 Step 1을 진행하세요.)

---

## Step 1. Cloud TPU v6e (`Flex-start`) 인스턴스 생성 & 대기

**`💻 [로컬 PC 터미널]`** — 부트 디스크 `100GB`, 최대 8시간 뒤 자동 삭제(`DELETE`) 안전장치가 포함된 명령어입니다.

```bash
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --provisioning-model=FLEX_START \
    --request-valid-for-duration=2h \
    --max-run-duration=8h \
    --instance-termination-action=DELETE \
    --boot-disk-size=100GB \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --maintenance-policy=TERMINATE
```

**할당 완료(`RUNNING`)까지 자동 대기 루프**

**`💻 [로컬 PC 터미널]`**
```bash
until [ "$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format='value(status)' 2>/dev/null)" = "RUNNING" ]; do
    echo "[$(date +%H:%M:%S)] Flex-start 큐에서 TPU v6e 할당 대기 중..."
    sleep 10
done
echo "✅ TPU v6e 인스턴스가 RUNNING 상태입니다! (SSH 부팅 대기 15초)"
sleep 15
```

---

## Step 2. 실습 코드 전송 & 포트 포워딩 SSH 접속

**`💻 [로컬 PC 터미널]`** — 프로젝트 폴더를 통째로 올린 뒤, XProf(`6006`)와 Jupyter/Colab(`8888`) 터널을 열어 접속합니다.

```bash
gcloud compute scp --recurse . "$VM_NAME":~/torchtpu-tensor-to-silicon --zone="$ZONE"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- -L 6006:localhost:6006 -L 8888:localhost:8888
```

---

## Step 3. TPU VM 내부 Python 3.12 + TorchTPU v0.2.0 원클릭 설치

이제 프롬프트가 TPU VM 내부로 바뀌었습니다! 아래 블록을 붙여넣으세요.

**`🚀 [TPU VM SSH 터미널]`**
```bash
cd ~/torchtpu-tensor-to-silicon
bash scripts/setup_v6e_vm.sh
source .venv/bin/activate
```
* 마지막에 `🎉 TorchTPU Ready! Device: TPU v6e ...`가 출력되면 준비 완료입니다.

---

## Step 4. [Lab 01] 4대 실행 모드 비교, Roofline 차트 & XProf 분석

### 4-1. 4가지 실행 모드 벤치마크 및 트레이스 캡처 실행

**`🚀 [TPU VM SSH 터미널]`**
```bash
mkdir -p outputs
python labs/lab01_execution_modes.py | tee outputs/lab01.log
```
* **저장되는 파일 (`~/torchtpu-tensor-to-silicon/outputs/`)**:
  1. `outputs/lab01.log` & `outputs/lab01_metrics.json`: 4가지 실행 모드 Latency / TFLOPS 표 및 수치 데이터
  2. `outputs/01_roofline_execution_modes.png`: TPU v6e Roofline 시각화 이미지
  3. `outputs/xprof_modes/{strict_eager,defer_and_fuse,torch_compile}/`: `.xplane.pb` XProf 하드웨어 트레이스

### 4-2. XProf 웹 대시보드 띄워서 눈으로 확인하기

**`🚀 [TPU VM SSH 터미널]`**
```bash
xprof --logdir=outputs/xprof_modes --port=6006
```
1. 실행한 상태에서 내 PC 웹 브라우저를 열고 **`http://localhost:6006`** 에 접속합니다.
2. 상단 Run 드롭다운에서 `strict_eager`, `defer_and_fuse`, `torch_compile`을 바꿔가며 **Trace Viewer**(Microbubbles 유무, 퓨전 블록 길이)와 **Roofline Analysis**(MXU 활용률 및 HBM 병목 탈출)를 비교합니다.
3. 다 확인했으면 터미널에서 **`Ctrl + C`** 를 눌러 XProf 서버를 종료합니다.

### 4-3. (보너스 CLI 실습) XLA 컴파일러가 실제로 퓨전한 HLO IR 텍스트 덤프 확인

**`🚀 [TPU VM SSH 터미널]`**
```bash
mkdir -p outputs/hlo_dump && rm -rf outputs/hlo_dump/*
TPU_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=outputs/hlo_dump" python -c '
import torch
d = torch.device("tpu")
fn = torch.compile(lambda x: torch.nn.functional.silu(x @ x) + x, backend="tpu")
_ = fn(torch.randn(1024, 1024, device=d, dtype=torch.bfloat16))
torch.accelerator.synchronize()
'
ls -lh outputs/hlo_dump/*after_optimizations.txt
grep -E "ENTRY|fusion" outputs/hlo_dump/*after_optimizations.txt | head -n 20
```

---

## Step 5. [Lab 02] Dynamic Shapes(`128`-버킷), Contiguity, `256×256` MXU 정렬 & `bf16` 정밀도

**`🚀 [TPU VM SSH 터미널]`**
```bash
python labs/lab02_shapes_memory_precision.py | tee outputs/lab02.log
```
* **저장되는 파일**: `outputs/lab02.log` (콘솔 출력 원문), `outputs/lab02_metrics.json` (4개 실험 전체 수치)
* **확인 포인트**:
  1. **`[1/4]` Bounded Dynamism**: `dynamic=False`(매 토큰 재컴파일 수십 ms) vs `dynamic=None`(초기 워밍업 후 **< 1 ms** 유지) 비교
  2. **`[2/4]` Contiguity**: `.t()` non-contiguous view 입력 시 PjRt 임시 복사 오버헤드 vs `.contiguous()` 비교
  3. **`[3/4]` TPU v6e `256×256` MXU 정렬**: `3997`(비정렬) → `3968`(`128` 홀수 배수) → **`4096`(`256` 배수)**로 갈 때의 유효 TFLOPS 상승 확인
  4. **`[4/4]` MXU 정밀도 모드**: `bfloat16`(1-pass) vs `fp32` `DEFAULT`(1-pass) / `HIGH`(3-pass, ~3배 시간) / `HIGHEST`(6-pass, ~6배 시간) 속도 및 최대 오차 비교

---

## Step 6. [Lab 03] 커스텀 Pallas 커널 (`torch.tpu.pallas.jax_op`) & 타일 크기 스윕

**`🚀 [TPU VM SSH 터미널]`**
```bash
python labs/lab03_custom_pallas_kernel.py | tee outputs/lab03.log
```
* **저장되는 파일**: `outputs/lab03.log` (콘솔 출력 원문), `outputs/lab03_metrics.json` (타일 스윕 및 대역폭 수치)
* **확인 포인트**:
  1. **`[1/3]` 수치 정합성 검증**: 순수 PyTorch 레퍼런스 대비 Pallas VMEM 커널의 오차(`torch.allclose` 통과) 확인
  2. **`[2/3]` HBM 대역폭 & VMEM 타일 스윕**: `Strict Eager` 대비 `torch.tpu.pallas.jax_op`의 실효 대역폭(GB/s) 향상 및 `block_m` (`8`, `64`, `128`, `256`, `512`) 변화에 따른 성능 추이 확인
  3. **`[3/3]` End-to-End `torch.compile` 인라이닝**: `nn.Module` 내부에 커스텀 Pallas 오퍼레이터를 넣고 `torch.compile(backend="tpu")`로 통째로 컴파일했을 때 그래프 브레이크 없이 동작하는지 확인

---

## Step 7. 모든 실험 결과(`outputs/`)를 내 로컬 PC로 회수하기 (VM 삭제 전 필수!)

클라우드 TPU VM은 Step 8에서 삭제하면 디스크도 함께 사라지므로, `outputs/` 폴더 하나에 모여 있는 **로그·JSON·Roofline 이미지·XProf 트레이스·HLO 덤프 전체**를 내 로컬 PC로 가져옵니다.

먼저 TPU VM SSH 세션에서 빠져나옵니다:

**`🚀 [TPU VM SSH 터미널]`**
```bash
exit
```

내 로컬 PC의 `results_from_vm/outputs/` 경로로 통째로 다운로드합니다:

**`💻 [로컬 PC 터미널]`**
```bash
mkdir -p ./results_from_vm
gcloud compute scp --recurse "$VM_NAME":~/torchtpu-tensor-to-silicon/outputs ./results_from_vm/ --zone="$ZONE"
ls -la ./results_from_vm/outputs
```
* *(팁)* 나중에 TPU VM을 삭제한 뒤에도 내 로컬 PC에서 `xprof --logdir=./results_from_vm/outputs/xprof_modes --port=6006`으로 언제든 트레이스를 다시 열어볼 수 있습니다!

> **(선택) 삭제 전 Colab 웹 UI로도 연결해보고 싶다면?**
> VM 내부에서 `jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0 --no-browser`를 실행하고 출력된 `http://localhost:8888/?token=...` 주소를 Colab 우측 상단 **`연결 ▼ → 로컬 런타임에 연결`**에 입력하면 됩니다.

---

## Step 8. 실습 종료 후 TPU VM 즉시 삭제 (크레딧 보호 필수!)

Flex-start 인스턴스는 일시정지(Stop)가 지원되지 않으므로, 실습이 끝나면 아래 명령어로 **반드시 삭제**해 크레딧 소모를 멈추세요.

**`💻 [로컬 PC 터미널]`**
```bash
gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
gcloud compute instances list
```
* `gcloud compute instances list` 결과에 실행 중인 TPU 인스턴스가 없으면 안전하게 정리된 것입니다!
