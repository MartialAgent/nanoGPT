# 08. 랩톱 WSL 환경 구축 및 명령어 레퍼런스

`laptop-wsl` 브랜치 기준으로, RTX 4060 Laptop 환경에서 nanoGPT를 실행하기 위한 설정 기록과 명령어 모음입니다.

- **브랜치**: `laptop-wsl` (base: `linux`)
- **런타임**: WSL2 Ubuntu (Windows 11 Pro)
- **작업 경로**: `/mnt/c/Study/260425 NanoGPT/nanoGPT`

---

## 환경 사양

### 하드웨어

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8,188 MiB (8GB) |
| Compute Capability | **8.9 (Ada Lovelace)** |
| bfloat16 하드웨어 지원 | ✅ |
| 드라이버 | 552.27 |
| CPU | Intel Core i7-14650HX |
| RAM | 15.7 GB |

### 소프트웨어 (WSL venv `~/venvs/nanogpt`)

| 패키지 | 버전 |
|---|---|
| Python | 3.12.3 |
| torch | 2.5.1+cu124 |
| triton | 3.1.0 |
| numpy | 2.5.2 |
| tiktoken | 0.13.0 |
| tqdm | 4.70.0 |
| transformers | 5.15.0 |
| datasets | 5.0.1 |

---

## 이전 머신(`pc` 브랜치)과의 차이

`pc` 브랜치는 RTX 2070 Desktop(Turing, CC 7.5) 기준으로 튜닝돼 있었습니다. Turing은 bfloat16을 하드웨어로 지원하지 않아 float16으로 고정돼 있었으나, Ada는 네이티브 지원하므로 bfloat16이 유리합니다(범위가 넓어 수치 불안정이 적고 GradScaler 불필요).

| | RTX 2070 Desktop (`pc`) | RTX 4060 Laptop (`laptop-wsl`) |
|---|---|---|
| Compute Capability | 7.5 (Turing) | 8.9 (Ada) |
| bfloat16 | ❌ 미지원 | ✅ 네이티브 |
| 권장 dtype | float16 + GradScaler | **bfloat16** |
| bf16 Tensor Core 피크 | — | ~126 TFLOPS |
| VRAM | 8GB | 8GB (동일) |

### 수정한 파일 (6개)

| 파일 | 변경 |
|---|---|
| `model.py:290`, `299-301` | `flops_promised` **60e12 → 126e12**, docstring |
| `train.py:50`, `74` | batch 주석, dtype 자동 감지 |
| `sample.py:21` | dtype 자동 감지 |
| `bench.py:12`, `18` | batch 주석, dtype 자동 감지 |
| `chat.py:9` | `bfloat16` |
| `config/finetune_agent.py:3`, `16` | 헤더, `bfloat16` |

dtype은 리터럴 고정 대신 upstream nanoGPT 원형인 자동 감지 방식을 사용합니다.

```python
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
```

> **`model.py`의 `flops_promised`를 반드시 함께 고쳐야 합니다.** 이 값은 MFU(Model FLOPs Utilization) 계산의 분모입니다. RTX 2070용 `60e12`를 그대로 두면 학습 로그의 MFU가 실제의 약 2.1배로 부풀려집니다.

---

## WSL 환경 구축

### 전제: sudo 비밀번호 문제

기본 WSL Ubuntu에는 `pip`도 `ensurepip`도 없어 `python3 -m venv`만으로는 venv에 pip이 생기지 않습니다. `sudo apt install python3-pip`이 정석이지만 비밀번호 입력이 필요하므로, root 권한이 필요 없는 **uv**로 우회합니다.

```bash
# 1) uv 설치 (~/.local/bin, sudo 불필요)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 2) venv 생성 — WSL 네이티브 fs에 배치할 것
uv venv ~/venvs/nanogpt --python 3.12

# 3) torch + 의존성
uv pip install --python ~/venvs/nanogpt/bin/python torch==2.5.1
uv pip install --python ~/venvs/nanogpt/bin/python numpy tiktoken tqdm transformers datasets
```

venv를 `/mnt/c`가 아닌 `~/venvs`에 두는 이유는 두 가지입니다. `/mnt/c`는 9p 프로토콜을 거쳐 순차 읽기가 61 MB/s인 반면 네이티브 fs는 2.7 GB/s이고, Windows venv는 `Scripts/`, Linux venv는 `bin/` 구조라 한 디렉터리를 공유할 수도 없습니다.

### 컴파일러 설치 (torch.compile 필수)

```bash
sudo apt update && sudo apt install -y build-essential
```

기본 Ubuntu에는 gcc·cc·clang·g++·make가 전혀 없습니다. `torch.compile`의 inductor 백엔드는 생성한 커널을 C로 컴파일하므로 이것 없이는 `--compile=True`가 실패합니다.

---

## 명령어 레퍼런스

### 1. 세션 시작

```bash
wsl -d Ubuntu
cd "/mnt/c/Study/260425 NanoGPT/nanoGPT"
source ~/venvs/nanogpt/bin/activate
```

환경 확인:

```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
git branch --show-current
```

### 2. 데이터 준비

```bash
python data/shakespeare_char/prepare.py   # 완료됨 (train.bin 1.9MB, val.bin 218KB)
python data/agent/prepare.py              # 미완 — chat.py 사용 전 필수
python data/shakespeare/prepare.py        # 미완 — GPT-2 BPE 버전
python data/openwebtext/prepare.py        # 수십 GB, 장시간
```

현재 `.bin`이 준비된 데이터셋은 `shakespeare_char`뿐입니다.

### 3. 학습

```bash
# 기본 (char-level, 5000 iters, 10.65M params)
python train.py config/train_shakespeare_char.py --compile=True

# 짧게 테스트 (체크포인트까지 확보하려면 300 이상)
python train.py config/train_shakespeare_char.py --max_iters=500 --compile=True

# 중단 후 재개
python train.py config/train_shakespeare_char.py --init_from=resume --compile=True

# VRAM 부족 시
python train.py config/train_shakespeare_char.py --batch_size=32 --block_size=128

# GPT-2 파인튜닝 (agent 데이터 전처리 후)
python train.py config/finetune_agent.py

# 백그라운드 + 로그
nohup python train.py config/train_shakespeare_char.py --compile=True > train.log 2>&1 &
tail -f train.log
```

`build-essential` 설치 전에는 `--compile=False`를 사용합니다.

### 4. 결과 확인

> **선행 조건**: `out-shakespeare-char/ckpt.pt`가 있어야 합니다. `train_shakespeare_char.py`는 `eval_interval = 250`, `always_save_checkpoint = False`이고 `train.py:279`가 `iter_num > 0`을 요구하므로, **첫 저장은 iteration 250**입니다. `--max_iters`가 300 미만이면 학습이 정상 종료돼도 체크포인트가 생기지 않아 `sample.py`가 `FileNotFoundError`로 실패합니다.

```bash
ls -lh out-shakespeare-char/

python sample.py --out_dir=out-shakespeare-char
python sample.py --out_dir=out-shakespeare-char --num_samples=3 --max_new_tokens=300
python sample.py --out_dir=out-shakespeare-char --start="ROMEO:" --temperature=0.7

# 체크포인트 메타 확인
python -c "import torch;c=torch.load('out-shakespeare-char/ckpt.pt',map_location='cpu');print('iter',c['iter_num'],'val_loss',c['best_val_loss'])"
```

### 5. 성능 확인

```bash
python bench.py                  # GPT-2 124M 기준 (batch 8, block 1024)
python bench.py --compile=False  # 컴파일 유무 비교
python bench.py --profile=True   # PyTorch profiler

watch -n 1 nvidia-smi
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 2
```

### 6. 대화

```bash
python chat.py                                    # out-agent-ft/ckpt.pt 로드
python chat.py --out_dir=out-shakespeare-char     # 다른 체크포인트
```

`exit` 입력 시 종료됩니다. `data/agent/prepare.py` → `train.py config/finetune_agent.py`를 마쳐야 `out-agent-ft/ckpt.pt`가 생성됩니다.

### 7. git

```bash
git status --short
git diff
git add -A && git commit -m "Adapt configs for RTX 4060 Laptop (Ada, bf16)"
git push -u origin laptop-wsl
```

---

## 트러블슈팅

### uv 의존성 해석 실패

```
× No solution found when resolving dependencies:
╰─▶ Because there is no version of nvidia-cudnn-cu12{...}==9.1.0.70 and
    torch>=2.5.1+cu121 depends on nvidia-cudnn-cu12{...}==9.1.0.70,
    we can conclude that torch>=2.5.1+cu121 cannot be used.
```

**원인**: uv에서 `--index-url`은 PyPI를 추가하는 게 아니라 **완전히 대체**합니다. PyTorch cu121 인덱스에는 해당 cuDNN 휠이 없어 해석이 막힙니다.

**해결**: `--index-url`을 빼고 PyPI 기본 휠(cu124)을 사용합니다. 드라이버 552.27이 CUDA 12.4를 지원하며, triton도 이쪽 의존성에 포함됩니다.

### torch.compile — C 컴파일러 부재

```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Failed to find C compiler. Please specify via CC environment variable.
```

**해결**: `sudo apt install -y build-essential`

### 진행바 s/it이 실제와 다름

`train.py:332`의 `pbar.update(1)`이 `if iter_num % log_interval == 0` 블록 안에 있습니다. 진행바가 `log_interval`마다 1칸씩만 전진하므로 다음 세 가지가 어긋납니다.

- 진행바가 `max_iters / log_interval`에서 종료 (100 iters 완주해도 `10/100` 표시)
- tqdm의 `s/it`이 실제의 `log_interval`배
- ETA가 같은 배율로 부풀려짐

수정하려면 `update`만 블록 밖으로 옮깁니다.

```python
        if master_process:
            pbar.set_postfix(loss=f"{lossf:.4f}", mfu=f"{running_mfu*100:.2f}%")
    if master_process:
        pbar.update(1)
    iter_num += 1
```

`train.py:339-340`에는 도달 불가능한 `break`가 중복돼 있습니다(동작에는 무영향).

---

## 검증 결과

### GPU 실측 (WSL)

| 항목 | 값 |
|---|---|
| matmul 4096² bfloat16 | 7.63 ms → **18.0 TFLOPS** |
| matmul 4096² float16 | 7.84 ms → 17.5 TFLOPS |
| matmul 4096² float32 | 37.68 ms → 3.6 TFLOPS |
| H2D 전송 1 GiB | 224 ms (~4.6 GB/s) |
| 소형 커널 1000회 | 51.4 ms |

WSL GPU 패스스루는 정상 동작합니다.

### 파일시스템 I/O

| | `/mnt/c` | WSL 네이티브 |
|---|---|---|
| memmap 랜덤 배치 (64×256) | 0.60 ms | 0.32 ms |
| 순차 읽기 (`dd`) | 61.2 MB/s | 2.7 GB/s |

학습 배치 로딩은 iteration당 1ms 미만이라 `/mnt/c`에 소스를 두어도 병목이 아닙니다. 다만 venv와 대용량 데이터셋은 네이티브 fs가 유리합니다.

### 라이브러리 호환성

| 항목 | 결과 |
|---|---|
| `GPT.from_pretrained` ↔ transformers 5.15.0 | ✅ 키 149/149 일치, shape assert 실패 0건 (Conv1D 유지) |
| `load_dataset` ↔ datasets 5.0.1 | ✅ `num_proc` 파라미터 유지 |
| numpy 2.5.2 ↔ torch 2.5.1 | ✅ memmap·interop 정상 |
| `torch.cuda.amp.GradScaler` | ⚠️ deprecation 경고만 (bf16에선 `enabled=False`라 무해) |

### 설치 소요

| 단계 | 시간 |
|---|---|
| torch 및 CUDA 런타임 (23개) | 5분 54초 |
| 나머지 의존성 (45개) | 13.4초 |
| 다운로드 총량 | 약 2.96 GiB |
