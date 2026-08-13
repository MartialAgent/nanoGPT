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

### 빌드 툴체인 설치 (torch.compile 필수)

```bash
sudo apt update && sudo apt install -y build-essential python3-dev
```

WSL Ubuntu 기본 이미지는 서버·컨테이너용 최소 구성이라 개발 도구가 전혀 없습니다.
`torch.compile`은 실행 시점에 코드를 생성해 **그 자리에서 컴파일**하므로 두 패키지가 모두 필요하며,
하나씩 순서대로 드러납니다.

| 패키지 | 제공하는 것 | 없으면 나는 오류 |
|---|---|---|
| `build-essential` | `gcc`, `g++`, `make`, `libc6-dev` | `Failed to find C compiler` |
| `python3-dev` | `Python.h` 등 C 확장 빌드용 헤더 | `fatal error: Python.h: No such file or directory` |

`python3-dev`가 필요한 이유는 triton이 `cuda_utils` 확장 모듈을 빌드하기 때문입니다.
venv가 시스템 CPython(`/usr/bin/python3.12`)에서 생성됐으므로 시스템 쪽 헤더를 참조합니다.

설치 후 실제 동작 확인:

```bash
python -c "import torch; f=torch.compile(lambda x: x*2); print(f(torch.ones(4,device='cuda')))"
```

`tensor([2., 2., 2., 2.], device='cuda:0')`가 나오면 컴파일 → GPU 실행 경로가 끝까지 뚫린 것입니다.
첫 실행은 컴파일 때문에 30초 안팎 걸립니다.

> `sudo`가 비밀번호를 요구하는데 비밀번호를 모른다면, Windows PowerShell에서 root로 우회할 수 있습니다.
> `wsl -d Ubuntu -u root -e apt install -y build-essential python3-dev`
> 비밀번호 재설정은 `wsl -d Ubuntu -u root passwd <사용자명>` (현재 비밀번호를 묻지 않음).
> 두 명령 모두 **Windows 프롬프트**(`PS C:\...>`)에서 실행해야 합니다. `wsl`은 Windows 명령어라
> Ubuntu 안(`user@HOST:~$`)에서는 `command not found`가 납니다.

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

### 4-1. 학습 로그 읽는 법

**시작 시**

| 출력 | 뜻 |
|---|---|
| `found vocab_size = 65` | 이 데이터의 고유 문자 종류 수 (`meta.pkl`에서 읽음) |
| `number of parameters: 10.65M` | 모델 가중치 개수. GPT-2 Small(124M)의 약 1/12 |
| `tokens per iteration will be: 16,384` | 1 iteration에 쓰는 토큰 수 (`batch_size 64 × block_size 256`) |
| `using fused AdamW: True` | 옵티마이저 연산을 CUDA 커널 하나로 융합 (더 빠름) |
| `compiling the model...` | inductor가 코드 생성·컴파일 중. 1~3분 멈춘 것처럼 보임 |

**진행 중**

```
step 0: train loss 4.2853, val loss 4.2842
saving checkpoint to out-shakespeare-char
```

- **train loss**: 학습 데이터에 대한 예측 오차
- **val loss**: 학습에 쓰지 않은 검증 데이터에 대한 오차 — 실제 성능 지표
- **시작값 해석**: `shakespeare_char`는 vocab이 65자이므로 무작위 추측의 이론적 손실이
  `ln(65) ≈ 4.17`입니다. 초기 4.28은 아직 아무것도 학습하지 않은 상태라는 뜻입니다
- 5000 iters 후 val loss **1.4~1.5** 부근이면 정상입니다
- `saving checkpoint`는 val loss가 이전 최저치를 갱신했다는 뜻입니다
  (`always_save_checkpoint = False`이므로 개선될 때만 저장)

**진행바**

```
Training: 10%|█ | 10/100 [02:40<24:06, loss=2.4599, mfu=1.44%]
```

- **mfu**: Model FLOPs Utilization. GPU 이론 성능(`model.py`의 `flops_promised`) 대비 실제 활용률.
  작은 모델은 GPU를 채우지 못해 낮게 나오는 것이 정상입니다
- **`it/s`·ETA**: 실제 iteration 속도입니다. `loss`·`mfu`는 `log_interval`마다만 갱신되므로
  그 사이에는 직전 값이 그대로 표시됩니다

**이상 신호**

| 증상 | 원인 / 대처 |
|---|---|
| val loss가 내려가다 다시 상승 | 과적합. 이 설정은 의도적으로 그렇게 되며 최저점만 저장됨 |
| loss가 `nan` | 수치 발산. bfloat16에서는 거의 발생하지 않음 |
| `CUDA out of memory` | `--batch_size=32` 등으로 낮출 것 |

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
python chat.py     # out-agent-ft/ckpt.pt 를 로드 (경로 고정)
```

> `chat.py`는 `configurator.py`를 호출하지 않으므로 **명령줄 인자를 받지 않습니다.**
> `--out_dir=...`을 붙여도 무시되고 `chat.py:7`의 `out-agent-ft`를 그대로 씁니다.
> 또한 `chat.py:35`가 `tiktoken.get_encoding("gpt2")`로 고정돼 있어 **GPT-2 계열 모델 전용**입니다.
> 문자 단위 모델(`shakespeare_char`, vocab 65)에 물리면 토큰 ID가 임베딩 범위를 벗어나 실패합니다.
> 다른 체크포인트를 쓰려면 `chat.py`를 직접 수정해야 합니다.

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

### torch.compile — Python 헤더 부재

`build-essential` 설치 후 gcc는 호출되지만 그다음 단계에서 막히는 경우입니다.

```
/tmp/tmpXXXX/main.c:5:10: fatal error: Python.h: No such file or directory
    5 | #include <Python.h>
...
subprocess.CalledProcessError: Command '['/usr/bin/gcc', ..., '-I/usr/include/python3.12']'
    returned non-zero exit status 1.
```

**원인**: gcc 명령줄에 `-I/usr/include/python3.12`가 있지만 그 디렉터리에 `Python.h`가 없습니다.
Ubuntu는 Python 런타임과 개발 헤더를 별도 패키지로 나눕니다.

**해결**: `sudo apt install -y python3-dev`

### 명령 실행 중 누른 키가 나중에 실행됨

```
$ python -c "..."
A                                    ← 실행 중 누른 키가 즉시 에코됨
tensor([2., 2., 2., 2.], ...)        ← 30초 뒤 실제 출력
A: command not found                 ← 종료 후 bash가 버퍼의 A를 명령으로 해석
```

**원인**: 앞 명령이 도는 동안 bash는 stdin을 읽지 않습니다. 누른 키는 화면에 에코되면서
커널 입력 큐에 쌓여 있다가, 명령이 끝나 bash가 복귀하면 명령줄로 처리됩니다.
개행(Enter)까지 들어갔다면 완성된 명령으로 실행됩니다.

**대처**: 긴 명령 실행 중에는 키를 누르지 않습니다. 위 사례는 `A`라는 명령이 없어 무해했지만,
버퍼에 쌓인 글자가 우연히 실제 명령을 이루면 그대로 실행되므로 원리상 주의가 필요합니다.

### 학습이 끝났는지 확인

진행바와 무관하게 프로세스와 체크포인트로 판단할 수 있습니다.

```bash
pgrep -af train.py     # 아무것도 안 나오면 종료된 것
python -c "import torch;c=torch.load('out-shakespeare-char/ckpt.pt',map_location='cpu');print(c['iter_num'], c['best_val_loss'])"
```

체크포인트의 `iter_num`이 `max_iters`보다 작은 것은 정상입니다. `always_save_checkpoint = False`이면
val loss가 갱신될 때만 저장하므로, 과적합이 시작된 이후 구간은 학습은 계속되지만 저장되지 않습니다.

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
