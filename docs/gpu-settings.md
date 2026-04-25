# GPU 설정 가이드 — RTX 2070 Desktop GPU 기준

## 현재 환경

| 항목 | 값 |
|---|---|
| GPU 모델 | NVIDIA GeForce RTX 2070 |
| VRAM | 8 GB |
| 드라이버 | 591.86 |
| CUDA | 13.1 |
| Compute Capability | 7.5 (Turing) |
| 아키텍처 | Turing (2세대 Tensor Core) |

---

## 수정된 파일 목록 및 변경 내용

### 1. `train.py` (학습 스크립트)

| 설정 | 변경 전 (laptop 브랜치) | 변경 후 (pc 브랜치) |
|---|---|---|
| `batch_size` | `8` (주석: RTX 4060 Laptop) | `8` (주석: RTX 2070 Desktop) |
| `dtype` | `'bfloat16' if ... else 'float16'` (자동 감지) | `'float16'` (고정) |

```python
# Line 49
batch_size = 8 # RTX 2070 Desktop 8GB VRAM: micro-batch 8 (effective batch = 60*8 = 480, same as original)

# Line 73
dtype = 'float16' # RTX 2070 (Turing, CC 7.5) → float16이 최적. 'float32', 'bfloat16', 'float16' 중 선택 가능
```

### 2. `model.py` (모델 정의)

| 설정 | 변경 전 | 변경 후 |
|---|---|---|
| `estimate_mfu()` docstring | RTX 4060 Laptop bfloat16 | RTX 2070 Desktop float16 |
| `flops_promised` | `126e12` (126 TFLOPS) | `60e12` (~60 TFLOPS) |

```python
# Line 290
""" estimate model flops utilization (MFU) in units of RTX 2070 Desktop float16 peak FLOPS """

# Line 299-301
# express our flops throughput as ratio of RTX 2070 Desktop float16 peak flops
flops_achieved = flops_per_iter * (1.0/dt) # per second
flops_promised = 60e12 # RTX 2070 Desktop GPU float16 Tensor Core peak flops is ~60 TFLOPS
```

### 3. `bench.py` (벤치마크 스크립트)

| 설정 | 변경 전 | 변경 후 |
|---|---|---|
| `batch_size` | `12` | `8` |
| `dtype` | `'bfloat16' if ... else 'float16'` (자동 감지) | `'float16'` (고정) |

```python
# Line 12
batch_size = 8 # RTX 2070 Desktop 8GB VRAM 기준

# Line 18
dtype = 'float16' # RTX 2070 (Turing, CC 7.5) → float16이 최적
```

### 4. `sample.py` (추론 스크립트)

| 설정 | 변경 전 | 변경 후 |
|---|---|---|
| `dtype` | `'bfloat16' if ... else 'float16'` (자동 감지) | `'float16'` (고정) |

```python
# Line 21
dtype = 'float16' # RTX 2070 (Turing, CC 7.5) → float16이 최적
```

### 5. `config/` (설정 파일들) — 변경 없음

아래 config 파일들은 `train.py`의 기본값을 오버라이드하는 용도이며, GPU 종속적인 설정(`dtype`, `flops_promised`)을 직접 지정하지 않으므로 수정하지 않았습니다.

- `config/train_gpt2.py` — `batch_size=12`는 8x A100 기준 주석이므로 그대로 유지
- `config/train_shakespeare_char.py` — 소형 모델(6레이어), 메모리 부담 없음
- `config/finetune_shakespeare.py` — `batch_size=1`로 이미 보수적
- `config/eval_gpt2*.py` — `batch_size=8`로 동일

---

## 설정값 변경 근거

### dtype: bfloat16 자동 감지 → float16 고정

| 아키텍처 | bfloat16 하드웨어 지원 | 권장 dtype |
|---|---|---|
| Turing (RTX 20xx, CC 7.5) | ❌ 미지원 | **float16** |
| Ampere (RTX 30xx, CC 8.0+) | ✅ 지원 | bfloat16 |
| Ada Lovelace (RTX 40xx, CC 8.9) | ✅ 지원 | bfloat16 |

- RTX 2070은 Turing 아키텍처로, bfloat16 연산을 하드웨어적으로 지원하지 않습니다.
- `torch.cuda.is_bf16_supported()`는 환경에 따라 True를 반환할 수 있으나, 실제로는 소프트웨어 에뮬레이션이므로 속도가 크게 저하됩니다.
- **float16 + GradScaler**가 RTX 2070에서 최적의 성능을 발휘합니다.

### flops_promised: 126e12 → 60e12

| GPU | FP16 Tensor Core 피크 | 출처 |
|---|---|---|
| RTX 2070 | ~60 TFLOPS | NVIDIA 공식 사양 |
| RTX 4060 Laptop | 126 TFLOPS | NVIDIA 공식 사양 |
| A100 (80GB) | 312 TFLOPS | NVIDIA 공식 사양 |

이 값은 `estimate_mfu()` 함수에서 GPU 활용률(MFU)을 계산하는 기준값입니다. GPU 모델에 맞게 설정해야 MFU 수치가 의미 있는 값이 됩니다.

### batch_size: bench.py만 12 → 8 변경

- `train.py`는 이미 laptop 브랜치에서 8로 설정되어 있었으므로 주석만 변경
- `bench.py`는 원본 기준(12)으로 남아 있어 8GB VRAM에서 OOM 위험이 있으므로 8로 변경

---

## 브랜치별 설정 비교

| 설정 | laptop 브랜치 (RTX 4060 Laptop) | pc 브랜치 (RTX 2070 Desktop) |
|---|---|---|
| GPU | RTX 4060 Laptop | RTX 2070 Desktop |
| VRAM | 8 GB | 8 GB |
| dtype | bfloat16 (자동 감지) | **float16 (고정)** |
| flops_promised | 126e12 | **60e12** |
| batch_size (train.py) | 8 | 8 |
| batch_size (bench.py) | 12 | **8** |
| compile | True | True |

---

## 참고: 다른 환경으로 전환 시

다른 GPU 환경에서 이 코드를 사용할 때 수정해야 할 항목:

1. **dtype** (`train.py`, `sample.py`, `bench.py`) — Ampere 이상이면 `'bfloat16'`으로 변경
2. **flops_promised** (`model.py:301`) — 해당 GPU의 FP16/BF16 피크 TFLOPS 값으로 변경
3. **batch_size** (`train.py`, `bench.py`) — VRAM 용량에 따라 조절
