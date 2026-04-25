# GPU 설정 가이드 — RTX 4060 Laptop GPU 기준

## 현재 환경

| 항목 | 값 |
|---|---|
| GPU 모델 | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8 GB |
| 드라이버 | 552.27 |
| Compute Capability | 8.9 (Ada Lovelace) |

---

## 수정된 설정값

### `train.py`

```python
batch_size = 8                        # VRAM 8GB 기준 안전한 micro-batch 크기
gradient_accumulation_steps = 5 * 12  # 실효 배치 480 유지 (60 × 8 = 480)
device = 'cuda'                        # NVIDIA GPU 사용
dtype = 'bfloat16'                     # 자동 감지 (Compute Cap 8.9 → bfloat16 지원)
compile = True                         # PyTorch 2.0 컴파일 최적화 (40xx에서 동작)
```

### `model.py`

```python
flops_promised = 126e12  # RTX 4060 Laptop BF16 피크 성능 (126 TFLOPS)
```

---

## 설정값 계산 근거

### batch_size: 12 → 8

학습 중 GPU 메모리는 다음 항목들이 차지합니다:

- 모델 파라미터 (GPT-2 Small 124M × 2바이트 = ~0.25 GB)
- Optimizer 상태 (파라미터의 2배 = ~0.5 GB)
- 활성화값 (Activation): **배치 크기에 비례해 증가** — 가장 큰 변수
- 그래디언트

8GB VRAM에서 `batch_size=12`, `block_size=1024`로 GPT-2 규모 학습 시 OOM(Out of Memory) 위험이 있습니다. `batch_size=8`은 여유 버퍼를 확보한 안전 값입니다.

### gradient_accumulation_steps: 40 → 60

`batch_size`를 줄이면 한 번에 보는 데이터가 줄어 학습 품질이 떨어질 수 있습니다.  
`gradient_accumulation_steps`는 **여러 micro-batch의 그래디언트를 쌓아** 큰 배치와 동일한 효과를 냅니다.

```
실효 배치 크기 = batch_size × gradient_accumulation_steps
변경 전: 12 × 40 = 480
변경 후:  8 × 60 = 480  ← 동일
```

메모리는 절약하면서 학습 안정성은 유지됩니다.

### dtype: bfloat16 자동 선택

RTX 4060 (Ada Lovelace, Compute Cap 8.9)은 bfloat16을 하드웨어 수준에서 지원합니다.  
`float32` 대비 메모리 절반, 속도 2배 이상이며, `float16` 대비 수치 안정성이 높습니다.

### flops_promised: 312e12 → 126e12

`estimate_mfu()`는 학습 중 GPU 활용률을 출력합니다. 기준값이 A100(312 TFLOPS)으로 고정되어 있으면 RTX 4060에서 MFU가 0.4 이상 나올 수 없어 수치가 의미 없어집니다.  
RTX 4060 Laptop의 실제 BF16 피크인 **126 TFLOPS**로 교체해 MFU가 실제 GPU 활용률을 반영하도록 했습니다.

---

## 사업담당자를 위한 설명

### "GPU"가 왜 중요한가?

GPT 같은 언어 모델은 수억 개의 숫자를 동시에 곱하고 더하는 연산을 반복합니다.  
CPU는 이 연산을 순서대로 처리하지만, GPU는 수천 개의 코어로 **병렬 처리**합니다.  
같은 학습을 CPU로 하면 몇 주, GPU로 하면 몇 시간~며칠로 줄어드는 이유입니다.

### "VRAM 8GB"의 의미

VRAM은 GPU 전용 메모리입니다. 학습 중 모델, 데이터, 중간 계산값이 모두 VRAM에 올라가야 합니다.  
VRAM이 부족하면 학습이 중단됩니다(OOM 에러).

- **8GB VRAM** → GPT-2 Small(124M 파라미터) 수준까지 학습 가능
- **GPT-2 Medium(345M)** → 배치 크기를 대폭 줄이면 가능하나 속도 저하
- **GPT-3(175B)** → 불가. 데이터센터 수십 개 GPU 필요

### "batch_size"와 "gradient_accumulation"이란?

**batch_size**는 모델이 한 번에 읽는 문장 묶음의 수입니다.  
많이 읽을수록 학습이 안정적이지만 메모리를 많이 씁니다.

**gradient_accumulation**은 메모리 제약을 우회하는 기법입니다.  
배치를 8개씩 60번 읽어 그래디언트를 쌓으면, 한 번에 480개를 읽는 것과 수학적으로 동일한 효과를 냅니다.  
마치 장바구니가 작아서 마트를 여러 번 다녀오는 것처럼, 결과는 같되 한 번에 드는 부담을 나누는 방식입니다.

### "bfloat16"이란?

숫자를 얼마나 정밀하게 저장하느냐의 단위입니다.

| 형식 | 비트 | 메모리 | 정밀도 | 비고 |
|---|---|---|---|---|
| float32 | 32 | 기준 | 높음 | 기본 과학 계산 |
| float16 | 16 | 절반 | 낮음 | 수치 불안정 가능 |
| bfloat16 | 16 | 절반 | float32와 동일한 범위 | AI 학습에 최적 |

RTX 4060은 bfloat16을 하드웨어에서 직접 지원하므로, float32 대비 메모리는 절반, 속도는 2배 이상 납니다. 학습 결과의 품질 차이는 거의 없습니다.

### "MFU(Model FLOPS Utilization)"란?

GPU가 이론상 낼 수 있는 최대 성능 대비 실제 활용률입니다.  
예를 들어 MFU 40%라면, GPU가 가진 연산 능력의 40%만 실제로 쓰이고 있다는 뜻입니다.  
나머지 60%는 데이터 이동, 대기 등에 소비됩니다.  

nanoGPT 수준의 단일 GPU 학습에서 MFU 30~50%는 정상 범위입니다.

### 비용 관점에서의 시사점

| 구분 | 내 RTX 4060 (랩탑) | 클라우드 A100 (80GB) |
|---|---|---|
| 피크 성능 | 126 TFLOPS | 312 TFLOPS |
| VRAM | 8 GB | 80 GB |
| 비용 | (이미 보유) | 약 $3~4/시간 |
| GPT-2 Small 학습 시간 (OpenWebText) | 수일 | 수 시간 |
| 적합한 용도 | 실험, 소규모 파인튜닝 | 본격 사전학습 |

현재 환경은 **알고리즘 이해, 소규모 실험, 파인튜닝 연구**에 충분합니다.  
실제 서비스 수준의 모델 학습이 목표라면 클라우드 GPU 비용을 별도 산정해야 합니다.
