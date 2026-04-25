# 06. 실습 가이드 (Hands-On)

처음부터 끝까지 직접 실행해보는 단계별 가이드입니다.

---

## 환경 설정

### 필수 패키지 설치

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

| 패키지 | 용도 |
|--------|------|
| torch | 딥러닝 프레임워크 |
| numpy | 배열 처리 |
| transformers | GPT-2 사전학습 가중치 로드 |
| datasets | OpenWebText 다운로드 |
| tiktoken | BPE 토크나이저 |
| wandb | 학습 모니터링 (선택) |
| tqdm | 진행 바 |

### PyTorch 버전 확인

```python
import torch
print(torch.__version__)         # 2.0 이상 권장
print(torch.cuda.is_available()) # True이어야 GPU 학습 가능
print(torch.cuda.get_device_name(0))
```

---

## 실습 1: 셰익스피어 문자 수준 모델 (입문)

가장 빠르게 GPT를 학습하고 텍스트를 생성해보는 실습입니다.

### 1단계: 데이터 준비

```bash
cd data/shakespeare_char
python prepare.py
```

약 5초 내로 완료됩니다. 생성 파일 확인:

```bash
ls -lh data/shakespeare_char/
# train.bin  (~2MB)
# val.bin    (~0.2MB)
# meta.pkl   (문자 매핑)
```

### 2단계: 학습 시작

```bash
python train.py config/train_shakespeare_char.py
```

**config/train_shakespeare_char.py 내용**:
```python
out_dir = 'out-shakespeare-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# 작은 GPT: 6 레이어, 6 헤드, 384 임베딩
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
```

**예상 학습 시간**:
- GPU (RTX 4060): ~5분
- CPU: ~30분 이상

**학습 로그 예시**:
```
step 0: train loss 4.2201, val loss 4.2168
step 250: train loss 2.0545, val loss 2.1254
step 500: train loss 1.7521, val loss 1.8901
...
step 5000: train loss 1.1234, val loss 1.4712
```

### 3단계: 텍스트 생성

```bash
python sample.py \
    --out_dir=out-shakespeare-char \
    --device=cuda \
    --num_samples=3 \
    --max_new_tokens=200
```

**예상 출력**:
```
HAMLET:
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...
---------------
KING RICHARD:
Now is the winter of our discontent
Made glorious summer by this sun of York...
```

---

## 실습 2: 사전학습된 GPT-2 평가

OpenAI의 GPT-2 가중치를 불러와 셰익스피어 텍스트를 파인튜닝합니다.

### GPT-2로 텍스트 생성 (파인튜닝 없이)

```bash
python sample.py \
    --init_from=gpt2 \
    --start="To be, or not to be" \
    --num_samples=2 \
    --max_new_tokens=100 \
    --temperature=0.8 \
    --top_k=200
```

처음 실행 시 HuggingFace에서 GPT-2 가중치를 자동으로 다운로드합니다 (~550MB).

### GPT-2를 셰익스피어로 파인튜닝

```bash
# 먼저 데이터 준비 (BPE 버전)
cd data/shakespeare
python prepare.py
cd ../..

# 파인튜닝
python train.py config/finetune_shakespeare.py
```

파인튜닝 후 셰익스피어 스타일에 맞는 텍스트가 생성됩니다.

---

## 실습 3: 하이퍼파라미터 실험

모델 크기와 학습 설정이 결과에 어떤 영향을 주는지 실험합니다.

### 작은 모델 vs 큰 모델

```bash
# 초소형 모델 (테스트용, 빠름)
python train.py config/train_shakespeare_char.py \
    --n_layer=2 --n_head=2 --n_embd=128 \
    --out_dir=out-tiny

# 중간 모델 (기본)
python train.py config/train_shakespeare_char.py \
    --out_dir=out-medium

# 큰 모델 (느리지만 더 좋은 결과)
python train.py config/train_shakespeare_char.py \
    --n_layer=12 --n_head=12 --n_embd=768 \
    --out_dir=out-large
```

### 드롭아웃 실험

```bash
# 과적합 방지를 위한 드롭아웃 추가
python train.py config/train_shakespeare_char.py \
    --dropout=0.2 \
    --out_dir=out-dropout
```

학습 손실과 검증 손실의 차이를 비교해보세요.

---

## 실습 4: 나만의 텍스트로 학습

직접 가져온 텍스트로 언어 모델을 학습합니다.

### 사용자 정의 데이터 준비

```python
# data/custom/prepare.py 작성
import numpy as np
import os

# 1. 텍스트 읽기
with open('my_text.txt', 'r', encoding='utf-8') as f:
    data = f.read()

print(f"데이터 길이: {len(data):,} 문자")

# 2. 어휘 생성 (문자 수준)
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"어휘 크기: {vocab_size}")

# 3. 인코딩
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

# 4. 분할 및 저장
n = int(0.9 * len(data))
train_ids = encode(data[:n])
val_ids = encode(data[n:])

np.array(train_ids, dtype=np.uint16).tofile('train.bin')
np.array(val_ids, dtype=np.uint16).tofile('val.bin')

import pickle
meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
pickle.dump(meta, open('meta.pkl', 'wb'))
print("완료!")
```

### 학습 설정

```bash
python train.py config/train_shakespeare_char.py \
    --dataset=custom \
    --out_dir=out-custom \
    --max_iters=3000
```

---

## 학습 모니터링

### 로그 해석

```
step 1000: train loss 1.8234, val loss 2.1456, lr 5.00e-04, mfu 12.34%
           ─────────────────  ──────────────  ──────────  ──────────
                 학습 손실         검증 손실    현재 학습률   GPU 활용률
```

**정상 패턴**:
- 학습 손실이 점점 낮아짐
- 검증 손실도 함께 낮아짐
- 두 손실의 차이가 크지 않음 (과적합 없음)

**문제 패턴**:
- 학습 손실 낮지만 검증 손실이 높아짐 → 과적합 (`dropout` 올리기)
- 두 손실 모두 정체 → 학습률 높이거나 모델 키우기
- Loss가 `nan` → 학습률 낮추기, `grad_clip` 확인

### WandB 연동 (선택사항)

```bash
# WandB 계정 연결
wandb login

# WandB 활성화하여 학습
python train.py config/train_shakespeare_char.py \
    --wandb_log=True \
    --wandb_project=nanogpt-study \
    --wandb_run_name=shakespeare-char-v1
```

브라우저에서 실시간으로 학습 곡선을 확인할 수 있습니다.

---

## GPU 메모리 부족 시 해결책

```bash
# 배치 크기 줄이기 + 그래디언트 누적으로 보완
python train.py config/train_shakespeare_char.py \
    --batch_size=32 \            # 기본 64에서 줄임
    --gradient_accumulation_steps=2  # 유효 배치 크기 유지

# bfloat16 사용 (Ampere 이상 GPU)
python train.py config/train_shakespeare_char.py \
    --dtype=bfloat16

# 더 작은 모델
python train.py config/train_shakespeare_char.py \
    --n_layer=4 --n_head=4 --n_embd=256
```

현재 RTX 4060 Laptop 최적 설정은 [gpu-settings.md](./gpu-settings.md)를 참고하세요.

---

## 벤치마크 실행

```bash
# 현재 GPU에서 처리량 측정
python bench.py

# 예상 출력:
# iter 0: loss 4.2416, time 89.87ms
# iter 1: loss 4.2010, time 75.23ms
# ...
# time per iteration: 76.12ms, MFU: 8.45%
```

---

## 자주 발생하는 오류

### CUDA out of memory

```
RuntimeError: CUDA out of memory.
```

→ `batch_size` 줄이기, `block_size` 줄이기, 또는 `dtype=bfloat16` 설정

### tiktoken 없음

```
ModuleNotFoundError: No module named 'tiktoken'
```

→ `pip install tiktoken`

### train.bin 없음

```
FileNotFoundError: data/shakespeare_char/train.bin
```

→ `cd data/shakespeare_char && python prepare.py` 먼저 실행

---

## 학습 완료 후 탐구 주제

1. **어텐션 시각화**: 각 레이어의 어텐션 패턴을 시각화해보세요 (`model.py`의 `attn_weights` 수정)

2. **임베딩 시각화**: t-SNE로 단어 임베딩을 2D로 줄여 비슷한 단어가 가깝게 위치하는지 확인

3. **모델 수술 (Model Surgery)**: `model.crop_block_size(512)`로 컨텍스트 창을 줄여보기

4. **프롬프팅 실험**: 다양한 프롬프트로 모델의 한계와 능력을 탐색

5. **GPT-2 분석**: `from_pretrained('gpt2')`로 불러온 모델의 가중치 분포 시각화

---

## 다음 학습 자료

이 시리즈를 모두 읽었다면 더 깊은 학습을 위한 자료입니다.

- **Andrej Karpathy의 강의**: "Let's build GPT from scratch" (YouTube)
- **원본 GPT 논문**: "Improving Language Understanding by Generative Pre-Training" (2018)
- **GPT-2 논문**: "Language Models are Unsupervised Multitask Learners" (2019)
- **Attention Is All You Need**: 트랜스포머 원논문 (2017)
- **nanochat**: nanoGPT의 후속 프로젝트 (더 현대적인 구조)
