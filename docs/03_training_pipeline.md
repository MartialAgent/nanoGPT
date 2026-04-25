# 03. 학습 파이프라인 (train.py)

`train.py`는 데이터 로딩부터 체크포인트 저장까지 전체 학습 과정을 담당합니다.

---

## 전체 흐름

```
1. 설정 로드          (configurator.py)
2. 분산 학습 초기화   (DDP setup)
3. 데이터 로드        (메모리맵 방식)
4. 모델 초기화        (scratch / resume / pretrained)
5. 혼합 정밀도 설정   (float16 / bfloat16)
6. 학습 루프 시작     ──────────────────┐
   ├─ 배치 가져오기                       │
   ├─ 그래디언트 누적                     │  반복
   ├─ 역전파                             │
   ├─ 그래디언트 클리핑                   │
   ├─ 파라미터 업데이트                   │
   └─ 검증/로깅/체크포인트 저장 ──────────┘
```

---

## 1. 설정 시스템

nanoGPT는 Python 파일을 설정으로 사용합니다. `configurator.py`가 실행 시 파일을 `exec()`로 평가합니다.

```bash
# 셰익스피어 학습 설정 사용
python train.py config/train_shakespeare_char.py

# 개별 값 오버라이드
python train.py config/train_shakespeare_char.py --batch_size=16 --device=cpu
```

**주요 하이퍼파라미터 (GPT-2 기본값)**:

```python
# 모델
n_layer = 12; n_head = 12; n_embd = 768
block_size = 1024; bias = False; dropout = 0.0

# 학습
batch_size = 12          # 마이크로 배치
gradient_accumulation_steps = 40   # 그래디언트 누적
max_iters = 600000       # 총 학습 스텝
learning_rate = 6e-4
min_lr = 6e-5            # 최저 학습률 (= lr/10)
warmup_iters = 2000
lr_decay_iters = 600000

# 정규화
weight_decay = 1e-1
beta1 = 0.9; beta2 = 0.95
grad_clip = 1.0
```

---

## 2. 데이터 로딩

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # 무작위 시작점 선택
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 입력: 위치 i부터 i+block_size
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # 타깃: 입력에서 1칸 앞 (다음 토큰 예측)
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y
```

**핵심 포인트**:
- `np.memmap` 사용: 전체 데이터를 메모리에 올리지 않고 디스크에서 직접 읽습니다 (17GB 데이터도 처리 가능)
- 타깃은 입력을 1칸 오른쪽으로 이동한 것: 위치 i에서 i+1을 예측

---

## 3. 모델 초기화 세 가지 방식

### 처음부터 학습 (`init_from='scratch'`)
```python
model = GPT(GPTConfig(**model_args))
```

### 체크포인트에서 재개 (`init_from='resume'`)
```python
checkpoint = torch.load(ckpt_path)
model = GPT(GPTConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
iter_num = checkpoint['iter_num']
```
이전 학습을 중단 없이 이어갑니다.

### 사전학습 GPT-2 불러오기 (`init_from='gpt2'`)
```python
model = GPT.from_pretrained('gpt2', override_args={'dropout': 0.1})
```
OpenAI 가중치를 불러와 파인튜닝 시작점으로 활용합니다.

---

## 4. 그래디언트 누적 (Gradient Accumulation)

메모리가 부족할 때 큰 배치 효과를 얻는 방법입니다.

```
실제 배치 크기: batch_size=12, gradient_accumulation_steps=40
유효 배치 크기 = 12 × 40 × 1024 토큰 = 491,520 토큰/스텝
```

```python
for micro_step in range(gradient_accumulation_steps):
    x, y = get_batch('train')
    with ctx:  # 혼합 정밀도 컨텍스트
        logits, loss = model(x, y)
        loss = loss / gradient_accumulation_steps  # 평균을 위해 나눔
    scaler.scale(loss).backward()  # 그래디언트 누적

# gradient_accumulation_steps 번 누적 후 한 번 업데이트
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

---

## 5. 혼합 정밀도 학습 (Mixed Precision)

```python
# dtype 선택: bfloat16 (Ampere GPU), float16 (이전 GPU), float32 (CPU)
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'

# 자동 형변환 컨텍스트
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# float16 용 그래디언트 스케일러
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

| dtype | 메모리 | 정밀도 | 사용 상황 |
|-------|--------|--------|---------|
| float32 | 기준(1×) | 높음 | CPU, 구형 GPU |
| bfloat16 | 절반(0.5×) | 충분 | Ampere 이상 (권장) |
| float16 | 절반(0.5×) | 제한적 | 구형 GPU |

**bfloat16이 권장되는 이유**: float16보다 범위가 넓어 수치 불안정 문제가 적습니다.

---

## 6. 학습률 스케줄

세 구간으로 나뉩니다:

```
학습률
  ↑
lr │      /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾──────────────
   │    /                                 ╲
   │  /  warmup                 cosine     ╲  min_lr
   │/   (2K steps)              decay       ─────────
   └────────────────────────────────────────→ 스텝
       0   2K                  600K
```

```python
def get_lr(it):
    # 1단계: 선형 워밍업
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2단계: min_lr 이하면 min_lr 고정
    if it > lr_decay_iters:
        return min_lr
    # 3단계: 코사인 감소
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 → 0
    return min_lr + coeff * (learning_rate - min_lr)
```

---

## 7. 그래디언트 클리핑

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # grad_clip=1.0
```

그래디언트의 L2 놈이 1.0을 초과하면 비율적으로 줄입니다. 폭주하는 그래디언트를 방지합니다.

---

## 8. 검증 및 체크포인트 저장

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # dropout 비활성화
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

```python
# eval_interval(기본 2000)마다 실행
if val_loss < best_val_loss:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': val_loss,
        'config': config,
    }
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
```

**검증 손실이 개선될 때만** 체크포인트를 저장합니다 (과적합 방지).

---

## 9. WandB 로깅 (선택사항)

```python
if wandb_log:
    wandb.log({
        "iter": iter_num,
        "train/loss": losses['train'],
        "val/loss": losses['val'],
        "lr": lr,
        "mfu": running_mfu * 100,  # GPU 활용률 (%)
    })
```

`wandb_log=True`로 설정하면 실시간으로 학습 곡선을 모니터링할 수 있습니다.

---

## 손실값 해석

| 모델 | 학습 데이터 | 목표 검증 손실 |
|------|-----------|--------------|
| 셰익스피어 (문자) | Shakespeare | ~1.47 |
| GPT-2 재현 | OpenWebText | ~2.85 |

손실이 낮을수록 더 나은 언어 모델입니다. 단, 학습 손실과 검증 손실 차이가 크면 과적합입니다.

---

## 다음 단계

데이터 준비 과정을 이해하려면 → [04_data_preparation.md](./04_data_preparation.md)
