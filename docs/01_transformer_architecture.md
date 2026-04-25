# 01. 트랜스포머 아키텍처 이해

## GPT의 전체 구조

GPT는 **Transformer Decoder** 기반 모델입니다. 입력 토큰 시퀀스를 받아서 다음 토큰의 확률 분포를 출력합니다.

```
입력 토큰 [t₁, t₂, ..., tₙ]
    ↓
[Token Embedding] + [Positional Embedding]
    ↓
[Transformer Block] × N회 반복
    ↓
[Final LayerNorm]
    ↓
[Linear 출력층] → vocab_size 크기의 logits
    ↓
Softmax → 다음 토큰 확률 분포
```

---

## 1단계: 임베딩 (Embedding)

### 토큰 임베딩 (Token Embedding)
단어(토큰)를 숫자 벡터로 변환합니다.

```
"hello" → 토큰 ID: 31373 → 768차원 벡터 [0.12, -0.34, ...]
```

- 테이블 크기: `vocab_size × n_embd` (GPT-2: 50257 × 768)
- 학습 중 자동으로 의미 있는 벡터를 찾아냅니다

### 위치 임베딩 (Positional Embedding)
트랜스포머는 순서 정보를 자체적으로 알 수 없어서, 위치 정보를 별도로 추가합니다.

```
위치 0 → [0.01, 0.99, ...] (768차원)
위치 1 → [0.45, 0.23, ...] (768차원)
...
```

- 테이블 크기: `block_size × n_embd` (GPT-2: 1024 × 768)
- nanoGPT는 학습 가능한 위치 임베딩을 사용합니다

**최종 입력 = 토큰 임베딩 + 위치 임베딩**

---

## 2단계: 트랜스포머 블록 (Transformer Block)

GPT-2 기준으로 이 블록을 12번 반복합니다. 각 블록의 구조:

```
x → LayerNorm → CausalSelfAttention → x + (결과)
              → LayerNorm → MLP → x + (결과)
```

**Pre-norm 구조**: LayerNorm을 앞에 적용합니다 (학습 안정성 향상).

---

## 3단계: 인과적 자기 어텐션 (Causal Self-Attention)

GPT의 핵심입니다. "이전 단어들만 보고" 현재 단어와의 관계를 계산합니다.

### 어텐션의 직관
> "고양이가 생선을 먹었다. 그것은 맛있었다."
> 
> "그것"이 무엇을 가리키는지 알려면 이전 단어들을 참조해야 합니다.  
> 어텐션은 이 참조 관계의 강도(가중치)를 학습합니다.

### Query, Key, Value
각 토큰을 세 가지 역할로 변환합니다:

- **Query (Q)**: "나는 무엇을 찾고 있나?"
- **Key (K)**: "나는 어떤 정보를 가지고 있나?"
- **Value (V)**: "실제로 전달할 정보"

```python
# 입력 x를 Q, K, V로 선형 변환
Q, K, V = linear(x).split(n_embd, dim=2)

# 어텐션 스코어 계산
attention = Q @ K.transpose(-2, -1) / sqrt(d_k)

# 인과 마스크: 미래 토큰은 볼 수 없도록
attention = attention.masked_fill(future_mask, -inf)

# 소프트맥스 → 가중치
attention = softmax(attention)

# 최종 출력
output = attention @ V
```

### 인과 마스크 (Causal Mask)
GPT는 **왼쪽에서 오른쪽**으로만 정보를 흘려야 합니다 (미래를 볼 수 없음).

```
           t1  t2  t3  t4
      t1 [ ✓   ✗   ✗   ✗ ]   t1은 자신만 볼 수 있음
      t2 [ ✓   ✓   ✗   ✗ ]   t2는 t1, t2를 볼 수 있음
      t3 [ ✓   ✓   ✓   ✗ ]   t3는 t1~t3를 볼 수 있음
      t4 [ ✓   ✓   ✓   ✓ ]   t4는 전부 볼 수 있음
```

### 멀티헤드 어텐션 (Multi-Head Attention)
여러 어텐션을 병렬로 계산합니다. 각 헤드가 다른 종류의 관계를 학습합니다.

- GPT-2: 12개 헤드, 각 헤드 차원 = 768 / 12 = **64**
- 헤드별로 독립적으로 Q, K, V 계산 → 합산

---

## 4단계: MLP (Feed-Forward Network)

어텐션 후 각 토큰을 독립적으로 처리합니다.

```
n_embd (768) → 4×n_embd (3072) → n_embd (768)
     Linear        GELU          Linear
```

**GELU 활성화 함수**: ReLU보다 부드러운 비선형성, GPT에서 표준적으로 사용됩니다.

---

## 5단계: 출력과 손실 계산

### 출력 계산
```
[배치, 시퀀스 길이, n_embd] → Linear → [배치, 시퀀스 길이, vocab_size]
```

이 값(logits)을 softmax하면 각 위치에서 다음 토큰의 확률이 됩니다.

### 손실 함수 (Cross-Entropy Loss)
모델이 예측한 확률 vs 실제 다음 토큰을 비교합니다.

```python
# logits: [B, T, vocab_size]
# targets: [B, T] (실제 다음 토큰)
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

**학습 = 이 loss를 최소화하는 것**

---

## 핵심 아이디어 요약

| 개념 | 역할 | 비유 |
|------|------|------|
| Token Embedding | 토큰 → 벡터 | 단어를 좌표로 표현 |
| Positional Embedding | 위치 정보 추가 | 문장 내 순서 표시 |
| Self-Attention | 토큰 간 관계 계산 | 단어 간 참조 관계 |
| Causal Mask | 미래 토큰 차단 | 예언 금지 |
| MLP | 개별 토큰 변환 | 토큰별 특징 추출 |
| LayerNorm | 값 정규화 | 학습 안정화 |
| Residual Connection | 이전 값 더하기 | 지름길 연결 |

---

## 다음 단계

이 개념들이 실제 코드에서 어떻게 구현되는지 보려면 → [02_model_implementation.md](./02_model_implementation.md)
