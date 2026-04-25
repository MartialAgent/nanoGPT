# 02. 모델 구현 분석 (model.py)

`model.py`는 약 330줄로 GPT 전체를 구현합니다. 클래스별로 분석합니다.

---

## 전체 클래스 구조

```
GPTConfig        ← 하이퍼파라미터 설정값 (dataclass)
LayerNorm        ← 정규화 레이어
CausalSelfAttention  ← 핵심 어텐션 메커니즘
MLP              ← Feed-Forward 네트워크
Block            ← 트랜스포머 블록 (Attention + MLP)
GPT              ← 최상위 모델 클래스
```

---

## GPTConfig (model.py:108)

```python
@dataclass
class GPTConfig:
    block_size: int = 1024    # 최대 시퀀스 길이 (컨텍스트 창)
    vocab_size: int = 50304   # 어휘 크기 (GPT-2: 50257, 64의 배수로 올림)
    n_layer: int = 12         # 트랜스포머 블록 수
    n_head: int = 12          # 어텐션 헤드 수
    n_embd: int = 768         # 임베딩 차원
    dropout: float = 0.0      # 드롭아웃 (사전학습: 0.0, 파인튜닝: 0.1)
    bias: bool = True         # 선형 레이어의 편향 사용 여부
```

> **vocab_size가 50304인 이유**: 원래 GPT-2는 50257인데, GPU 연산 효율을 위해 64의 배수인 50304로 올립니다. 추가된 토큰은 학습 중 사용되지 않습니다.

---

## LayerNorm (model.py:18)

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

PyTorch 기본 LayerNorm과 동일하지만, `bias=False` 옵션을 지원합니다.

**역할**: 각 토큰 벡터의 값을 정규화 (평균=0, 분산=1)하여 학습을 안정화합니다.

---

## CausalSelfAttention (model.py:29)

어텐션의 핵심 구현입니다.

### 초기화

```python
def __init__(self, config):
    # Q, K, V를 한 번에 계산하는 선형 레이어
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    # 출력 투영
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    # 인과 마스크 등록 (학습 파라미터 아님)
    self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                      .view(1, 1, block_size, block_size))
```

`c_attn`은 입력 768차원을 받아 3×768=2304 차원을 출력하고, 이를 Q/K/V로 분할합니다.

### Forward (핵심 계산)

```python
def forward(self, x):
    B, T, C = x.size()  # 배치, 시퀀스 길이, 임베딩 차원

    # Q, K, V 계산 및 분리
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

    # 멀티헤드를 위해 reshape: [B, n_head, T, head_dim]
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    # Flash Attention (PyTorch 2.0+): 메모리 효율적인 구현
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=self.dropout, is_causal=True
    )
```

**Flash Attention**: 표준 어텐션과 수학적으로 동일하지만, GPU 메모리를 훨씬 효율적으로 사용합니다. PyTorch 2.0부터 내장되어 있습니다.

---

## MLP (model.py:78)

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)    # 768 → 3072
        x = self.gelu(x)    # 비선형 활성화
        x = self.c_proj(x)  # 3072 → 768
        x = self.dropout(x)
        return x
```

4배 확장 후 축소하는 구조로, 모델이 더 풍부한 표현을 학습하도록 합니다.

---

## Block (model.py:94)

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Pre-norm + Attention + Residual
        x = x + self.mlp(self.ln_2(x))   # Pre-norm + MLP + Residual
        return x
```

**Residual Connection (잔차 연결)**: `x + f(x)` 형태로 입력을 출력에 더합니다.  
→ 기울기 소실(vanishing gradient) 문제를 방지하고, 매우 깊은 네트워크 학습을 가능하게 합니다.

---

## GPT (model.py:118)

### 전체 구조

```python
self.transformer = nn.ModuleDict({
    'wte': nn.Embedding(vocab_size, n_embd),      # 토큰 임베딩
    'wpe': nn.Embedding(block_size, n_embd),      # 위치 임베딩
    'drop': nn.Dropout(dropout),
    'h': nn.ModuleList([Block(config) for _ in range(n_layer)]),  # 블록들
    'ln_f': LayerNorm(n_embd, bias=bias),          # 최종 LayerNorm
})
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # 출력층

# 가중치 공유: 토큰 임베딩 ↔ 출력층
self.transformer.wte.weight = self.lm_head.weight
```

**가중치 공유 (Weight Tying)**: 입력 임베딩과 출력 레이어가 같은 가중치를 사용합니다.
- 파라미터 절약 (~30M 절감)
- "비슷한 토큰은 비슷한 벡터"라는 제약이 학습을 안정화

### Forward

```python
def forward(self, idx, targets=None):
    B, T = idx.size()
    pos = torch.arange(0, T, device=device)  # [0, 1, 2, ..., T-1]

    # 임베딩
    tok_emb = self.transformer.wte(idx)   # [B, T, n_embd]
    pos_emb = self.transformer.wpe(pos)   # [T, n_embd]
    x = self.drop(tok_emb + pos_emb)

    # 트랜스포머 블록들 통과
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)

    # 손실 계산 여부
    if targets is not None:
        logits = self.lm_head(x)  # 학습 시: 모든 위치
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    else:
        logits = self.lm_head(x[:, [-1], :])  # 추론 시: 마지막 위치만
        loss = None

    return logits, loss
```

> **추론 최적화**: 새 토큰 생성 시 마지막 위치의 logits만 계산하면 충분합니다 (`x[:, [-1], :]`).

### 가중치 초기화 (model.py:167)

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**잔차 투영의 특별한 초기화**: `std = 0.02 / sqrt(2 * n_layer)`  
깊은 네트워크에서 잔차 경로의 값이 누적되지 않도록 GPT-2 논문에서 제안된 방식입니다.

### AdamW 옵티마이저 설정 (model.py:218)

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # 가중치 감쇠를 2D 이상 텐서에만 적용
    decay_params    = [p for p in param_dict.values() if p.dim() >= 2]
    no_decay_params = [p for p in param_dict.values() if p.dim() < 2]
```

**Weight Decay**: 과적합 방지를 위해 가중치를 정규화합니다.
- 2D 이상 (행렬): 감쇠 적용 → `weight_decay=0.1`
- 1D (편향, LayerNorm 파라미터): 감쇠 적용 안 함

### 텍스트 생성 (model.py:301)

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 컨텍스트가 block_size를 초과하면 잘라냄
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature  # 온도 적용

        if top_k is not None:
            # top_k 이외의 토큰을 -inf로 마스킹
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # 샘플링
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

**Temperature**: logits를 나누어 분포의 날카로움을 조절합니다.
- `temperature=1.0`: 원래 분포
- `temperature < 1.0`: 더 결정적 (높은 확률 토큰 선호)
- `temperature > 1.0`: 더 무작위 (다양한 출력)

---

## 파라미터 수 계산

GPT-2 (124M) 기준:

| 컴포넌트 | 계산 | 파라미터 수 |
|---------|------|-----------|
| 토큰 임베딩 | 50304 × 768 | 38.6M |
| 위치 임베딩 | 1024 × 768 | 0.79M |
| 어텐션 (×12) | 4 × 768² × 12 | 28.3M |
| MLP (×12) | 2 × 4 × 768² × 12 | 56.6M |
| LayerNorm | 768 × 2 × 13 | 0.02M |
| **합계** | | **~124M** |

> `from_pretrained()` 메서드로 OpenAI가 공개한 실제 GPT-2 가중치를 불러올 수 있습니다.

---

## 다음 단계

모델이 어떻게 학습되는지 보려면 → [03_training_pipeline.md](./03_training_pipeline.md)
