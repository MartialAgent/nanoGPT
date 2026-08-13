# 05. 추론과 텍스트 생성 (sample.py)

학습이 완료된 모델(또는 사전학습된 GPT-2)로 텍스트를 생성하는 방법을 다룹니다.

---

## 자동회귀 생성 (Autoregressive Generation)

GPT는 **한 번에 토큰 하나씩** 생성합니다.

```
1. 프롬프트 입력: "Once upon a"
2. 모델 실행 → 다음 토큰 확률 계산
3. 확률에 따라 토큰 샘플링: "time"
4. "Once upon a time" → 다시 입력
5. 다음 토큰 샘플링: "there"
6. ...반복...
```

이를 **자동회귀(autoregressive)** 생성이라 합니다: 이전 출력이 다음 입력이 됩니다.

### 첫 토큰은 반드시 주어져야 한다

모델은 입력이 있어야 다음 토큰의 확률 분포를 계산할 수 있습니다. 완전한 무에서는 시작할 수 없으므로
`start`가 그 **씨앗** 역할을 합니다. 사용자가 프롬프트를 주면 그것이 씨앗이고, 주지 않으면
기본값인 개행 문자 하나(`'\n'`)가 들어갑니다.

즉 아래 두 명령은 같습니다.

```bash
python sample.py --out_dir=out-shakespeare-char
python sample.py --out_dir=out-shakespeare-char --start="\n"
```

개행 하나만 받아도 그럴듯한 출력이 나오는 이유는, 셰익스피어 원문이 `이름:\n대사` 형식이라
모델이 "개행 다음엔 등장인물 이름이 온다"는 패턴을 학습했기 때문입니다. 실제 출력도
`Clown:` 같은 이름으로 시작합니다 — 그 뒤 전부가 모델이 스스로 만든 것입니다.

### 명령어에 아무것도 안 줬을 때 실제로 들어가는 값

```bash
python sample.py --out_dir=out-shakespeare-char --num_samples=2 --max_new_tokens=250
```

`--start`가 없지만 입력은 존재합니다. `sample.py` 안의 기본값이 그대로 쓰이기 때문입니다.

```python
start = "\n"                                    # sample.py:14  ← 여기서 결정
...
start_ids = encode(start)                       # sample.py:80
x = torch.tensor(start_ids, ...)[None, ...]     # sample.py:81
y = model.generate(x, max_new_tokens, ...)      # sample.py:87
```

`encode`는 체크포인트에 기록된 `dataset` 이름으로 `meta.pkl`을 찾아 결정됩니다 (`sample.py:58-68`).

```python
encode = lambda s: [stoi[c] for c in s]         # 문자 단위 (meta.pkl 있음)
```

`meta.pkl`의 `stoi`에서 `'\n'`은 **0번**입니다. 따라서 최종적으로 모델에 들어가는 값은 이것뿐입니다.

```
start      "\n"
  ↓ encode
start_ids  [0]
  ↓ tensor
x          shape (1, 1)   ← 토큰 단 1개
  ↓ generate(250)
y          shape (1, 251) ← 250개를 이어붙임
```

**토큰 하나로 시작해 250개를 스스로 만든 것**입니다. `--start="ROMEO:"`를 주면 `[30, 27, 25, 17, 27, 10]`
처럼 6개 토큰으로 시작할 뿐, 이후 과정은 동일합니다.

> `meta.pkl`이 없는 GPT-2 계열 모델이면 `else` 분기로 가서 `tiktoken`의 BPE 인코더를 씁니다
> (`sample.py:69-74`). 같은 `sample.py`가 두 방식을 모두 처리합니다.

---

## sample.py 실행 방법

### 사전학습된 GPT-2 사용

```bash
# GPT-2 Small (124M) 불러오기
python sample.py \
    --init_from=gpt2 \
    --start="What is the answer to life" \
    --num_samples=3 \
    --max_new_tokens=100
```

### 직접 학습한 모델 사용

```bash
python sample.py \
    --out_dir=out-shakespeare-char \
    --start="\n" \
    --num_samples=5 \
    --max_new_tokens=200
```

### 파일에서 프롬프트 읽기

```bash
python sample.py --init_from=gpt2 --start="FILE:prompt.txt"
```

---

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `init_from` | `'resume'` | `'resume'` 또는 `'gpt2'`, `'gpt2-medium'` 등 |
| `out_dir` | `'out'` | 체크포인트 폴더 경로 |
| `start` | `'\n'` | 시작 프롬프트 텍스트 |
| `num_samples` | 10 | 생성할 샘플 수 |
| `max_new_tokens` | 500 | 최대 생성 토큰 수 |
| `temperature` | 0.8 | 무작위성 조절 (0.0~2.0) |
| `top_k` | 200 | 상위 k개 토큰만 고려 |
| `device` | `'cuda'` | 사용할 디바이스 |
| `dtype` | `'bfloat16'` | 연산 정밀도 |

---

## 샘플링 전략

### Temperature (온도)

logits를 temperature로 나누어 확률 분포의 날카로움을 조절합니다.

```python
logits = logits / temperature
probs = F.softmax(logits, dim=-1)
```

```
temperature=0.5:  더 결정적, 반복적 (높은 확률 토큰을 더 자주 선택)
temperature=1.0:  원래 모델 확률 (기본값)
temperature=1.5:  더 창의적, 때로 말이 안 될 수 있음
```

**비유**: 온도가 낮으면 가장 많이 득표한 후보만 당선, 높으면 낮은 득표 후보도 당선 가능성.

### Top-K 필터링

상위 k개 토큰을 제외한 나머지를 확률 0으로 만듭니다.

```python
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')
```

```
전체 vocab 50257개 중에서
top_k=200이면 → 상위 200개 토큰만 고려
나머지 50057개는 확률 0
```

**장점**: 말이 안 되는 토큰(확률이 극히 낮은)이 생성되는 것을 방지합니다.

### Temperature + Top-K 조합 권장값

| 목적 | Temperature | Top-K |
|------|------------|-------|
| 일관된 텍스트 | 0.5 | 50 |
| 균형 잡힌 생성 | 0.8 | 200 |
| 창의적 텍스트 | 1.2 | 500 |

---

## 내부 동작: generate() 함수

```python
@torch.no_grad()  # 그래디언트 계산 비활성화 (추론 시 불필요)
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 컨텍스트 창 초과 시 잘라냄 (block_size=1024)
        idx_cond = idx if idx.size(1) <= self.config.block_size \
                   else idx[:, -self.config.block_size:]

        # 모델 실행
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]  # 마지막 위치만 필요

        # Temperature 적용
        logits = logits / temperature

        # Top-K 필터링
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 확률 분포로 변환
        probs = F.softmax(logits, dim=-1)

        # 샘플링: 확률에 따라 토큰 하나 선택
        idx_next = torch.multinomial(probs, num_samples=1)

        # 생성된 토큰을 시퀀스에 추가
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

**`@torch.no_grad()`**: 추론 시 역전파가 필요 없으므로 그래디언트를 계산하지 않습니다. 메모리와 속도 모두 절약됩니다.

---

## 컨텍스트 창 제한

GPT-2는 최대 1024 토큰을 볼 수 있습니다. 생성이 길어지면 가장 오래된 토큰을 버립니다.

```
[토큰1, 토큰2, ..., 토큰1024] → 새 토큰 생성
[토큰2, 토큰3, ..., 토큰1024, 새토큰] → 새 토큰 생성
[토큰3, ...]
```

이로 인해 매우 긴 텍스트를 생성할 때 앞 내용을 "기억"하지 못할 수 있습니다.

---

## 토크나이저 처리

### GPT-2 BPE 사용 시
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

# 프롬프트 인코딩
start_ids = enc.encode(start)
x = torch.tensor(start_ids, dtype=torch.long).unsqueeze(0)  # [1, T]

# 생성
y = model.generate(x, max_new_tokens, temperature, top_k)

# 디코딩
print(enc.decode(y[0].tolist()))
```

### 문자 수준 모델 사용 시
```python
# meta.pkl에서 매핑 로드
with open('data/shakespeare_char/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

---

## 생성 품질 향상 팁

1. **프롬프트를 구체적으로**: 짧은 프롬프트보다 문맥이 있는 프롬프트가 일관성 있는 결과를 냅니다.

2. **temperature와 top_k 조정**:
   - 반복되는 텍스트가 나오면 → temperature 올리기
   - 말이 안 되는 텍스트가 나오면 → temperature 낮추기

3. **더 큰 모델 사용**: gpt2-xl이 gpt2보다 훨씬 자연스러운 텍스트를 생성합니다.

4. **파인튜닝**: 특정 도메인의 텍스트를 원한다면 해당 도메인 데이터로 파인튜닝합니다.

---

## 다음 단계

실제로 학습부터 생성까지 실행해보려면 → [06_hands_on.md](./06_hands_on.md)
