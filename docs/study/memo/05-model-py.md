# 5장 · 모델 구현 분석 (model.py) — 학습 로그

본문: [`README.md` 5장](../README.md#5-모델-구현-분석-modelpy) · 색인: [`memo/README.md`](./README.md)

---

## LOG-0001 · vocab padding과 타일 양자화

**날짜** 2026-08-20 · **걸친 장** 5장(주), 3장·6장·7장 · **상태** 개념 확인 완료 / 벤치마크 미측정

### 개념정리

**vocab_size 50304란 무엇인가** — 실제 GPT-2 어휘 수는 50257개다. `50304`는 그것을
**64의 배수로 올림한 패딩값**으로, 늘어난 47칸에는 대응하는 토큰이 없다(토크나이저는 여전히
`0~50256`만 출력). 토큰을 새로 정의한 게 아니라 **모델 쪽 텐서만 크게 잡은 것**이다.
그런데도 연산량은 오히려 늘어나면서 **더 빨라진다** — 이유는 아래 GPU 정렬(§왜 빨라지는가) 참고.

**전체 과정에서의 위치** — from-scratch로 모델을 초기화할 때 `GPTConfig.vocab_size`
기본값(`model.py:111`) 또는 `meta.pkl` 유무에 따른 `train.py`의 fallback(`:155-156`)에서
50304가 정해진다. 반대로 사전학습 GPT-2 체크포인트를 불러올 때는 OpenAI 가중치 shape과 맞춰야
하므로 `model.py:222-223`에서 50257로 강제 고정된다 — **패딩은 from-scratch 학습 전용**이다.

### 근거 — 저장소 안 실제 위치

| 값        | 위치                                                  | 맥락                                   |
| --------- | ----------------------------------------------------- | -------------------------------------- |
| `50257`   | [`.venv/…/tiktoken_ext/openai_public.py:26`](../../.venv/Lib/site-packages/tiktoken_ext/openai_public.py) | `explicit_n_vocab` — 토크나이저 실측값 |
| `50257`   | [`model.py:222-223`](../../model.py#L222-L223)        | 사전학습 GPT-2 로드 시 **강제 고정**   |
| `50304`   | [`model.py:111`](../../model.py#L111)                 | `GPTConfig` 기본값 + 이유가 주석에 있음 |
| `50304`   | [`train.py:155-156`](../../train.py#L155-L156)        | `meta.pkl` 없을 때의 fallback           |
| `50304`   | [`bench.py:46-47`](../../bench.py#L46-L47)            | 벤치마크용 더미 입력                    |

```python
# model.py:111
vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
```

```python
# train.py:154-156
if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
```

즉 vocab_size의 출처는 **두 갈래**다.

```
data/<셋>/meta.pkl 있음  →  그 안의 vocab_size 사용   (shakespeare_char = 65)
meta.pkl 없음            →  50304                     (shakespeare BPE, openwebtext)
```

### 그 47칸은 실제로 무엇인가

파라미터는 **실제로 할당된다.** 개념상의 빈칸이 아니라 메모리를 쓰는 진짜 행이다.

- `wte = nn.Embedding(vocab_size, n_embd)` — [`model.py:127`](../../model.py#L127)
- `lm_head = nn.Linear(n_embd, vocab_size, bias=False)` — [`model.py:133`](../../model.py#L133)
- **가중치 공유(weight tying)** 로 이 둘은 같은 텐서 — [`model.py:138`](../../model.py#L138)

```python
self.transformer.wte.weight = self.lm_head.weight
```

47 × 768 ≈ 36K 파라미터. 124M 모델 기준 0.03% 수준이라 무시할 만하다.
공유되어 있으므로 임베딩·출력층에 중복으로 잡히지도 않는다.

동작은 이렇게 갈린다.

| 방향        | 47칸에 일어나는 일                                             |
| ----------- | -------------------------------------------------------------- |
| 입력(`wte`) | 그 id가 데이터에 없으므로 **조회되지 않음** → 직접 gradient 없음 |
| 출력(`lm_head`) | 정답이 된 적이 없으므로 **로짓을 계속 내리는 gradient만** 받음 |

결과적으로 학습 후 그 47칸의 확률은 0에 수렴한다. 생성 결과에 튀어나오지 않는다.

### 왜 빨라지는가 — 핵심 역설

> **곱셈을 더 하는데 더 빠르다.**

GPU는 큰 행렬곱을 고정 크기 **타일**(예: 128×128)로 쪼개 SM에 배분한다.
차원이 타일 크기로 나누어떨어지지 않으면, 마지막 타일이 거의 비어 있어도
**온전한 타일 하나만큼의 비용**을 낸다. 50257은 소수라 정렬이 최악이다.

```
vocab 50257 →  마지막 타일이 거의 빈 채로 한 칸 차지 + 텐서코어 정렬 조건 깨짐 → 느린 커널
vocab 50304 →  64(및 128)의 배수 → 정렬된 빠른 커널 경로
       ↑ 47열 더 계산하지만 전체는 빨라짐
```

lm_head는 `[B·T, n_embd] × [n_embd, vocab_size]`로 모델 전체에서 가장 큰 행렬곱 중 하나라
이 정렬 하나가 전체 step time에 드러난다.

### 용어

| 층위             | 이름                                        | 설명                                                 |
| ---------------- | ------------------------------------------- | ---------------------------------------------------- |
| 피하려는 현상    | **타일 양자화 (tile quantization)**         | 차원이 타일 크기로 안 나눠떨어져 생기는 낭비. NVIDIA 공식 용어 |
| 형제 개념        | **웨이브 양자화 (wave quantization)**       | 타일 **개수**가 SM 개수로 안 나눠떨어져 생기는 낭비   |
| 해결 기법(어휘)  | **vocab padding / padded vocab size**       | Megatron-LM엔 `--make-vocab-size-divisible-by` 플래그가 있음(기본 128) |
| 해결 기법(일반)  | **정렬 패딩 (alignment padding)**, shape padding | 어휘뿐 아니라 hidden dim, head dim 등 모든 축에 적용 |
| 남는 47칸        | **unused / dummy token slot**               | 토크나이저에 존재조차 하지 않는 자리                  |

한 문장으로 부른다면 → **"타일 양자화를 피하기 위한 vocab padding"**, 짧게는 **"vocab size alignment"**.

### 부가사항

- `tiktoken.get_encoding("gpt2")` 호출은 [`data/shakespeare/prepare.py:22`](../../data/shakespeare/prepare.py#L22)
  **한 곳뿐**이다. `model.py`는 토크나이저가 무엇인지 전혀 모르고 `vocab_size`라는 **정수 하나만** 받는다.
  그래서 같은 `model.py`가 char(65)든 BPE(50304)든 그대로 돌아간다.
- 여기서 말하는 tile/wave quantization은 int8·fp8 같은 **모델 양자화(quantization)와는 무관한 동음이의어**다.
  검색할 때 섞이기 쉬우니 구분해서 찾을 것.
- NLP의 `<pad>`는 배치 길이를 맞추는 실재하는 특수토큰이지만, 여기 47칸은 토크나이저 자체에
  존재하지 않는 자리라 성격이 다르다 — 같은 "padding"이라는 말을 써도 가리키는 대상이 다르다.
- `prepare.py`는 토큰을 `np.uint16`으로 저장하므로, vocab_size가 65535를 넘는 토크나이저에는
  이 코드 그대로는 못 쓴다 (GPT-2 50257, char 65 모두 이 범위 안).

### 남은 질문

- [ ] **실제 속도 이득이 이 저장소·이 GPU에서 몇 %인가.** 대화 중 "~25%"라고 말했으나 출처를 특정하지 못했다.
      공개 자료마다 수 %~수십 %로 편차가 크고 하드웨어·모델 크기에 따라 달라진다.
      → [`bench.py`](../../bench.py)를 `50257` / `50304` 두 값으로 돌려 **직접 측정**할 것. 측정 전까지 수치를 인용하지 말 것.
- [ ] 64가 아니라 128의 배수로 올리면 더 빨라지는가. (Megatron 기본값은 128, nanoGPT는 64 기준으로 서술)
- [ ] `torch.compile` 사용 시 이 이득이 유지되는가, 아니면 컴파일러가 흡수해버리는가.

### 관련

- 본문 [5.2 GPTConfig](../README.md#52-gptconfig-modelpy108) — `vocab_size` 기본값이 선언된 곳
- 본문 [3.5 torch.compile이 하는 일](../README.md#35-torchcompile이-하는-일) — 커널 선택과 관련
- 본문 [6.2 세 가지 데이터셋](../README.md#62-세-가지-데이터셋) — `meta.pkl` 유무가 갈리는 지점
- 본문 [7.4 모델 초기화 세 가지 방식](../README.md#74-모델-초기화-세-가지-방식) — `init_from='gpt2'`에서 50257 강제
- [`QNA.md`](../QNA.md) — `data/` 폴더 관련 입문 문답
