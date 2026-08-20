# 6장 · 데이터 준비 — 학습 로그

본문: [`README.md` 6장](../README.md#6-데이터-준비) · 색인: [`memo/README.md`](./README.md)

---

## LOG-0002 · train.bin/val.bin 분할과 char vs BPE 인코딩 순서

**날짜** 2026-08-20 · **걸친 장** 6장 · **상태** 개념 확인 완료

### 개념정리

**train.bin/val.bin이란** — 데이터 준비(`data/*/prepare.py`) 단계의 최종 산출물이다.
사람이 읽는 텍스트가 아니라, 토큰 시퀀스를 그대로 `np.uint16` 정수 배열로 직렬화한 순수 바이너리 파일이다.

**전체 과정에서의 위치** — nanoGPT 파이프라인은 크게 세 단계다.

```
① 데이터 준비 (data/*/prepare.py)   →   ② 학습 (train.py)              →   ③ 추론 (sample.py)
   원문 텍스트                            train.bin/val.bin을 np.memmap으로       학습된 가중치로
   → 분할 → 인코딩                        읽어 get_batch()가 배치 생성            텍스트 생성
   → train.bin/val.bin 저장
```

train.bin/val.bin은 **①의 출력이자 ②의 입력**이다. 만들어지는 순서는:

1. 원본 텍스트(문자열)를 먼저 분할한다 — 아직 토큰화 전 상태.
2. 분할된 각 조각을 따로 인코딩(토큰화)한다.
3. 인코딩 결과를 `np.uint16` 배열로 변환해 파일로 저장한다.

**분할 비율** — shakespeare / shakespeare_char는 텍스트를 앞 90% / 뒤 10%로 순서대로 자른다(셔플 없음).
openwebtext는 예외로 `test_size=0.0005`(0.05%)를 무작위로 뽑아 val로 삼는다 — 9:1이 아니다.
데이터셋 규모가 커지면 val 비율을 굳이 크게 둘 필요가 없기 때문이다.

**인코딩 단위 두 가지 (관련 용어)**

| 구분 | 단위 | vocab 출처 | vocab 크기 | 저장되는 매핑 |
| --- | --- | --- | --- | --- |
| char 모드 (`shakespeare_char`) | 문자(character) 하나 | 이 텍스트에서 새로 생성 | 65 | `meta.pkl`에 `stoi`/`itos` 저장 |
| BPE 모드 (`shakespeare`, `openwebtext`) | 서브워드(subword) 조각 | GPT-2 사전학습 때 이미 고정된 사전 | 50257 | 없음 — `tiktoken`의 GPT-2 인코더를 그대로 사용 |

즉 **토큰화(tokenization)**란 텍스트를 정수 시퀀스로 바꾸는 과정 전체를 가리키고,
**vocab(어휘 사전)**은 그 정수가 무엇을 의미하는지 정의하는 표다. char 모드는 vocab을 데이터셋마다
새로 만들고, BPE 모드는 데이터셋과 무관하게 이미 정해진 vocab을 그대로 재사용한다는 점이 핵심 차이다.

### 근거 — 저장소 안 실제 위치

| 파일 | 위치 | 내용 |
| --- | --- | --- |
| `data/shakespeare_char/prepare.py` | [`:38-40`](../../data/shakespeare_char/prepare.py#L38-L40) | `train_data = data[:int(n*0.9)]` — 문자열 상태에서 분할 |
| `data/shakespeare_char/prepare.py` | [`:29-33`](../../data/shakespeare_char/prepare.py#L29-L33) | `stoi`/`itos` — 이 텍스트 고유 문자(65개) 매핑표 |
| `data/shakespeare_char/prepare.py` | [`:42-44`](../../data/shakespeare_char/prepare.py#L42-L44) | 분할 **후** `encode()` 호출 |
| `data/shakespeare_char/prepare.py` | [`:63-68`](../../data/shakespeare_char/prepare.py#L63-L68) | 결과: train 1,003,854 / val 111,540 토큰(=문자) |
| `data/shakespeare/prepare.py` | [`:15-17`](../../data/shakespeare/prepare.py#L15-L17) | 동일하게 문자열 상태에서 9:1 분할 |
| `data/shakespeare/prepare.py` | [`:19-22`](../../data/shakespeare/prepare.py#L19-L22) | 분할 **후** `tiktoken.get_encoding("gpt2")`로 인코딩 |
| `data/shakespeare/prepare.py` | [`:32-33`](../../data/shakespeare/prepare.py#L32-L33) | 결과: train 301,966 / val 36,059 토큰(BPE) |
| `data/openwebtext/prepare.py` | [`:26`](../../data/openwebtext/prepare.py#L26) | `train_test_split(test_size=0.0005, seed=2357, shuffle=True)` |

### 부가사항

- BPE는 텍스트 길이 비율이 9:1이어도 토큰 개수 비율은 정확히 9:1로 떨어지지 않는다
  (301,966:36,059 ≈ 8.37:1). 서브워드 경계가 주변 문맥에 따라 달라지기 때문이다.
- BPE vocab(50257)은 이 저장소의 vocab_size 기본값 50304와는 다른 논의다 —
  50304는 GPU 연산 효율을 위한 패딩값이고, 실제 토크나이저가 쓰는 값은 여전히 50257이다.
  자세한 내용은 [`05-model-py.md`](./05-model-py.md) 참고.

### 관련

- 본문 [6.2 세 가지 데이터셋](../README.md#62-세-가지-데이터셋)
- 본문 [6.3 저장 형식: .bin 파일](../README.md#63-저장-형식-bin-파일)
- [`05-model-py.md`](./05-model-py.md) — vocab_size 50257 vs 50304 논의

---

## LOG-0004 · meta.pkl의 양방향 매핑과 모델 안쪽의 wte/lm_head

**날짜** 2026-08-20 · **걸친 장** 6, 8, 5 · **상태** 개념 확인 완료

### 개념정리

**어휘 사전(vocabulary)이란** — 정수 ID와 사람이 읽는 기호를 이어주는 룩업 표다.
char 모드는 이 표를 데이터셋에서 직접 만들어 `meta.pkl`로 저장하고, BPE 모드는 `tiktoken`이 들고 있는
GPT-2 표를 그대로 쓰기 때문에 따로 저장하지 않는다.

**양방향 매핑이 필요한 위치** — 파이프라인에서 변환은 서로 반대인 두 지점에서 일어난다.

| 표 | 방향 | 쓰이는 지점 |
| --- | --- | --- |
| `stoi` (string→int) | 문자 → ID | ① 데이터 준비에서 원문을 `train.bin`으로 인코딩 / ③ 추론에서 프롬프트(`--start`)를 ID로 변환 |
| `itos` (int→string) | ID → 문자 | ③ 추론에서 모델이 생성한 정수 시퀀스를 텍스트로 복원 |

```
"ROMEO"  --stoi-->  [30,27,25,17,27]  --wte-->  벡터  →ㅡ트랜스포머ㅡ→  x
                                                                        ↓ lm_head
"R"      <--itos--  30  <--샘플링--  [65개 확률]  <--softmax--     [65개 logit]
```

**모델 경계 — 두 겹의 변환** — 위 그림에서 변환 층이 두 겹이라는 점이 핵심이다.
트랜스포머는 정수만 받고 정수 위의 점수만 내보낸다. `'R'`이 문자라는 사실을 모델은 알지 못한다.
`stoi`/`itos`는 그 정수를 사람이 읽는 기호와 이어주는 **모델 바깥의 전·후처리 표**이고,
모델 **안쪽**에서 ID와 의미 벡터를 잇는 것은 임베딩 `wte`와 출력층 `lm_head`다.

| 층위 | 하는 일 | 사는 곳 | 학습 여부 |
| --- | --- | --- | --- |
| `stoi` / `itos` | 문자 ↔ 정수 ID | `meta.pkl` (모델 밖) | 고정 — 학습되지 않음 |
| `wte` / `lm_head` | 정수 ID ↔ 의미 벡터 | `model.py` (모델 안) | 학습되는 파라미터 |

**weight tying** — 바깥에서는 `stoi`/`itos` 두 장을 모두 저장하지만, 안쪽에서는 반대다.
nanoGPT는 `wte.weight`와 `lm_head.weight`를 **같은 텐서 하나로 묶어** 입력 방향과 출력 방향에
동일한 행렬을 앞뒤로 사용한다. 같은 토큰의 입력 표현과 출력 표현이 같은 의미 공간에 놓이는 것이
자연스럽고, 파라미터도 그만큼 줄어든다.

**meta.pkl의 유무가 신호로 쓰인다** — 학습·추론 코드는 `meta.pkl`이 있으면 커스텀 어휘로,
없으면 GPT-2 표준 BPE로 간주한다. char 모드에서 `vocab_size`가 65로 잡히는 것도 이 경로를 탄 결과다.

### 근거 — 저장소 안 실제 위치

| 파일 | 위치 | 내용 |
| --- | --- | --- |
| `data/shakespeare_char/prepare.py` | [`:30-35`](../../data/shakespeare_char/prepare.py#L30-L35) | `stoi`/`itos` 생성과 `encode`/`decode` 정의 |
| `data/shakespeare_char/prepare.py` | [`:55-61`](../../data/shakespeare_char/prepare.py#L55-L61) | `vocab_size`·`itos`·`stoi` 세 개를 `meta.pkl`로 저장 |
| `sample.py` | [`:56-68`](../../sample.py#L56-L68) | `meta.pkl`을 찾아 `stoi`/`itos`를 모두 꺼내 `encode`/`decode` 구성 |
| `sample.py` | [`:69-74`](../../sample.py#L69-L74) | 없을 때 GPT-2 인코딩으로 폴백 |
| `sample.py` | [`:80`](../../sample.py#L80), [`:88`](../../sample.py#L88) | 프롬프트에 `encode`, 생성 결과에 `decode` — 양방향이 실제로 모두 호출됨 |
| `train.py` | [`:139-145`](../../train.py#L139-L145) | `meta.pkl`이 있으면 `meta_vocab_size`를 읽음 |
| `train.py` | [`:150-156`](../../train.py#L150-L156) | 없으면 기본값 50304로 대체 |
| `model.py` | [`:127`](../../model.py#L127) | `wte = nn.Embedding(vocab_size, n_embd)` — ID → 벡터 |
| `model.py` | [`:133`](../../model.py#L133) | `lm_head = nn.Linear(n_embd, vocab_size, bias=False)` — 벡터 → ID별 점수 |
| `model.py` | [`:138`](../../model.py#L138) | `self.transformer.wte.weight = self.lm_head.weight` — weight tying |
| `model.py` | [`:177`](../../model.py#L177), [`:186`](../../model.py#L186) | forward에서 `wte(idx)` → … → `lm_head(x)` 순서 |

### 부가사항

- `itos`는 원리적으로 `stoi`에서 파생할 수 있다. char 모드의 매핑은 `enumerate(chars)`의 단순 순번이라
  정렬된 문자 리스트 하나만 있어도 복원된다. 그럼에도 두 표를 모두 굽는 이유는 두 가지다.
  (1) dict 조회는 O(1)이지만 한 표만 두고 역방향을 찾으면 토큰마다 사전 전체를 훑어야 한다.
  (2) 매핑이 단순 순번이 아닌 인코더로 확장할 여지를 남긴다 — `sample.py:65`에 그 취지의 TODO 주석이 있다.
- weight tying으로 절약되는 파라미터는 `vocab_size × n_embd`다. char 설정(`n_embd=384`, vocab 65)에서는
  약 2.5만 개로 미미하지만, GPT-2 small(`n_embd=768`, vocab 50257) 규모에서는 약 3,860만 개가 된다.
- 어휘 사전의 크기(65 / 50257)와 모델이 실제로 잡는 `vocab_size`(50304)가 갈리는 이유는
  [LOG-0001](./05-model-py.md#log-0001--vocab-padding과-타일-양자화) 참고.

### 관련

- 본문 [6. 데이터 준비](../README.md#6-데이터-준비)
- 본문 [8. 추론과 텍스트 생성 (sample.py)](../README.md#8-추론과-텍스트-생성-samplepy)
- [LOG-0002](#log-0002--trainbinvalbin-분할과-char-vs-bpe-인코딩-순서) — char/BPE 인코딩 분기
- [`05-model-py.md`](./05-model-py.md) — `vocab_size` 패딩 논의
