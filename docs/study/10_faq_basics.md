# 10. 입문 Q&A — 데이터와 파일 구조

`00_overview.md`를 읽다가 막히는 지점들을 문답으로 정리했습니다.
실제 파일을 열어 확인한 내용이라 값은 모두 이 저장소 기준 실측치입니다.

> 대상: `data/` 폴더와 파이프라인 산출물. 모델 내부는 `02_model_implementation.md` 참조.

---

## Q1. `input.txt`는 안내문인가?

**아니다. 모델이 학습할 원문 텍스트 그 자체다.**

```
data/agent/input.txt            1,126,182 바이트 (약 1.1MB)
data/shakespeare_char/input.txt 1,155,394 바이트
```

`data/agent/input.txt` 앞부분:

```
# AutoGPT: Build, Deploy, and Run AI Agents

[![Discord Follow](https://img.shields.io/badge/...
```

AutoGPT 프로젝트 문서를 긁어모은 것이다. 모델은 이 글의 문체와 내용을 흉내 내도록 학습된다.

> 헷갈릴 수 있는 이유: `data/shakespeare_char/`에는 진짜 안내문인 `readme.md`가 따로 있다.
> **`readme.md` = 안내, `input.txt` = 데이터.**

---

## Q2. `input.txt`는 폴더마다 다른 파일인가?

**그렇다. 이름만 같고 내용은 전혀 다르다.**

```
data/shakespeare_char/input.txt   ← 셰익스피어 희곡
data/shakespeare/input.txt        ← 셰익스피어 희곡 (BPE판)
data/agent/input.txt              ← AutoGPT 문서
```

`prepare.py`는 항상 **자기 폴더의** `input.txt`만 읽는다.

```python
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
                               #  ↑ "이 스크립트가 있는 폴더"
```

섞이지 않는다.

---

## Q3. 왜 셰익스피어로 연습하나?

**작아서 빨리 돌기 때문이다.** 배우는 대상이 아니라 파이프라인 검증용 재료다.

| | `shakespeare_char` | `agent` |
|---|---|---|
| 출처 | 원본 nanoGPT | 이 저장소 추가 ★ |
| 크기 | 1.1MB | 1.1MB |
| 학습 시간 | 약 3분 (GPU) | 파인튜닝 수십 분 |
| 모델 | 10.6M (scratch) | 124M (GPT-2 이어받기) |
| 용도 | 환경 검증 · 디버깅 · 학습 예제 | 실제 실험 |

용도 3가지:

1. **환경 검증** — GPU/CUDA/PyTorch 설치가 정상인지 3분 만에 확인
2. **디버깅** — 코드 수정 후 망가졌는지 빠르게 확인
3. **학습용 예제** — 문자 단위(vocab 65)라 구조가 단순

`openwebtext/`도 원본에 딸려온 것이며 이 저장소에서는 사용하지 않았다.

---

## Q4. 왜 텍스트를 "학습"시키나?

**모델에 지식을 넣는 수단이 그것뿐이기 때문이다.** 초기 모델은 랜덤 숫자 덩어리다.

```
"AI agents can execute tasks autonomously"
        ↓  이 한 문장에서 뽑히는 문제들
"AI" 다음은?            → "agents"    (틀림 → 가중치 수정)
"AI agents" 다음은?     → "can"       (틀림 → 가중치 수정)
"AI agents can" 다음은? → "execute"   (틀림 → 가중치 수정)
```

정답 라벨을 사람이 붙일 필요가 없다. **텍스트 자체가 정답지다** → 자기지도학습(self-supervised).

먹인 텍스트가 곧 출력의 성격을 결정한다.

```
셰익스피어 먹임    → "Thou art a villain!"
AutoGPT 문서 먹임  → "The agent executes the task..."
```

---

## Q5. 결국 말투 흉내 아닌가?

**10M짜리 셰익스피어 모델은 정확히 그렇다.** 내용은 헛소리다.

```
KING RICHARD III:
And that the world of the streath of the sea,
```

등장인물 이름, 줄바꿈 형식, 고어체 — 껍데기만 배웠다.

**핵심은 규모가 커지면 껍데기가 내용으로 바뀐다는 점이다.**

| 모델 | 데이터 | 배우는 것 |
|---|---|---|
| 10M (셰익스피어) | 1MB | 말투·철자 |
| 124M (GPT-2) | 40GB | 문법 + 상식 + 일부 지식 |
| 1.5B (GPT-2 XL) | 40GB | 위 + 어설픈 추론 |
| 수백B (상용 LLM) | 수십 TB | 추론·코딩·대화 |

**학습 방식은 넷 다 동일하다.** "다음 토큰 맞히기" 하나뿐이며, 바뀌는 것은 모델 크기와 데이터 양이다.

이 저장소의 `agent` 실험은 한 단계 위다 — 처음부터 배우는 게 아니라 이미 언어를 아는 모델(GPT-2)에 전문 분야를 붙인다. 그래서 1MB로도 효과가 난다.

---

## Q6. 셰익스피어용 코드와 agent용 코드는 다른가?

**완전히 같다. 다른 것은 설정 파일 하나뿐이다.**

```bash
python train.py config/train_shakespeare_char.py   # 셰익스피어
python train.py config/finetune_agent.py           # agent
        ↑                    ↑
   같은 파일             이것만 다름
```

`train.py`, `model.py`는 한 글자도 바뀌지 않는다.

| 설정 | `train_shakespeare_char.py` | `finetune_agent.py` |
|---|---|---|
| `dataset` | `'shakespeare_char'` | `'agent'` |
| `init_from` | (기본 `'scratch'`) | `'gpt2'` |
| `n_layer/n_head/n_embd` | 6 / 6 / 384 | GPT-2가 결정 (12/12/768) |
| `max_iters` | 5000 | 500 |
| `learning_rate` | 1e-3 | 3e-5 |
| `out_dir` | `out-shakespeare-char` | `out-agent-ft` |

`train.py`는 문자열 하나로 데이터 경로를 조립한다.

```python
data_dir = os.path.join('data', dataset)      # dataset='agent' → 'data/agent'
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), ...)
```

`train.py`는 그 안이 셰익스피어인지 AutoGPT 문서인지 **모른다.** 그냥 숫자다.

```
                    train.py / model.py   ← 공유
        ┌───────────────────┴───────────────────┐
train_shakespeare_char.py              finetune_agent.py
   dataset='shakespeare_char'             dataset='agent'
        ↓                                      ↓
   data/shakespeare_char/                 data/agent/
        ↓                                      ↓
   out-shakespeare-char/ckpt.pt           out-agent-ft/ckpt.pt
```

**엔진 하나에 연료통만 갈아 끼우는 구조.** nanoGPT가 300줄로 끝나는 이유다.

---

## Q7. 두 실험의 가중치와 데이터는 호환되나?

**전혀 안 된다. 크기 자체가 다르다.**

| | 셰익스피어 모델 | agent 모델 |
|---|---|---|
| `vocab_size` | 65 | 50,257 |
| `n_embd` | 384 | 768 |
| `n_layer` | 6 | 12 |
| `block_size` | 256 | 1024 |
| 파라미터 | 10.77M | 124M |

로드를 시도하면 즉시 에러가 난다.

```
RuntimeError: size mismatch for lm_head.weight:
  copying a param with shape [65, 384] from checkpoint,
  the shape in current model is [50257, 768]
```

`.bin`도 마찬가지다. 같은 형식이지만 **숫자의 의미가 다르다.**

```
셰익스피어 .bin의 "42"  → meta.pkl 사전   → 'e' 라는 글자
agent .bin의 "42"      → tiktoken 사전   → ' the' 라는 단어조각
```

바꿔 넣어도 에러가 안 나고 **그냥 쓰레기를 학습한다.** 더 위험하다.

**코드만 공유, 나머지는 전부 독립.** 실험을 여러 개 동시에 돌려도 안전하다.

---

## Q8. 파이프라인 산출물 전체 목록

### 1막 — `prepare.py` 실행 결과

| 파일 | 내용 | 형식 | 크기 (셰익스피어) |
|---|---|---|---|
| `train.bin` | 학습용 토큰 ID (앞 90%) | `uint16` 바이너리 | 2,007,708 B |
| `val.bin` | 검증용 토큰 ID (뒤 10%) | `uint16` 바이너리 | 223,080 B |
| `meta.pkl` | 글자↔숫자 사전 | pickle | 703 B |

- `.bin`은 헤더도 구분자도 없다. 숫자만 이어붙인 날것이라 `np.memmap`으로 바로 읽을 수 있다.
- `uint16` = 토큰당 2바이트. 최대 65,535 → GPT-2 vocab 50,257이 딱 들어간다.
  ```
  2,007,708 바이트 ÷ 2 = 1,003,854 토큰  ✓
  ```
- **`meta.pkl`은 문자 단위에서만 생성된다.** (Q10 참조)

### 2막 — `train.py` 실행 결과

`out_dir/ckpt.pt` 하나. 내부 6개 항목:

| 키 | 내용 | 추론에 필요? |
|---|---|---|
| `model` | 가중치 전부 | ✓ |
| `optimizer` | AdamW 상태 | ✗ (학습 재개용) |
| `model_args` | 구조 설계값 | ✓ (모델 재조립) |
| `iter_num` | 진행 스텝 수 | ✗ |
| `best_val_loss` | 최고 검증 성적 | ✗ |
| `config` | 설정 스냅샷 | ✗ |

> 파일은 하나뿐이며 계속 덮어써진다. 검증 손실이 개선될 때만 저장한다
> (`always_save_checkpoint=False`인 경우). 스텝별 히스토리는 남지 않는다.

> **`iter_num`이 `max_iters`보다 작은 것은 정상이다.** 과적합이 시작되면 val loss가 더는
> 개선되지 않아 저장이 멈추고, 학습만 끝까지 계속된다. 즉 `iter_num`은 "어디서 멈췄나"가 아니라
> **"가장 좋았던 시점이 언제인가"**를 뜻한다.

> 10.77M 파라미터인데 파일이 129MB인 이유: 파라미터 자체는 약 43MB(fp32)이고,
> **AdamW가 파라미터당 상태값 2개를 더 들고 있어서** 약 3배가 된다.

### 3막 — `sample.py` / `chat.py`

**파일을 만들지 않는다.** 화면 출력뿐이다.

```bash
python sample.py --out_dir=out-shakespeare-char > result.txt   # 저장은 직접
```

### 부수적으로 생기는 것

| 대상 | 위치 | 시점 |
|---|---|---|
| `__pycache__/` | 프로젝트 루트 | Python 자동 생성 |
| `wandb/` | 프로젝트 루트 | `wandb_log=True`일 때만 |
| GPT-2 원본 가중치 (~500MB) | `~/.cache/huggingface/` | `init_from='gpt2'` 최초 1회 |

### git에 올라가는 것

`.gitignore`가 **산출물**을 제외한다 (`*.bin`, `*.pt`, `*.pkl`, `out-*/`, `.venv/`).
**원본 텍스트인 `data/*/input.txt`는 추적한다.**

| | git | 이유 |
|---|---|---|
| `input.txt` | ✓ 추적 | 원본. 재생성이 외부 URL에 의존 |
| `train.bin` / `val.bin` / `meta.pkl` | ✗ 제외 | `prepare.py`로 재생성 |
| `ckpt.pt` | ✗ 제외 | 학습으로 재생성 (129MB) |

> **원칙: 재생성 가능한 것은 올리지 않는다.** `input.txt`와 코드만 있으면
> 명령 두 번으로 나머지를 전부 복구할 수 있다.
> ```bash
> python data/agent/prepare.py
> python train.py config/finetune_agent.py
> ```

`input.txt`가 예외인 이유는 이 원칙의 전제가 성립하지 않기 때문이다. `.bin`은 `input.txt`만
있으면 언제든 만들 수 있지만, `input.txt` 자체의 재생성은 **외부 URL에 의존**한다
(`shakespeare_char`는 2015년 저장소인 `karpathy/char-rnn`). 그 URL이 사라지면 복구 경로가 없고,
이 문서의 실측치를 대조할 대상도 함께 사라진다.

> `.gitignore`에 있던 `!data/agent/input.txt`는 실제로는 아무 동작도 하지 않는 줄이었다.
> `*.bin`·`*.pkl`은 `.txt`를 매칭하지 않으므로 `input.txt`는 애초에 무시된 적이 없고,
> 무시되지 않은 것을 negation으로 되살릴 수는 없다. `git check-ignore -v`로 확인할 수 있다.

---

## Q9. `val.bin`은 "테스트용"인가?

**"검증(validation)"이 정확한 표현이다.** 테스트와 다르다.

| | 검증(val) | 테스트(test) |
|---|---|---|
| 시점 | 학습 **중에 계속** | 학습 종료 후 1회 |
| 목적 | 과적합 감지, 체크포인트 저장 판단 | 최종 성적 발표 |
| nanoGPT | ✓ 있음 | ✗ 없음 |

**`val.bin`으로는 절대 학습하지 않는다. 채점만 한다.**
`eval_interval`마다(셰익스피어 설정은 250스텝) 두 값을 비교한다.

```
train loss 1.2 / val loss 1.5   → 정상
train loss 0.8 / val loss 1.9   → 과적합. 데이터를 외워버린 것
```

시험 범위를 외운 건지 진짜 아는 건지 구분하려고 **보여주지 않은 문제**를 남겨두는 것이다.

> **뒤 10%인 이유**: 자르기 편해서다 (`data[int(n*0.9):]`). 무작위로 섞지 않는다.
> val 구간이 특정 챕터에 치우칠 수 있지만 nanoGPT는 단순함을 택했다.

### 쪼개는 시점 — 인코딩 **전**이다

```python
train_data = data[:int(n*0.9)]     # ① 먼저 텍스트 상태로 자르고
val_data   = data[int(n*0.9):]
train_ids  = encode(train_data)    # ② 각각 따로 인코딩
val_ids    = encode(val_data)
```

단, 사전(`stoi`)은 **자르기 전 전체 텍스트**에서 만든다.
그래서 val 구간에만 등장하는 글자도 사전에 포함된다.

---

## Q10. `meta.pkl`이 숫자를 부여하는 건가?

**아니다. 부여는 `prepare.py`가 하고, `meta.pkl`은 결과를 보관만 한다.**

```
prepare.py 실행 중:
  1. 고유 문자 65개 수집          ← 여기서 번호 부여
  2. stoi = {'\n':0, ' ':1, ...}
  3. 이 사전으로 텍스트를 숫자 변환 → train.bin
  4. 사전을 파일로 저장           → meta.pkl   ← 단순 백업
```

프로그램이 끝나면 메모리의 `stoi`는 사라진다. 나중에 되돌릴 수 있도록 디스크에 남기는 것이다.

```
meta.pkl 없으면 → 모델이 뱉은 46이 뭔지 알 방법이 없음
meta.pkl 있으면 → itos로 되돌림 → 't'
```

정확히는 "바이너리 파일"이 아니라 **pickle** — 파이썬 dict를 그대로 굳힌 것이다.

| | `train.bin` | `meta.pkl` |
|---|---|---|
| 생성 도구 | `numpy.tofile()` | `pickle.dump()` |
| 내용 | 숫자 배열만 | dict 구조 통째로 |
| 크기 | 2 MB | 703 B |

### `meta.pkl`의 존재 여부가 모델 크기를 결정한다

`train.py:138-156`:

```python
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    meta_vocab_size = meta['vocab_size']        # 있으면 → 65
...
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
                                              # 없으면 → 50304 (GPT-2 기본)
```

| 데이터셋 | `meta.pkl` | 출력층 | 모델 크기 |
|---|---|---|---|
| `shakespeare_char` | ✓ | 65 | 10.77M |
| `agent` | ✗ | 50,304 | 124M |

---

## Q11. `stoi` / `itos`라는 이름과 중복의 이유

```
stoi = string to integer   (문자 → 숫자)
itos = integer to string   (숫자 → 문자)
```

`to`를 `2`로 줄여 쓰는 관습(`str2int`)을 더 압축한 표기다. Karpathy가 자신의 프로젝트에서
일관되게 쓰는 이름이라 nanoGPT에도 그대로 들어왔다.

### ⚠️ 흔한 오해: "하나는 사람용, 하나는 컴퓨터용"

**아니다. 둘 다 컴퓨터가 쓰는 도구다.** 사람은 `stoi`도 `itos`도 보지 않는다.

두 dict는 **같은 짝**(`'F'`, `18`)을 담고 있다. 정보량은 완전히 동일하며 **조회 방향만 다르다.**

```
        [사람 영역]                      [기계 영역]
                    ┌─────────┐
   "First"  ───────►│  stoi   │─────────► [18,47,56,57,58]
                    └─────────┘                  │
                                                 ▼
                                             모델 계산
                                                 │
                    ┌─────────┐                  ▼
   "First"  ◄───────│  itos   │◄───────── [18,47,56,57,58]
                    └─────────┘
```

사람이 읽는 것은 양 끝의 `"First"`뿐이다. 두 dict는 모두 **코드가 호출하는 변환표**다.

> 비유: 한→영 사전과 영→한 사전. 둘 다 번역가(컴퓨터)의 도구지 독자를 위한 것이 아니다.

### 쓰이는 시점이 정반대다

```
학습 준비 (prepare.py)          생성 (sample.py)
─────────────────              ─────────────────
"First"                         [18, 47, 56, 57, 58]
   ↓  stoi                          ↓  itos
[18,47,56,57,58]                "First"
```

- `stoi` — 사람의 글을 모델에 넣을 때 (**입구**)
- `itos` — 모델이 뱉은 숫자를 사람에게 보일 때 (**출구**)

`stoi`만 있으면 학습은 되지만 결과를 읽을 수 없고, `itos`만 있으면 프롬프트를 넣을 수 없다.

### dict 하나로는 안 되나?

파이썬 dict는 **키 → 값** 조회만 O(1)이다. 역방향은 전체 순회가 필요하다.

```python
stoi['F']                              # 즉시 → 18       O(1)
for k, v in stoi.items():
    if v == 18: return k               # 전부 뒤짐        O(n)
```

| | 문자 단위 (65) | BPE (50,257) |
|---|---|---|
| `itos` 사용 | 1회 조회 | 1회 조회 |
| 역탐색 | 평균 32회 | 평균 25,000회 |
| 500토큰 생성 시 | 16,000회 | **1,250만 회** |

`itos`를 미리 만드는 비용은 703바이트짜리 dict 하나다. **메모리 2배를 쓰고 속도 수만 배를 얻는
전형적인 시간-공간 트레이드오프.**

생성 코드는 같은 리스트를 두 번 순회할 뿐이다 (`prepare.py:30-31`):

```python
stoi = { ch:i for i,ch in enumerate(chars) }   # 'F' → 18
itos = { i:ch for i,ch in enumerate(chars) }   # 18 → 'F'
                ↑ 키와 값의 위치만 교환
```

---

## Q12. 여기서 말하는 GPT는 ChatGPT인가?

**아니다. GPT는 제품이 아니라 구조와 학습 전략의 이름이다.**

```
Generative     생성하는
Pre-trained    미리 학습된
Transformer    트랜스포머
```

**"Pre-trained"는 아키텍처가 아니라 방법론을 가리킨다.** 구조 자체는 decoder-only 트랜스포머일
뿐이고, "먼저 대량의 텍스트로 학습시켜 놓고 나중에 용도별로 다듬는다"는 전략까지 포함한 이름이다.
`Pre`가 붙었다는 것은 **뒤에 무언가 더 올 것을 전제**한다는 뜻이다.

이 저장소의 두 실험이 정확히 그 두 단계다.

| 단계 | 내용 | 이 저장소 |
|---|---|---|
| **P**re-training | 난수에서 시작, 언어 자체를 습득 | `train_shakespeare_char.py` |
| Fine-tuning | 학습된 가중치에 특정 도메인을 얹음 | `finetune_agent.py` |

ChatGPT는 그 위에 두 층이 더 쌓인 **제품**이다.

```
GPT 사전학습 → instruction tuning → RLHF → 서비스 계층(안전 필터·세션·UI)
   ↑ nanoGPT는 여기까지
```

정리하면 `model.py`는 **구조 정의**, `train.py`는 **그 구조를 GPT 방식으로 학습시키는 절차**,
산출물 `ckpt.pt`가 **학습된 모델**이다.

---

## Q13. 셰익스피어 모델도 GPT-2 가중치를 받아서 쓰나?

**아니다. 난수에서 시작했다. 다운로드는 없었다.**

```
Initializing a new model from scratch
number of parameters: 10.65M          ← GPT-2는 124M
```

`train_shakespeare_char.py`는 `init_from`이 기본값 `'scratch'`다. 영어 문법도 단어도 모르는
빈 모델이 셰익스피어 1MB만 보고 글자 조합 패턴을 익힌 것이다. `"The tractor and ten will my father"`
같은 출력이 나오는 이유가 이것이다.

GPT-2 가중치 다운로드(~500MB)는 `init_from='gpt2'`인 파인튜닝 설정에서만 일어난다.

| | 셰익스피어 | agent |
|---|---|---|
| 출발점 | **난수 (scratch)** | GPT-2 124M 공개 가중치 |
| 다운로드 | 없음 | ~500MB (`~/.cache/huggingface/`) |
| 아는 것 | 셰익스피어 철자 패턴뿐 | 영어 문법·상식 |

---

## Q14. 셰익스피어 말투는 코드에 하드코딩된 것인가?

**아니다. 코드에 셰익스피어와 관련된 것은 한 글자도 없다.**

`model.py`에는 등장인물 이름 목록도, "대사 앞에 콜론을 붙여라" 같은 규칙도, 고어체 사전도 없다.
행렬 곱과 softmax뿐이다.

| | 내용 | 어디에 |
|---|---|---|
| 사람이 정함 | 레이어 6개, 헤드 6개, 차원 384, 학습률 | `config/`, `model.py` |
| 데이터가 정함 | vocab 65자와 그 구성 | `meta.pkl` (`prepare.py`가 추출) |
| **학습이 정함** | **말투·철자·형식 전부** | `ckpt.pt`의 숫자 10,770,048개 |

사람이 정한 것은 **틀**뿐이다. 무엇을 배울지는 지정하지 않았다.

**근거 셋**

1. **같은 코드가 전혀 다른 것을 학습한다.** `train.py`·`model.py`는 한 글자도 바뀌지 않고
   `dataset` 문자열 하나만 달라진다 (Q6 참조). 한국어를 넣으면 한국어 패턴을 배운다
2. **vocab 65도 하드코딩이 아니다.** `sorted(set(text))`로 원문에서 세어 만든 값이다.
   셰익스피어에 숫자가 `'3'` 하나뿐이라 사전에도 `'3'`만 들어갔다
3. **체크포인트에 텍스트가 없다.** 129MB 전체가 소수점이다 (부록 A 참조).
   학습 시작 시점엔 전부 난수였다

말투는 1,077만 개 숫자에 분산돼 있고, 어느 하나를 짚어 "여기가 콜론 규칙"이라고 말할 수 없다.

---

## Q15. 파인튜닝하면 대화가 되나?

**안 된다. 파인튜닝이 바꾸는 것은 이어쓰기의 내용과 문체다.**

| | 이어쓰는 방식 |
|---|---|
| 셰익스피어 (scratch) | 영어를 모름 → 형식만 맞는 헛소리 |
| GPT-2 파인튜닝 | 영어를 앎 + 도메인 문체 → 말이 되는 문장 |

둘 다 하는 일은 **다음 토큰 이어붙이기** 하나로 동일하다. GPT-2에서 출발하면 결과가 그럴듯해질 뿐이다.

`"What is an AI agent?"`를 입력해도 모델은 답하지 않는다. 그 줄 뒤에 올 법한 텍스트를 이어쓴다.
학습 문서에는 질문 뒤에 답변이 오는 구조가 드물고 제목·목차가 이어지는 경우가 많다.

```
What is an AI agent?
## Getting Started
### Installation
```

질문을 이해한 것이 아니라 "이런 줄 다음엔 이런 줄이 오더라"를 재현한 것이다.

진짜 질의응답이 되려면 질문-답변 쌍으로 훈련하는 **instruction tuning**이 필요하다.
nanoGPT에는 없다.

### `chat.py`의 제약

`chat.py`는 이어쓰기를 `input()` 루프로 감싸고 입력 부분을 잘라내 보여줄 뿐이다.
대화처럼 **보이게** 만드는 껍데기다. 추가로 두 가지 제약이 있다.

- `configurator.py`를 호출하지 않아 **명령줄 인자를 받지 않는다.** `--out_dir=...`은 무시되고
  `chat.py:7`의 `out-agent-ft`를 그대로 쓴다
- `chat.py:35`가 `tiktoken.get_encoding("gpt2")`로 고정돼 **GPT-2 계열 전용**이다.
  문자 단위 모델(vocab 65)에 물리면 토큰 ID가 임베딩 범위를 벗어나 실패한다

---

## Q16. 세 데이터셋의 입력과 결과는?

세 경우 모두 **유저 입력 뒤를 이어쓰는 것**이고, 달라지는 것은 출발점 가중치와 학습 데이터뿐이다.

| | `shakespeare_char` | `shakespeare` | `agent` |
|---|---|---|---|
| 원본 데이터 | 셰익스피어 희곡 1.1MB | 같은 희곡 | AutoGPT 문서 1.1MB |
| 토큰화 | 문자 단위 | BPE | BPE |
| `vocab_size` | 65 | 50,257 | 50,257 |
| 출발점 | 난수 (scratch) | GPT-2 XL 1.5B | GPT-2 124M |
| `meta.pkl` | ✓ 생성 | ✗ | ✗ |
| 설정 파일 | `train_shakespeare_char.py` | `finetune_shakespeare.py` | `finetune_agent.py` |
| 실행 방식 | `sample.py --start` | `sample.py --start` | `chat.py` |
| 나오는 것 | 형식만 맞는 헛소리 | 말이 되는 셰익스피어풍 | 기술문서 문체 이어쓰기 |

### 차이는 두 축으로 환원된다

**① 출발점** — 난수냐, 사전학습된 가중치냐. **출력 품질**을 가른다.
`shakespeare_char`와 `shakespeare`는 데이터가 같은데 결과가 갈리므로, 사전학습의 유무가
무엇을 바꾸는지 직접 비교할 수 있다.

**② 토큰화** — 문자 단위냐 BPE냐. **모델 크기**를 결정한다. `meta.pkl` 유무가 여기서 갈린다
(Q10 참조).

```
meta.pkl 있음 → vocab 65     → 출력층 (65,384)     → 10.65M 모델
meta.pkl 없음 → vocab 50,304 → 출력층 (50304,768)  → 124M 모델
```

코드(`train.py`, `model.py`)는 세 경우 모두 **한 글자도 다르지 않다.** 설정 파일 하나만 바뀐다.

### `shakespeare_char` — 실측

```bash
python sample.py --out_dir=out-shakespeare-char        # start 기본값 "\n"
```

```
Clown:
So, who is he so fear me? what was the army to my country?
Lord Marshal:
The tractor and ten will my father, when my father
Make it of me and now in a grief,--
```

형식(이름·콜론·줄바꿈·고어체)은 배웠으나 내용은 무의미하다. 셰익스피어에 트랙터는 나오지 않는다.

### `shakespeare` — 예상

같은 원문이지만 BPE로 자르고 `init_from='gpt2-xl'`이다. 이미 영어를 아는 1.5B 모델에
문체만 얹으므로 문장이 실제로 말이 된다.

```
ROMEO:
I would not lose thee for the world, sweet love;
```

같은 데이터인데 결과가 갈리는 이유는 **출발점**이다.

> `gpt2-xl`(1.5B)은 8GB VRAM에서 OOM 가능성이 높다. `--init_from=gpt2`로 낮춰야 할 수 있다.

### `agent` — 예상

```
User: What is an AI agent?
AI Expert: ## Getting Started
           ### Installation
```

문서 이어쓰기 형태로 넣으면 더 자연스럽다.

```
User: An AI agent is
AI Expert:  a system that can autonomously execute tasks by breaking
            down a goal into subtasks and calling external tools.
```

---

## 부록 A. 실제 파일 내부 (셰익스피어 기준 실측)

### `input.txt`

```
'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n'
```

### `meta.pkl`

```python
keys: ['vocab_size', 'itos', 'stoi']
vocab_size: 65

stoi: {'\n':0, ' ':1, '!':2, '$':3, '&':4, "'":5, ',':6, '-':7, '.':8, '3':9, ':':10, ';':11, ...}
itos: {0:'\n', 1:' ', 2:'!', 3:'$', 4:'&', 5:"'", 6:',', 7:'-', 8:'.', 9:'3', 10:':', 11:';', ...}
```

**정렬 순서가 곧 번호다.** `sorted(set(text))` 결과에 0부터 매긴다. 개행이 0번인 것은
ASCII 순서상 앞에 있어서일 뿐 의미는 없다.

> 숫자 문자는 `'3'` 하나뿐이다 — 셰익스피어 원문에 숫자가 거의 나오지 않는다는 뜻.

### `train.bin`

```
총 토큰: 1,003,854  |  dtype: uint16  |  파일: 2,007,708 bytes

앞 60개:
[18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 0, 14, 43, 44, 53, 56,
 43, 1, 61, 43, 1, 54, 56, 53, 41, 43, 43, 42, 1, 39, 52, 63, 1, 44, 59, 56, ...]

itos로 되돌리면:
'First Citizen:\nBefore we proceed any further, hear me speak.'
```

`input.txt` 첫 문장과 정확히 일치한다. 1:1 치환일 뿐이다.

```
F → 18      1 → ' '(공백)
i → 47      : → 10
r → 56      \n → 0
```

### `val.bin`

```
총 토큰: 111,540
앞 40개: [12, 0, 0, 19, 30, 17, 25, 21, 27, 10, 0, 19, 53, 53, 42, ...]
되돌리면: '?\n\nGREMIO:\nGood morrow, neighbour Baptis'
```

**문장 중간에서 잘렸다.** 무작위 분할이 아니라 위치로 자른다는 증거다.

### `ckpt.pt`

```
최상위 키: ['model', 'optimizer', 'model_args', 'iter_num', 'best_val_loss', 'config']

model_args   : {n_layer:6, n_head:6, n_embd:384, block_size:256,
                bias:False, vocab_size:65, dropout:0.2}
iter_num     : 1750          ← 마지막으로 val loss가 개선된 스텝 (중단된 것이 아님)
best_val_loss: 1.4667        ← 목표치 1.47 달성
총 파라미터  : 10,770,048 (10.77M)
가중치 텐서  : 40개
```

가중치 이름과 모양:

```
_orig_mod.transformer.wte.weight              (65, 384)     ← 토큰 임베딩
_orig_mod.transformer.wpe.weight              (256, 384)    ← 위치 임베딩
──────────────────── 블록 0 ────────────────────
_orig_mod.transformer.h.0.ln_1.weight         (384,)        ← LayerNorm
_orig_mod.transformer.h.0.attn.c_attn.weight  (1152, 384)   ← Q,K,V 한 번에 (384×3)
_orig_mod.transformer.h.0.attn.c_proj.weight  (384, 384)    ← 어텐션 출력
_orig_mod.transformer.h.0.ln_2.weight         (384,)
_orig_mod.transformer.h.0.mlp.c_fc.weight     (1536, 384)   ← 4배 확장 (384×4)
_orig_mod.transformer.h.0.mlp.c_proj.weight   (384, 1536)   ← 축소
──────────────────── 블록 1~5 동일 반복 ────────────────────
_orig_mod.transformer.ln_f.weight             (384,)        ← 최종 정돈
_orig_mod.lm_head.weight                      (65, 384)     ← 출력층
```

**설정값이 그대로 텐서 모양에 박혀 있다:**

| 설정 | 값 | 나타나는 곳 |
|---|---|---|
| `vocab_size` | 65 | `wte (65,384)`, `lm_head (65,384)` |
| `block_size` | 256 | `wpe (256,384)` |
| `n_embd` | 384 | 전부 |
| `n_layer` | 6 | `h.0` ~ `h.5` |
| `bias=False` | — | **`.bias` 항목이 하나도 없음** |
| `n_head=6` | — | **모양에 안 보임** (실행 중에만 쪼갬) |

> `wte`와 `lm_head`가 둘 다 `(65,384)`인 것은 우연이 아니다 —
> 가중치 공유(weight tying). `02_model_implementation.md` 참조.

실제 값을 열어보면:

```
'\n' 토큰의 임베딩 벡터 (384차원 중 앞 8개):
[-0.0191, 0.0604, -0.0116, -0.0108, -0.017, 0.0516, 0.0547, -0.0104]
```

이것이 "학습된 지식"의 실체다. 의미를 알 수 없는 소수점 1,077만 개다.

> **`_orig_mod.` 접두사**: `torch.compile`을 쓰면 자동으로 붙는다.
> 그래서 `chat.py`에 이를 제거하는 코드가 들어 있다. 떼지 않으면 키 이름이 안 맞아 로드가 실패한다.

---

## 부록 B. 문자 단위 vs BPE — 같은 문장을 쪼개보면

### BPE란

**Byte Pair Encoding** (바이트 쌍 인코딩). 이름 그대로 **가장 자주 붙어 나오는 쌍(pair)을
하나로 합치는** 작업을 반복해 사전을 만드는 알고리즘이다.

```
시작:   l o w   l o w e r   l o w e s t
        ↓ 'l'+'o' 가 가장 빈번 → 병합
1회:    lo w    lo w e r    lo w e s t
        ↓ 'lo'+'w' 가 가장 빈번 → 병합
2회:    low     low e r     low e s t
        ↓ 반복
결과:   low, er, est ... 가 사전에 등록됨
```

GPT-2는 이 병합을 사전 크기가 50,257이 될 때까지 돌린 결과를 쓴다. 그래서 흔한 단어는 통째로
1토큰이 되고, 드문 단어는 조각으로 남는다.

**"Byte"인 이유**: 원래 BPE는 1994년에 나온 데이터 압축 알고리즘이었고 문자 단위로 동작했다.
GPT-2는 이를 **바이트 단위**로 바꿔 적용했다. 유니코드 문자는 수십만 개라 전부 사전에 넣을 수 없지만
바이트는 256가지뿐이고, 모든 텍스트는 바이트로 표현된다. 따라서 사전에 없는 문자가 들어와도
최소한 바이트 단위로는 쪼갤 수 있어 **처리 불가능한 입력이 존재하지 않는다.**
한글·이모지·중국어도 GPT-2 토크나이저가 처리할 수 있는 이유다.

대신 사전에 없는 문자는 토큰을 많이 소모한다. 한글은 글자당 3바이트라 영어보다 훨씬 비효율적이다.

이 저장소는 `tiktoken` 라이브러리로 GPT-2의 BPE 사전을 그대로 불러다 쓴다
(`data/shakespeare/prepare.py`, `data/agent/prepare.py`, `sample.py`).

### 실측 비교

`agent/input.txt` 앞부분을 GPT-2 BPE로 토큰화한 실측 결과:

```
원문: '# AutoGPT: Build, Deploy, and Run AI Agents\n\n[![Discord Follow](https://...'

토큰: ['#', ' Auto', 'G', 'PT', ':', ' Build', ',', ' Deploy', ',', ' and',
       ' Run', ' AI', ' Agents', '\n', '\n', '[', '!', '[', 'Disc', 'ord',
       ' Follow', '](', 'https', '://', 'img', '.', 'shield', 's', '.', 'io']

ID  : [2, 11160, 38, 11571, 25, 10934, 11, 34706, 11, 290, 5660, 9552, 28295, ...]
```

| | 문자 단위 | BPE |
|---|---|---|
| `"Agents"` | 6토큰 (`A`,`g`,`e`,`n`,`t`,`s`) | **1토큰** (`' Agents'`) |
| 압축률 | 1글자 = 1토큰 | **3.6글자 = 1토큰** |
| 앞 공백 | 별도 토큰 | 단어에 포함 (`' Auto'`) |
| vocab | 65 | 50,257 |

```
agent/input.txt: 1,108,254 글자 → 307,470 토큰 (3.6배 압축)
```

**같은 `block_size`로 3.6배 긴 문맥을 볼 수 있다.** 이것이 BPE를 쓰는 이유다.

> `AutoGPT`가 `' Auto'` + `'G'` + `'PT'` 3조각으로 쪼개졌다.
> GPT-2 사전이 만들어진 2019년에 없던 단어라 기존 조각을 조합할 수밖에 없다.

---

## 부록 C. 한 장 요약

```
input.txt      "First Citizen:\nBefore we..."       사람이 읽음
     ↓ prepare.py
meta.pkl       {'F':18, 'i':47, 'r':56, ...}        번역표 (문자 단위만)
train.bin      [18, 47, 56, 57, 58, 1, 15, ...]     기계가 읽음
val.bin        (뒤 10%, 채점 전용)
     ↓ train.py  (1750 스텝)
ckpt.pt        wte(65,384) = [-0.0191, 0.0604, ...]  기계가 배운 것
               텐서 40개 / 파라미터 10,770,048개
     ↓ sample.py / chat.py
(화면 출력)     "KING RICHARD III:\nAnd that the..."
```

---

## 다음 단계

- 모델 내부 구조 → [02_model_implementation.md](./02_model_implementation.md)
- `.bin`에서 배치를 뽑는 방법 (`get_batch`) → [03_training_pipeline.md](./03_training_pipeline.md)
- 직접 실행해보기 → [06_hands_on.md](./06_hands_on.md)
