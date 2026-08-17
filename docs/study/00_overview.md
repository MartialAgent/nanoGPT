# NanoGPT 학습 가이드 - 전체 개요

## 이 문서 시리즈의 목적

nanoGPT는 Andrej Karpathy가 만든 GPT 언어 모델의 **최소한의 구현체**입니다.  
약 300줄짜리 `model.py`와 `train.py`로 실제 GPT-2(124M 파라미터)를 재현합니다.

> **왜 nanoGPT인가?**  
> 코드가 짧고 명확하기 때문에 "실제로 어떻게 동작하는지"를 처음 배우기에 최적입니다.  
> 논문을 읽는 것보다, 코드를 직접 보고 실행하면서 이해하는 방식을 취합니다.

---

## 학습 순서 (이 문서 시리즈 구성)

| 번호 | 파일 | 핵심 내용 |
|------|------|----------|
| 01 | [트랜스포머 아키텍처](./01_transformer_architecture.md) | GPT의 구조를 개념 수준에서 이해 |
| 02 | [모델 구현](./02_model_implementation.md) | `model.py` 코드 한 줄씩 분석 |
| 03 | [학습 파이프라인](./03_training_pipeline.md) | `train.py` - 데이터부터 최적화까지 |
| 04 | [데이터 준비](./04_data_preparation.md) | 텍스트를 토큰으로 변환하는 과정 |
| 05 | [추론과 텍스트 생성](./05_inference.md) | 학습된 모델로 글 생성하기 |
| 06 | [실습 가이드](./06_hands_on.md) | 직접 실행해보는 단계별 실습 |
| 07 | [사전학습과 데이터 처리](./07_pretraining_and_data.md) | 원본 텍스트 → `.bin` 변환 과정 상세 |
| 08 | [랩톱 WSL 환경 구축](./08_laptop_wsl_setup.md) | RTX 4060 Laptop + WSL 설정과 명령어 레퍼런스 |
| 09 | [GPU · CUDA · 텐서 기초](./09_gpu_cuda_tensor.md) | `cuda`, `tensor`, `dtype`, `torch.compile`이 뭔지 |
| 10 | [입문 Q&A](./10_faq_basics.md) | `input.txt`·`.bin`·`meta.pkl`·`ckpt.pt`가 각각 뭔지 (실측 덤프 포함) |
| 11 | [Q·K·V 차원 워크북](./11_qkv_dimension_workbook.ipynb) | 노트북. 어텐션의 B·T·C 변형을 빈칸 채우며 직접 검산 |

> 실행 환경 세팅·GPU 튜닝·실험 결과 기록은 별도로 [`docs/test/`](../test/00_system_setup.md)에 있습니다.

---

## 전체 파이프라인 한눈에 보기

```
[원본 텍스트]
    ↓  prepare.py (토크나이징)
[train.bin / val.bin]
    ↓  train.py (학습)
[checkpoint.pt]
    ↓  sample.py (추론)
[생성된 텍스트]
```

---

## 프로젝트 파일 구조

`★` 표시는 원본 nanoGPT에 없는 **이 저장소에서 추가한 파일**입니다.

```
nanoGPT/
├── model.py              ← GPT 모델 정의 (~330줄)
├── train.py              ← 학습 루프 (~343줄, 원본 337줄 + tqdm)
├── sample.py             ← 텍스트 생성 (단발성)
├── chat.py            ★  ← 학습된 모델과 대화하는 REPL
├── bench.py              ← 속도 벤치마크
├── configurator.py       ← 설정 관리
│
├── config/               ← 시나리오별 하이퍼파라미터
│   ├── train_shakespeare_char.py   ← 입문용 (빠른 학습)
│   ├── train_gpt2.py               ← GPT-2 전체 재현
│   ├── finetune_shakespeare.py     ← 파인튜닝 예시
│   └── finetune_agent.py        ★  ← AI Agent 문서 파인튜닝
│
├── data/
│   ├── shakespeare_char/  ← 입문용 데이터 (1MB)
│   ├── shakespeare/       ← BPE 토큰화된 셰익스피어
│   ├── openwebtext/       ← 대규모 학습 데이터 (9B 토큰)
│   └── agent/          ★  ← AI Agent 문서 모음 (약 17,000줄)
│
└── docs/
    ├── study/         ★  ← 지금 읽고 있는 학습 자료
    └── test/          ★  ← 환경 세팅 / GPU 튜닝 / 실험 기록
```

---

## 원본 nanoGPT와의 차이점

이 저장소는 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)의 fork이며,
업스트림 커밋 `3adf61e` 위에 아래 변경이 얹혀 있습니다.
**학습 문서를 읽을 때 원본 코드와 다른 부분이므로 미리 알아둘 것.**

> 이 절은 변경이 생길 때마다 갱신하는 **살아있는 기록**입니다. 갱신 방법은 맨 아래 "차이 목록 갱신하기" 참조.

### 0. 변경된 파일 요약

CRLF 노이즈를 제외한 실제 내용 변경입니다 (`git diff 3adf61e --ignore-cr-at-eol --stat`).

| 파일 | 변경량 | 성격 |
|------|--------|------|
| `train.py` | 21줄 | tqdm 진행바, 배치 조정, 종료 조건 |
| `model.py` | 6줄 | `estimate_mfu()` 기준 GPU 변경 |
| `bench.py` | 4줄 | 배치 조정, 주석 |
| `sample.py` | 2줄 | 주석만 |
| `chat.py` ★ | 신규 57줄 | 대화형 REPL |
| `config/finetune_agent.py` ★ | 신규 27줄 | Agent 파인튜닝 설정 |
| `data/agent/prepare.py` ★ | 신규 33줄 | Agent 데이터 토크나이징 |
| `data/agent/input.txt` ★ | 신규 17,001줄 | Agent 문서 데이터 |
| `.gitignore` | 33줄 | 체크포인트·venv 제외 |
| `docs/` ★ | 신규 17개 | 학습 자료(노트북 1개 포함) + 실험 기록 |

`★` = 원본에 없는 신규 파일. 코드 변경은 실질적으로 `train.py`·`model.py` 두 개에 집중돼 있습니다.

### 1. GPU 하드웨어에 맞춘 튜닝

| 위치 | 원본 nanoGPT | 이 저장소 (`laptop-wsl`) | 기능 차이 |
|------|-------------|------------------------|----------|
| `train.py` 배치 | `batch_size=12`, `grad_accum=5*8` | `batch_size=8`, `grad_accum=5*12` | 8GB VRAM 대응. 실효 배치는 480으로 **동일** |
| `bench.py` 배치 | `batch_size=12` | `batch_size=8` | 8GB VRAM 대응 |
| `model.py` `estimate_mfu()` | `flops_promised = 312e12` (A100) | `126e12` (RTX 4060 Laptop bf16) | **있음** — MFU 분모가 바뀜 |
| `train.py`/`sample.py`/`bench.py` dtype | bf16 지원 시 bf16, 아니면 fp16 (자동 감지) | 동일한 자동 감지 식 | **없음** — 주석만 다름 |

> **dtype에 대한 오해 주의**: 원본 nanoGPT는 이미 `torch.cuda.is_bf16_supported()`로 자동 감지합니다.
> `laptop-wsl` 브랜치는 이 원본 동작을 그대로 쓰며 주석만 하드웨어에 맞게 고쳤습니다.
> 반면 **`pc` 브랜치는 `dtype = 'float16'`으로 고정**했는데, 이는 RTX 2070(Turing, CC 7.5)이
> bfloat16을 하드웨어 지원하지 않기 때문입니다. 브랜치별로 이 값이 다르다는 점에 유의하세요.

> **MFU 수치 해석 주의**: 분모가 A100(312 TFLOPS)이 아니라 로컬 GPU 성능으로 바뀌었기 때문에,
> 여기서 찍히는 MFU % 는 원본 README의 수치와 직접 비교할 수 없습니다.
> 브랜치별 값: `laptop-wsl` = `126e12`, `pc` = `60e12`, 원본 = `312e12`.

### 2. 학습 루프에 진행바 추가 (`train.py`)

원본은 iteration마다 `print(f"iter {iter_num}: loss ...")`로 한 줄씩 출력합니다. 이를 tqdm 진행바로 대체했습니다.

| 항목 | 원본 | 이 저장소 |
|------|------|----------|
| 진행 출력 | `print(f"iter ...")` | `pbar.set_postfix(loss=..., mfu=...)` (`log_interval`마다) |
| 진행바 전진 | 해당 없음 | `pbar.update(1)` (**매 iteration**) |
| eval / 체크포인트 로그 | `print(...)` | `tqdm.write(...)` (진행바를 깨지 않음) |
| 종료 조건 | `if iter_num > max_iters` | `if iter_num >= max_iters` |

- `from tqdm import tqdm` 의존성이 추가되었으나 원본 README의 설치 목록에는 없습니다 → `pip install tqdm` 별도 필요
- 종료 조건 변경으로 총 iteration이 **1회 감소**합니다 (원본은 `max_iters + 1`회 실행)

### 3. AI Agent 파인튜닝 실험 추가

원본에는 없는 "GPT-2를 특정 도메인 문서로 파인튜닝하고 대화해보는" 실험 세트입니다.

```
data/agent/prepare.py  →  config/finetune_agent.py  →  chat.py
 (문서 → 토큰)              (GPT-2 124M 파인튜닝)        (대화형 추론)
```

- 데이터: AI 에이전트 관련 영문 문서 약 17,000줄
- 설정: `init_from='gpt2'`, `learning_rate=3e-5`, `decay_lr=False`, `max_iters=500`, `batch_size=4`, `grad_accum=8`
- `chat.py`: 체크포인트를 로드해 `input()` 루프로 대화. `_orig_mod.` 접두사(torch.compile 흔적)를 제거하는 처리가 들어 있음
- 결과 기록: [`docs/test/04_gpt2_finetuning_experiment.md`](../test/04_gpt2_finetuning_experiment.md),
  [`docs/test/03_chat_interaction_test.md`](../test/03_chat_interaction_test.md)

### 4. 기타

- `.gitignore` 확장: `.venv/`, `out-*/`, `*.pt`, `*.bin`, `*.pkl` 등 추가 (`data/agent/input.txt`는 예외로 추적)
- 전체 파일이 CRLF 줄바꿈으로 변환됨 → `git diff`에서 README·노트북·LICENSE 등이 대량 변경된 것처럼
  보이지만 실제 내용 차이는 없습니다. 비교할 때는 반드시 `--ignore-cr-at-eol`을 붙이세요
  (붙이지 않으면 41개 파일 21,114줄, 붙이면 22개 파일 19,281줄)

### 5. 알려진 버그 — 원본에는 없는 문제

이 저장소의 수정 과정에서 생긴 문제입니다. 원본 nanoGPT에는 해당하지 않습니다.

**`data/agent/prepare.py` — 셰익스피어 스크립트 복사본**

`data/shakespeare/prepare.py`를 복사해 만든 탓에 다운로드 URL이 아직 tinyshakespeare를 가리킵니다.
`input.txt`가 이미 있으면 다운로드를 건너뛰므로 현재는 정상 동작하지만, `input.txt`가 없는 상태에서
실행하면 엉뚱하게 셰익스피어를 받아옵니다. 파일 하단의 토큰 수 주석(301,966 / 36,059)도
셰익스피어 기준이라 실제 값과 다릅니다.

### 차이 목록 갱신하기

원본 대비 차이가 바뀌면 아래로 다시 뽑아 이 절을 갱신합니다.

```bash
# 파일별 변경량 (CRLF 노이즈 제외)
git diff 3adf61e --ignore-cr-at-eol --stat

# 코드 파일 상세 diff
git diff 3adf61e --ignore-cr-at-eol -U2 -- '*.py'

# 신규 파일만
git diff 3adf61e --ignore-cr-at-eol --diff-filter=A --name-only
```

`3adf61e`("Update README to mention nanochat and deprecation")가 이 fork의 업스트림 마지막 커밋입니다.

---

## 핵심 개념 미리보기

### GPT가 하는 일
GPT는 **다음 토큰 예측기**입니다. 앞에 나온 단어들을 보고 다음에 올 단어를 맞추도록 학습합니다.

```
입력:  "The quick brown fox"
출력:  "jumps" (가장 높은 확률의 다음 단어)
```

이것만 반복해서 학습하면, 모델은 자연스럽게 언어 구조를 배웁니다.

### 파라미터 크기별 모델 비교

| 모델 | 레이어 | 헤드 | 임베딩 차원 | 파라미터 |
|------|--------|------|------------|---------|
| 입문용 (셰익스피어) | 6 | 6 | 384 | ~10M |
| GPT-2 Small | 12 | 12 | 768 | 124M |
| GPT-2 Medium | 24 | 16 | 1024 | 350M |
| GPT-2 Large | 36 | 20 | 1280 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 1558M |

---

## 학습 전 권장 배경지식

- Python, PyTorch 기초
- 행렬 연산 (곱셈, 전치) 개념
- 신경망 기초 (역전파, 손실함수)

> 트랜스포머를 전혀 모른다면 → `01_transformer_architecture.md`부터 시작하세요.  
> 바로 코드를 보고 싶다면 → `02_model_implementation.md`로 이동하세요.  
> 직접 실행해보고 싶다면 → `06_hands_on.md`로 이동하세요.
