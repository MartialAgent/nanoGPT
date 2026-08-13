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

### 1. GPU 하드웨어에 맞춘 튜닝

원본은 A100(bfloat16) 기준으로 값을 자동 감지하지만, 이 저장소는 로컬 GPU 값으로 고정했습니다.

| 위치 | 원본 nanoGPT | 이 저장소 |
|------|-------------|----------|
| `train.py` / `sample.py` / `bench.py` | `dtype = bf16 지원시 bf16, 아니면 fp16` | `dtype = 'bfloat16'` 고정 |
| `train.py` 배치 | `batch_size=12`, `grad_accum=5*8` | `batch_size=8`, `grad_accum=5*12`<br>(8GB VRAM 대응, 실효 배치 480은 동일) |
| `model.py` `estimate_mfu()` | `flops_promised = 312e12` (A100) | `126e12` (RTX 4060 Laptop bf16) |

> **MFU 수치 해석 주의**: 분모가 A100이 아니라 로컬 GPU 성능으로 바뀌었기 때문에,
> 여기서 찍히는 MFU % 는 원본 README에 나오는 수치와 직접 비교할 수 없습니다.

### 2. 학습 루프에 진행바 추가 (`train.py`)

- `tqdm` 진행바를 도입해 iteration마다 loss/MFU를 `set_postfix`로 표시
- eval 결과와 체크포인트 저장 로그는 진행바를 깨지 않도록 `tqdm.write` 사용
- 종료 조건이 `iter_num > max_iters` → `iter_num >= max_iters` 로 변경 (총 iteration 1회 감소)
- `tqdm` 의존성이 추가되었으나 requirements에는 미반영 → 별도 `pip install tqdm` 필요

### 3. AI Agent 파인튜닝 실험 추가

원본에는 없는 "GPT-2를 특정 도메인 문서로 파인튜닝하고 대화해보는" 실험 세트입니다.

```
data/agent/prepare.py  →  config/finetune_agent.py  →  chat.py
 (문서 → 토큰)              (GPT-2 124M 파인튜닝)        (대화형 추론)
```

- 데이터: AutoGPT 등 AI 에이전트 관련 영문 문서 약 17,000줄
- 설정: `init_from='gpt2'`, `learning_rate=3e-5`, `decay_lr=False`, `max_iters=500`
- 결과 기록: [`docs/test/04_gpt2_finetuning_experiment.md`](../test/04_gpt2_finetuning_experiment.md),
  [`docs/test/03_chat_interaction_test.md`](../test/03_chat_interaction_test.md)

> ⚠️ **알려진 문제**: `data/agent/prepare.py`는 `data/shakespeare/prepare.py`를 복사해 만든 것이라
> 다운로드 URL이 아직 tinyshakespeare를 가리킵니다. `input.txt`가 이미 있으면 다운로드를 건너뛰므로
> 현재는 정상 동작하지만, `input.txt`가 없는 상태에서 실행하면 엉뚱하게 셰익스피어를 받아옵니다.
> 파일 하단의 토큰 수 주석(301,966 / 36,059)도 셰익스피어 기준이라 실제 값과 다릅니다.

### 4. 기타

- `.gitignore` 확장: `.venv/`, `out-*/`, `*.pt` 등 추가 (`data/agent/input.txt`는 예외로 추적)
- 전체 파일이 CRLF 줄바꿈으로 변환됨 → `git diff`에서 README·노트북 등이 대량 변경된 것처럼
  보이지만 실제 내용 차이는 없습니다. 비교할 때는 `git diff --ignore-cr-at-eol` 사용

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
