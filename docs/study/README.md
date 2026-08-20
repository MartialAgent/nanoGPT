# nanoGPT 학습 가이드

nanoGPT는 Andrej Karpathy가 만든 GPT 언어 모델의 **최소한의 구현체**입니다.
약 300줄짜리 `model.py`와 `train.py`로 실제 GPT-2(124M 파라미터)를 재현합니다.

> **왜 nanoGPT인가?**
> 코드가 짧고 명확하기 때문에 "실제로 어떻게 동작하는지"를 처음 배우기에 최적입니다.
> 논문을 읽는 것보다, 코드를 직접 보고 실행하면서 이해하는 방식을 취합니다.

이 문서는 저장소 전체를 한 편으로 설명합니다. 문답 형식으로 정리된 입문 질문은
[`QNA.md`](./QNA.md)에, 어텐션의 차원 변형을 직접 검산해보는 노트북은
[`qkv_dimension_workbook.ipynb`](./qkv_dimension_workbook.ipynb)에 따로 있습니다.
실행 환경 세팅·GPU 튜닝·실험 결과 기록은 [`docs/test/`](../test/00_system_setup.md)에 있습니다.
읽다가 막힌 지점과 그 해소 과정을 장별로 쌓아둔 학습 로그는 [`memo/`](./memo/README.md)에 있습니다.

---

## 목차

| 장                                                                                  | 내용                                      |
| ----------------------------------------------------------------------------------- | ----------------------------------------- |
| [1. 전체 개요](#1-전체-개요)                                                         | 파이프라인, 파일 구조, 핵심 개념 미리보기 |
| [2. 원본 nanoGPT와의 차이점](#2-원본-nanogpt와의-차이점)                             | fork 이후 얹힌 변경의 살아있는 기록       |
| [3. GPU와 CUDA 그리고 텐서](#3-gpu와-cuda-그리고-텐서)                               | 코드를 읽기 전 최소한의 배경지식          |
| [4. 트랜스포머 아키텍처](#4-트랜스포머-아키텍처)                                     | GPT의 구조를 개념 수준에서 이해           |
| [5. 모델 구현 분석 (model.py)](#5-모델-구현-분석-modelpy)                            | 코드 한 줄씩 분석                         |
| [6. 데이터 준비](#6-데이터-준비)                                                     | 텍스트를 토큰으로 변환하는 과정           |
| [7. 학습 파이프라인 (train.py)](#7-학습-파이프라인-trainpy)                          | 데이터 로딩부터 최적화까지                |
| [8. 추론과 텍스트 생성 (sample.py)](#8-추론과-텍스트-생성-samplepy)                  | 학습된 모델로 글 생성하기                 |
| [9. 실습 가이드](#9-실습-가이드)                                                     | 직접 실행해보는 단계별 실습               |
| [10. 랩톱 WSL 환경 구축과 명령어 레퍼런스](#10-랩톱-wsl-환경-구축과-명령어-레퍼런스) | RTX 4060 Laptop + WSL 설정                |
| [11. 더 읽을 자료](#11-더-읽을-자료)                                                 | 다음 단계                                 |

> 트랜스포머를 전혀 모른다면 → [4장](#4-트랜스포머-아키텍처)부터.
> 바로 코드를 보고 싶다면 → [5장](#5-모델-구현-분석-modelpy)으로.
> 직접 실행해보고 싶다면 → [9장](#9-실습-가이드)으로.

---

# 1. 전체 개요

## 1.1 파이프라인 한눈에 보기

전체 과정은 **3막**입니다. 각 막에서 어떤 파일이 읽히고 무엇이 만들어지는지가 아래 그림입니다.

```
①  데이터 준비                    ②  학습                        ③  사용
────────────────────────         ───────────────────────       ──────────────────

data/<셋>/input.txt               config/<설정>.py                out-*/ckpt.pt
   (원본 텍스트)                    (하이퍼파라미터)                   │
        │                               │                             ├─► sample.py
        │                               ▼                             │      └─► 화면 출력
        ▼                        configurator.py                      │
data/<셋>/prepare.py              (CLI·설정 병합)                      └─► chat.py
        │                               │                                    └─► 대화 REPL
        ▼                               ▼
   train.bin ─────────────────────► train.py ◄──────── model.py
   val.bin                              │             (GPT 클래스 정의)
   meta.pkl (문자 단위만)                 │
                                        ▼
                                  out-*/ckpt.pt
                                  (가중치 + 옵티마이저 상태)
```

**막별 상세 — 실행 파일과 입출력**

| 막            | 실행 명령                            | 읽는 파일                                                                                                                                 | 만드는 파일                                           |
| ------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| ① 토크나이징 | `python data/<셋>/prepare.py`      | `input.txt` (없으면 외부 URL에서 다운로드)                                                                                              | `train.bin`, `val.bin`, `meta.pkl`(문자 단위만) |
| ② 학습       | `python train.py config/<설정>.py` | `config/<설정>.py`, `configurator.py`, `model.py`, `train.bin`·`val.bin`·`meta.pkl` / `init_from='gpt2'`면 HF 캐시 가중치 | `<out_dir>/ckpt.pt`                                 |
| ③-a 생성     | `python sample.py --out_dir=...`   | `ckpt.pt`, `configurator.py`, `model.py`, `meta.pkl` 또는 tiktoken BPE 사전                                                       | 없음 (화면 출력만)                                    |
| ③-b 대화     | `python chat.py`                   | `out-agent-ft/ckpt.pt`, `model.py`, tiktoken BPE 사전                                                                                 | 없음 (화면 출력만)                                    |

- **`model.py`와 `configurator.py`는 막을 가로지르는 공용 부품**입니다. `model.py`는 ②③ 모두에서
  `import`되고, `configurator.py`는 `train.py`·`sample.py`·`bench.py`가 `exec()`로 불러 씁니다
  (`chat.py`만 예외 — 그래서 명령줄 인자를 받지 않습니다).
- **③은 파일을 만들지 않습니다.** 저장하려면 `> result.txt`로 직접 리다이렉트해야 합니다.
- 부수적으로 `__pycache__/`, `wandb/`(로깅 켤 때), `~/.cache/huggingface/`(GPT-2 가중치 ~500MB)가 생깁니다.

**파이프라인에 직접 참여하지 않는 파일**

| 파일                         | 역할                                                       | 주의                                                                                                                                                                              |
| ---------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bench.py`                 | 속도·MFU 측정                                             | 기본값이`real_data=True` / `dataset='openwebtext'`라 **`data/openwebtext/train.bin`을 요구**합니다. 준비하지 않았다면 `--real_data=False`로 난수 배치를 써야 합니다 |
| `config/eval_gpt2*.py`     | 학습이 아니라`eval_only=True`로 공개 GPT-2의 손실만 측정 | `train.py`에 넘기지만 첫 eval 직후 종료됩니다 (`train.py:290`)                                                                                                                |
| `scaling_laws.ipynb`       | Chinchilla 스케일링 법칙 재현                              | 원본 부속 분석 노트북                                                                                                                                                             |
| `transformer_sizing.ipynb` | FLOPs·파라미터·메모리 이론 추정                          | 원본 부속 분석 노트북                                                                                                                                                             |
| `docs/`                    | 학습 자료와 실험 기록                                      | 이 저장소에서 추가                                                                                                                                                                |

## 1.2 프로젝트 파일 구조

저장소의 **모든 파일**입니다.
`★` = 원본 nanoGPT에 없는 이 저장소의 추가물, `⚙` = 실행으로 생기는 산출물(git 제외),
`📦` = 실행 경로에서 빼내 `archive/` 아래로 격리한 파일(2.4 참조).

```
nanoGPT/
│
├── model.py                     330줄   GPT 정의 (GPTConfig·LayerNorm·CausalSelfAttention·MLP·Block·GPT)
├── train.py                     344줄   학습 루프 (원본 337줄 + tqdm)
├── sample.py                     89줄   텍스트 생성 (단발성)
├── bench.py                     117줄   속도·MFU 벤치마크
├── configurator.py               47줄   설정 파일·CLI 인자 오버라이드 (exec 기반)
│
├── config/                              시나리오별 하이퍼파라미터
│   ├── train_shakespeare_char.py 37줄   입문용 scratch 학습 (10.65M, 5000 iters)
│   ├── train_gpt2.py             25줄   GPT-2 124M 전체 재현 (8×A100, ~5일)
│   ├── finetune_shakespeare.py   25줄   GPT-2 XL(1.5B) → 셰익스피어 파인튜닝
│   ├── eval_gpt2.py               8줄   ┐ 학습 없이 eval_only=True로
│   ├── eval_gpt2_medium.py        8줄   │ 공개 GPT-2의 손실만 측정
│   ├── eval_gpt2_large.py         8줄   │ (124M / 350M / 774M / 1558M)
│   └── eval_gpt2_xl.py            8줄   ┘
│
├── data/
│   ├── shakespeare_char/                문자 단위, vocab 65 — 입문용
│   │   ├── prepare.py            68줄
│   │   ├── readme.md                    안내문 (input.txt와 혼동 주의)
│   │   ├── input.txt            1.1MB   prepare.py가 외부 URL에서 다운로드
│   │   ├── train.bin      ⚙   2,007,708 B  (1,003,854 토큰)
│   │   ├── val.bin        ⚙     223,080 B  (111,540 토큰)
│   │   └── meta.pkl       ⚙       703 B    stoi/itos — 문자 단위에서만 생성
│   ├── shakespeare/                     같은 원문의 BPE 판 — prepare 미실행
│   │   ├── prepare.py            33줄
│   │   └── readme.md
│   └── openwebtext/                     대규모 (~54GB / 9B 토큰) — 이 저장소 미사용
│       ├── prepare.py            81줄
│       └── readme.md
│
├── docs/                  ★
│   ├── study/             ★             학습 자료 (지금 읽는 문서)
│   │   ├── README.md      ★             이 문서 — 저장소 전체 설명
│   │   ├── QNA.md         ★             입문 Q&A 16문답 + 실측 덤프
│   │   └── qkv_dimension_workbook.ipynb ★  어텐션 B·T·C 차원 검산 노트북
│   └── test/              ★             환경 세팅 / GPU 튜닝 / 실험 기록
│       ├── 00_system_setup.md
│       ├── 01_gpu_optimization.md
│       └── 02_training_report_rtx2070.md
│
├── archive/               ★📦           실행 경로에서 격리한 실험 보관소
│   └── agent-experiment/  ★📦
│       ├── README.md      ★📦           보관 경위·복원 절차·미완 사항
│       └── repo/          ★📦           저장소 루트 기준 원래 경로를 그대로 보존
│           ├── chat.py                57줄   대화형 REPL
│           ├── config/finetune_agent.py 27줄 GPT-2 124M → AI Agent 파인튜닝
│           ├── data/agent/prepare.py   33줄  (셰익스피어 복사본 — 2.5의 알려진 버그)
│           ├── data/agent/input.txt 17,001줄 AutoGPT 문서 1.1MB
│           └── docs/test/03·04.md            대화·파인튜닝 실험 기록
│
├── scaling_laws.ipynb                   Chinchilla 스케일링 재현 (원본 부속)
├── transformer_sizing.ipynb             FLOPs·파라미터·메모리 이론 추정 (원본 부속)
│
├── assets/                              원본 README용 이미지
│   ├── nanogpt.jpg
│   └── gpt2_124M_loss.png
│
├── README.md                            원본 nanoGPT README
├── LICENSE                              MIT
├── .gitignore                           산출물·venv 제외 (2.5 참조)
├── .gitattributes                       `*.ipynb linguist-generated` — 언어 통계 보정
│
├── out-shakespeare-char/  ⚙             학습 산출물 디렉터리
│   └── ckpt.pt            ⚙   128,986,325 B  (10.77M 파라미터 + AdamW 상태)
├── out/                   ⚙             기본 out_dir (현재 비어 있음)
├── __pycache__/           ⚙             Python 바이트코드
└── .venv/                 ⚙             Windows용 venv (WSL은 ~/venvs/nanogpt 사용)
```

> `⚙` 항목은 `.gitignore`에 걸려 있어 clone 직후에는 존재하지 않습니다.
> `input.txt`와 코드만 있으면 `prepare.py` → `train.py` 두 번으로 전부 복구됩니다
> (`data/shakespeare_char/input.txt`도 `prepare.py`가 내려받으므로 마찬가지입니다).
> 자세한 추적 정책은 [`QNA.md` Q8](./QNA.md#q8-파이프라인-산출물-전체-목록) 참조.

## 1.3 핵심 개념 미리보기

### GPT가 하는 일

GPT는 **다음 토큰 예측기**입니다. 앞에 나온 단어들을 보고 다음에 올 단어를 맞추도록 학습합니다.

```
입력:  "The quick brown fox"
출력:  "jumps" (가장 높은 확률의 다음 단어)
```

이것만 반복해서 학습하면, 모델은 자연스럽게 언어 구조를 배웁니다.

### 파라미터 크기별 모델 비교

| 모델                | 레이어 | 헤드 | 임베딩 차원 | 파라미터 |
| ------------------- | ------ | ---- | ----------- | -------- |
| 입문용 (셰익스피어) | 6      | 6    | 384         | ~10M     |
| GPT-2 Small         | 12     | 12   | 768         | 124M     |
| GPT-2 Medium        | 24     | 16   | 1024        | 350M     |
| GPT-2 Large         | 36     | 20   | 1280        | 774M     |
| GPT-2 XL            | 48     | 25   | 1600        | 1558M    |

## 1.4 권장 배경지식

- Python, PyTorch 기초
- 행렬 연산 (곱셈, 전치) 개념
- 신경망 기초 (역전파, 손실함수)

`cuda`·`tensor`·`dtype`이 처음이라면 [3장](#3-gpu와-cuda-그리고-텐서)에서 최소한만 짚고 갑니다.

---

# 2. 원본 nanoGPT와의 차이점

이 저장소는 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)의 fork이며,
업스트림 커밋 `3adf61e` 위에 아래 변경이 얹혀 있습니다.
**학습 문서를 읽을 때 원본 코드와 다른 부분이므로 미리 알아둘 것.**

> 이 장은 변경이 생길 때마다 갱신하는 **살아있는 기록**입니다. 갱신 방법은 [2.6](#26-차이-목록-갱신하기) 참조.

## 2.1 변경된 파일 요약

CRLF 노이즈를 제외한 실제 내용 변경입니다 (`git diff 3adf61e --ignore-cr-at-eol --stat`).

| 파일                            | 변경량        | 성격                                           |
| ------------------------------- | ------------- | ---------------------------------------------- |
| `train.py`                    | 21줄          | tqdm 진행바, 배치 조정, 종료 조건              |
| `model.py`                    | 6줄           | `estimate_mfu()` 기준 GPU 변경               |
| `bench.py`                    | 4줄           | 배치 조정, 주석                                |
| `sample.py`                   | 2줄           | 주석만                                         |
| `chat.py` ★📦                | 신규 57줄     | 대화형 REPL — **격리됨**                      |
| `config/finetune_agent.py` ★📦 | 신규 27줄   | Agent 파인튜닝 설정 — **격리됨**              |
| `data/agent/prepare.py` ★📦  | 신규 33줄     | Agent 데이터 토크나이징 — **격리됨**          |
| `data/agent/input.txt` ★📦   | 신규 17,001줄 | Agent 문서 데이터 — **격리됨**                |
| `.gitignore`                  | 33줄          | 체크포인트·venv 제외                          |
| `docs/` ★                    | 신규 6개      | 학습 자료 3개(노트북 1개 포함) + 실험 기록 3개 |
| `archive/` ★📦               | 신규 7개      | 격리 보관소 — 실행 경로에 관여하지 않음       |

`★` = 원본에 없는 신규 파일, `📦` = **격리됨** — 실행 경로에서 빼내 `archive/agent-experiment/`로 옮긴 파일.
격리된 파일은 저장소에 남아 있지만 **어떤 실행 경로도 참조하지 않습니다.**
따라서 실행에 관여하는 원본 대비 차이는 `train.py`·`model.py`·`bench.py`·`sample.py` 4개, 21줄이 전부입니다.

## 2.2 GPU 하드웨어에 맞춘 튜닝

| 위치                                          | 원본 nanoGPT                               | 이 저장소 (`laptop-wsl`)            | 기능 차이                                        |
| --------------------------------------------- | ------------------------------------------ | ------------------------------------- | ------------------------------------------------ |
| `train.py` 배치                             | `batch_size=12`, `grad_accum=5*8`      | `batch_size=8`, `grad_accum=5*12` | 8GB VRAM 대응. 실효 배치는 480으로**동일** |
| `bench.py` 배치                             | `batch_size=12`                          | `batch_size=8`                      | 8GB VRAM 대응                                    |
| `model.py` `estimate_mfu()`               | `flops_promised = 312e12` (A100)         | `126e12` (RTX 4060 Laptop bf16)     | **있음** — MFU 분모가 바뀜                |
| `train.py`/`sample.py`/`bench.py` dtype | bf16 지원 시 bf16, 아니면 fp16 (자동 감지) | 동일한 자동 감지 식                   | **없음** — 주석만 다름                    |

> **dtype에 대한 오해 주의**: 원본 nanoGPT는 이미 `torch.cuda.is_bf16_supported()`로 자동 감지합니다.
> `laptop-wsl` 브랜치는 이 원본 동작을 그대로 쓰며 주석만 하드웨어에 맞게 고쳤습니다.
> 반면 **`pc` 브랜치는 `dtype = 'float16'`으로 고정**했는데, 이는 RTX 2070(Turing, CC 7.5)이
> bfloat16을 하드웨어 지원하지 않기 때문입니다. 브랜치별로 이 값이 다르다는 점에 유의하세요.

> **MFU 수치 해석 주의**: 분모가 A100(312 TFLOPS)이 아니라 로컬 GPU 성능으로 바뀌었기 때문에,
> 여기서 찍히는 MFU % 는 원본 README의 수치와 직접 비교할 수 없습니다.
> 브랜치별 값: `laptop-wsl` = `126e12`, `pc` = `60e12`, 원본 = `312e12`.

## 2.3 학습 루프에 진행바 추가 (`train.py`)

원본은 iteration마다 `print(f"iter {iter_num}: loss ...")`로 한 줄씩 출력합니다. 이를 tqdm 진행바로 대체했습니다.

| 항목                   | 원본                        | 이 저장소                                                      |
| ---------------------- | --------------------------- | -------------------------------------------------------------- |
| 진행 출력              | `print(f"iter ...")`      | `pbar.set_postfix(loss=..., mfu=...)` (`log_interval`마다) |
| 진행바 전진            | 해당 없음                   | `pbar.update(1)` (**매 iteration**)                    |
| eval / 체크포인트 로그 | `print(...)`              | `tqdm.write(...)` (진행바를 깨지 않음)                       |
| 종료 조건              | `if iter_num > max_iters` | `if iter_num > max_iters` (**원본과 동일**)                |

- `from tqdm import tqdm` 의존성이 추가되었으나 원본 README의 설치 목록에는 없습니다 → `pip install tqdm` 별도 필요
- 종료 조건은 원본 그대로입니다 → 총 iteration은 원본과 같은 `max_iters + 1`회.
  (한때 `>=`로 바꿔 1회 적게 돌던 시기가 있었으나 원본에 맞춰 되돌렸습니다.
  이에 맞춰 진행바 총량도 `tqdm(total=max_iters + 1)`로 잡습니다.)

## 2.4 AI Agent 파인튜닝 실험 추가 📦 격리됨

원본에는 없는 "GPT-2를 특정 도메인 문서로 파인튜닝하고 대화해보는" 실험 세트입니다.

> **📦 이 절의 파일은 실행 경로에서 격리되었습니다.**
> 원본 대비 차이를 "로컬 환경 대응 + 실행성 수정"으로만 한정하기 위해
> [`archive/agent-experiment/repo/`](../../archive/agent-experiment/) 아래로 옮겼습니다
> (삭제가 아니라 격리 — 저장소에는 그대로 남아 있고, 다만 어떤 실행 경로도
> 이들을 참조하지 않습니다).
> 보관 경위·복원 절차·미완 사항은
> [`archive/agent-experiment/README.md`](../../archive/agent-experiment/README.md) 참조.
> 아래 설명은 **격리 시점의 상태 기록**으로 남겨 둡니다.

```
data/agent/prepare.py  →  config/finetune_agent.py  →  chat.py
 (문서 → 토큰)              (GPT-2 124M 파인튜닝)        (대화형 추론)
```

- 데이터: AI 에이전트 관련 영문 문서 약 17,000줄
- 설정: `init_from='gpt2'`, `learning_rate=3e-5`, `decay_lr=False`, `max_iters=500`, `batch_size=4`, `grad_accum=8`
- `chat.py`: 체크포인트를 로드해 `input()` 루프로 대화. `_orig_mod.` 접두사(torch.compile 흔적)를 제거하는 처리가 들어 있음
- 결과 기록: 둘 다 함께 격리되어 `archive/agent-experiment/repo/docs/test/` 아래에 있습니다 —
  [`04_gpt2_finetuning_experiment.md`](../../archive/agent-experiment/repo/docs/test/04_gpt2_finetuning_experiment.md),
  [`03_chat_interaction_test.md`](../../archive/agent-experiment/repo/docs/test/03_chat_interaction_test.md)

## 2.5 기타 변경과 알려진 버그

- `.gitignore` 확장: `.venv/`, `out-*/`, `*.pt`, `*.bin`, `*.pkl` 등 추가
  (`data/shakespeare_char/input.txt`는 예외로 추적 — 원본은 이를 무시하고 `prepare.py`가
  매번 내려받지만, `prepare.py`에 존재 여부 가드가 있어 **실행 결과는 원본과 동일**합니다)
- 전체 파일이 CRLF 줄바꿈으로 변환됨 → `git diff`에서 README·노트북·LICENSE 등이 대량 변경된 것처럼
  보이지만 실제 내용 차이는 없습니다. 비교할 때는 반드시 `--ignore-cr-at-eol`을 붙이세요
  (붙이지 않으면 41개 파일 21,114줄, 붙이면 22개 파일 19,281줄)

**알려진 버그 📦 — `data/agent/prepare.py`가 셰익스피어 스크립트 복사본입니다.**
(해당 파일은 2.4와 함께 `archive/agent-experiment/repo/data/agent/prepare.py`로 격리되어
실행 경로에는 없습니다. 실험을 재개할 때 먼저 고쳐야 하므로 기록을 남깁니다.)
이 저장소의 수정 과정에서 생긴 문제로, 원본 nanoGPT에는 해당하지 않습니다.
`data/shakespeare/prepare.py`를 복사해 만든 탓에 다운로드 URL이 아직 tinyshakespeare를 가리킵니다.
`input.txt`가 이미 있으면 다운로드를 건너뛰므로 현재는 정상 동작하지만, `input.txt`가 없는 상태에서
실행하면 엉뚱하게 셰익스피어를 받아옵니다. 파일 하단의 토큰 수 주석(301,966 / 36,059)도
셰익스피어 기준이라 실제 값과 다릅니다.

## 2.6 차이 목록 갱신하기

원본 대비 차이가 바뀌면 아래로 다시 뽑아 이 장을 갱신합니다.

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

# 3. GPU와 CUDA 그리고 텐서

nanoGPT를 실행하다 보면 `cuda`, `tensor`, `device='cuda:0'` 같은 것들이 계속 나옵니다.
코드를 읽기 전에 알아두면 좋은 최소한의 개념을 정리합니다.

## 3.1 GPU와 CUDA의 관계

**GPU는 하드웨어, CUDA는 그 하드웨어를 계산에 쓰게 해주는 소프트웨어 층**입니다.

```
당신의 코드          train.py
      ↓
PyTorch             torch 2.5.1
      ↓
라이브러리          cuDNN 9.1, cuBLAS 12.4   ← 딥러닝 연산 최적화
      ↓
CUDA 런타임         CUDA 12.4                ← GPU에 명령을 보내는 규약
      ↓
드라이버            552.27                   ← OS와 GPU 통신
      ↓
GPU 하드웨어        RTX 4060 Laptop          ← 실제 계산하는 칩
```

### 왜 CUDA가 필요한가

GPU는 원래 그래픽 전용 장치였습니다. 화면에 삼각형을 그리고 픽셀에 색을 칠하는 용도였습니다.

그런데 "픽셀 수백만 개를 동시에 처리하는 능력이면 숫자 수백만 개도 동시에 처리할 수 있지 않나"라는
발상이 나왔습니다. 문제는 GPU에 "행렬을 곱해라"라고 시킬 수단이 없었다는 점입니다. 초기 연구자들은
계산을 그래픽 연산으로 위장했습니다 — 데이터를 텍스처 이미지로 바꾸고 결과를 픽셀 색상으로 읽는 식입니다.

2007년 NVIDIA가 CUDA를 발표하면서 **그래픽이 아닌 일반 계산을 GPU에 직접 시키는 통로**가 열렸고,
이것이 딥러닝이 폭발한 결정적 계기가 됐습니다.

### 이 저장소에서의 실제 구성

CUDA 툴킷을 따로 설치하지 않았습니다. `uv pip install torch`가 전부 끌고 왔습니다.

```
nvidia-cublas-cu12    12.4.5.8    행렬 곱 라이브러리
nvidia-cudnn-cu12     9.1.0.70    딥러닝 전용 연산
nvidia-cuda-runtime   12.4.127    CUDA 런타임
```

약 2.9 GiB 다운로드의 대부분이 이것들입니다. PyTorch 휠이 CUDA를 통째로 품고 있어 별도 설치가 불필요합니다.

**WSL의 특이점**: Linux 안에 NVIDIA 드라이버를 설치한 적이 없는데도 `nvidia-smi`가 동작합니다.
`/usr/lib/wsl/lib/`에 Windows 드라이버로 연결되는 통로가 마련돼 있어, Linux 쪽 CUDA 호출이
그 경로를 타고 Windows 드라이버로 전달됩니다. 실제 계산은 같은 물리 GPU에서 일어납니다.

### 헷갈리기 쉬운 것

- **"CUDA 코어"**는 하드웨어 연산 유닛의 이름이기도 합니다 (RTX 4060 Laptop에 3,072개).
  소프트웨어 CUDA와 이름만 같습니다
- **CUDA는 NVIDIA 전용**입니다. AMD는 ROCm, Apple Silicon은 Metal(MPS)을 씁니다.
  `train.py`의 `device` 선택지가 `'cuda'` / `'mps'` / `'cpu'`인 이유입니다

## 3.2 텐서 (Tensor)

**숫자를 담는 다차원 배열**입니다.

| 차원       | 이름   | 예시                   |
| ---------- | ------ | ---------------------- |
| 0차원      | 스칼라 | `5`                  |
| 1차원      | 벡터   | `[1, 2, 3]`          |
| 2차원      | 행렬   | `[[1,2], [3,4]]`     |
| 3차원 이상 | 텐서   | 행렬을 여러 장 쌓은 것 |

파이썬 리스트와 비슷해 보이지만 두 가지가 결정적으로 다릅니다.

- **GPU에서 돌아갑니다.** 수백만 개 숫자에 같은 연산을 동시에 적용할 수 있습니다
- **미분이 자동으로 됩니다.** 학습은 "오차를 줄이는 방향"을 계산하는 일인데,
  텐서는 자신이 거쳐온 연산 경로를 기억해 그 방향을 역으로 계산해냅니다 (autograd)

### 출력 읽는 법

```
tensor([2., 2., 2., 2.], device='cuda:0')
```

| 부분                 | 뜻                                                            |
| -------------------- | ------------------------------------------------------------- |
| `tensor(...)`      | 이것은 텐서다                                                 |
| `[2., 2., 2., 2.]` | 값 4개                                                        |
| `2.` (점에 주의)   | **실수**라는 표시. `2.0`과 같으며 정수 `2`와 구분됨 |
| `device='cuda:0'`  | 이 데이터가 0번 GPU 메모리에 있다                             |

`device=`는 CPU에 있을 때 생략됩니다(기본값이라서). 즉 이 표시가 찍혔다는 것 자체가
**GPU까지 제대로 갔다는 증거**입니다.

### nanoGPT에서의 텐서

학습 중 오가는 모든 데이터가 텐서입니다.

```python
X, Y = get_batch('train')
```

`X`는 `(64, 256)` 모양의 2차원 텐서입니다.

- `64` = 한 번에 처리하는 문장 개수 (`batch_size`)
- `256` = 문장 하나의 길이 (`block_size`)

문자 16,384개가 정수 ID로 변환돼 GPU에 올라가 있는 상태이며, 모델은 여기에 행렬 곱을 반복해
"다음 글자는 무엇인가"를 예측합니다.

## 3.3 device — `cuda`와 `cuda:0`

PyTorch는 GPU를 **0번부터** 셉니다. GPU가 하나면 `cuda:0` 하나뿐이고, 넷이면 `cuda:0` ~ `cuda:3`입니다.

```
GPU 1개 (이 랩톱)        GPU 4개 (서버)
┌─────────┐              ┌────┬────┬────┬────┐
│ cuda:0  │              │ :0 │ :1 │ :2 │ :3 │
└─────────┘              └────┴────┴────┴────┘
```

GPU가 하나여도 번호를 떼지 않는 이유는, 코드가 몇 대짜리 환경에서든 동일하게 동작해야 하기 때문입니다.

| 표기         | 뜻                                          |
| ------------ | ------------------------------------------- |
| `'cuda'`   | 기본 GPU에 배치 → 이 랩톱에선 자동으로 0번 |
| `'cuda:0'` | 0번 GPU에 명시적으로 배치                   |

GPU가 하나인 환경에서는 둘의 결과가 같습니다. `train.py`는 `'cuda'`를 쓰지만 출력은 `cuda:0`으로
나오는데, PyTorch가 실제 배치 위치를 구체적으로 알려주기 때문입니다.

### 여러 대를 쓰는 경우

nanoGPT에도 다중 GPU 코드가 있습니다 (`train.py:87-90`).

```python
ddp_local_rank = int(os.environ['LOCAL_RANK'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
```

`torchrun --nproc_per_node=4`로 실행하면 프로세스 4개가 뜨고 각자 `LOCAL_RANK`를 0~3으로 받아
서로 다른 GPU를 잡습니다(DDP, Distributed Data Parallel). 단일 GPU 환경에서는 이 블록이
실행되지 않습니다.

확인:

```bash
python -c "import torch; print(torch.cuda.device_count())"   # → 1
```

> 이 랩톱에는 i7-14650HX의 내장 그래픽도 있지만, PyTorch의 CUDA는 NVIDIA GPU만 세므로
> 목록에 잡히지 않고 연산에도 쓰이지 않습니다.

## 3.4 dtype — 숫자를 몇 비트로 담을 것인가

같은 숫자라도 정밀도를 낮추면 메모리가 줄고 계산이 빨라집니다. 딥러닝은 약간의 정밀도 손실을
감수하고 속도를 택하는 것이 일반적입니다.

| dtype        | 비트 | 표현 범위               | 정밀도 | 하드웨어 요구       |
| ------------ | ---- | ----------------------- | ------ | ------------------- |
| `float32`  | 32   | 넓음                    | 높음   | 모든 GPU            |
| `bfloat16` | 16   | float32와**동일** | 낮음   | Ampere(CC 8.0) 이상 |
| `float16`  | 16   | 좁음                    | 중간   | 대부분의 GPU        |

**bfloat16이 권장되는 이유**는 지수부 비트를 float32와 똑같이 유지하기 때문입니다. 표현 범위가
같으므로 값이 넘치거나 0으로 사라지는 문제가 거의 없습니다. 반면 float16은 범위가 좁아
학습 중 발산하기 쉬워서, 손실값을 인위적으로 키웠다 되돌리는 **GradScaler**라는 보정 장치가 필요합니다.

`train.py`는 하드웨어를 보고 자동으로 고릅니다.

```python
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
```

RTX 4060 Laptop은 Ada(CC 8.9)라 bfloat16이 선택되고, GradScaler는 자동으로 비활성화됩니다.
RTX 2070(Turing, CC 7.5)에서는 bfloat16을 하드웨어 지원하지 않아 float16 + GradScaler 경로를 탑니다.

## 3.5 torch.compile이 하는 일

`--compile=True`를 붙이면 파이썬 코드를 그대로 실행하지 않습니다. 연산 그래프를 분석해
**더 빠른 코드를 새로 생성**하고, 그것을 그 자리에서 컴파일해 실행합니다.

```
model(X, Y)  →  [Dynamo: 그래프 추출]  →  [Inductor: 코드 생성]
                                              ↓
                                      C++ 코드 + Triton GPU 커널
                                              ↓
                                      gcc / g++ 로 컴파일       ← 빌드 도구가 필요한 지점
                                              ↓
                                      최적화된 실행 코드
```

컴파일이 이득인 이유는 주로 **커널 융합**입니다. `a + b`, `* c`, `relu` 같은 연산을 따로 실행하면
매번 GPU 메모리를 오가야 하지만, 융합하면 한 번의 커널 실행으로 끝납니다. 보통 1.3~2배 빨라집니다.

대신 첫 실행에 컴파일 시간(1~3분)이 들고, 빌드 도구가 없으면 아예 실패합니다. 이 저장소에서
`build-essential`과 `python3-dev`가 필요했던 이유가 이것입니다
([10.2 빌드 툴체인 설치](#102-빌드-툴체인-설치-torchcompile-필수) 참조).

가장 작은 확인 방법:

```bash
python -c "import torch; f=torch.compile(lambda x: x*2); print(f(torch.ones(4,device='cuda')))"
```

`torch.ones(4, device='cuda')`로 GPU에 `[1,1,1,1]`을 만들고, `lambda x: x*2`(입력에 2를 곱하는
이름 없는 함수)를 컴파일해 적용합니다. `tensor([2., 2., 2., 2.], device='cuda:0')`가 나오면
코드 생성 → 컴파일 → GPU 실행 경로가 전부 정상이라는 뜻입니다.

> 학습 루프에서의 `torch.compile` 모드 선택과 ON/OFF 기준은 [7.10](#710-pytorch-20-최적화-torchcompile)에 있습니다.

---

# 4. 트랜스포머 아키텍처

## 4.1 GPT의 전체 구조

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

## 4.2 1단계: 임베딩 (Embedding)

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

## 4.3 2단계: 트랜스포머 블록 (Transformer Block)

GPT-2 기준으로 이 블록을 12번 반복합니다. 각 블록의 구조:

```
x → LayerNorm → CausalSelfAttention → x + (결과)
              → LayerNorm → MLP → x + (결과)
```

**Pre-norm 구조**: LayerNorm을 앞에 적용합니다 (학습 안정성 향상).

## 4.4 3단계: 인과적 자기 어텐션 (Causal Self-Attention)

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

> B·T·C 차원이 어떻게 변형되는지 직접 손으로 검산해보려면
> [`qkv_dimension_workbook.ipynb`](./qkv_dimension_workbook.ipynb)를 여세요.

## 4.5 4단계: MLP (Feed-Forward Network)

어텐션 후 각 토큰을 독립적으로 처리합니다.

```
n_embd (768) → 4×n_embd (3072) → n_embd (768)
     Linear        GELU          Linear
```

**GELU 활성화 함수**: ReLU보다 부드러운 비선형성, GPT에서 표준적으로 사용됩니다.

## 4.6 5단계: 출력과 손실 계산

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

## 4.7 핵심 아이디어 요약

| 개념                 | 역할              | 비유               |
| -------------------- | ----------------- | ------------------ |
| Token Embedding      | 토큰 → 벡터      | 단어를 좌표로 표현 |
| Positional Embedding | 위치 정보 추가    | 문장 내 순서 표시  |
| Self-Attention       | 토큰 간 관계 계산 | 단어 간 참조 관계  |
| Causal Mask          | 미래 토큰 차단    | 예언 금지          |
| MLP                  | 개별 토큰 변환    | 토큰별 특징 추출   |
| LayerNorm            | 값 정규화         | 학습 안정화        |
| Residual Connection  | 이전 값 더하기    | 지름길 연결        |

---

# 5. 모델 구현 분석 (model.py)

`model.py`는 약 330줄로 GPT 전체를 구현합니다. 클래스별로 분석합니다.

## 5.1 전체 클래스 구조

```
GPTConfig        ← 하이퍼파라미터 설정값 (dataclass)
LayerNorm        ← 정규화 레이어
CausalSelfAttention  ← 핵심 어텐션 메커니즘
MLP              ← Feed-Forward 네트워크
Block            ← 트랜스포머 블록 (Attention + MLP)
GPT              ← 최상위 모델 클래스
```

## 5.2 GPTConfig (model.py:108)

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

## 5.3 LayerNorm (model.py:18)

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

## 5.4 CausalSelfAttention (model.py:29)

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

## 5.5 MLP (model.py:78)

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

## 5.6 Block (model.py:94)

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Pre-norm + Attention + Residual
        x = x + self.mlp(self.ln_2(x))   # Pre-norm + MLP + Residual
        return x
```

**Residual Connection (잔차 연결)**: `x + f(x)` 형태로 입력을 출력에 더합니다.
→ 기울기 소실(vanishing gradient) 문제를 방지하고, 매우 깊은 네트워크 학습을 가능하게 합니다.

## 5.7 GPT (model.py:118)

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

## 5.8 파라미터 수 계산

GPT-2 (124M) 기준:

| 컴포넌트       | 계산                  | 파라미터 수     |
| -------------- | --------------------- | --------------- |
| 토큰 임베딩    | 50304 × 768          | 38.6M           |
| 위치 임베딩    | 1024 × 768           | 0.79M           |
| 어텐션 (×12)  | 4 × 768² × 12      | 28.3M           |
| MLP (×12)     | 2 × 4 × 768² × 12 | 56.6M           |
| LayerNorm      | 768 × 2 × 13        | 0.02M           |
| **합계** |                       | **~124M** |

> `from_pretrained()` 메서드로 OpenAI가 공개한 실제 GPT-2 가중치를 불러올 수 있습니다.

---

# 6. 데이터 준비

텍스트 데이터를 모델이 학습할 수 있는 숫자 배열로 변환하는 과정을 다룹니다.

## 6.1 토크나이제이션이란?

텍스트를 정수 ID의 시퀀스로 변환하는 과정입니다.

```
"Hello, world!" → [15496, 11, 995, 0]  (GPT-2 BPE 토크나이저)
"Hello, world!" → [20, 5, 12, 12, 15, 2, 0, 23, 15, 18, 12, 4, 1]  (문자 수준)
```

컴퓨터(GPU)는 글자 'A'나 'B'를 직접 계산할 수 없습니다. 따라서 모든 글자를 고유한 **숫자(ID)**로
매핑해야 합니다. 문자 수준에서 번호를 매기는 원칙은 다음 세 단계입니다.

1. **고유 문자 추출**: 전체 텍스트(`input.txt`)를 모두 읽어 그 안에 들어있는 모든 종류의 글자를
   중복 없이 추출합니다. (셰익스피어 데이터의 경우 총 65종류)
2. **정렬(Sorting)**: 추출된 글자들을 일정한 순서(보통 아스키 코드 순서)로 나열합니다.
3. **인덱싱(Indexing)**: 나열된 순서대로 0번부터 번호를 부여합니다. 이것이 모델의 **사전(Vocabulary)**이 됩니다.
   - 예: `\n`(0번), (1번), `F`(18번), `i`(47번) ...

## 6.2 세 가지 데이터셋

| 데이터셋             | 토크나이저  | 크기  | 학습 시간      | 목적            |
| -------------------- | ----------- | ----- | -------------- | --------------- |
| `shakespeare_char` | 문자 수준   | 1MB   | ~3분 (A100)    | 빠른 실험, 입문 |
| `shakespeare`      | BPE (GPT-2) | 1MB   | 단시간         | BPE 파인튜닝    |
| `openwebtext`      | BPE (GPT-2) | ~54GB | ~4일 (8×A100) | GPT-2 재현      |

### Shakespeare Char (문자 수준)

**경로**: `data/shakespeare_char/prepare.py`

```python
# 1. 데이터 다운로드
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

# 2. 어휘 집합 생성
chars = sorted(list(set(data)))  # 65개 고유 문자
vocab_size = len(chars)  # 65

# 3. 문자 ↔ 정수 매핑
stoi = {ch: i for i, ch in enumerate(chars)}  # 문자 → 정수
itos = {i: ch for i, ch in enumerate(chars)}  # 정수 → 문자

# 4. 인코딩
encode = lambda s: [stoi[c] for c in s]

# 5. 학습/검증 분할 (90% / 10%)
n = int(0.9 * len(data))
train_ids = encode(data[:n])
val_ids = encode(data[n:])

# 6. uint16 numpy 배열로 저장
np.array(train_ids, dtype=np.uint16).tofile('train.bin')
np.array(val_ids, dtype=np.uint16).tofile('val.bin')

# 7. 메타데이터 저장
meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
pickle.dump(meta, open('meta.pkl', 'wb'))
```

**출력물**:

- `train.bin`: ~1M 토큰 (uint16, 2바이트씩)
- `val.bin`: ~111K 토큰
- `meta.pkl`: 문자 매핑 정보

**어휘 예시 (65개)**:

```
\n ! $ & ' , - . 3 : ; ? A B C ... Z a b c ... z
```

### Shakespeare BPE

**경로**: `data/shakespeare/prepare.py`

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
```

**차이점**: 같은 셰익스피어 텍스트지만 BPE로 토크나이징하므로 토큰 수가 더 적습니다.

|           | 문자 수준 | BPE   |
| --------- | --------- | ----- |
| 학습 토큰 | ~1M       | ~302K |
| 검증 토큰 | ~111K     | ~36K  |
| 어휘 크기 | 65        | 50257 |

BPE는 자주 나오는 글자 조합을 하나의 토큰으로 묶어 더 압축합니다.

### OpenWebText (대규모)

**경로**: `data/openwebtext/prepare.py`

Reddit에서 추천받은 웹 페이지를 크롤링한 데이터셋입니다.

```python
from datasets import load_dataset

# 8M+ 문서 다운로드 (~54GB)
dataset = load_dataset("openwebtext", num_proc=8)

# 학습/검증 분할
split_dataset = dataset["train"].train_test_split(
    test_size=0.0005, seed=2357, shuffle=True
)
# → train: 8M 문서, val: 4007 문서

# 병렬 토크나이징
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)  # 문서 종료 토큰 추가
    return {'ids': ids, 'len': len(ids)}

tokenized = split_dataset.map(
    process,
    num_proc=num_proc,  # 멀티프로세싱
)

# 하나의 큰 바이너리 파일로 저장
arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
```

**출력물**:

- `train.bin`: ~9B 토큰 (~17GB)
- `val.bin`: ~4M 토큰 (~8.5MB)

## 6.3 저장 형식: `.bin` 파일

토큰을 `uint16` (2바이트 부호 없는 정수)로 저장합니다.

```
왜 uint16?
- GPT-2 vocab_size = 50257
- uint16 최대값 = 65535
- 50257 < 65535 → uint16으로 충분
- uint32보다 파일 크기 절반
```

원본 텍스트를 그대로 학습에 쓰지 않고 `.bin`으로 변환하는 이유는 두 가지입니다.

- **속도**: 텍스트 파일은 읽는 속도가 느리지만, 숫자로 변환된 이진 데이터는 GPU 메모리로
  직접 로드하기에 훨씬 빠르고 효율적입니다.
- **일관성**: 모든 문자가 동일한 크기(2바이트)의 숫자로 저장되어 있어, 모델이 랜덤하게
  데이터 덩어리(Batch)를 집어오기가 매우 수월합니다.

파일 구성은 `train.bin`(전체의 90%, 실제 학습용)과 `val.bin`(나머지 10%, 학습하지 않은 문제를
얼마나 잘 맞히는지 검증하는 용도)입니다.

**메모리맵 방식 (`np.memmap`)**:

```python
data = np.memmap('train.bin', dtype=np.uint16, mode='r')
```

파일 전체를 RAM에 올리지 않고, 필요한 부분만 디스크에서 읽습니다. 17GB 파일도 8GB RAM에서 처리할 수 있습니다.

## 6.4 데이터 배치 구조

`get_batch()`가 반환하는 텐서 구조:

```
batch_size=4, block_size=8 일 때:

x (입력):
[[ 1, 23, 45, 67, 89, 12, 34, 56],
 [78, 90, 11, 22, 33, 44, 55, 66],
 [99, 88, 77, 66, 55, 44, 33, 22],
 [11, 22, 33, 44, 55, 66, 77, 88]]

y (타깃, x를 1칸 오른쪽으로 이동):
[[23, 45, 67, 89, 12, 34, 56, 91],  ← x[0]에서 1칸 뒤
 [90, 11, 22, 33, 44, 55, 66, 77],
 [88, 77, 66, 55, 44, 33, 22, 11],
 [22, 33, 44, 55, 66, 77, 88, 99]]

모델은 x의 각 위치에서 y의 같은 위치를 맞춰야 합니다.
```

## 6.5 원문 한 줄이 학습 문제로 바뀌는 과정

`First Citizen:` 이라는 문장을 모델이 어떻게 학습하는지 단계별로 살펴봅니다.

### 1단계: 숫자 데이터 스트림 (Input tokens)

`prepare.py`에 의해 위 문장은 아래와 같은 숫자 열로 변환되어 `train.bin`에 들어있습니다.

```
[18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10]
```

### 2단계: 퀴즈 생성 (Sliding Window)

모델은 이 숫자 열을 가지고 수천 개의 퀴즈를 스스로 만듭니다.

- **퀴즈 1**: `[18]` (F)를 보여주고, 정답은 `47` (i)이라고 알려줌.
- **퀴즈 2**: `[18, 47]` (Fi)를 보여주고, 정답은 `56` (r)이라고 알려줌.
- **퀴즈 3**: `[18, 47, 56]` (Fir)를 보여주고, 정답은 `57` (s)이라고 알려줌.
- **퀴즈 4**: `[18, 47, 56, 57, 58, 1]` (First )를 보여주고, 정답은 `15` (C)라고 알려줌.

### 3단계: 사전학습의 핵심 (Learning)

모델은 위 퀴즈를 풀면서 다음과 같은 통계적 확률을 파라미터에 저장합니다.

> *"데이터를 보니 `F`, `i`, `r`, `s`, `t` 다음에는 공백(`1`)이 올 확률이 압도적으로 높고,
> 그 공백 다음에는 대문자 `C`가 나올 가능성이 크구나!"*

이런 과정을 5,000번 반복하면서 모델은 단어의 철자뿐만 아니라, 인물 뒤에 콜론(`:`)이 붙는
연극 대본의 형식까지 완벽하게 체득하게 됩니다. 이것이 바로 **사전학습(Pre-training)**의 본질입니다.

정답 라벨을 사람이 붙일 필요가 없다는 점이 핵심입니다 — **텍스트 자체가 정답지**이며,
이를 자기지도학습(self-supervised)이라 부릅니다.

## 6.6 BPE (Byte Pair Encoding) 이란?

GPT-2가 사용하는 토크나이저입니다.

### 기본 아이디어

자주 등장하는 바이트 쌍을 반복적으로 새 토큰으로 합칩니다.

```
1. 문자 수준 시작: "hello" → h e l l o
2. 자주 나오는 쌍 합치기:
   - "l" + "l" → "ll"   : "hello" → h e ll o
   - "e" + "ll" → "ell" : "hello" → h ell o
   ...
3. 최종 어휘 크기가 될 때까지 반복
```

### 장점

- 모르는 단어(OOV)가 없음: 어떤 텍스트도 기존 토큰으로 분해 가능
- 효율적: 문자 수준보다 시퀀스 길이가 짧음
- 언어에 독립적

### tiktoken 사용법

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

# 인코딩
ids = enc.encode("Hello, world!")  # [15496, 11, 995, 0]

# 디코딩
text = enc.decode([15496, 11, 995, 0])  # "Hello, world!"
```

> BPE가 왜 "Byte" Pair인지, 같은 문장을 문자 단위와 BPE로 쪼개면 얼마나 차이가 나는지는
> [`QNA.md` 부록 B](./QNA.md#부록-b-문자-단위-vs-bpe--같은-문장을-쪼개보면)에 실측과 함께 있습니다.

## 6.7 직접 실행하기

```bash
# 셰익스피어 (문자) - 가장 빠름
cd data/shakespeare_char
python prepare.py

# 셰익스피어 (BPE)
cd data/shakespeare
python prepare.py

# OpenWebText - 수 시간 소요, 대용량 저장공간 필요
cd data/openwebtext
python prepare.py
```

---

# 7. 학습 파이프라인 (train.py)

`train.py`는 데이터 로딩부터 체크포인트 저장까지 전체 학습 과정을 담당합니다.

## 7.1 전체 흐름

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

## 7.2 설정 시스템

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

## 7.3 데이터 로딩

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

## 7.4 모델 초기화 세 가지 방식

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

## 7.5 그래디언트 누적 (Gradient Accumulation)

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

## 7.6 혼합 정밀도 학습 (Mixed Precision)

```python
# dtype 선택: bfloat16 (Ampere GPU), float16 (이전 GPU), float32 (CPU)
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'

# 자동 형변환 컨텍스트
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# float16 용 그래디언트 스케일러
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

| dtype    | 메모리      | 정밀도 | 사용 상황          |
| -------- | ----------- | ------ | ------------------ |
| float32  | 기준(1×)   | 높음   | CPU, 구형 GPU      |
| bfloat16 | 절반(0.5×) | 충분   | Ampere 이상 (권장) |
| float16  | 절반(0.5×) | 제한적 | 구형 GPU           |

**bfloat16이 권장되는 이유**: float16보다 범위가 넓어 수치 불안정 문제가 적습니다.
자세한 배경은 [3.4 dtype](#34-dtype--숫자를-몇-비트로-담을-것인가) 참조.

## 7.7 학습률 스케줄

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

## 7.8 그래디언트 클리핑

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # grad_clip=1.0
```

그래디언트의 L2 놈이 1.0을 초과하면 비율적으로 줄입니다. 폭주하는 그래디언트를 방지합니다.

## 7.9 검증과 체크포인트 저장

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

### WandB 로깅 (선택사항)

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

### 손실값 해석

| 모델              | 학습 데이터 | 목표 검증 손실 |
| ----------------- | ----------- | -------------- |
| 셰익스피어 (문자) | Shakespeare | ~1.47          |
| GPT-2 재현        | OpenWebText | ~2.85          |

손실이 낮을수록 더 나은 언어 모델입니다. 단, 학습 손실과 검증 손실 차이가 크면 과적합입니다.

## 7.10 PyTorch 2.0 최적화 (torch.compile)

`torch.compile`은 파이썬 코드를 하드웨어에 최적화된 커널로 변환하여 학습 속도를 10~20% 향상시킵니다.
내부 동작은 [3.5](#35-torchcompile이-하는-일) 참조.

### 코드 적용 위치

```python
model = GPT(gptconf)
model.to(device)

if compile:
    print("Compiling model...")
    model = torch.compile(model) # 학습 루프 진입 전 한 번만 실행
```

### 주요 모드 (Modes) 및 옵션

`torch.compile(model, mode='...')`를 통해 최적화 전략을 선택할 수 있습니다.

| 모드                          | 특징                                     | 권장 상황                             |
| :---------------------------- | :--------------------------------------- | :------------------------------------ |
| **`default`**         | 최적화 성능과 컴파일 시간의 균형         | 일반적인 학습 시 (기본값)             |
| **`reduce-overhead`** | 파이썬 실행 오버헤드를 대폭 감소         | 작은 모델 또는 추론 위주 작업         |
| **`max-autotune`**    | 최고의 성능을 위해 최적의 커널 자동 선택 | RTX 30/40 시리즈 이상, 장시간 학습 시 |

### ON/OFF 선택 가이드

- **`True`로 설정할 때**:
  - 본격적인 학습(Training)을 시작할 때 (반복 횟수가 많을수록 유리)
  - RTX 20/30/40 시리즈 등 최신 텐서 코어가 탑재된 GPU 사용 시
- **`False`로 설정할 때**:
  - 짧은 코드 테스트나 디버깅 시 (컴파일 대기 시간이 더 아까울 때)
  - VRAM이 극도로 부족할 때 (컴파일 과정에서 추가 메모리가 소요될 수 있음)
  - `sample.py` 등 짧은 추론 작업 시

> 실제 하드웨어 최적화 리포트: [`../test/01_gpu_optimization.md`](../test/01_gpu_optimization.md)

---

# 8. 추론과 텍스트 생성 (sample.py)

학습이 완료된 모델(또는 사전학습된 GPT-2)로 텍스트를 생성하는 방법을 다룹니다.

## 8.1 자동회귀 생성 (Autoregressive Generation)

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

## 8.2 sample.py 실행 방법

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

## 8.3 주요 파라미터

| 파라미터           | 기본값         | 설명                                               |
| ------------------ | -------------- | -------------------------------------------------- |
| `init_from`      | `'resume'`   | `'resume'` 또는 `'gpt2'`, `'gpt2-medium'` 등 |
| `out_dir`        | `'out'`      | 체크포인트 폴더 경로                               |
| `start`          | `'\n'`       | 시작 프롬프트 텍스트                               |
| `num_samples`    | 10             | 생성할 샘플 수                                     |
| `max_new_tokens` | 500            | 최대 생성 토큰 수                                  |
| `temperature`    | 0.8            | 무작위성 조절 (0.0~2.0)                            |
| `top_k`          | 200            | 상위 k개 토큰만 고려                               |
| `device`         | `'cuda'`     | 사용할 디바이스                                    |
| `dtype`          | `'bfloat16'` | 연산 정밀도                                        |

## 8.4 샘플링 전략

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

| 목적           | Temperature | Top-K |
| -------------- | ----------- | ----- |
| 일관된 텍스트  | 0.5         | 50    |
| 균형 잡힌 생성 | 0.8         | 200   |
| 창의적 텍스트  | 1.2         | 500   |

## 8.5 내부 동작: generate() 함수

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

## 8.6 컨텍스트 창 제한

GPT-2는 최대 1024 토큰을 볼 수 있습니다. 생성이 길어지면 가장 오래된 토큰을 버립니다.

```
[토큰1, 토큰2, ..., 토큰1024] → 새 토큰 생성
[토큰2, 토큰3, ..., 토큰1024, 새토큰] → 새 토큰 생성
[토큰3, ...]
```

이로 인해 매우 긴 텍스트를 생성할 때 앞 내용을 "기억"하지 못할 수 있습니다.

## 8.7 토크나이저 처리

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

## 8.8 생성 품질 향상 팁

1. **프롬프트를 구체적으로**: 짧은 프롬프트보다 문맥이 있는 프롬프트가 일관성 있는 결과를 냅니다.
2. **temperature와 top_k 조정**:

   - 반복되는 텍스트가 나오면 → temperature 올리기
   - 말이 안 되는 텍스트가 나오면 → temperature 낮추기
3. **더 큰 모델 사용**: gpt2-xl이 gpt2보다 훨씬 자연스러운 텍스트를 생성합니다.
4. **파인튜닝**: 특정 도메인의 텍스트를 원한다면 해당 도메인 데이터로 파인튜닝합니다.

---

# 9. 실습 가이드

처음부터 끝까지 직접 실행해보는 단계별 가이드입니다.

## 9.1 환경 설정

### 필수 패키지 설치

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

| 패키지       | 용도                       |
| ------------ | -------------------------- |
| torch        | 딥러닝 프레임워크          |
| numpy        | 배열 처리                  |
| transformers | GPT-2 사전학습 가중치 로드 |
| datasets     | OpenWebText 다운로드       |
| tiktoken     | BPE 토크나이저             |
| wandb        | 학습 모니터링 (선택)       |
| tqdm         | 진행 바                    |

### PyTorch 버전 확인

```python
import torch
print(torch.__version__)         # 2.0 이상 권장
print(torch.cuda.is_available()) # True이어야 GPU 학습 가능
print(torch.cuda.get_device_name(0))
```

> WSL에서 uv로 환경을 구축하는 절차는 [10장](#10-랩톱-wsl-환경-구축과-명령어-레퍼런스)에 있습니다.

## 9.2 실습 1: 셰익스피어 문자 수준 모델 (입문)

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

## 9.3 실습 2: 사전학습된 GPT-2 평가

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

## 9.4 실습 3: 하이퍼파라미터 실험

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

## 9.5 실습 4: 나만의 텍스트로 학습

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

## 9.6 학습 모니터링

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

> 실제 tqdm 진행바 출력과 시작 로그를 한 줄씩 읽는 법은 [10.5](#105-학습-로그-읽는-법)에 있습니다.

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

## 9.7 GPU 메모리 부족 시 해결책

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

현재 RTX 4060 Laptop 최적 설정은 [`../test/01_gpu_optimization.md`](../test/01_gpu_optimization.md)를 참고하세요.

## 9.8 벤치마크 실행

```bash
# 현재 GPU에서 처리량 측정 (난수 배치 — 데이터 준비 불필요)
python bench.py --real_data=False

# 예상 출력:
# iter 0: loss 4.2416, time 89.87ms
# iter 1: loss 4.2010, time 75.23ms
# ...
# time per iteration: 76.12ms, MFU: 8.45%
```

> `bench.py`의 기본값은 `real_data=True`이고 그 경로는 `data/openwebtext/train.bin`을 읽습니다
> (`bench.py:33-36`). OpenWebText를 준비하지 않았다면 위처럼 `--real_data=False`를 붙여
> 난수 배치로 돌려야 `FileNotFoundError`를 피할 수 있습니다.

## 9.9 자주 발생하는 오류

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

> WSL 특유의 문제(uv 의존성 해석 실패, torch.compile 빌드 도구 부재 등)는 [10.7](#107-트러블슈팅) 참조.

## 9.10 학습 완료 후 탐구 주제

1. **어텐션 시각화**: 각 레이어의 어텐션 패턴을 시각화해보세요 (`model.py`의 `attn_weights` 수정)
2. **임베딩 시각화**: t-SNE로 단어 임베딩을 2D로 줄여 비슷한 단어가 가깝게 위치하는지 확인
3. **모델 수술 (Model Surgery)**: `model.crop_block_size(512)`로 컨텍스트 창을 줄여보기
4. **프롬프팅 실험**: 다양한 프롬프트로 모델의 한계와 능력을 탐색
5. **GPT-2 분석**: `from_pretrained('gpt2')`로 불러온 모델의 가중치 분포 시각화

---

# 10. 랩톱 WSL 환경 구축과 명령어 레퍼런스

`laptop-wsl` 브랜치 기준으로, RTX 4060 Laptop 환경에서 nanoGPT를 실행하기 위한 설정 기록과 명령어 모음입니다.

- **브랜치**: `laptop-wsl` (base: `linux`)
- **런타임**: WSL2 Ubuntu (Windows 11 Pro)
- **작업 경로**: `/mnt/c/Study/260425 NanoGPT/nanoGPT`

## 10.1 환경 사양

### 하드웨어

| 항목                   | 값                                 |
| ---------------------- | ---------------------------------- |
| GPU                    | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM                   | 8,188 MiB (8GB)                    |
| Compute Capability     | **8.9 (Ada Lovelace)**       |
| bfloat16 하드웨어 지원 | ✅                                 |
| 드라이버               | 552.27                             |
| CPU                    | Intel Core i7-14650HX              |
| RAM                    | 15.7 GB                            |

### 소프트웨어 (WSL venv `~/venvs/nanogpt`)

| 패키지       | 버전        |
| ------------ | ----------- |
| Python       | 3.12.3      |
| torch        | 2.5.1+cu124 |
| triton       | 3.1.0       |
| numpy        | 2.5.2       |
| tiktoken     | 0.13.0      |
| tqdm         | 4.70.0      |
| transformers | 5.15.0      |
| datasets     | 5.0.1       |

### 이전 머신(`pc` 브랜치)과의 차이

`pc` 브랜치는 RTX 2070 Desktop(Turing, CC 7.5) 기준으로 튜닝돼 있었습니다. Turing은 bfloat16을 하드웨어로 지원하지 않아 float16으로 고정돼 있었으나, Ada는 네이티브 지원하므로 bfloat16이 유리합니다(범위가 넓어 수치 불안정이 적고 GradScaler 불필요).

|                       | RTX 2070 Desktop (`pc`) | RTX 4060 Laptop (`laptop-wsl`) |
| --------------------- | ------------------------- | -------------------------------- |
| Compute Capability    | 7.5 (Turing)              | 8.9 (Ada)                        |
| bfloat16              | ❌ 미지원                 | ✅ 네이티브                      |
| 권장 dtype            | float16 + GradScaler      | **bfloat16**               |
| bf16 Tensor Core 피크 | —                        | ~126 TFLOPS                      |
| VRAM                  | 8GB                       | 8GB (동일)                       |

**수정한 파일 (6개)**

| 파일                                   | 변경                                                    |
| -------------------------------------- | ------------------------------------------------------- |
| `model.py:290`, `299-301`          | `flops_promised` **60e12 → 126e12**, docstring |
| `train.py:50`, `74`                | batch 주석, dtype 자동 감지                             |
| `sample.py:21`                       | dtype 자동 감지                                         |
| `bench.py:12`, `18`                | batch 주석, dtype 자동 감지                             |
| `chat.py:9`                          | `bfloat16`                                            |
| `config/finetune_agent.py:3`, `16` | 헤더,`bfloat16`                                       |

dtype은 리터럴 고정 대신 upstream nanoGPT 원형인 자동 감지 방식을 사용합니다.

```python
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
```

> **`model.py`의 `flops_promised`를 반드시 함께 고쳐야 합니다.** 이 값은 MFU(Model FLOPs Utilization) 계산의 분모입니다. RTX 2070용 `60e12`를 그대로 두면 학습 로그의 MFU가 실제의 약 2.1배로 부풀려집니다.

## 10.2 빌드 툴체인 설치 (torch.compile 필수)

### 전제: sudo 비밀번호 문제

기본 WSL Ubuntu에는 `pip`도 `ensurepip`도 없어 `python3 -m venv`만으로는 venv에 pip이 생기지 않습니다. `sudo apt install python3-pip`이 정석이지만 비밀번호 입력이 필요하므로, root 권한이 필요 없는 **uv**로 우회합니다.

```bash
# 1) uv 설치 (~/.local/bin, sudo 불필요)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 2) venv 생성 — WSL 네이티브 fs에 배치할 것
uv venv ~/venvs/nanogpt --python 3.12

# 3) torch + 의존성
uv pip install --python ~/venvs/nanogpt/bin/python torch==2.5.1
uv pip install --python ~/venvs/nanogpt/bin/python numpy tiktoken tqdm transformers datasets
```

venv를 `/mnt/c`가 아닌 `~/venvs`에 두는 이유는 두 가지입니다. `/mnt/c`는 9p 프로토콜을 거쳐 순차 읽기가 61 MB/s인 반면 네이티브 fs는 2.7 GB/s이고, Windows venv는 `Scripts/`, Linux venv는 `bin/` 구조라 한 디렉터리를 공유할 수도 없습니다.

### 빌드 도구

```bash
sudo apt update && sudo apt install -y build-essential python3-dev
```

WSL Ubuntu 기본 이미지는 서버·컨테이너용 최소 구성이라 개발 도구가 전혀 없습니다.
`torch.compile`은 실행 시점에 코드를 생성해 **그 자리에서 컴파일**하므로 두 패키지가 모두 필요하며,
하나씩 순서대로 드러납니다.

| 패키지              | 제공하는 것                               | 없으면 나는 오류                                     |
| ------------------- | ----------------------------------------- | ---------------------------------------------------- |
| `build-essential` | `gcc`, `g++`, `make`, `libc6-dev` | `Failed to find C compiler`                        |
| `python3-dev`     | `Python.h` 등 C 확장 빌드용 헤더        | `fatal error: Python.h: No such file or directory` |

`python3-dev`가 필요한 이유는 triton이 `cuda_utils` 확장 모듈을 빌드하기 때문입니다.
venv가 시스템 CPython(`/usr/bin/python3.12`)에서 생성됐으므로 시스템 쪽 헤더를 참조합니다.

설치 후 실제 동작 확인:

```bash
python -c "import torch; f=torch.compile(lambda x: x*2); print(f(torch.ones(4,device='cuda')))"
```

`tensor([2., 2., 2., 2.], device='cuda:0')`가 나오면 컴파일 → GPU 실행 경로가 끝까지 뚫린 것입니다.
첫 실행은 컴파일 때문에 30초 안팎 걸립니다.

> `sudo`가 비밀번호를 요구하는데 비밀번호를 모른다면, Windows PowerShell에서 root로 우회할 수 있습니다.
> `wsl -d Ubuntu -u root -e apt install -y build-essential python3-dev`
> 비밀번호 재설정은 `wsl -d Ubuntu -u root passwd <사용자명>` (현재 비밀번호를 묻지 않음).
> 두 명령 모두 **Windows 프롬프트**(`PS C:\...>`)에서 실행해야 합니다. `wsl`은 Windows 명령어라
> Ubuntu 안(`user@HOST:~$`)에서는 `command not found`가 납니다.

## 10.3 세션 시작과 데이터 준비

```bash
wsl -d Ubuntu
cd "/mnt/c/Study/260425 NanoGPT/nanoGPT"
source ~/venvs/nanogpt/bin/activate
```

환경 확인:

```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
git branch --show-current
```

데이터 준비:

```bash
python data/shakespeare_char/prepare.py   # 완료됨 (train.bin 1.9MB, val.bin 218KB)
python data/agent/prepare.py              # 미완 — chat.py 사용 전 필수
python data/shakespeare/prepare.py        # 미완 — GPT-2 BPE 버전
python data/openwebtext/prepare.py        # 수십 GB, 장시간
```

현재 `.bin`이 준비된 데이터셋은 `shakespeare_char`뿐입니다.

## 10.4 학습과 결과 확인

```bash
# 기본 (char-level, 5000 iters, 10.65M params)
python train.py config/train_shakespeare_char.py --compile=True

# 짧게 테스트 (체크포인트까지 확보하려면 300 이상)
python train.py config/train_shakespeare_char.py --max_iters=500 --compile=True

# 중단 후 재개
python train.py config/train_shakespeare_char.py --init_from=resume --compile=True

# VRAM 부족 시
python train.py config/train_shakespeare_char.py --batch_size=32 --block_size=128

# GPT-2 파인튜닝 (agent 데이터 전처리 후)
python train.py config/finetune_agent.py

# 백그라운드 + 로그
nohup python train.py config/train_shakespeare_char.py --compile=True > train.log 2>&1 &
tail -f train.log
```

`build-essential` 설치 전에는 `--compile=False`를 사용합니다.

> **결과 확인의 선행 조건**: `out-shakespeare-char/ckpt.pt`가 있어야 합니다. `train_shakespeare_char.py`는 `eval_interval = 250`, `always_save_checkpoint = False`이고 `train.py:279`가 `iter_num > 0`을 요구하므로, **첫 저장은 iteration 250**입니다. `--max_iters`가 300 미만이면 학습이 정상 종료돼도 체크포인트가 생기지 않아 `sample.py`가 `FileNotFoundError`로 실패합니다.

```bash
ls -lh out-shakespeare-char/

python sample.py --out_dir=out-shakespeare-char
python sample.py --out_dir=out-shakespeare-char --num_samples=3 --max_new_tokens=300
python sample.py --out_dir=out-shakespeare-char --start="ROMEO:" --temperature=0.7

# 체크포인트 메타 확인
python -c "import torch;c=torch.load('out-shakespeare-char/ckpt.pt',map_location='cpu');print('iter',c['iter_num'],'val_loss',c['best_val_loss'])"
```

## 10.5 학습 로그 읽는 법

**시작 시**

| 출력                                     | 뜻                                                               |
| ---------------------------------------- | ---------------------------------------------------------------- |
| `found vocab_size = 65`                | 이 데이터의 고유 문자 종류 수 (`meta.pkl`에서 읽음)            |
| `number of parameters: 10.65M`         | 모델 가중치 개수. GPT-2 Small(124M)의 약 1/12                    |
| `tokens per iteration will be: 16,384` | 1 iteration에 쓰는 토큰 수 (`batch_size 64 × block_size 256`) |
| `using fused AdamW: True`              | 옵티마이저 연산을 CUDA 커널 하나로 융합 (더 빠름)                |
| `compiling the model...`               | inductor가 코드 생성·컴파일 중. 1~3분 멈춘 것처럼 보임          |

**진행 중**

```
step 0: train loss 4.2853, val loss 4.2842
saving checkpoint to out-shakespeare-char
```

- **train loss**: 학습 데이터에 대한 예측 오차
- **val loss**: 학습에 쓰지 않은 검증 데이터에 대한 오차 — 실제 성능 지표
- **시작값 해석**: `shakespeare_char`는 vocab이 65자이므로 무작위 추측의 이론적 손실이
  `ln(65) ≈ 4.17`입니다. 초기 4.28은 아직 아무것도 학습하지 않은 상태라는 뜻입니다
- 5000 iters 후 val loss **1.4~1.5** 부근이면 정상입니다
- `saving checkpoint`는 val loss가 이전 최저치를 갱신했다는 뜻입니다
  (`always_save_checkpoint = False`이므로 개선될 때만 저장)

**진행바**

```
Training: 10%|█ | 10/100 [02:40<24:06, loss=2.4599, mfu=1.44%]
```

- **mfu**: Model FLOPs Utilization. GPU 이론 성능(`model.py`의 `flops_promised`) 대비 실제 활용률.
  작은 모델은 GPU를 채우지 못해 낮게 나오는 것이 정상입니다
- **`it/s`·ETA**: 실제 iteration 속도입니다. `loss`·`mfu`는 `log_interval`마다만 갱신되므로
  그 사이에는 직전 값이 그대로 표시됩니다

**이상 신호**

| 증상                          | 원인 / 대처                                              |
| ----------------------------- | -------------------------------------------------------- |
| val loss가 내려가다 다시 상승 | 과적합. 이 설정은 의도적으로 그렇게 되며 최저점만 저장됨 |
| loss가`nan`                 | 수치 발산. bfloat16에서는 거의 발생하지 않음             |
| `CUDA out of memory`        | `--batch_size=32` 등으로 낮출 것                       |

## 10.6 성능 확인, 대화, git

```bash
python bench.py --real_data=False                  # GPT-2 124M 기준 (batch 8, block 1024)
python bench.py --real_data=False --compile=False  # 컴파일 유무 비교
python bench.py --real_data=False --profile=True   # PyTorch profiler

watch -n 1 nvidia-smi
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 2
```

```bash
python chat.py     # out-agent-ft/ckpt.pt 를 로드 (경로 고정)
```

> `chat.py`는 `configurator.py`를 호출하지 않으므로 **명령줄 인자를 받지 않습니다.**
> `--out_dir=...`을 붙여도 무시되고 `chat.py:7`의 `out-agent-ft`를 그대로 씁니다.
> 또한 `chat.py:35`가 `tiktoken.get_encoding("gpt2")`로 고정돼 있어 **GPT-2 계열 모델 전용**입니다.
> 문자 단위 모델(`shakespeare_char`, vocab 65)에 물리면 토큰 ID가 임베딩 범위를 벗어나 실패합니다.
> 다른 체크포인트를 쓰려면 `chat.py`를 직접 수정해야 합니다.

`exit` 입력 시 종료됩니다. `data/agent/prepare.py` → `train.py config/finetune_agent.py`를 마쳐야 `out-agent-ft/ckpt.pt`가 생성됩니다.

```bash
git status --short
git diff
git add -A && git commit -m "Adapt configs for RTX 4060 Laptop (Ada, bf16)"
git push -u origin laptop-wsl
```

## 10.7 트러블슈팅

### uv 의존성 해석 실패

```
× No solution found when resolving dependencies:
╰─▶ Because there is no version of nvidia-cudnn-cu12{...}==9.1.0.70 and
    torch>=2.5.1+cu121 depends on nvidia-cudnn-cu12{...}==9.1.0.70,
    we can conclude that torch>=2.5.1+cu121 cannot be used.
```

**원인**: uv에서 `--index-url`은 PyPI를 추가하는 게 아니라 **완전히 대체**합니다. PyTorch cu121 인덱스에는 해당 cuDNN 휠이 없어 해석이 막힙니다.

**해결**: `--index-url`을 빼고 PyPI 기본 휠(cu124)을 사용합니다. 드라이버 552.27이 CUDA 12.4를 지원하며, triton도 이쪽 의존성에 포함됩니다.

### torch.compile — C 컴파일러 부재

```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Failed to find C compiler. Please specify via CC environment variable.
```

**해결**: `sudo apt install -y build-essential`

### torch.compile — Python 헤더 부재

`build-essential` 설치 후 gcc는 호출되지만 그다음 단계에서 막히는 경우입니다.

```
/tmp/tmpXXXX/main.c:5:10: fatal error: Python.h: No such file or directory
    5 | #include <Python.h>
...
subprocess.CalledProcessError: Command '['/usr/bin/gcc', ..., '-I/usr/include/python3.12']'
    returned non-zero exit status 1.
```

**원인**: gcc 명령줄에 `-I/usr/include/python3.12`가 있지만 그 디렉터리에 `Python.h`가 없습니다.
Ubuntu는 Python 런타임과 개발 헤더를 별도 패키지로 나눕니다.

**해결**: `sudo apt install -y python3-dev`

### 명령 실행 중 누른 키가 나중에 실행됨

```
$ python -c "..."
A                                    ← 실행 중 누른 키가 즉시 에코됨
tensor([2., 2., 2., 2.], ...)        ← 30초 뒤 실제 출력
A: command not found                 ← 종료 후 bash가 버퍼의 A를 명령으로 해석
```

**원인**: 앞 명령이 도는 동안 bash는 stdin을 읽지 않습니다. 누른 키는 화면에 에코되면서
커널 입력 큐에 쌓여 있다가, 명령이 끝나 bash가 복귀하면 명령줄로 처리됩니다.
개행(Enter)까지 들어갔다면 완성된 명령으로 실행됩니다.

**대처**: 긴 명령 실행 중에는 키를 누르지 않습니다. 위 사례는 `A`라는 명령이 없어 무해했지만,
버퍼에 쌓인 글자가 우연히 실제 명령을 이루면 그대로 실행되므로 원리상 주의가 필요합니다.

### 학습이 끝났는지 확인

진행바와 무관하게 프로세스와 체크포인트로 판단할 수 있습니다.

```bash
pgrep -af train.py     # 아무것도 안 나오면 종료된 것
python -c "import torch;c=torch.load('out-shakespeare-char/ckpt.pt',map_location='cpu');print(c['iter_num'], c['best_val_loss'])"
```

체크포인트의 `iter_num`이 `max_iters`보다 작은 것은 정상입니다. `always_save_checkpoint = False`이면
val loss가 갱신될 때만 저장하므로, 과적합이 시작된 이후 구간은 학습은 계속되지만 저장되지 않습니다.

## 10.8 검증 결과

### GPU 실측 (WSL)

| 항목                   | 값                              |
| ---------------------- | ------------------------------- |
| matmul 4096² bfloat16 | 7.63 ms →**18.0 TFLOPS** |
| matmul 4096² float16  | 7.84 ms → 17.5 TFLOPS          |
| matmul 4096² float32  | 37.68 ms → 3.6 TFLOPS          |
| H2D 전송 1 GiB         | 224 ms (~4.6 GB/s)              |
| 소형 커널 1000회       | 51.4 ms                         |

WSL GPU 패스스루는 정상 동작합니다.

### 파일시스템 I/O

|                            | `/mnt/c` | WSL 네이티브 |
| -------------------------- | ---------- | ------------ |
| memmap 랜덤 배치 (64×256) | 0.60 ms    | 0.32 ms      |
| 순차 읽기 (`dd`)         | 61.2 MB/s  | 2.7 GB/s     |

학습 배치 로딩은 iteration당 1ms 미만이라 `/mnt/c`에 소스를 두어도 병목이 아닙니다. 다만 venv와 대용량 데이터셋은 네이티브 fs가 유리합니다.

### 라이브러리 호환성

| 항목                                           | 결과                                                       |
| ---------------------------------------------- | ---------------------------------------------------------- |
| `GPT.from_pretrained` ↔ transformers 5.15.0 | ✅ 키 149/149 일치, shape assert 실패 0건 (Conv1D 유지)    |
| `load_dataset` ↔ datasets 5.0.1             | ✅`num_proc` 파라미터 유지                               |
| numpy 2.5.2 ↔ torch 2.5.1                     | ✅ memmap·interop 정상                                    |
| `torch.cuda.amp.GradScaler`                  | ⚠️ deprecation 경고만 (bf16에선`enabled=False`라 무해) |

### 설치 소요

| 단계                        | 시간        |
| --------------------------- | ----------- |
| torch 및 CUDA 런타임 (23개) | 5분 54초    |
| 나머지 의존성 (45개)        | 13.4초      |
| 다운로드 총량               | 약 2.96 GiB |

---

# 11. 더 읽을 자료

## 이 저장소 안에서

- [`QNA.md`](./QNA.md) — `input.txt`·`.bin`·`meta.pkl`·`ckpt.pt`가 각각 뭔지, 실측 덤프 포함 문답 16개
- [`qkv_dimension_workbook.ipynb`](./qkv_dimension_workbook.ipynb) — 어텐션의 B·T·C 변형을 빈칸 채우며 직접 검산
- [`../test/`](../test/00_system_setup.md) — 환경 세팅 / GPU 튜닝 / 실험 기록

## 바깥 자료

- **Andrej Karpathy의 강의**: "Let's build GPT from scratch" (YouTube)
- **원본 GPT 논문**: "Improving Language Understanding by Generative Pre-Training" (2018)
- **GPT-2 논문**: "Language Models are Unsupervised Multitask Learners" (2019)
- **Attention Is All You Need**: 트랜스포머 원논문 (2017)
- **nanochat**: nanoGPT의 후속 프로젝트 (더 현대적인 구조)
