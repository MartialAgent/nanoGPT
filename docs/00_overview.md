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

```
nanoGPT/
├── model.py              ← GPT 모델 정의 (~330줄)
├── train.py              ← 학습 루프 (~337줄)
├── sample.py             ← 텍스트 생성
├── bench.py              ← 속도 벤치마크
├── configurator.py       ← 설정 관리
│
├── config/               ← 시나리오별 하이퍼파라미터
│   ├── train_shakespeare_char.py   ← 입문용 (빠른 학습)
│   ├── train_gpt2.py               ← GPT-2 전체 재현
│   └── finetune_shakespeare.py     ← 파인튜닝 예시
│
├── data/
│   ├── shakespeare_char/  ← 입문용 데이터 (1MB)
│   ├── shakespeare/       ← BPE 토큰화된 셰익스피어
│   └── openwebtext/       ← 대규모 학습 데이터 (9B 토큰)
│
└── docs/                  ← 지금 읽고 있는 학습 자료
```

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
