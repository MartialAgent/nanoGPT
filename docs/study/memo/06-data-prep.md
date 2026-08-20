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
