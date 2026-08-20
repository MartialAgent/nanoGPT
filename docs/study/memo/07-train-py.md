# 7장 · 학습 파이프라인 (train.py) — 학습 로그

본문: [`README.md` 7장](../README.md#7-학습-파이프라인-trainpy) · 색인: [`memo/README.md`](./README.md)

---

## LOG-0003 · ckpt.pt의 내부 구조와 파일 크기, 그리고 `_orig_mod.` 접두사

**날짜** 2026-08-20 · **걸친 장** 7장, 5장, 8장 · **상태** 실측 확인 완료

### 개념정리

**체크포인트(checkpoint)란** — 학습 도중의 상태를 파일 하나로 얼려둔 것이다. 흔히 "학습된 모델"이라고
부르지만, nanoGPT가 저장하는 `ckpt.pt`는 **모델 가중치만 들어 있는 파일이 아니다.** 학습을 중단한
지점에서 그대로 재개할 수 있도록, 가중치 외에 옵티마이저 상태와 재현에 필요한 설정까지 함께 담는다.

**전체 과정에서의 위치** — 파이프라인에서 `ckpt.pt`는 ②의 출력이자 ③의 입력이고, 동시에 ②로 되돌아가는
입력이기도 하다(학습 재개).

```
① 데이터 준비            ② 학습 (train.py)                    ③ 추론 (sample.py)
   train.bin/val.bin  →     주기적으로 val loss 평가       →      ckpt.pt 읽어
   meta.pkl                 → 갱신되면 ckpt.pt 저장               텍스트 생성
                            ↑______ init_from='resume' ______|
```

**저장되는 6개 키** — `torch.save()`에 넘기는 dict의 구조가 곧 파일의 구조다.

| 키              | 담기는 것                             | 용도                            |
| --------------- | ------------------------------------- | ------------------------------- |
| `model`         | 가중치 텐서 40개 (state_dict)         | 추론·재개 모두 필요             |
| `optimizer`     | AdamW의 모멘트 상태                   | **재개에만** 필요               |
| `model_args`    | n_layer/n_head/n_embd/block_size/…    | 같은 형태의 빈 모델을 다시 조립 |
| `iter_num`      | 저장 시점의 스텝 수                   | 재개 시 이어붙일 지점           |
| `best_val_loss` | 그때까지의 최저 검증 손실 (0-dim 텐서) | 다음 저장 여부 판단 기준        |
| `config`        | 학습에 쓰인 하이퍼파라미터 전체       | 사후 재현·추적용                |

`model_args`가 함께 저장되는 이유가 중요하다. `state_dict`는 **텐서 값의 모음일 뿐 구조 정보가 아니다.**
불러올 때는 `model_args`로 `GPTConfig`를 복원해 동일한 형태의 모델을 먼저 만들고, 거기에 값을 부어넣는다.

**파일 크기가 파라미터의 약 3배인 이유** — AdamW는 파라미터마다 1차 모멘트(`exp_avg`)와
2차 모멘트(`exp_avg_sq`)를 파라미터와 같은 크기로 들고 있다. 그래서 `파라미터 1벌 + 옵티마이저 2벌 = 3벌`이다.
이 저장소의 shakespeare_char 체크포인트를 실제로 열어 재본 값:

| 항목                          | 실측                                      |
| ----------------------------- | ----------------------------------------- |
| `model` (fp32 40텐서)         | 10,770,048 params → 43,080,192 B          |
| `optimizer` (39개 파라미터 × 2) | 85,960,704 B                              |
| 파일 전체                     | 128,986,325 B (≈123 MiB, 가중치의 **2.99배**) |

옵티마이저 항목이 39개인데 가중치 텐서가 40개인 것은 **가중치 묶기(weight tying)** 때문이다 —
`transformer.wte.weight`와 `lm_head.weight`는 같은 텐서를 공유하므로 옵티마이저 입장에서는 하나다.

**`_orig_mod.` 접두사** — 이 저장소의 체크포인트는 키 이름이 `_orig_mod.transformer.wte.weight`처럼
시작한다. `torch.compile(model)`이 원래 모듈을 `OptimizedModule`로 감싸면서 하위 모듈 이름 앞에
`_orig_mod.`을 붙이기 때문이다. `raw_model`은 **DDP 래퍼만** 벗겨내고 compile 래퍼는 벗기지 않으므로,
접두사가 붙은 채로 저장된다. 불러오는 쪽(`train.py`의 resume, `sample.py`)이 문자열을 잘라내 정리한다.

### 근거 — 저장소 안 실제 위치

| 파일        | 위치                                                   | 내용                                                        |
| ----------- | ------------------------------------------------------ | ----------------------------------------------------------- |
| `train.py`  | [`:281-288`](../../train.py#L281-L288)                 | 저장되는 6개 키의 정의                                      |
| `train.py`  | [`:290`](../../train.py#L290)                          | `torch.save(checkpoint, out_dir/'ckpt.pt')`                 |
| `train.py`  | [`:278`](../../train.py#L278)                          | 저장 조건 — val loss 갱신 시, 또는 `always_save_checkpoint` |
| `train.py`  | [`:208-209`](../../train.py#L208-L209)                 | `model = torch.compile(model)` — 접두사가 생기는 지점       |
| `train.py`  | [`:254`](../../train.py#L254)                          | `raw_model = model.module if ddp else model` — DDP만 해제   |
| `train.py`  | [`:175-178`](../../train.py#L175-L178)                 | resume 시 `_orig_mod.` 접두사 제거                          |
| `sample.py` | [`:42-45`](../../sample.py#L42-L45)                    | 추론 시 동일한 접두사 제거                                  |
| `model.py`  | [`:138`](../../model.py#L138)                          | `wte.weight = lm_head.weight` — 가중치 묶기                 |
| `model.py`  | [`:280-284`](../../model.py#L280-L284)                 | AdamW 생성 (모멘트 2벌이 여기서 생김)                       |

### 부가사항

- **체크포인트를 여는 전용 뷰어는 없다시피 하다.** `ckpt.pt`는 traced 그래프가 아니라 dict + state_dict라
  Netron 같은 모델 뷰어로 열어도 연산 그래프가 나오지 않는다. `.bin`은 헤더 없는 `uint16` 배열이라
  hex 뷰어로는 바이트만 보이고, `meta.pkl`은 파이썬 pickle이라 텍스트 에디터로는 깨진다.
  세 포맷을 함께 덤프하는 스크립트를 [`tools/inspect_ckpt.py`](../../tools/inspect_ckpt.py)에 두었다.
- **추론만 할 거라면 `optimizer` 키는 없어도 된다.** 실제로 파일의 2/3가 옵티마이저 상태이므로,
  배포용으로는 `model`/`model_args`만 남겨 다시 저장하면 크기가 1/3로 줄어든다.
  단, 그렇게 만든 파일로는 `init_from='resume'` 학습 재개를 할 수 없다.
- **저장 시점이 마지막 스텝이 아닐 수 있다.** 기본 설정(`always_save_checkpoint=False`)에서는
  검증 손실이 갱신될 때만 저장하므로, `iter_num`은 "최저 val loss를 찍은 스텝"이지 "학습을 끝낸 스텝"이 아니다.
- `best_val_loss`는 파이썬 float이 아니라 0-dim 텐서로 저장된다(`tensor(1.4667)`). 수치 비교에는
  문제가 없지만 그대로 출력하면 `tensor(...)` 형태로 찍힌다.

### 남은 질문

- `torch.compile` 없이 학습한 체크포인트(`compile=False`)와 접두사가 붙은 체크포인트를 서로 바꿔
  불러올 때 실제로 어떤 경고/에러가 나는지는 확인하지 않았다.
- fp32 외의 dtype으로 저장되는 경우(예: bf16 가중치를 그대로 저장)의 크기 비율은 확인하지 않았다.
  이 저장소는 `dtype='bfloat16'`으로 학습했지만 저장된 텐서는 전부 fp32였다 — autocast가 마스터
  가중치를 fp32로 유지하기 때문으로 보이나 코드로 확정하지는 못했다.

### 관련

- 본문 [7장 학습 파이프라인](../README.md#7-학습-파이프라인-trainpy)
- [`06-data-prep.md`](./06-data-prep.md) — `train.bin`/`meta.pkl`의 포맷 (LOG-0002)
- [`05-model-py.md`](./05-model-py.md) — 파라미터 수 세는 방식과 vocab padding
