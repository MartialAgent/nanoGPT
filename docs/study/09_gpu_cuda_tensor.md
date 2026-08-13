# 09. GPU · CUDA · 텐서 기초

nanoGPT를 실행하다 보면 `cuda`, `tensor`, `device='cuda:0'` 같은 것들이 계속 나옵니다.
코드를 읽기 전에 알아두면 좋은 최소한의 개념을 정리합니다.

---

## GPU와 CUDA의 관계

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

---

## 텐서 (Tensor)

**숫자를 담는 다차원 배열**입니다.

| 차원 | 이름 | 예시 |
|---|---|---|
| 0차원 | 스칼라 | `5` |
| 1차원 | 벡터 | `[1, 2, 3]` |
| 2차원 | 행렬 | `[[1,2], [3,4]]` |
| 3차원 이상 | 텐서 | 행렬을 여러 장 쌓은 것 |

파이썬 리스트와 비슷해 보이지만 두 가지가 결정적으로 다릅니다.

- **GPU에서 돌아갑니다.** 수백만 개 숫자에 같은 연산을 동시에 적용할 수 있습니다
- **미분이 자동으로 됩니다.** 학습은 "오차를 줄이는 방향"을 계산하는 일인데,
  텐서는 자신이 거쳐온 연산 경로를 기억해 그 방향을 역으로 계산해냅니다 (autograd)

### 출력 읽는 법

```
tensor([2., 2., 2., 2.], device='cuda:0')
```

| 부분 | 뜻 |
|---|---|
| `tensor(...)` | 이것은 텐서다 |
| `[2., 2., 2., 2.]` | 값 4개 |
| `2.` (점에 주의) | **실수**라는 표시. `2.0`과 같으며 정수 `2`와 구분됨 |
| `device='cuda:0'` | 이 데이터가 0번 GPU 메모리에 있다 |

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

---

## device — `cuda`와 `cuda:0`

PyTorch는 GPU를 **0번부터** 셉니다. GPU가 하나면 `cuda:0` 하나뿐이고, 넷이면 `cuda:0` ~ `cuda:3`입니다.

```
GPU 1개 (이 랩톱)        GPU 4개 (서버)
┌─────────┐              ┌────┬────┬────┬────┐
│ cuda:0  │              │ :0 │ :1 │ :2 │ :3 │
└─────────┘              └────┴────┴────┴────┘
```

GPU가 하나여도 번호를 떼지 않는 이유는, 코드가 몇 대짜리 환경에서든 동일하게 동작해야 하기 때문입니다.

| 표기 | 뜻 |
|---|---|
| `'cuda'` | 기본 GPU에 배치 → 이 랩톱에선 자동으로 0번 |
| `'cuda:0'` | 0번 GPU에 명시적으로 배치 |

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

---

## dtype — 숫자를 몇 비트로 담을 것인가

같은 숫자라도 정밀도를 낮추면 메모리가 줄고 계산이 빨라집니다. 딥러닝은 약간의 정밀도 손실을
감수하고 속도를 택하는 것이 일반적입니다.

| dtype | 비트 | 표현 범위 | 정밀도 | 하드웨어 요구 |
|---|---|---|---|---|
| `float32` | 32 | 넓음 | 높음 | 모든 GPU |
| `bfloat16` | 16 | float32와 **동일** | 낮음 | Ampere(CC 8.0) 이상 |
| `float16` | 16 | 좁음 | 중간 | 대부분의 GPU |

**bfloat16이 권장되는 이유**는 지수부 비트를 float32와 똑같이 유지하기 때문입니다. 표현 범위가
같으므로 값이 넘치거나 0으로 사라지는 문제가 거의 없습니다. 반면 float16은 범위가 좁아
학습 중 발산하기 쉬워서, 손실값을 인위적으로 키웠다 되돌리는 **GradScaler**라는 보정 장치가 필요합니다.

`train.py`는 하드웨어를 보고 자동으로 고릅니다.

```python
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
```

RTX 4060 Laptop은 Ada(CC 8.9)라 bfloat16이 선택되고, GradScaler는 자동으로 비활성화됩니다.
RTX 2070(Turing, CC 7.5)에서는 bfloat16을 하드웨어 지원하지 않아 float16 + GradScaler 경로를 탑니다.

---

## torch.compile이 하는 일

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
([08_laptop_wsl_setup.md](./08_laptop_wsl_setup.md) 참조).

가장 작은 확인 방법:

```bash
python -c "import torch; f=torch.compile(lambda x: x*2); print(f(torch.ones(4,device='cuda')))"
```

`torch.ones(4, device='cuda')`로 GPU에 `[1,1,1,1]`을 만들고, `lambda x: x*2`(입력에 2를 곱하는
이름 없는 함수)를 컴파일해 적용합니다. `tensor([2., 2., 2., 2.], device='cuda:0')`가 나오면
코드 생성 → 컴파일 → GPU 실행 경로가 전부 정상이라는 뜻입니다.
