# 04. 데이터 준비 (Data Preparation)

텍스트 데이터를 모델이 학습할 수 있는 숫자 배열로 변환하는 과정을 다룹니다.

---

## 토크나이제이션이란?

텍스트를 정수 ID의 시퀀스로 변환하는 과정입니다.

```
"Hello, world!" → [15496, 11, 995, 0]  (GPT-2 BPE 토크나이저)
"Hello, world!" → [20, 5, 12, 12, 15, 2, 0, 23, 15, 18, 12, 4, 1]  (문자 수준)
```

---

## 세 가지 데이터셋

| 데이터셋 | 토크나이저 | 크기 | 학습 시간 | 목적 |
|---------|----------|------|---------|------|
| `shakespeare_char` | 문자 수준 | 1MB | ~3분 (A100) | 빠른 실험, 입문 |
| `shakespeare` | BPE (GPT-2) | 1MB | 단시간 | BPE 파인튜닝 |
| `openwebtext` | BPE (GPT-2) | ~54GB | ~4일 (8×A100) | GPT-2 재현 |

---

## 1. Shakespeare Char (문자 수준)

**경로**: `data/shakespeare_char/prepare.py`

### 처리 과정

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

---

## 2. Shakespeare BPE

**경로**: `data/shakespeare/prepare.py`

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
```

**차이점**: 같은 셰익스피어 텍스트지만 BPE로 토크나이징하므로 토큰 수가 더 적습니다.

| | 문자 수준 | BPE |
|-|---------|-----|
| 학습 토큰 | ~1M | ~302K |
| 검증 토큰 | ~111K | ~36K |
| 어휘 크기 | 65 | 50257 |

BPE는 자주 나오는 글자 조합을 하나의 토큰으로 묶어 더 압축합니다.

---

## 3. OpenWebText (대규모)

**경로**: `data/openwebtext/prepare.py`

Reddit에서 추천받은 웹 페이지를 크롤링한 데이터셋입니다.

### 처리 과정

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

---

## 저장 형식: `.bin` 파일

토큰을 `uint16` (2바이트 부호 없는 정수)로 저장합니다.

```
왜 uint16?
- GPT-2 vocab_size = 50257
- uint16 최대값 = 65535
- 50257 < 65535 → uint16으로 충분
- uint32보다 파일 크기 절반
```

**메모리맵 방식 (`np.memmap`)**:
```python
data = np.memmap('train.bin', dtype=np.uint16, mode='r')
```

파일 전체를 RAM에 올리지 않고, 필요한 부분만 디스크에서 읽습니다. 17GB 파일도 8GB RAM에서 처리할 수 있습니다.

---

## 데이터 배치 구조

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

---

## BPE (Byte Pair Encoding) 이란?

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

---

## 직접 실행하기

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

## 다음 단계

학습이 완료된 후 텍스트를 생성하는 방법 → [05_inference.md](./05_inference.md)
