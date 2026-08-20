# AI Agent 파인튜닝 실험 — 격리 보관

이 폴더는 저장소의 **실행 경로에서 빼내되 버리지는 않은** 신규 기능 일체다.
카파시 원본 nanoGPT에 없던 추가 기능이라, 원본 대비 차이를 "로컬 환경 대응 +
실행성 수정"으로만 한정하기 위해 `archive/` 아래로 옮겼다.
저장소 안에 있지만 어떤 실행 경로도 이 폴더를 참조하지 않는다 — 보관 전용이다.

- 격리 일자: 2026-08-20
- 격리 시점 리포 HEAD: `87fa4e27ce2f79188734ca4490b0a1d6bfb4f2c7`
- 업스트림 기준 커밋: `3adf61e` (karpathy/nanoGPT 최종 커밋 — 이 위로 포크 커밋 15개)
- 도입 커밋: `f9863c9` "feat: complete AI Agent experiment with tqdm and interactive chat"
  (이후 `71b1de5`에서 RTX 4060 Laptop 설정으로 조정)

## 무엇이 들어있나

`repo/` 아래는 원래 저장소 루트 기준 상대 경로를 그대로 유지한다.
복원할 때 `repo/` 내용을 저장소 루트에 그대로 덮어쓰면 된다.

| 보관 경로 | 원래 경로 | 줄 수 | 내용 |
|---|---|---|---|
| `repo/chat.py` | `chat.py` | 57 | 학습된 체크포인트를 로드해 `input()` 루프로 대화하는 REPL. `out-agent-ft/ckpt.pt` 경로가 하드코딩되어 있고 CLI 인자를 받지 않는다. `_orig_mod.` 접두사(torch.compile 흔적) 제거 처리 포함 |
| `repo/config/finetune_agent.py` | `config/finetune_agent.py` | 27 | GPT-2 124M(`init_from='gpt2'`) → `agent` 데이터셋 파인튜닝 설정. `max_iters=500`, `lr=3e-5`, `decay_lr=False`, `batch_size=4`, `grad_accum=8` |
| `repo/data/agent/prepare.py` | `data/agent/prepare.py` | 33 | GPT-2 BPE 토크나이징 → `train.bin` / `val.bin` |
| `repo/data/agent/input.txt` | `data/agent/input.txt` | 17,001 | 학습 원문 |
| `repo/docs/test/03_chat_interaction_test.md` | 〃 | 72 | 대화 실험 리포트 |
| `repo/docs/test/04_gpt2_finetuning_experiment.md` | 〃 | 40 | 파인튜닝 실험 리포트 |

## 격리 시점의 미완 사항 (실험 재개 시 먼저 고칠 것)

**`repo/data/agent/prepare.py`는 `data/shakespeare/prepare.py`의 복사본이다.**
`input.txt`가 없을 때 tinyshakespeare를 내려받는 코드가 그대로 남아 있고
(`data_url = .../tinyshakespeare/input.txt`), 파일 말미 주석의 토큰 수
(`train.bin has 301,966 tokens`)도 셰익스피어 기준 값이다.
`input.txt`가 함께 보관되어 있으므로 다운로드 분기는 타지 않지만,
주석과 URL은 agent 데이터에 맞게 고쳐야 한다.

`train.bin` / `val.bin`은 `.gitignore` 대상이라 생성된 적이 없고 여기에도 없다.
`out-agent-ft/` 체크포인트도 생성된 적이 없다. 즉 **실험은 파인튜닝 실행 전
단계에서 멈춰 있었다.**

## 실험 재개 절차

저장소 루트에서:

```bash
# 1. 실행 경로에 복원
cp -r archive/agent-experiment/repo/. .

# 2. 데이터 토크나이징 (prepare.py 주석/URL 먼저 수정 권장)
python data/agent/prepare.py

# 3. 파인튜닝 (out-agent-ft/ckpt.pt 생성)
python train.py config/finetune_agent.py

# 4. 대화
python chat.py
```

## 격리해도 리포가 깨지지 않는 근거

격리 대상 6개 파일을 참조하는 코드는 `config/finetune_agent.py:11`의
`dataset = 'agent'` 한 줄뿐이고, 그 파일도 함께 격리했다.
`chat.py`는 어떤 모듈도 import하지 않으며, `train.py` / `sample.py` /
`bench.py` / `model.py`는 이들을 전혀 참조하지 않는다.
따라서 셰익스피어 계열 학습·샘플링 경로의 실행 결과는 격리 전후로 동일하다.

문서 쪽 참조는 `docs/study/README.md`에 남아 있으며,
해당 항목들에는 삭제가 아니라 **격리 표시**를 달아 두었다.
