# RTX 2070 기반 nanoGPT 학습 성능 및 결과 보고서

본 보고서는 NVIDIA RTX 2070 환경에서 nanoGPT의 모델 학습 성적과 최적화 결과를 기록합니다.

## 1. 테스트 환경 (System Environment)
- **GPU**: NVIDIA GeForce RTX 2070 Desktop (8GB VRAM)
- **OS**: Windows 11 / WSL2 (Ubuntu)
- **Library**: PyTorch (CUDA 지원), Transformers, Tiktoken
- **Optimization**: `dtype = 'float16'`, `batch_size = 8` (RTX 2070 최적화 설정)

## 2. 학습 모델 성적 (Training Metrics)
다음 수치는 `out-shakespeare-char/ckpt.pt` 파일 내부에서 직접 추출한 결과입니다.

| 항목 | 수치 | 비고 |
| :--- | :--- | :--- |
| **최종 손실값 (Best Val Loss)** | **1.4686** | 캐릭터 단위 모델 기준 안정적 수준 |
| **학습 반복 횟수 (Iter)** | **1,500** | 충분한 학습이 진행됨 |
| **추론 장치** | CUDA (RTX 2070) | GPU 가속 정상 작동 확인 |
| **VRAM 점유율** | 약 2.5GB | 8GB 중 여유 공간 충분함 |

## 3. 수치 추출 방법 (How to verify)
별도의 로그 파일 없이 모델 파일(`.pt`)에서 직접 수치를 확인하려면 아래 명령어를 사용합니다.

```bash
python3 -c "import torch; c=torch.load('out-shakespeare-char/ckpt.pt', map_location='cpu'); print(f'최종 손실값: {c.get(\"best_val_loss\")}\n학습 횟수: {c.get(\"iter_num\")}')"
```

## 4. 생성 결과물 평가 및 심층 분석 (Sample Analysis)
`sample.py`를 통해 생성된 결과물에 대한 정성적 분석 결과입니다.

### 📊 분석 결과 요약
| 분석 항목 | 평가 | 세부 내용 |
| :--- | :--- | :--- |
| **구조적 재현율** | **우수 (High)** | `LUCIO:`, `ISABELLA:` 등 인물 이름과 대화 형식을 100% 가깝게 유지함. |
| **어휘적 유사도** | **우수 (High)** | `thou`, `shalt`, `thee`, `beseech` 등 셰익스피어 데이터셋의 고어 어휘를 적재적소에 배치함. |
| **문맥 일관성** | **보통 (Mid)** | 문장 간의 문법적 연결은 자연스러우나, 전체적인 대화의 주제 일관성은 10~20단어 이후 다소 낮아짐. |

### 🔍 심층 분석 내용
1. **캐릭터 단위 모델의 한계 극복**:
   - 단어 단위가 아닌 글자 단위(Character-level) 학습임에도 불구하고, 영어 단어의 철자 오류가 거의 발견되지 않음. 이는 1500번의 Iteration을 통해 모델이 영문 구조와 어휘를 충분히 내면화했음을 의미함.
   
2. **희곡 형식의 이해**:
   - 단순히 텍스트를 나열하는 것이 아니라, 대화 사이의 여백과 인물 전환을 이해하고 있음. 특히 `Clown:`, `First Citizen:` 등 단역 인물들의 명칭까지 데이터셋에서 학습하여 적절히 활용함.

3. **Loss 수치와의 상관관계**:
   - 최종 Val Loss **1.4686**은 모델이 문장의 기본 구조를 넘어서 '스타일'을 흉내 내기 시작하는 지점임을 결과물로 증명함. 1.0 이하로 내려갈 경우 문맥 유지 능력이 더 향상될 것으로 기대됨.

### ⚠️ 발견된 한계점
- **무한 루프 현상**: 드물게 특정 단어나 문장 구조(예: `LUCIO: ... LUCIO: ...`)가 반복되는 패턴이 나타남. 이는 모델 크기(10.6M)가 작아 복잡한 문맥 정보를 모두 담지 못해 발생하는 현상으로 판단됨.

## 5. 결론
RTX 2070은 nanoGPT급 모델의 학습과 추론에 매우 적합하며, 특히 `float16` 최적화를 통해 VRAM 효율을 높인 결과 안정적인 성능을 보여주었습니다.
