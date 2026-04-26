# 실험 시스템 설정 및 로깅 가이드 (System Setup)

본 문서는 모든 nanoGPT 실험에 공통적으로 적용되는 하드웨어 최적화 설정과 로깅 방식을 정의합니다.

## 1. 진행도 시각화 (tqdm 도입)
학습 과정에서 남은 시간과 진행률을 실시간으로 확인하기 위해 \	qdm\ 라이브러리를 사용합니다.

- **설정 방식**: \	rain.py\의 메인 학습 루프를 \	qdm\으로 래핑(Wrapping)하여 사용.
- **표시 항목**: 현재 Iteration, Loss, 학습 속도(it/s), 예상 완료 시간(ETA).

## 2. 로그 관리 및 저장 (Logging)
터미널에 출력되는 모든 로그는 분석을 위해 자동으로 파일에 저장됩니다.

- **명령어 예시**: \python train.py ... | tee -a docs/test/experiment_log.txt- **로그 구조**:
    - 시스템 사양 (GPU VRAM 등)
    - 하이퍼파라미터 설정값
    - Iteration별 Loss 및 시간 데이터

## 3. 실험 시간 측정 기준
이전 실험(Shakespeare-char) 결과 분석을 바탕으로 다음과 같은 기준을 적용합니다.

- **단기 실험**: 1시간 이내 (Finetuning 등)
- **중기 실험**: 3~12시간 (Small model pretraining)
- **장기 실험**: 1일 이상 (Full pretraining)

---
*마지막 업데이트: 2026-04-26*
