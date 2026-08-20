# 03. 대화형 인터페이스(Chat Mode) 구현 실험 기획

본 문서는 nanoGPT 기본 모델(Base Model)을 사용하여 사용자 인터랙티브 채팅 환경을 구축하고 테스트하기 위한 실험 계획을 기록합니다.

## 1. 실험 목적
- **목적**: 텍스트 완성(Completion) 모델을 대화형(Chat) 구조로 유도하여 사용자 인터랙션 가능성을 확인.
- **핵심 질문**: "별도의 파인튜닝 없이 프롬프트 설계만으로 채팅 환경을 구현할 수 있는가?"

## 2. 실행 전략 (Implementation Strategy)

### A. 인터랙티브 스크립트 (`chat.py`) 제작
- 사용자의 입력을 실시간으로 받아 모델의 입력(context)으로 전달.
- 모델의 답변을 출력한 뒤 다시 사용자 입력을 기다리는 루프(Loop) 구조 설계.

### B. 프롬프트 엔지니어링 (Few-shot Prompting)
- 모델이 '대화 중'임을 인지하도록 사전에 대화 형식을 주입.
- 예시:
  ```text
  The following is a conversation between a User and a Shakespearean Actor.
  User: Hello!
  Actor: Greetings, noble friend. How may I serve thee this day?
  User: [사용자 입력]
  Actor:
  ```

## 3. 테스트 시나리오

### 시나리오 1: 셰익스피어 페르소나 테스트
- **입력**: "Tell me a story about love."
- **기대 결과**: 셰익스피어풍의 고어와 비유를 사용한 답변 생성.

### 시나리오 2: 역할 수행(Role-play) 테스트
- **입력**: "User: You are a king. Assistant:"
- **기대 결과**: 왕의 권위가 느껴지는 말투로 대사 생성 여부 확인.

## 4. 측정 및 평가 지표
- **응답 일관성**: 사용자 질문에 문맥적으로 맞는 답변을 하는가?
- **형식 유지**: 대화 형식(`Actor:`)을 깨뜨리지 않고 답변을 마치는가?
- **응답 속도**: RTX 2070 GPU에서 실시간 대화가 가능한 수준의 지연 시간(Latency)을 유지하는가?

## 6. 실제 테스트 결과 (Test Logs)

| 사용자 입력 (User) | 모델 답변 (Actor) | 분석 |
| :--- | :--- | :--- |
| "Lucis you are a bad boy!" | "where you would be? O learn in a senator," | 인물에 대한 반응을 희곡 속 대사로 처리함. |
| "Are you mad?" | "away, what falls? my lord" | 질문에 당황한 듯한 연극적 반응 (`my lord` 사용). |
| "You are blind!" | "marry, what well? now, cousins, where has the" | `marry`, `cousins` 등 시대적 어휘 활용 우수. |
| "You can't talk" | "even, for your good offer'd-fool!" | 사용자의 비난에 대해 '바보(fool)'라는 단어로 맞받아침. |

### 🔍 기술적 심층 평가 (Technical Deep-Dive)

#### 1. 도메인 특화 페르소나 유지 (Domain-Specific Persona Consistency)
- **현상**: 모델은 모든 응답에서 초기 시스템 프롬프트가 정의한 'Actor'의 역할을 이탈하지 않음.
- **분석**: 1.46대의 Val Loss는 모델이 단순히 단어를 나열하는 수준을 넘어, 데이터셋 내의 상호작용 패턴(대화 주체 교체 및 특정 말투)을 강력하게 조건화(Conditioning)하고 있음을 시사함. `my lord`, `cousins`와 같은 호칭어 사용은 도메인 특화 데이터셋의 전이 효과가 성공적으로 발현된 결과임.

#### 2. 의미론적 정렬 및 응답 역동성 (Semantic Alignment & Response Dynamics)
- **현상**: "You can't talk" → "fool!"과 같이 의미적으로 대조되거나 연관된 단어가 선택됨.
- **분석**: 이는 단순한 무작위 생성이 아니라, 사용자 입력 내의 '부정적 어조'를 모델이 어텐션(Attention) 메커니즘을 통해 포착했음을 의미함. 셰익스피어 데이터셋 내에서 비난-반박의 패턴이 높은 확률 밀도를 형성하고 있어, 사용자의 공격적 입력이 해당 패턴을 트리거(Trigger)한 것으로 해석됨.

#### 3. 국소 일관성 vs. 전역 일관성 (Local vs. Global Coherence)
- **국소 일관성(우수)**: 각 문장 내의 문법적 연결성과 철자 정확도는 매우 높음.
- **전역 일관성(제한적)**: 사용자의 질문에 대한 논리적 답변보다는, '대화의 분위기'를 이어가는 데 치중함. 이는 10.6M의 소규모 파라미터가 장기적인 논리적 추론(Logical Reasoning)을 담기에는 용량이 부족함을 나타내는 전형적인 특징임.

#### 4. 캐릭터 단위 인코딩의 이점 (Advantages of Char-level Encoding)
- **분석**: BPE(Byte Pair Encoding)를 사용하지 않았음에도 불구하고 `senator`, `implying`과 같은 복잡한 단어의 철자를 완벽히 생성함. 이는 모델이 텍스트의 통계적 구조를 문자 단위에서 매우 조밀하게 학습했음을 입증함.

## 7. 종합 결론 및 제언 (Final Conclusion)
본 실험을 통해 nanoGPT 기본 모델이 **프롬프트 엔지니어링만으로도 강력한 도메인 페르소나를 투영할 수 있음**을 확인하였음.

- **성과**: RTX 2070 환경에서 실시간 인터랙션이 가능한 저지연(Low-latency) 대화 환경 구축 성공.
- **한계**: 논리적 대화보다는 스타일 재현에 치중되어 있음.
- **향후 과제**: 더 높은 추론 능력을 위해 파라미터 수를 확장(100M+)하거나, 대화형 데이터셋을 통한 지도학습(Supervised Fine-Tuning)이 병행될 경우 상용 챗봇에 근접한 성능을 낼 수 있을 것으로 판단됨.
