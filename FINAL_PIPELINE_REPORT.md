# Final Pipeline Report

## 1) 단위 테스트 실패 1건 보정

- 실패 원인: 시스템 프롬프트 문구 변경 이후 테스트 기대 문자열이 이전 값을 참조.
- 조치: `tests/test_audio_manager.py`의 기대 문자열을 최신 프롬프트 문구로 동기화.
- 결과: `python -m unittest -v tests.test_audio_manager` 전체 통과.

검증 결과 요약:
- 총 22개 테스트
- 실패 0, 에러 0
- 상태: OK

## 2) TTS 스트리밍 평활화 (문장 단위 버퍼링)

대상 파일: `audio_manager.py`

적용 로직:
- 스트리밍 delta 텍스트를 즉시 전달하지 않고 `sentence_buffer`에 누적.
- 문장 경계 인식 시점에만 TTS로 전달:
  - 종결 기호: `.`, `!`, `?`, `。`, `！`, `？`, 줄바꿈(`\n`)
  - 쉼표(`,`, `，`)는 문장이 너무 길어질 때(임계 길이 도달) 유동 분할
- 스트림 종료 시 버퍼 잔여 텍스트를 마지막으로 flush.
- 중복 문장은 직전 emitted 문장과 비교하여 제거.

가상 스트리밍 주입 런타임 검증 결과:
- 입력 청크는 형태소/단어 단위로 잘게 분할된 누적형 스트림 형태.
- 출력은 문장 단위로 묶여 전달됨:
  1. 안녕!
  2. 오늘은 날씨가 아주 좋아, 산책 가기 딱 좋아요!
  3. 물도 챙기고 모자도 쓰면 더 좋아요.
  4. 즐거운 하루 보내요.

해석:
- TTS 엔진이 단문/문장 단위 입력을 받도록 안정화되어 오디오 끊김 체감이 크게 줄어드는 구조로 개선됨.

## 3) QLoRA 커스텀 데이터셋 생성

대상 파일: `generate_dataset.py`

카테고리 구성:
- Category A: function_call (기기 제어/도구 호출 JSON)
- Category B: general_knowledge (일반 상식 자연어 답변)
- Category C: ood_refusal (모르는 질문 방어)

OOD 방어 문구(요청 반영):
- "헤헤, 미안해! 그건 지니가 아직 공부를 못해서 잘 몰라."
- "외계인 언어 번역해줘" 샘플 포함

생성 실행:
- `python generate_dataset.py --output jetson_dataset.jsonl --count 150`

생성 결과 검증:
- TOTAL: 150
- CATEGORIES:
  - function_call: 50
  - general_knowledge: 50
  - ood_refusal: 50

## 4) 추가 정합성 조치

- `train_qlora.py`에서 `Dataset` 중복 import 제거.
- 수정 파일 대상 compileall 통과.

## 5) 최종 상태

- 단위 테스트: PASSED (22/22)
- TTS 문장 버퍼링: 적용 및 런타임 로그 검증 완료
- 데이터셋: `jetson_dataset.jsonl` 생성 완료 (150쌍, 3카테고리 균형)
- 파이프라인 수렴 상태: 완료
