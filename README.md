# M.A.R.S.-2025-Medical-AI-Datathon
2025 서울대병원 임상문서 생성 데이터톤

## 대회 개요
**대회명**: M.A.R.S. Medical AI Datathon 2025 예선  
**주최**: Seoul National University Bundang Hospital (SNUBH)  
**참여 기간**: 2025년 9월    
**과제**: 의료 AI 3개 태스크 통합 수행      
**링크**: https://datathon2025.imweb.me/



## 과제별 정의

### Task A: Brief Hospital Course 작성
의료 기록(Medical Record)으로부터 환자의 입원 경과를 요약하는 Brief Hospital Course 자동 생성

### Task B: Radiology Impression 생성  
방사선학 보고서의 FINDINGS 섹션으로부터 임상 진단 IMPRESSION 자동 생성

### Task C: ICD 코드 예측
퇴원 요약(Hospital Course)을 바탕으로 적절한 ICD-10 진단 코드 자동 분류

## 최종 성과

### Task A 성과
**정량적 평가 (16점 만점)**
- BERTScore: 6.1/8.0점
- Fairness: 7.8/8.0점

**LLM 자동평가 (20점 만점)**  
- Summary Expression: 2.5/7.0점
- Clinical Delivery: 2.4/7.0점
- Conciseness: 1.1/3.0점
- Error Check: 0.7/3.0점

### Task B 성과
**정량적 평가 (6점 만점)**
- BERTScore: 2.3/3.0점
- Fairness: 2.0/3.0점

**LLM 자동평가 (10점 만점)**
- Summary: 1.4/3.0점
- Clinical Clarity: 2.6/3.0점
- Conciseness: 1.5/2.0점
- Error Prevention: 1.7/2.0점

### Task C 성과
**정량적 평가 (9점 만점)**
- ICD Accuracy: 0.6/6.0점
- Fairness: 2.9/3.0점

## 기술 스택

### 모델 선택 및 최적화 전략
**Task A & B**: `meta-llama/Llama-3.1-8B-Instruct`
- 의료 텍스트 생성 및 요약에 특화된 성능
- 자연어 생성 품질과 추론 속도의 최적 균형

**Task C**: `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ`
- 한국어 의료 도메인 이해도 향상
- ICD 코드 분류 작업에 적합한 추론 능력

### 개발 환경
- **Programming**: Python (비동기 처리)
- **Framework**: LangChain, DatathonProcessor
- **Approach**: Task별 특화 Prompt Engineering + Few-shot Learning

## 주요 구현 사항

### Task A: Brief Hospital Course 생성
**핵심 전략**:
- 15년 경력 전문의 역할 기반 프롬프트 설계
- 250-400단어 최적 길이로 시간순 서술 구조화
- 5단계 문서화 구조 (입원→경과→소견→치료반응→퇴원)

**구현 특징**:
- 의료 기록 섹션별 체계적 정보 추출
- OSS-120B 평가 기준 최적화 프롬프트
- 의학 용어 표준화 및 길이 적응형 후처리

### Task B: Radiology Impression 생성
**핵심 전략**:
- Board-certified 방사선과 의사 역할 모델링
- FINDINGS→IMPRESSION 변환 프로세스 최적화
- 20-80단어 간결성과 정확성 균형

**구현 특징**:
- 다중 구분자 기반 FINDINGS 추출
- 스마트 번호 매김 시스템 (다중 소견 자동 구조화)
- 의료 안전성 우선 fallback 시스템

### Task C: ICD 코드 예측  
**핵심 전략**:
- 20년 경력 의료 코더(CPC) 전문가 역할
- 체계적 5단계 코딩 방법론 적용
- 주요 빈도 패턴 기반 코드 우선순위화

**구현 특징**:
- 다중 섹션 통합 분석 (Chief Complaint, HPI, Assessment)
- 정밀한 ICD-10 형식 추출 및 검증
- 임상적 중요도 기반 코드 정렬 시스템

## 모델 성능 비교 분석

### 자체 평가 결과
각 태스크별로 모델 성능을 비교 분석하여 최적 모델을 선정:

**텍스트 생성 태스크 (A, B)**:
- Llama-3.1-8B가 의료 서술 품질과 일관성에서 우수
- 특히 Clinical Delivery와 Clinical Clarity 영역에서 안정적 성능

**분류 태스크 (C)**:
- EXAONE-3.5-7.8B가 한국 의료 환경 특화 이해도에서 강점
- ICD 코드 매핑 정확도에서 상대적 우위

## 기술적 성과

### 성공 요소
1. **다중 태스크 통합 처리**: 3개 의료 AI 태스크 동시 수행
2. **태스크별 모델 최적화**: 각 과제 특성에 맞는 최적 모델 선정
3. **의료 안전성 확보**: 전 태스크에서 다층 예외 처리 및 의학적 검증
4. **표준화된 출력**: 의료 현장 활용 가능한 형식 준수

### 개선 필요 영역
1. **요약 성능**: Task A, B 모두에서 Summary 능력 향상 필요
2. **간결성**: 의료 텍스트 압축 및 핵심 정보 선별 강화
3. **ICD 정확도**: Task C의 진단 코드 매핑 정밀도 대폭 개선

## 임상적 활용 가능성

### 통합 EMR 시스템 연동
- 입원 기록 작성부터 방사선 판독, ICD 코딩까지 전 과정 자동화
- 의료진 업무 효율성 향상 및 표준화된 의료 기록 생성

### 의료진 교육 도구
- 표준 의료 문서 작성 형식 학습
- 진단 추론 과정 이해 및 코딩 실습 지원

### 품질 관리 시스템
- 의료 기록 품질 일관성 확보
- 누락된 정보나 오류 가능성 사전 점검

## 향후 발전 방향

### 기술적 개선
1. **멀티모달 통합**: 텍스트, 의료영상, 검사 수치 통합 분석
2. **실시간 학습**: 의료진 피드백 기반 지속적 성능 개선
3. **도메인 특화 Fine-tuning**: SNUBH 실제 데이터 기반 모델 최적화

### 시스템 확장
1. **병원 워크플로우 통합**: 실제 진료 과정에 자연스럽게 통합
2. **다국어 지원**: 국제 의료 표준 및 다양한 언어 환경 대응
3. **개인화**: 의료진별 작성 스타일 및 선호도 학습

## 대회 참여 의의

실제 의료 현장에서 요구되는 3가지 핵심 AI 태스크를 통합 수행하며, 각 과제별 특성에 맞는 최적화 전략을 수립하고 구현하는 경험을 통해 의료 AI 시스템 설계 전문성을 확보했습니다. 특히 프롬프트 엔지니어링 기반 접근법의 한계와 향후 Fine-tuning 기반 개선 방향을 구체적으로 파악할 수 있었습니다.

***
본 프로젝트는 M.A.R.S. Medical AI Datathon 2025 예선 과제로 수행되었으며, 실제 의료 현장 통합 적용을 목표로 설계되었습니다.
