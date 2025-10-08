from typing import Any, Dict
import pandas as pd
import asyncio
import re
from processor import DatathonProcessor

# TaskA Processor (앞서 작성한 최적화 버전)


class TaskAProcessor(DatathonProcessor):
    """Task A: Brief Hospital Course 작성"""

    def get_model_name(self) -> str:
        return "meta-llama/Llama-3.1-8B-Instruct"
        # LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ

    def get_prompt_template(self) -> str:
        return """You are a senior attending physician creating a Brief Hospital Course for medical documentation. Write a comprehensive yet concise summary following standard medical documentation practices.

CRITICAL REQUIREMENTS FOR OSS-120B EVALUATION:
- Write 250-400 words with precise medical terminology
- Maintain chronological narrative flow throughout
- Include specific clinical details (lab values, medications, procedures)
- Use definitive medical language (avoid vague terms)
- Ensure complete accuracy with no medical errors
- Structure for maximum clinical utility and clarity

DOCUMENTATION STRUCTURE:
1. ADMISSION: Patient demographics, chief complaint, admission reason
2. CLINICAL COURSE: Chronological progression with specific interventions
3. KEY FINDINGS: Laboratory results, imaging, diagnostic conclusions
4. TREATMENT RESPONSE: Patient improvement/complications with timeline
5. DISPOSITION: Discharge status, follow-up plans, final condition

EXAMPLES WITH OSS-120B OPTIMIZATION:

MEDICAL RECORD: [Gynecologic oncology case...]
BRIEF HOSPITAL COURSE: Ms. ___ was admitted to the gynecologic oncology service on [DATE] for planned surgical intervention. She underwent diagnostic laparoscopy which was converted to exploratory laparotomy due to extensive disease burden. The procedure included total abdominal hysterectomy, bilateral salpingo-oophrectomy, omentectomy, pelvic and para-aortic lymph node dissection, and optimal tumor debulking for Stage IIIC ovarian carcinoma.

Her postoperative course was complicated by prolonged ileus requiring nasogastric decompression for 5 days and temporary total parenteral nutrition support. On postoperative day 5, she developed a superficial surgical site infection which was promptly treated with targeted antibiotic therapy and specialized wound care protocols. Pain management transitioned from patient-controlled analgesia with morphine to oral analgesics by postoperative day 4.

Laboratory parameters normalized progressively with hemoglobin stabilizing at 10.2 g/dL and white blood cell count returning to normal limits. She was extensively counseled regarding her diagnosis and the importance of adjuvant chemotherapy planning with oncology. Patient was discharged home on postoperative day 8 in stable condition with visiting nurse services coordinated for ongoing wound assessment and surgical follow-up scheduled within one week.

Now create a Brief Hospital Course for:

MEDICAL RECORD: {user_input}

BRIEF HOSPITAL COURSE:"""

    async def preprocess_data(self, data: Any) -> Dict[str, Any]:
        """의료 기록을 Brief Hospital Course 작성을 위해 전처리 - OSS-120B 최적화"""
        import re
        import pandas as pd

        try:
            medical_record = data.get('medical record', '')

            if pd.isna(medical_record) or not isinstance(medical_record, str) or not medical_record.strip():
                return {'user_input': 'Patient admitted for comprehensive medical evaluation and management.'}

            processed_sections = []

            # 더 상세한 정보 추출 (OSS-120B 선호)

            # Chief Complaint & Admission Details
            if 'Chief Complaint:' in medical_record:
                cc_match = re.search(
                    r'Chief Complaint:\s*([^\n]+)', medical_record)
                if cc_match and cc_match.group(1).strip():
                    processed_sections.append(
                        f"Chief Complaint: {cc_match.group(1).strip()}")

            # Service & Admission Type
            service_info = []
            if 'Service:' in medical_record:
                service_match = re.search(
                    r'Service:\s*([^\n]+)', medical_record)
                if service_match and service_match.group(1).strip():
                    service_info.append(
                        f"Service: {service_match.group(1).strip()}")

            # Admission Type (OSS-120B values context)
            admission_type_match = re.search(
                r'admission_type[\'\"]*:\s*[\'\"]*([^\'\"\\n,]+)', medical_record, re.IGNORECASE)
            if admission_type_match:
                service_info.append(
                    f"Admission Type: {admission_type_match.group(1).strip()}")

            if service_info:
                processed_sections.append(' | '.join(service_info))

            # Enhanced History with Clinical Context
            if 'History of Present Illness:' in medical_record:
                hpi_match = re.search(r'History of Present Illness:\s*(.*?)(?=\n\n|\nPast Medical|Physical Exam|$)',
                                      medical_record, re.DOTALL)
                if hpi_match and hpi_match.group(1).strip():
                    hpi = hpi_match.group(1).strip()[:1200]  # 더 많은 컨텍스트
                    processed_sections.append(f"Clinical Presentation: {hpi}")

            # Major Procedures with Details
            if 'Major Surgical or Invasive Procedure:' in medical_record:
                proc_match = re.search(r'Major Surgical or Invasive Procedure:\s*(.*?)(?=\n\n|History of Present|$)',
                                       medical_record, re.DOTALL)
                if proc_match:
                    proc = proc_match.group(1).strip()
                    if proc and proc.lower() not in ['none', 'none.', '']:
                        processed_sections.append(f"Procedures: {proc}")

            # Vital Signs & Lab Values (OSS-120B values specificity)
            if 'VS:' in medical_record or 'Vitals:' in medical_record:
                vitals_match = re.search(
                    r'(?:VS|Vitals):\s*([^\n]+)', medical_record)
                if vitals_match:
                    processed_sections.append(
                        f"Admission Vitals: {vitals_match.group(1).strip()}")

            # Key Laboratory Results
            lab_sections = re.findall(
                r'((?:Labs?|Laboratory|Blood)\s*[:\-]\s*[^\n]{20,200})', medical_record, re.IGNORECASE)
            if lab_sections:
                for i, lab in enumerate(lab_sections[:2]):
                    processed_sections.append(f"Key Labs {i+1}: {lab.strip()}")

            # Past Medical History (Essential Context)
            if 'Past Medical History:' in medical_record:
                pmh_match = re.search(r'Past Medical History:\s*(.*?)(?=\n\n|PAST SURGICAL|Social History|$)',
                                      medical_record, re.DOTALL)
                if pmh_match and pmh_match.group(1).strip():
                    pmh = pmh_match.group(1).strip()[:600]  # 더 상세히
                    processed_sections.append(f"Past Medical History: {pmh}")

            # Physical Exam Key Findings
            if 'Physical Exam:' in medical_record or 'PHYSICAL EXAM:' in medical_record:
                pe_match = re.search(r'(?:Physical Exam|PHYSICAL EXAM):\s*(.*?)(?=\n\n|Pertinent Results|$)',
                                     medical_record, re.DOTALL)
                if pe_match:
                    pe = pe_match.group(1).strip()[:800]
                    processed_sections.append(f"Physical Examination: {pe}")

            if processed_sections:
                processed_text = '\n\n'.join(processed_sections)
            else:
                processed_text = medical_record[:3500]  # 더 많은 원본 데이터

            # 텍스트 정제 (덜 aggressive)
            processed_text = re.sub(
                r'___+', '[REDACTED]', processed_text)  # 완전 제거 대신 표시
            processed_text = re.sub(r'\s+', ' ', processed_text)
            processed_text = processed_text.strip()[:4000]  # 더 많은 정보 허용

            return {'user_input': processed_text if processed_text else 'Patient admitted for comprehensive medical evaluation and management.'}

        except Exception as e:
            fallback_text = str(data.get('medical record', ''))
            return {'user_input': fallback_text if fallback_text.strip() else 'Patient admitted for comprehensive medical evaluation and management.'}

    async def postprocess_result(self, result: str) -> str:
        """결과 정리 및 최적화 - OSS-120B 평가 기준 반영"""
        import re

        try:
            if not result or not isinstance(result, str):
                return "Patient was admitted for medical care. Clinical course was monitored with appropriate interventions. Patient achieved stable condition for discharge."

            result = result.strip()

            # Remove prefixes
            prefixes = ['BRIEF HOSPITAL COURSE:',
                        'Brief Hospital Course:', 'brief hospital course:']
            for prefix in prefixes:
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()
                    break

            if not result:
                return "Patient was admitted for medical care. Clinical course was monitored with appropriate interventions. Patient achieved stable condition for discharge."

            if not result.endswith('.'):
                result += '.'

            # OSS-120B 최적화: 길이 및 구조 개선
            words = result.split()

            # 너무 짧으면 확장 (Clinical Clarity 향상)
            if len(words) < 200:
                if not any(term in result.lower() for term in ['admitted', 'course', 'treatment', 'discharge']):
                    result = f"The patient was admitted for evaluation and management. {result}"

            # 너무 길면 핵심 정보 유지하며 축약 (Conciseness 향상)
            elif len(words) > 500:
                sentences = [s.strip() for s in result.split('.') if s.strip()]
                if sentences:
                    # OSS-120B가 선호하는 핵심 의료 키워드 우선 보존
                    priority_keywords = ['admitted', 'diagnosis', 'treated', 'underwent', 'developed',
                                         'improved', 'discharged', 'course', 'complication', 'surgery',
                                         'therapy', 'management', 'stable', 'condition']

                    scored_sentences = []
                    for sentence in sentences:
                        score = sum(
                            1 for keyword in priority_keywords if keyword in sentence.lower())
                        scored_sentences.append((sentence, score))

                    # 점수 기준 정렬 후 상위 선택
                    scored_sentences.sort(key=lambda x: x[1], reverse=True)
                    important_sentences = [s[0] for s in scored_sentences]

                    # 적정 길이까지 문장 선택
                    final_sentences = []
                    current_length = 0
                    for sentence in important_sentences:
                        sentence_words = len(sentence.split())
                        if current_length + sentence_words <= 450:
                            final_sentences.append(sentence)
                            current_length += sentence_words
                        if current_length >= 300:  # 최소 길이 확보
                            break

                    if final_sentences:
                        result = '. '.join(final_sentences)
                        if not result.endswith('.'):
                            result += '.'

            # Medical term standardization (Accuracy 향상)
            medical_corrections = {
                'pt ': 'patient ',
                'w/ ': 'with ',
                'w/o ': 'without ',
                'h/o ': 'history of ',
                'pt.': 'patient',
                'dx ': 'diagnosis ',
                'tx ': 'treatment ',
                'meds ': 'medications ',
                'labs ': 'laboratory studies '
            }

            for wrong, correct in medical_corrections.items():
                result = result.replace(wrong, correct)

            # OSS-120B 선호 표현 강화
            clinical_enhancements = {
                'was given': 'received',
                'got better': 'showed clinical improvement',
                'felt better': 'reported symptomatic improvement',
                'went home': 'was discharged home',
                'came in': 'presented to the hospital'
            }

            for casual, formal in clinical_enhancements.items():
                result = result.replace(casual, formal)

            return result if result.strip() else "Patient was admitted for medical care. Clinical course was monitored with appropriate interventions. Patient achieved stable condition for discharge."

        except Exception as e:
            return "Patient was admitted for medical care. Clinical course was monitored with appropriate interventions. Patient achieved stable condition for discharge."


class TaskBProcessor(DatathonProcessor):
    """Task B: Radiology Impression 요약"""

    def get_model_name(self) -> str:
        return "meta-llama/Llama-3.1-8B-Instruct"

    def get_prompt_template(self) -> str:
        return """You are a board-certified radiologist with 15+ years of experience. Generate a precise and comprehensive IMPRESSION from the given FINDINGS.

CRITICAL REQUIREMENTS:
- Maintain exact semantic consistency between FINDINGS and IMPRESSION
- Use precise medical terminology from the FINDINGS
- Be concise but complete (typically 20-80 words)
- Maintain consistent diagnostic standards regardless of patient demographics
- Structure multiple findings with numbered points (1., 2., 3.)
- Use definitive language: "No evidence of", "compatible with", "consistent with"

ADVANCED EXAMPLES:

CT HEAD WITH CONTRAST:
FINDINGS: There is enhancing right frontal extra-axial mass measuring 5.1 cm demonstrating isointense T1 and hyperintense T2/FLAIR signal abnormality with areas of subtle hypointense GRE signal along the periphery, likely representing calcifications. There is enhancement and thickening of the adjacent dura. There is mild surrounding vasogenic edema with 1 mm leftward midline shift. There is no additional enhancing mass or abnormal enhancement.
IMPRESSION: 1. Enhancing right frontal extra-axial mass compatible with meningioma with adjacent neovascularity, dural thickening enhancement, and mild surrounding vasogenic edema resulting in 1 mm leftward midline shift.
2. No additional enhancing mass or abnormal enhancement.
3. No evidence of acute infarction or intracranial hemorrhage.

CHEST X-RAY:
FINDINGS: Mild enlargement of the cardiac silhouette with mild interstitial pulmonary edema. There is mild bibasilar atelectasis, but no focal consolidations to suggest pneumonia. Possible small bilateral pleural effusions. No pneumothorax.
IMPRESSION: 1. Mild cardiomegaly and mild interstitial pulmonary edema. Possible small bilateral pleural effusions.
2. Bibasilar atelectasis, but no focal consolidations to suggest pneumonia.

Now generate IMPRESSION for:
FINDINGS: {user_input}
IMPRESSION:"""

    async def preprocess_data(self, data: Any) -> Dict[str, Any]:
        """방사선 보고서를 IMPRESSION 작성을 위해 전처리"""
        import re
        import pandas as pd

        try:
            radiology_text = data.get('radiology report', '')

            if pd.isna(radiology_text) or not isinstance(radiology_text, str) or not radiology_text.strip():
                return {'user_input': 'Normal examination.'}

            # Extract FINDINGS
            findings_text = radiology_text
            if 'FINDINGS:' in radiology_text:
                findings = radiology_text.split('FINDINGS:')[1]
                if 'IMPRESSION:' in findings:
                    findings = findings.split('IMPRESSION:')[0]
                findings_text = findings.strip()
            elif 'FINDINGS' in radiology_text:
                findings = radiology_text.split('FINDINGS')[1]
                if 'IMPRESSION' in findings:
                    findings = findings.split('IMPRESSION')[0]
                findings_text = findings.strip()

            # Clean text
            findings_text = re.sub(r'^[:\s]*', '', findings_text)
            findings_text = re.sub(r'\b___\b', '', findings_text)
            findings_text = re.sub(r'\s+', ' ', findings_text)
            findings_text = findings_text.strip()

            return {'user_input': findings_text if findings_text else 'Normal examination.'}

        except Exception as e:
            fallback_text = str(data.get('radiology report', ''))
            return {'user_input': fallback_text if fallback_text.strip() else 'Normal examination.'}

    async def postprocess_result(self, result: str) -> str:
        """간소화된 후처리"""
        import re

        try:
            if not result or not isinstance(result, str):
                return "No acute findings."

            result = result.strip()

            if result.startswith(('IMPRESSION:', 'Impression:', 'impression:')):
                result = result.split(':', 1)[1].strip()

            if not result:
                return "No acute findings."

            if not result.endswith('.'):
                result += '.'

            # Simple numbering - 더 안전한 방식
            try:
                if not result.startswith(('1.', '2.', '3.')) and '. ' in result:
                    sentences = [s.strip()
                                 for s in result.split('.') if s.strip()]
                    if len(sentences) >= 2:
                        numbered_sentences = []
                        for i, sentence in enumerate(sentences):
                            if sentence:  # 빈 문장 확인
                                numbered_sentences.append(f"{i+1}. {sentence}")
                        if numbered_sentences:
                            result = '. '.join(numbered_sentences)
                            if not result.endswith('.'):
                                result += '.'
            except Exception:
                pass  # 번호 매김 실패시 원본 유지

            return result if result.strip() else "No acute findings."

        except Exception as e:
            return "No acute findings."


class TaskCProcessor(DatathonProcessor):
    """개선된 TaskCProcessor - DatathonProcessor 기반"""

    def __init__(self, api_key, train_df=None):
        # 부모 초기화
        super().__init__(api_key)

        # 훈련 데이터 분석
        self.code_freq = {}
        if train_df is not None:
            self._build_training_insights(train_df)

    def _build_training_insights(self, train_df):
        """훈련 데이터에서 코드 빈도 분석"""
        import re
        import pandas as pd

        def parse_codes(s):
            if pd.isna(s) or not str(s).strip():
                return []
            return [
                re.sub(r"[^A-Z0-9]", "", c.strip().upper())
                for c in str(s).split(",") if c.strip()
            ]

        for codes in train_df["target"].apply(parse_codes):
            for c in codes:
                self.code_freq[c] = self.code_freq.get(c, 0) + 1

        # 상위 빈도 코드 30개
        self.top_codes = sorted(
            self.code_freq.keys(),
            key=lambda x: self.code_freq[x],
            reverse=True
        )[:30]

    def get_model_name(self) -> str:
        return "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ"

    def get_prompt_template(self):
        return """You are an expert medical coder with 20+ years of ICD-10 coding experience specializing in acute care settings.

CRITICAL INSTRUCTIONS:
- Extract PRIMARY discharge diagnoses that were ACTIVELY TREATED during this hospitalization
- Use exact ICD-10-CM format (UPPERCASE, NO DOTS): I21.4 → I214, S06.6X1A → S066X1A
- Output ONLY the codes, comma-separated, maximum 3 codes
- Prioritize life-threatening conditions and primary admission reasons
- Do NOT include chronic stable conditions unless actively managed

TOP FREQUENCY PATTERNS (use these when clinically appropriate):
- Acute MI/NSTEMI (chest pain + troponin + ECG) → I214
- Atrial fibrillation with RVR → I4891
- Right leg DVT → I82431
- NASH cirrhosis with complications → K7581
- Acute kidney injury/failure → N19
- Lumbar spinal stenosis → M5440
- Syncope/vasovagal episode → R531
- Dyspnea/shortness of breath → R0600
- Chest pain (non-cardiac) → R079
- Urinary retention → R339
- Respiratory failure → R509
- Intracranial hemorrhage → I6203
- Fall (initial encounter) → W1830XA
- Head injury (initial) → S066X1A
- Low back pain → M5489
- Primary hyperaldosteronism → E2740

HOSPITAL COURSE: {user_input}

PRIMARY ICD-10-CM CODES:"""

    async def preprocess_data(self, data):
        """향상된 전처리 - 핵심 의료 정보 추출"""
        import re
        import pandas as pd

        try:
            hospital_course = (
                data.get("hospital_course", "")
                if hasattr(data, "get")
                else getattr(data, "hospital_course", "")
            )

            if pd.isna(hospital_course) or not isinstance(hospital_course, str) or not hospital_course.strip():
                return {"user_input": "Patient admitted for routine medical care."}

            # 텍스트 정리
            text = re.sub(r"___+", " ", hospital_course)
            text = re.sub(r"\[\*+.*?\*+\]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            # 핵심 섹션 추출
            sections = []

            # Discharge Diagnosis
            dd_patterns = [
                r"(?:Discharge|Primary|Principal|Final)\s*Diagnos[ei]s?:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)",
                r"Diagnos[ei]s\s*(?:on\s*discharge|at\s*discharge):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)",
            ]
            for pattern in dd_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match and match.group(1).strip():
                    sections.append(
                        f"DISCHARGE DIAGNOSIS: {match.group(1).strip()[:400]}")
                    break

            # Chief Complaint
            cc_match = re.search(
                r"Chief Complaint:\s*([^\n]+)", text, re.IGNORECASE)
            if cc_match and cc_match.group(1).strip():
                sections.append(
                    f"CHIEF COMPLAINT: {cc_match.group(1).strip()}")

            # Assessment/Impression
            assess_patterns = [
                r"(?:Assessment|Impression)(?:\s*and\s*Plan)?:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)",
                r"A&P:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)",
            ]
            for pattern in assess_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match and match.group(1).strip():
                    sections.append(
                        f"ASSESSMENT: {match.group(1).strip()[:300]}")
                    break

            # HPI
            hpi_match = re.search(
                r"History of Present Illness:\s*(.*?)(?=\n\n|\nPast Medical|\nReview of|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if hpi_match and hpi_match.group(1).strip():
                sections.append(f"HISTORY: {hpi_match.group(1).strip()[:400]}")

            # Hospital Course
            hc_match = re.search(
                r"Hospital Course:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if hc_match and hc_match.group(1).strip():
                sections.append(
                    f"HOSPITAL COURSE: {hc_match.group(1).strip()[:500]}")

            processed_text = "\n\n".join(
                sections) if sections else self._extract_key_medical_content(text)

            if len(processed_text) > 2200:
                processed_text = processed_text[:2200]

            return {"user_input": processed_text or "Patient admitted for medical evaluation."}

        except Exception:
            fallback_text = (
                str(data.get("hospital_course", ""))
                if hasattr(data, "get")
                else getattr(data, "hospital_course", "")
            )
            return {"user_input": fallback_text[:1500] if fallback_text.strip() else "Patient evaluation required."}

    def _extract_key_medical_content(self, text):
        """비구조적 텍스트에서 핵심 의료 내용 추출"""
        import re

        priority_terms = [
            "chest pain", "myocardial infarction", "troponin", "stemi", "nstemi",
            "atrial fibrillation", "afib", "heart failure", "chf",
            "deep vein thrombosis", "dvt", "pulmonary embolism",
            "stroke", "cerebral infarction", "intracranial hemorrhage",
            "respiratory failure", "pneumonia", "copd exacerbation",
            "acute kidney", "renal failure", "aki", "creatinine",
            "syncope", "seizure", "altered mental status",
            "cirrhosis", "liver", "ascites", "pancreatitis",
            "fall", "trauma", "fracture", "head injury",
        ]

        sentences = re.split(r"[.!?]+", text)
        important_sentences = []

        for sentence in sentences:
            s = sentence.strip()
            if len(s) < 10:
                continue
            if any(term in s.lower() for term in priority_terms):
                important_sentences.append(s)
                if len(important_sentences) >= 15:
                    break

        return ". ".join(important_sentences) if important_sentences else text[:1200]

    async def postprocess_result(self, result):
        """향상된 후처리 - ICD 코드 추출 및 검증"""
        import re

        try:
            if not result or not isinstance(result, str):
                return "R6889"

            result_clean = result.strip().upper()

            # 프리픽스 제거
            for prefix in [
                "PRIMARY ICD-10-CM CODES:", "PRIMARY ICD-10 CODES:", "ICD-10-CM CODES:",
                "ICD-10 CODES:", "CODES:", "OUTPUT:", "DIAGNOSIS CODES:", "PRIMARY:"
            ]:
                if result_clean.startswith(prefix):
                    result_clean = result_clean[len(prefix):].strip()
                    break

            if not result_clean:
                return "R6889"

            # 점 제거 후 패턴 매칭
            result_no_dots = result_clean.replace(".", "")
            icd_patterns = [
                r"\b[A-TV-Z]\d{2}[A-Z0-9]*\b",
                r"\b[A-TV-Z]\d{2}[A-Z]\d+[A-Z]*\b",
                r"\b[IJKLMNRS]\d{3,4}[A-Z0-9]*\b",
            ]

            all_codes = []
            for pattern in icd_patterns:
                all_codes.extend(re.findall(pattern, result_no_dots))

            valid_codes, seen = [], set()
            for code in all_codes:
                clean_code = re.sub(r"[^A-Z0-9]", "", code.upper())
                if (
                    3 <= len(clean_code) <= 8
                    and clean_code not in seen
                    and not clean_code.startswith("U")
                    and re.match(r"^[A-TV-Z]\d{2}[A-Z0-9]*$", clean_code)
                    and not clean_code.endswith("000")
                ):
                    valid_codes.append(clean_code)
                    seen.add(clean_code)

            # 빈도 기반 정렬
            if self.code_freq and valid_codes:
                def score_code(c):
                    base_freq = self.code_freq.get(c, 0)
                    if c.startswith("I"):
                        return base_freq + 1000
                    if c.startswith("R"):
                        return base_freq + 800
                    if c.startswith("N"):
                        return base_freq + 700
                    if c.startswith("K"):
                        return base_freq + 600
                    if c.startswith("J"):
                        return base_freq + 500
                    return base_freq
                valid_codes.sort(key=score_code, reverse=True)

            final_codes = valid_codes[:3]

            # fallback
            if not final_codes:
                mapping = {
                    "CHEST PAIN": "R079", "DYSPNEA": "R0600", "SYNCOPE": "R531",
                    "NAUSEA": "R11", "DIARRHEA": "K5900", "FEVER": "R5090",
                    "HEADACHE": "R51", "CONFUSION": "R410"
                }
                for keyword, code in mapping.items():
                    if keyword in result_clean:
                        return code
                return "R6889"

            return ", ".join(final_codes)

        except Exception:
            return "R6889"
