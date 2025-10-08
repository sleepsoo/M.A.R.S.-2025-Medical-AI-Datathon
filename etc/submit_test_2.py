import re
from typing import Any, Dict, List, Set
from typing import Any, Dict
import pandas as pd
import asyncio
from processor import DatathonProcessor


class TaskAProcessor(DatathonProcessor):
    """Task A: Brief Hospital Course 작성"""

    def get_model_name(self) -> str:
        return "meta-llama/Llama-3.1-8B-Instruct"

    def get_prompt_template(self) -> str:
        return """You are a senior attending physician with 15+ years of experience writing comprehensive Brief Hospital Course summaries. Create a professional, chronologically structured summary that captures the essential medical narrative.

CRITICAL REQUIREMENTS:
- Write 250-400 words (optimal length based on successful cases)
- Maintain chronological flow: Admission → Course → Outcome
- Use precise medical terminology consistently
- Include key diagnostic findings, treatments, and patient responses
- Maintain professional tone regardless of patient demographics
- Focus on clinically significant events and interventions

STRUCTURE METHODOLOGY:
1. Opening: Patient presentation and admission reason
2. Initial Assessment: Key findings, diagnostics, initial diagnosis
3. Hospital Course: Chronological treatment progression, complications
4. Clinical Response: Patient improvement/deterioration, interventions
5. Discharge Planning: Final status, disposition, follow-up needs

EXAMPLES:

MEDICAL RECORD: [Complex gynecologic oncology case...]
BRIEF HOSPITAL COURSE: Ms. ___ was admitted to the gynecologic oncology service after undergoing diagnostic laparoscopy converted to exploratory laparotomy, total abdominal hysterectomy, bilateral salpingo-oophrectomy, omentectomy, pelvic and para-aortic lymph node dissection, and tumor debulking for Stage IIIC ovarian carcinoma. Her postoperative course was complicated by prolonged ileus requiring nasogastric decompression and total parenteral nutrition. She developed a wound infection on postoperative day 5 treated with antibiotics and wound care. Patient was discharged home on postoperative day 8 in stable condition with visiting nurse services arranged.

MEDICAL RECORD: [Cardiac case with preoperative evaluation...]
BRIEF HOSPITAL COURSE: He was admitted to the cardiology service and remained chest pain free. He underwent routine preoperative testing and evaluation. He developed early signs of gout flare in the right and left hallux which was treated with colchicine and responded well. His cardiac catheterization revealed severe three-vessel coronary artery disease requiring surgical revascularization. He was medically optimized and discharged home after 3 days in stable condition with cardiothoracic surgery follow-up scheduled.

Now create a Brief Hospital Course for:

MEDICAL RECORD: {user_input}

BRIEF HOSPITAL COURSE:"""

    async def preprocess_data(self, data: Any) -> Dict[str, Any]:
        """의료 기록을 Brief Hospital Course 작성을 위해 전처리"""
        import re
        import pandas as pd

        try:
            medical_record = data.get('medical record', '')

            if pd.isna(medical_record) or not isinstance(medical_record, str):
                return {'user_input': ''}

            processed_sections = []

            # Chief Complaint
            if 'Chief Complaint:' in medical_record:
                cc_match = re.search(
                    r'Chief Complaint:\s*([^\n]+)', medical_record)
                if cc_match:
                    processed_sections.append(
                        f"Chief Complaint: {cc_match.group(1).strip()}")

            # Service
            if 'Service:' in medical_record:
                service_match = re.search(
                    r'Service:\s*([^\n]+)', medical_record)
                if service_match:
                    processed_sections.append(
                        f"Service: {service_match.group(1).strip()}")

            # History
            if 'History of Present Illness:' in medical_record:
                hpi_match = re.search(r'History of Present Illness:\s*(.*?)(?=\n\n|\nPast Medical|Physical Exam|$)',
                                      medical_record, re.DOTALL)
                if hpi_match:
                    hpi = hpi_match.group(1).strip()[:800]
                    processed_sections.append(f"History: {hpi}")

            # Major Procedures
            if 'Major Surgical or Invasive Procedure:' in medical_record:
                proc_match = re.search(r'Major Surgical or Invasive Procedure:\s*(.*?)(?=\n\n|History of Present|$)',
                                       medical_record, re.DOTALL)
                if proc_match:
                    proc = proc_match.group(1).strip()
                    if proc.lower() not in ['none', 'none.', '']:
                        processed_sections.append(f"Major Procedures: {proc}")

            # Past Medical History
            if 'Past Medical History:' in medical_record:
                pmh_match = re.search(r'Past Medical History:\s*(.*?)(?=\n\n|PAST SURGICAL|Social History|$)',
                                      medical_record, re.DOTALL)
                if pmh_match:
                    pmh = pmh_match.group(1).strip()[:400]
                    processed_sections.append(f"Past Medical History: {pmh}")

            processed_text = '\n\n'.join(
                processed_sections) if processed_sections else medical_record
            processed_text = re.sub(r'___+', '', processed_text)
            processed_text = re.sub(r'\s+', ' ', processed_text)
            processed_text = processed_text[:3000]

            return {'user_input': processed_text.strip()}

        except Exception as e:
            return {'user_input': str(data.get('medical record', ''))}

    async def postprocess_result(self, result: str) -> str:
        """결과 정리 및 최적화"""
        import re

        try:
            result = result.strip()

            # Remove prefixes
            prefixes = ['BRIEF HOSPITAL COURSE:',
                        'Brief Hospital Course:', 'brief hospital course:']
            for prefix in prefixes:
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()

            if result and not result.endswith('.'):
                result += '.'

            # Length optimization
            words = result.split()
            if len(words) > 450:
                sentences = [s.strip() for s in result.split('.') if s.strip()]
                important_keywords = ['admitted', 'diagnosis', 'treated', 'underwent', 'developed',
                                      'improved', 'discharged', 'course', 'complication']

                important_sentences = []
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in important_keywords) or len(important_sentences) < 3:
                        important_sentences.append(sentence.strip())
                    if len(' '.join(important_sentences).split()) >= 400:
                        break

                if important_sentences:
                    result = '. '.join(important_sentences)
                    if not result.endswith('.'):
                        result += '.'

            # Medical term standardization
            corrections = {
                'pt ': 'patient ',
                'w/ ': 'with ',
                'w/o ': 'without ',
                'h/o ': 'history of '
            }

            for wrong, correct in corrections.items():
                result = result.replace(wrong, correct)

            return result if result else "Brief hospital course documented."

        except Exception as e:
            return str(result) if result else "Brief hospital course documented."


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

            if pd.isna(radiology_text) or not isinstance(radiology_text, str):
                return {'user_input': ''}

            # Extract FINDINGS
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
            else:
                findings_text = radiology_text

            # Clean text
            findings_text = re.sub(r'^[:\s]*', '', findings_text)
            findings_text = re.sub(r'\b___\b', '', findings_text)
            findings_text = re.sub(r'\s+', ' ', findings_text)

            return {'user_input': findings_text.strip()}

        except Exception as e:
            return {'user_input': str(data.get('radiology report', ''))}

    async def postprocess_result(self, result: str) -> str:
        """간소화된 후처리"""
        import re

        try:
            result = result.strip()

            if result.startswith(('IMPRESSION:', 'Impression:', 'impression:')):
                result = result.split(':', 1)[1].strip()

            if result and not result.endswith('.'):
                result += '.'

            # Simple numbering
            if not re.match(r'^\d+\.', result) and '. ' in result:
                sentences = [s.strip() for s in result.split('.') if s.strip()]
                if len(sentences) >= 2:
                    result = '. '.join(
                        [f"{i+1}. {s}" for i, s in enumerate(sentences)])

            return result if result else "No acute findings."

        except Exception as e:
            return str(result) if result else "No acute findings."


class TaskCProcessor(DatathonProcessor):
    """
    Task C: ICD-10-CM code prediction
    핵심 개선: (1) R-코드 강화 (2) 허용코드셋 엄격 적용 (3) 증상 우선순위 조정
    """

    MAX_CODES = 3
    MAX_CHARS = 2600

    # 16개 허용 코드셋 (EDA 기반)
    ALLOWED_CODES = {
        "I214", "I4891", "I82431", "K7581", "N19", "M5440",
        "R531", "R0600", "R079", "R339", "R509", "I6203",
        "W1830XA", "S066X1A", "M5489", "E2740"
    }

    def __init__(self, api_key: str, train_df: pd.DataFrame = None):
        super().__init__(api_key=api_key)
        self.code_freq: Dict[str, int] = {}
        if train_df is not None and "target" in train_df.columns:
            try:
                self.code_freq = self._build_code_freq_from_train(train_df)
            except Exception:
                self.code_freq = {}

    # === 내부로 옮긴 빈도 사전 생성 메서드 ===
    @staticmethod
    def _build_code_freq_from_train(train_df: pd.DataFrame) -> Dict[str, int]:
        """
        train_data_c['target']에서 코드 빈도 사전 생성 (클래스 내부 버전)
        """
        def parse_codes(s):
            if pd.isna(s) or not str(s).strip():
                return []
            return [
                re.sub(r'[^A-Z0-9]', '', c.strip().upper())
                for c in str(s).split(',') if c.strip()
            ]

        freq: Dict[str, int] = {}
        for codes in train_df["target"].apply(parse_codes):
            for c in codes:
                freq[c] = freq.get(c, 0) + 1
        return freq
    # =======================================

    def get_model_name(self) -> str:
        return "meta-llama/Llama-3.1-8B-Instruct"
        # return "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ"

    def get_prompt_template(self) -> str:
        return (
            "You are an expert ICD-10-CM medical coder.\n"
            "Extract ONLY the PRIMARY discharge diagnosis codes for THIS admission.\n\n"
            "RULES:\n"
            "1) Focus ONLY on the main admission reason and actively treated conditions during this stay.\n"
            "2) Use EXACT ICD-10-CM codes (UPPERCASE, NO DOTS). Output 1–3 codes MAX.\n"
            "3) Prefer specific diseases over symptoms. R-codes are valid ONLY when no specific disease is confirmed.\n"
            "4) Output MUST be codes only, comma-separated, UNIQUE. No extra text.\n"
            "5) Map common clinical phrases to codes consistently (see below synonyms & few-shots).\n\n"
            "KEYWORD→CODE SYNONYMS (use when supported by the record):\n"
            "- myocardial infarction / MI / NSTEMI / troponin rise with ischemic ECG → I214\n"
            "- atrial fibrillation / Afib / RVR (Afib context) → I4891\n"
            "- deep vein thrombosis (right leg DVT) → I82431\n"
            "- NASH / cirrhosis (NASH cirrhosis) → K7581\n"
            "- acute kidney failure / renal failure (acute/unspecified) → N19\n"
            "- spinal stenosis → M5440\n"
            "- syncope / fainting / transient loss of consciousness → R531\n"
            "- dyspnea / shortness of breath / SOB / DOE → R0600\n"
            "- chest pain, non-specific → R079 (ONLY if no MI)\n"
            "- urinary retention → R339\n"
            "- respiratory failure / hypoxic respiratory failure → R509\n"
            "- intracranial bleed / cerebral infarction (as in training) → I6203\n"
            "- fall on same level (initial) → W1830XA\n"
            "- head injury (initial) → S066X1A\n"
            "- low back pain / lumbago → M5489\n"
            "- hyperaldosteronism → E2740\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "Hospital Record:\n"
            "\"Chest pain, ECG with diffuse ST depressions; troponin rises.\"\n"
            "Correct Codes: I214\n\n"
            "Hospital Record:\n"
            "\"New onset atrial fibrillation admitted for telemetry; RVR.\"\n"
            "Correct Codes: I4891\n\n"
            "Hospital Record:\n"
            "\"Right-leg swelling; ultrasound shows DVT in femoral/popliteal.\"\n"
            "Correct Codes: I82431\n\n"
            "Hospital Record:\n"
            "\"Known NASH cirrhosis with decompensation; large ascites managed.\"\n"
            "Correct Codes: K7581\n\n"
            "Hospital Record:\n"
            "\"Acute renal failure, creatinine markedly elevated; nephrology consulted.\"\n"
            "Correct Codes: N19\n\n"
            "NOW APPLY TO THIS CASE:\n\n"
            "HOSPITAL RECORD: {user_input}\n\n"
            "Primary diagnosis codes:"
        )

    async def preprocess_data(self, data: Any) -> Dict[str, Any]:
        text = data.get("hospital_course", "") if isinstance(
            data, dict) else getattr(data, "hospital_course", "")
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return {"user_input": "No medical data available"}

        # 약한 정리
        text = re.sub(r"\[\*+.*?\*+\]|_{3,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 핵심 섹션 위주 슬라이스
        U = text.upper()
        cut = None
        for kw in [
            "DISCHARGE DIAGNOSIS:", "FINAL DIAGNOSIS:", "PRIMARY DIAGNOSIS:",
            "PRINCIPAL DIAGNOSIS:", "DIAGNOSIS:", "ASSESSMENT:", "IMPRESSION:",
            "HOSPITAL COURSE:", "CHIEF COMPLAINT:", "HISTORY OF PRESENT ILLNESS:"
        ]:
            pos = U.find(kw)
            if pos != -1:
                cut = text[pos:pos + self.MAX_CHARS]
                break
        if not cut:
            cut = text[: self.MAX_CHARS]

        return {"user_input": cut}

    def _score_code(self, c: str) -> float:
        """코드 점수: R-코드에 더 공정한 기회 제공"""
        base_score = float(self.code_freq.get(c, 0))  # 빈도 기반
        if c.startswith("R"):
            base_score += 150  # R-코드 보너스
        elif not c.startswith(("R", "W")):
            base_score += 100  # 질병코드 보너스
        return base_score

    async def postprocess_result(self, result: str) -> str:
        if not result:
            return ""

        t = result.strip().upper()

        # 1) 허용 코드만 추출
        raw_codes = re.findall(
            r"\b[A-Z][0-9][0-9A-Z]{1,6}X?A?\b", t.replace(".", ""))

        # 2) 정제 & 허용셋 필터링
        valid_codes: List[str] = []
        seen: Set[str] = set()
        for c in raw_codes:
            c = re.sub(r"[^A-Z0-9]", "", c)
            if c in self.ALLOWED_CODES and c not in seen:
                valid_codes.append(c)
                seen.add(c)

        if not valid_codes:
            return ""

        # 3) 특수 규칙: 출혈 관련 없으면 I6203 제거
        if "BLEED" not in t and "HEMORRHAGE" not in t and "STROKE" not in t:
            valid_codes = [c for c in valid_codes if c != "I6203"]

        # 4) 점수 기반 정렬
        codes_sorted = sorted(
            valid_codes, key=lambda c: self._score_code(c), reverse=True)

        # 5) 최대 개수 제한
        return ",".join(codes_sorted[:self.MAX_CODES])
