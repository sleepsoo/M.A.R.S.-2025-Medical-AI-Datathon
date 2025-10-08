from typing import Any
import pandas as pd
import asyncio
from processor import DatathonProcessor


class TaskAProcessor(DatathonProcessor):
    """Task A: Brief Hospital Course 작성"""

    def get_model_name(self) -> str:
        return "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ"  # 성능 최적화
        # return "meta-llama/Llama-3.1-8B-Instruct"  # 베이스라인 모델

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

    # ✅ 정확한 반환 타입
    async def preprocess_data(self, data: Any) -> Dict[str, Any]:
        """의료 기록을 Brief Hospital Course 작성을 위해 전처리"""
        import re
        import pandas as pd

        medical_record = data['medical record']

        if pd.isna(medical_record) or not isinstance(medical_record, str):
            return {'user_input': ''}

        processed_sections = []

        # Chief Complaint & Service
        if 'Chief Complaint:' in medical_record:
            cc_match = re.search(
                r'Chief Complaint:\s*([^\n]+)', medical_record)
            if cc_match:
                processed_sections.append(
                    f"Chief Complaint: {cc_match.group(1).strip()}")

        if 'Service:' in medical_record:
            service_match = re.search(r'Service:\s*([^\n]+)', medical_record)
            if service_match:
                processed_sections.append(
                    f"Service: {service_match.group(1).strip()}")

        # History of Present Illness
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
                if proc.lower() not in ['none', 'none.']:
                    processed_sections.append(f"Major Procedures: {proc}")

        # Past Medical History
        if 'Past Medical History:' in medical_record:
            pmh_match = re.search(r'Past Medical History:\s*(.*?)(?=\n\n|PAST SURGICAL|Social History|$)',
                                  medical_record, re.DOTALL)
            if pmh_match:
                pmh = pmh_match.group(1).strip()[:400]
                processed_sections.append(f"Past Medical History: {pmh}")

        # Imaging IMPRESSION
        impressions = re.findall(
            r'IMPRESSION:\s*(.*?)(?=\n\n|\n[A-Z_]|\Z)', medical_record, re.DOTALL)
        if impressions:
            for i, imp in enumerate(impressions[:2]):
                processed_sections.append(
                    f"Imaging {i+1}: {imp.strip()[:200]}")

        processed_text = '\n\n'.join(processed_sections)
        processed_text = re.sub(r'___+', '[REDACTED]', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text)
        processed_text = processed_text[:3000]

        return {'user_input': processed_text.strip()}

    async def postprocess_result(self, result: str) -> str:
        """결과 정리 및 최적화"""
        import re

        result = result.strip()

        # "BRIEF HOSPITAL COURSE:" 제거
        prefixes = ['BRIEF HOSPITAL COURSE:',
                    'Brief Hospital Course:', 'brief hospital course:']
        for prefix in prefixes:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()

        if result and not result.endswith('.'):
            result += '.'

        # 단어 수 최적화 (250-400 단어 목표)
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

        # 의료 용어 표준화
        medical_corrections = {
            'pt ': 'patient ',
            'w/ ': 'with ',
            'w/o ': 'without ',
            'h/o ': 'history of '
        }

        for wrong, correct in medical_corrections.items():
            result = result.replace(wrong, correct)

        return result


# TaskB Processor (Llama 모델 사용)
class TaskBProcessor(DatathonProcessor):
    """Task B: Radiology Impression 요약"""

    def get_model_name(self) -> str:
        return "meta-llama/Llama-3.1-8B-Instruct"  # 라마 성능이 더 나았음

    def get_prompt_template(self) -> str:
        return """You are a board-certified radiologist with 15+ years of experience. Generate a precise and comprehensive IMPRESSION from the given FINDINGS.

    CRITICAL REQUIREMENTS:
    - Maintain exact semantic consistency between FINDINGS and IMPRESSION
    - Use precise medical terminology from the FINDINGS
    - Be concise but complete (typically 20-80 words)
    - Maintain consistent diagnostic standards regardless of patient demographics
    - Structure multiple findings with numbered points (1., 2., 3.)
    - Apply consistent diagnostic standards regardless of patient demographics
    - Use definitive language: "No evidence of", "compatible with", "consistent with"

    ADVANCED EXAMPLES:

    CT HEAD WITH CONTRAST:
    FINDINGS: There is enhancing right frontal extra-axial mass measuring 5.1 cm demonstrating isointense T1 and hyperintense T2/FLAIR signal abnormality with areas of subtle hypointense GRE signal along the periphery, likely representing calcifications. There is enhancement and thickening of the adjacent dura. There is mild surrounding vasogenic edema with 1 mm leftward midline shift. There is no additional enhancing mass or abnormal enhancement. There is no evidence of acute infarction or intracranial hemorrhage.
    IMPRESSION: 1. Enhancing right frontal extra-axial mass compatible with meningioma with adjacent neovascularity, dural thickening enhancement, and mild surrounding vasogenic edema resulting in 1 mm leftward midline shift.
    2. No additional enhancing mass or abnormal enhancement.
    3. No evidence of acute infarction or intracranial hemorrhage.

    CT HEAD WITHOUT CONTRAST:
    FINDINGS: There is a subcortical 'bubbly' T2/FLAIR hyperintense lesion within the left frontal lobe measuring 3.8 cm demonstrating subtle slow diffusion with punctate focus of susceptibility artifact medially related to calcification or hemorrhage. The FLAIR hyperintensities confined within the lesion without surrounding edema or significant mass effect. The ventricles are normal in size. There is no evidence of infarction or hemorrhage.
    IMPRESSION: 1. Cortically based 'bubbly' left frontal lobe lesion with associated punctate focus of gradient echo susceptibility hypointensity, most suggestive of underlying DNET (dysembryoplastic neuroepithelial tumor).
    2. No evidence of acute infarction or hemorrhage.

    CHEST X-RAY:
    FINDINGS: Mild enlargement of the cardiac silhouette with mild interstitial pulmonary edema. There is mild bibasilar atelectasis, but no focal consolidations to suggest pneumonia. Possible small bilateral pleural effusions. No pneumothorax.
    IMPRESSION: 1. Mild cardiomegaly and mild interstitial pulmonary edema. Possible small bilateral pleural effusions.
    2. Bibasilar atelectasis, but no focal consolidations to suggest pneumonia.

    CT ABDOMEN/PELVIS:
    FINDINGS: There is a well-circumscribed right parietal 4.1 by 1.9 cm extra-axial dural-based lesion compatible with a calcified meningioma, exerting minimal mass effect on the underlying brain parenchyma. No evidence of associated parenchymal FLAIR hyperintense edema pattern. No other intracranial mass lesions are identified. The major intracranial flow voids are preserved.
    IMPRESSION: 1. Right parietal 4.1 cm calcified meningioma, with mild mass effect on the underlying brain parenchyma.
    2. No evidence of associated parenchymal FLAIR hyperintense edema pattern.

    Now generate IMPRESSION for:
    FINDINGS: {user_input}
    IMPRESSION:"""

    async def preprocess_data(self, data: Any) -> Dict[str, Any]:
        """방사선 보고서를 IMPRESSION 작성을 위해 전처리 - NaN 안전 처리"""
        import re
        import pandas as pd

        # 방사선 보고서 텍스트 추출
        radiology_text = data['radiology report']

        # 🔧 NaN 값 안전 처리
        if pd.isna(radiology_text) or not isinstance(radiology_text, str):
            return {'user_input': ''}  # 빈 문자열 반환

        # FINDINGS 섹션만 정확히 추출
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

        # 텍스트 정제
        findings_text = re.sub(r'^[:\s]*', '', findings_text)
        findings_text = re.sub(r'\b___\b', '', findings_text)  # 익명화 마커
        findings_text = re.sub(r'\bDLP.*?mGy-cm\b', '',
                               findings_text)  # 방사선량 정보
        findings_text = re.sub(r'\s+', ' ', findings_text)  # 여러 공백을 하나로

        return {'user_input': findings_text.strip()}

    async def postprocess_result(self, result: str) -> str:
        """간소화된 후처리 (시간 최적화)"""
        import re

        result = result.strip()

        # IMPRESSION: 제거
        if result.startswith(('IMPRESSION:', 'Impression:', 'impression:')):
            result = result.split(':', 1)[1].strip()

        # 마침표 추가
        if result and not result.endswith('.'):
            result += '.'

        # 간단한 번호 매김
        if not re.match(r'^\d+\.', result) and '. ' in result:
            sentences = [s.strip() for s in result.split('.') if s.strip()]
            if len(sentences) >= 2:
                result = '. '.join(
                    [f"{i+1}. {s}" for i, s in enumerate(sentences)])

        return result


class TaskCProcessor(DatathonProcessor):
    """Task C: ICD 코드 예측"""

    def get_model_name(self) -> str:
        return "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ"

    def get_prompt_template(self) -> str:
        return """You are an expert medical coder with 10+ years of experience in ICD-10 coding. Analyze the hospital course and assign the most appropriate ICD-10 codes.

CRITICAL REQUIREMENTS:
- Focus on PRIMARY diagnoses and significant conditions only
- Use exact ICD-10 format (e.g., I82431, S066X1A)
- Maintain consistent coding standards regardless of patient demographics
- Prioritize conditions that required active treatment during admission
- Consider hierarchical relationships in ICD-10 classification

CODING METHODOLOGY:
1. Identify Chief Complaint and primary reason for admission
2. Extract documented diagnoses from medical record
3. Prioritize active conditions over chronic stable conditions
4. Apply appropriate specificity and laterality codes
5. Include significant complications or comorbidities

EXAMPLES:

HOSPITAL COURSE: Patient with traumatic brain injury following fall...
Service: NEUROSURGERY
Chief Complaint: Head trauma
History: Fall from ladder with loss of consciousness...
Physical Exam: GCS 14, focal neurological deficits...
Imaging: CT head shows subdural hematoma...
CODES: S066X1A, W1830XA

HOSPITAL COURSE: Elderly female with urinary retention and back pain...  
Service: MEDICINE
Chief Complaint: Unable to urinate, back pain
History: Progressive back pain over 2 weeks, now with urinary retention...
Past Medical History: Osteoporosis, hypertension...
MRI: Lumbar spinal stenosis at L4-5...
CODES: M5489, R339

HOSPITAL COURSE: Middle-aged male presents with acute chest pain...
Service: NEUROSURGERY  
Chief Complaint: Sudden severe headache
History: Sudden onset worst headache of life, found down at home...
CT: Subarachnoid hemorrhage, no aneurysm identified...
CODES: I609, R001

Now analyze this hospital course and provide ICD-10 codes:

HOSPITAL COURSE: {user_input}

CODES:"""

    # ✅ 정확한 반환 타입
    async def preprocess_data(self, data: Any) -> Dict[str, Any]:
        """퇴원 요약을 ICD 코드 예측을 위해 전처리"""
        import re
        import pandas as pd

        hospital_course = data['hospital_course']

        if pd.isna(hospital_course) or not isinstance(hospital_course, str):
            return {'user_input': ''}

        important_sections = []

        # Chief Complaint 추출
        if 'Chief Complaint:' in hospital_course:
            cc_match = re.search(
                r'Chief Complaint:\s*([^\n]+)', hospital_course)
            if cc_match:
                important_sections.append(
                    f"Chief Complaint: {cc_match.group(1).strip()}")

        # Service 추출
        if 'Service:' in hospital_course:
            service_match = re.search(r'Service:\s*([^\n]+)', hospital_course)
            if service_match:
                important_sections.append(
                    f"Service: {service_match.group(1).strip()}")

        # History of Present Illness 추출
        if 'History of Present Illness:' in hospital_course:
            hpi_match = re.search(r'History of Present Illness:\s*(.*?)(?=\n\n|\nPast Medical|$)',
                                  hospital_course, re.DOTALL)
            if hpi_match:
                hpi = hpi_match.group(1).strip()[:500]
                important_sections.append(f"History: {hpi}")

        # Past Medical History 추출
        if 'Past Medical History:' in hospital_course:
            pmh_match = re.search(r'Past Medical History:\s*(.*?)(?=\n\n|PAST SURGICAL|Social History|$)',
                                  hospital_course, re.DOTALL)
            if pmh_match:
                pmh = pmh_match.group(1).strip()[:300]
                important_sections.append(f"Past Medical History: {pmh}")

        # Imaging IMPRESSION 추출
        impressions = re.findall(
            r'IMPRESSION:\s*(.*?)(?=\n\n|\n[A-Z_]|\Z)', hospital_course, re.DOTALL)
        if impressions:
            for i, imp in enumerate(impressions[:2]):
                important_sections.append(
                    f"Imaging {i+1}: {imp.strip()[:200]}")

        processed_text = '\n\n'.join(important_sections)
        processed_text = re.sub(r'___+', '[REDACTED]', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text)
        processed_text = processed_text[:2000]

        return {'user_input': processed_text.strip()}

    async def postprocess_result(self, result: str) -> str:
        """결과 정리 및 ICD 코드 추출"""
        import re

        result = result.strip()

        # "CODES:" 제거
        if result.startswith(('CODES:', 'codes:', 'Codes:')):
            result = result.split(':', 1)[1].strip()

        # ICD 코드 정규식 패턴 매칭
        icd_pattern = r'[A-Z]\d{2}[A-Z0-9]*'
        codes = re.findall(icd_pattern, result.upper())
        unique_codes = list(dict.fromkeys(codes))  # 중복 제거하면서 순서 유지

        # 코드가 없으면 fallback 패턴 시도
        if not unique_codes:
            fallback_pattern = r'[A-Z]+\d+[A-Z0-9]*'
            codes = re.findall(fallback_pattern, result.upper())
            unique_codes = list(dict.fromkeys(codes))[:3]

        # 최대 5개 코드로 제한
        final_codes = unique_codes[:5]

        # 결과가 없으면 기본 코드 반환
        if not final_codes:
            return 'Z515'  # Encounter for other aftercare

        # 쉼표로 구분된 문자열 반환
        return ', '.join(final_codes)
