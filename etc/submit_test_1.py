from typing import Any, Dict
 import pandas as pd
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

                if pd.isna(medical_record) or not isinstance(medical_record, str) or not medical_record.strip():
                    return {'user_input': 'Patient admitted for medical care.'}

                processed_sections = []

                # Chief Complaint
                if 'Chief Complaint:' in medical_record:
                    cc_match = re.search(
                        r'Chief Complaint:\s*([^\n]+)', medical_record)
                    if cc_match and cc_match.group(1).strip():
                        processed_sections.append(
                            f"Chief Complaint: {cc_match.group(1).strip()}")

                # Service
                if 'Service:' in medical_record:
                    service_match = re.search(
                        r'Service:\s*([^\n]+)', medical_record)
                    if service_match and service_match.group(1).strip():
                        processed_sections.append(
                            f"Service: {service_match.group(1).strip()}")

                # History
                if 'History of Present Illness:' in medical_record:
                    hpi_match = re.search(r'History of Present Illness:\s*(.*?)(?=\n\n|\nPast Medical|Physical Exam|$)',
                                          medical_record, re.DOTALL)
                    if hpi_match and hpi_match.group(1).strip():
                        hpi = hpi_match.group(1).strip()[:800]
                        processed_sections.append(f"History: {hpi}")

                # Major Procedures
                if 'Major Surgical or Invasive Procedure:' in medical_record:
                    proc_match = re.search(r'Major Surgical or Invasive Procedure:\s*(.*?)(?=\n\n|History of Present|$)',
                                           medical_record, re.DOTALL)
                    if proc_match:
                        proc = proc_match.group(1).strip()
                        if proc and proc.lower() not in ['none', 'none.', '']:
                            processed_sections.append(
                                f"Major Procedures: {proc}")

                # Past Medical History
                if 'Past Medical History:' in medical_record:
                    pmh_match = re.search(r'Past Medical History:\s*(.*?)(?=\n\n|PAST SURGICAL|Social History|$)',
                                          medical_record, re.DOTALL)
                    if pmh_match and pmh_match.group(1).strip():
                        pmh = pmh_match.group(1).strip()[:400]
                        processed_sections.append(
                            f"Past Medical History: {pmh}")

                if processed_sections:
                    processed_text = '\n\n'.join(processed_sections)
                else:
                    processed_text = medical_record[:2000]  # 원본 데이터 사용

                processed_text = re.sub(r'___+', '', processed_text)
                processed_text = re.sub(r'\s+', ' ', processed_text)
                processed_text = processed_text.strip()[:3000]

                return {'user_input': processed_text if processed_text else 'Patient admitted for medical care.'}

            except Exception as e:
                fallback_text = str(data.get('medical record', ''))
                return {'user_input': fallback_text if fallback_text.strip() else 'Patient admitted for medical care.'}

        async def postprocess_result(self, result: str) -> str:
            """결과 정리 및 최적화"""
            import re

            try:
                if not result or not isinstance(result, str):
                    return "Brief hospital course documented."

                result = result.strip()

                # Remove prefixes
                prefixes = ['BRIEF HOSPITAL COURSE:',
                            'Brief Hospital Course:', 'brief hospital course:']
                for prefix in prefixes:
                    if result.startswith(prefix):
                        result = result[len(prefix):].strip()
                        break

                if not result:
                    return "Brief hospital course documented."

                if not result.endswith('.'):
                    result += '.'

                # Length optimization
                words = result.split()
                if len(words) > 450:
                    sentences = [s.strip()
                                 for s in result.split('.') if s.strip()]
                    if sentences:
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

                return result if result.strip() else "Brief hospital course documented."

            except Exception as e:
                return "Brief hospital course documented."

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
                                    numbered_sentences.append(
                                        f"{i+1}. {sentence}")
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
    CODES: S066X1A, W1830XA

    HOSPITAL COURSE: Elderly female with urinary retention and back pain...  
    Service: MEDICINE
    Chief Complaint: Unable to urinate, back pain
    History: Progressive back pain over 2 weeks, now with urinary retention...
    CODES: M5489, R339

    Now analyze this hospital course and provide ICD-10 codes:

    HOSPITAL COURSE: {user_input}

    CODES:"""

        async def preprocess_data(self, data: Any) -> Dict[str, Any]:
            """퇴원 요약을 ICD 코드 예측을 위해 전처리"""
            import re
            import pandas as pd

            try:
                hospital_course = data.get('hospital_course', '')

                if pd.isna(hospital_course) or not isinstance(hospital_course, str) or not hospital_course.strip():
                    return {'user_input': 'Patient admitted for medical evaluation.'}

                important_sections = []

                # Chief Complaint
                if 'Chief Complaint:' in hospital_course:
                    cc_match = re.search(
                        r'Chief Complaint:\s*([^\n]+)', hospital_course)
                    if cc_match and cc_match.group(1).strip():
                        important_sections.append(
                            f"Chief Complaint: {cc_match.group(1).strip()}")

                # Service
                if 'Service:' in hospital_course:
                    service_match = re.search(
                        r'Service:\s*([^\n]+)', hospital_course)
                    if service_match and service_match.group(1).strip():
                        important_sections.append(
                            f"Service: {service_match.group(1).strip()}")

                # History
                if 'History of Present Illness:' in hospital_course:
                    hpi_match = re.search(r'History of Present Illness:\s*(.*?)(?=\n\n|\nPast Medical|$)',
                                          hospital_course, re.DOTALL)
                    if hpi_match and hpi_match.group(1).strip():
                        hpi = hpi_match.group(1).strip()[:500]
                        important_sections.append(f"History: {hpi}")

                if important_sections:
                    processed_text = '\n\n'.join(important_sections)
                else:
                    processed_text = hospital_course[:1500]  # 원본 사용

                processed_text = re.sub(r'___+', '', processed_text)
                processed_text = re.sub(r'\s+', ' ', processed_text)
                processed_text = processed_text.strip()[:2000]

                return {'user_input': processed_text if processed_text else 'Patient admitted for medical evaluation.'}

            except Exception as e:
                fallback_text = str(data.get('hospital_course', ''))
                return {'user_input': fallback_text if fallback_text.strip() else 'Patient admitted for medical evaluation.'}

        async def postprocess_result(self, result: str) -> str:
            """결과 정리 및 ICD 코드 추출"""
            import re

            try:
                if not result or not isinstance(result, str):
                    return 'Z515'

                result = result.strip()

                if result.startswith(('CODES:', 'codes:', 'Codes:')):
                    result = result.split(':', 1)[1].strip()

                if not result:
                    return 'Z515'

                # ICD code extraction
                icd_pattern = r'[A-Z]\d{2}[A-Z0-9]*'
                codes = re.findall(icd_pattern, result.upper())
                unique_codes = []
                seen = set()
                for code in codes:
                    if code not in seen:
                        unique_codes.append(code)
                        seen.add(code)

                if not unique_codes:
                    fallback_pattern = r'[A-Z]+\d+[A-Z0-9]*'
                    codes = re.findall(fallback_pattern, result.upper())
                    unique_codes = []
                    seen = set()
                    for code in codes[:3]:  # 최대 3개만
                        if code not in seen:
                            unique_codes.append(code)
                            seen.add(code)

                final_codes = unique_codes[:5]  # 최대 5개

                return ', '.join(final_codes) if final_codes else 'Z515'

            except Exception as e:
                return 'Z515'
