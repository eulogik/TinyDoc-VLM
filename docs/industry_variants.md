# TinyDoc-VLM: Industry-Specific Fine-Tuned Variants

## Executive Summary

This document outlines the strategy for developing industry-specific fine-tuned variants of TinyDoc-VLM. Each variant targets a vertical market with domain-specific document types, terminology, and output schemas. By combining synthetic data generation, parameter-efficient fine-tuning (LoRA/QLoRA), and domain-specific evaluation benchmarks, each model achieves production-grade accuracy targeted at its target domain while maintaining TinyDoc-VLM's core efficiency advantages.

---

## 1. TinyDoc-Finance

### Industry Overview & Pain Points

Financial services organizations process millions of invoices, receipts, tax forms, and statements daily. Current OCR solutions struggle with:
- Multi-currency invoices with mixed symbol placement
- Tables with merged cells or irregular layouts (common in SEC filings)
- Handwritten amounts on checks and receipts
- Low-quality scans from mobile devices or fax machines
- Strict audit-trail requirements demanding >99% accuracy on monetary fields

### Document Types & Formats

| Document | Format | Complexity |
|----------|--------|------------|
| Vendor invoices | PDF, PNG, JPEG | Medium-High |
| Receipts | JPEG (mobile) | Low-Medium |
| Financial statements | PDF (structured) | High |
| SEC filings (10-K, 10-Q) | PDF | Very High |
| Bank statements | PDF | Medium |
| Tax forms (W-2, 1099) | PDF | Medium |
| Checks (front/back) | JPEG, TIFF | Low |

### Special Tokens

```
Currency: [USD], [EUR], [GBP], [JPY], [CNY], [CHF], [CAD], [AUD]
Financial: [ACCOUNT], [ROUTING], [INVOICE_NUM], [PO_NUM], [TAX_ID], [SWIFT_IBAN]
Table: [TABLE_START], [TABLE_END], [ROW], [CELL], [HEADER]
Amount: [CURRENCY_AMOUNT], [PERCENTAGE], [DEBIT], [CREDIT]
```

### Output Schema

```json
{
  "invoice": {
    "vendor": {
      "name": "string",
      "address": "string",
      "tax_id": "string"
    },
    "line_items": [
      {
        "description": "string",
        "quantity": "float",
        "unit_price": "float",
        "amount": "float"
      }
    ],
    "subtotal": "float",
    "tax_rate": "float",
    "tax_amount": "float",
    "total": "float",
    "currency": "string (ISO 4217)",
    "invoice_date": "date",
    "due_date": "date",
    "invoice_number": "string",
    "po_number": "string",
    "payment_terms": "string"
  }
}
```

### Training Data Strategy

**Volume**: 50,000 synthetic financial documents

**Generation pipeline**:
1. **Template bank** (200+ templates): Invoice layouts from 50 common accounting platforms (QuickBooks, Xero, SAP, Oracle, FreshBooks)
2. **SEC filing scraper**: Import 5,000 real 10-K/10-Q filings as structural templates, populate with synthetic data
3. **Receipt generator**: Apply receipt-style noise (curved paper, shadows, low light) to synthetic receipt text
4. **Multi-currency layer**: Randomize currency symbols, formatting conventions (1,000.00 vs 1.000,00)
5. **Augmentation**: Rotation (±15°), Gaussian noise, JPEG compression artifacts, Gaussian blur

**Data distribution**:
- Invoices: 20,000
- Receipts: 10,000
- Financial statements: 5,000
- SEC filings (pages): 5,000
- Tax forms: 5,000
- Bank statements: 5,000

### Fine-Tuning Recipe

| Parameter | Value |
|-----------|-------|
| Base model | TinyDoc-VLM |
| Method | LoRA |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Learning rate | 2e-4 (cosine schedule) |
| Epochs | 5 |
| Batch size | 32 (effective 128 with gradient accumulation) |
| Max sequence length | 4096 |
| Warmup ratio | 0.03 |
| Optimizer | AdamW (β1=0.9, β2=0.999) |
| Weight decay | 0.01 |
| FP precision | BF16 |
| Training hardware | 8× A100 80GB |
| Estimated training time | ~48 hours |

### Evaluation Plan

**Custom benchmark**: `TinyDoc-Bench-Finance` (2,000 annotated documents held out from synthetic generation)

**Metrics**:
- Field Extraction Accuracy (F1 per field)
- Monetary Amount Accuracy (exact match)
- Line Item Accuracy (row-level F1)
- End-to-End Document Accuracy (all critical fields correct)

**Target accuracy**:
| Document Type | Clean | Noisy Scans |
|--------------|-------|-------------|
| Invoices | >97% | >87% |
| Receipts | >93% | >82% |
| Tax forms | >96% | >88% |
| Financial statements | >94% | >83% |

**Real-world validation**: Partner with 2-3 AP automation companies for blind evaluation on their production data (NDA-protected).

### Deployment Considerations

- **Hardware target**: On-premise server (Intel Xeon / AMD EPONYC) or private cloud
- **Latency**: <2s per page on CPU (quantized INT8), <500ms on GPU
- **Integration**: REST API with pre-built connectors to QuickBooks, Xero, NetSuite
- **Model size**: ~4GB (INT8 quantized LoRA variant)

### Compliance & Privacy

- SOC 2 Type II deployment environments
- Data residency: US/EU/APAC regional deployment options
- No training on customer data without explicit consent
- Audit logging for all extraction results (for financial audit trails)

### Estimated Cost

| Item | Cost |
|------|------|
| Synthetic data generation | $2,000 |
| LoRA fine-tuning (compute) | $3,840 (8× A100 × 48h × $2/hr) |
| Benchmark creation | $1,500 |
| Validation & QA | $2,000 |
| **Total** | **~$9,340** |

---

## 2. TinyDoc-Healthcare

### Industry Overview & Pain Points

Healthcare organizations face massive document processing backlogs:
- Insurance claim denial rates of 13% often stem from manual data entry errors
- Prior authorization processing takes an average of 20 minutes per request
- Clinical note summarization consumes 2+ hours of physician time daily
- HIPAA compliance limits cloud-based OCR solutions
- Handwritten prescriptions remain error-prone even with digital tools

### Document Types & Formats

| Document | Format | Complexity |
|----------|--------|------------|
| Medical intake forms | PDF, paper scan | Medium |
| Prescriptions | Handwritten/print | Medium |
| Lab results | PDF (structured tables) | High |
- Insurance cards (front/back) | JPEG | Low |
| CMS-1500 claim forms | PDF (scanned) | Medium |
| UB-04 institutional claims | PDF | Medium |
| Clinical notes (SOAP) | Handwritten notes | Very High |

### Special Tokens

```
Medical: [MEDICATION], [DOSAGE], [FREQUENCY], [ROUTE], [NDC_CODE]
Diagnosis: [ICD10_CODE], [DIAGNOSIS], [SYMPTOM], [ONSET]
Labs: [LAB_VALUE], [UNIT], [REFERENCE_RANGE], [FLAG]
Identifiers: [NPI], [MRN], [INSURANCE_ID], [GROUP_NUM]
Anatomy: [BODY_PART], [LATERALITY]
```

### Output Schemas

**Patient Information Extraction:**
```json
{
  "patient": {
    "name": "string",
    "dob": "date",
    "mrn": "string",
    "insurance": {
      "provider": "string",
      "member_id": "string",
      "group_number": "string"
    }
  },
  "medications": [
    {
      "name": "string",
      "ndc_code": "string",
      "dosage": "string",
      "frequency": "string",
      "route": "string",
      "prescriber_npi": "string"
    }
  ],
  "diagnoses": [
    {
      "description": "string",
      "icd10_code": "string",
      "date": "date"
    }
  ],
  "lab_results": [
    {
      "test_name": "string",
      "value": "float",
      "unit": "string",
      "reference_range": "string",
      "abnormal_flag": "boolean"
    }
  ]
}
```

### Training Data Strategy

**Volume**: 30,000 synthetic medical documents

**Generation pipeline**:
1. **Form templates**: 150+ CMS-1500, UB-04, and common intake form layouts
2. **Prescription generator**: Synthesize realistic prescriptions with varied handwriting styles (using handwriting synthesis models)
3. **Lab report generator**: Create structured lab panels (CBC, CMP, lipid panel) with realistic value distributions
4. **Clinical note generator**: GPT-4-based SOAP note generation with medical terminology
5. **Insurance card generator**: Front/back with realistic carrier designs
6. **Augmentation**: Strikethrough marks, checkboxes, handwritten annotations, fax-quality degradation

**Data distribution**:
- Medical forms: 8,000
- Prescriptions: 5,000
- Lab reports: 5,000
- Insurance cards: 4,000
- Clinical notes: 4,000
- Claim forms: 4,000

### Fine-Tuning Recipe

| Parameter | Value |
|-----------|-------|
| Base model | TinyDoc-VLM |
| Method | QLoRA (4-bit quantized base) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj, gate_proj, up_proj, down_proj |
| Learning rate | 1e-4 (cosine schedule) |
| Epochs | 8 |
| Batch size | 16 (effective 64 with gradient accumulation) |
| Max sequence length | 4096 |
| Warmup ratio | 0.05 |
| Optimizer | Paged AdamW 8-bit |
| Weight decay | 0.01 |
| FP precision | NF4 + BF16 compute |
| Training hardware | 4× A100 80GB |
| Estimated training time | ~36 hours |

### Evaluation Plan

**Custom benchmark**: `TinyDoc-Bench-Healthcare` (1,500 annotated documents)

**Metrics**:
- Field-level F1 score
- Critical field accuracy (patient ID, medication name, dosage)
- ICD-10 code prediction accuracy
- End-to-end claim form accuracy

**Target accuracy**:
| Document Type | Clean | Noisy |
|--------------|-------|-------|
| Insurance cards | >98% | >92% |
| Lab reports | >96% | >88% |
| Prescriptions | >93% | >80% |
| Medical forms | >95% | >85% |
| Clinical notes | >90% | >78% |

**Real-world validation**: Partner with 1-2 RCM (Revenue Cycle Management) companies for de-identified data testing.

### Deployment Considerations

- **Hardware target**: On-premise only (HIPAA requirement) — edge device or hospital server
- **Latency**: <3s per document on edge device (NVIDIA Jetson / Intel NUC)
- **Integration**: HL7 FHIR output format, Epic/Cerner connector plugins
- **Model size**: ~2.5GB (QLoRA 4-bit quantized)
- **Key advantage**: Full on-device inference — no PHI leaves the premises

### Compliance & Privacy

- **HIPAA**: Full compliance via on-device processing — no data transmission
- **BAA**: Not required for on-device deployment (no PHI to processor)
- **HITRUST**: Architecture supports HITRUST certification pathway
- **Data handling**: Zero PHI in training data (100% synthetic)
- **Audit**: All extractions logged with user attribution

### Estimated Cost

| Item | Cost |
|------|------|
| Synthetic data generation | $3,000 |
| QLoRA fine-tuning (compute) | $2,880 (4× A100 × 36h × $2/hr) |
| Benchmark creation | $2,000 |
| Validation & QA | $2,500 |
| **Total** | **~$10,380** |

---

## 3. TinyDoc-Legal

### Industry Overview & Pain Points

Legal document review is notoriously time-intensive:
- Contract review averages $300/hour for associates
- NDA analysis requires identifying 12-15 key clauses per document
- Lease abstraction for commercial real estate costs $500-1500 per lease
- Court document filing errors cause 5% of rejected filings
- Law firms handle document formats from opposing counsel with wildly varying layouts

### Document Types & Formats

| Document | Format | Complexity |
|----------|--------|------------|
| Contracts (various types) | PDF | High |
| NDAs | PDF | Medium |
| Lease agreements | PDF | High |
| Court filings | PDF | Medium-High |
| Wills & trusts | PDF | High |
| Powers of attorney | PDF | Medium |
| Service agreements | PDF | Medium |

### Special Tokens

```
Legal: [PARTY], [EFFECTIVE_DATE], [TERMINATION_DATE], [GOVERNING_LAW]
Clauses: [CLAUSE_REF], [INDEMNIFICATION], [LIMITATION_OF_LIABILITY], [FORCE_MAJEURE]
Monetary: [CONSIDERATION], [PENALTY], [LIQUIDATED_DAMAGES]
Obligations: [SHALL], [SHALL_NOT], [COVENANT], [REPRESENTATION]
References: [HEREIN], [HEREINAFTER], [WHEREAS], [WITNESS]
```

### Output Schema

```json
{
  "contract": {
    "document_type": "string",
    "parties": [
      {
        "name": "string",
        "role": "string",
        "address": "string"
      }
    ],
    "dates": {
      "effective_date": "date",
      "execution_date": "date",
      "termination_date": "date",
      "renewal_date": "date"
    },
    "key_clauses": [
      {
        "clause_type": "string",
        "clause_reference": "string",
        "summary": "string",
        "risk_level": "low|medium|high"
      }
    ],
    "obligations": [
      {
        "party": "string",
        "description": "string",
        "deadline": "date"
      }
    ],
    "monetary_terms": [
      {
        "type": "string",
        "amount": "float",
        "currency": "string"
      }
    ],
    "governing_law": "string",
    "signature_blocks": [
      {
        "party": "string",
        "signatory": "string",
        "date": "date"
      }
    ]
  }
}
```

### Training Data Strategy

**Volume**: 25,000 synthetic legal documents

**Generation pipeline**:
1. **Contract template library**: 100+ templates covering NDA, MSA, SaaS, employment, lease, franchise
2. **Clause injection**: Generate variations of key clauses with different risk profiles
3. **Party randomization**: Diverse entity names, jurisdictions, and roles
4. **Date logic**: Consistent date chains (execution → effective → termination)
5. **Augmentation**: Signature stamps, initials on each page, exhibit references, marginal notes

**Data distribution**:
- Contracts (various): 10,000
- NDAs: 5,000
- Lease agreements: 4,000
- Court filings: 3,000
- Service agreements: 3,000

### Fine-Tuning Recipe

| Parameter | Value |
|-----------|-------|
| Base model | TinyDoc-VLM |
| Method | LoRA |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| Target modules | q_proj, v_proj, k_proj, o_proj, gate_proj |
| Learning rate | 1.5e-4 (cosine schedule) |
| Epochs | 6 |
| Batch size | 32 (effective 128) |
| Max sequence length | 8192 (legal docs are longer) |
| Warmup ratio | 0.03 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| FP precision | BF16 |
| Training hardware | 8× A100 80GB |
| Estimated training time | ~40 hours |

### Evaluation Plan

**Custom benchmark**: `TinyDoc-Bench-Legal` (1,200 annotated documents)

**Metrics**:
- Party extraction accuracy
- Date extraction accuracy
- Clause classification F1
- Risk level assignment accuracy
- End-to-end contract abstraction accuracy

**Target accuracy**:
| Document Type | Clean | Noisy |
|--------------|-------|-------|
| NDAs | >94% | >83% |
| Service agreements | >92% | >80% |
| Lease agreements | >90% | >78% |
| Court filings | >93% | >82% |

### Deployment Considerations

- **Hardware target**: Private cloud or on-premise (client confidentiality)
- **Latency**: <5s per page (legal docs are longer, multi-page)
- **Integration**: API with connectors to Clio, iManage, NetDocuments
- **Model size**: ~4GB (INT8 quantized)

### Compliance & Privacy

- Attorney-client privilege protection via on-device processing
- ABA Model Rule 1.6 compliance (confidentiality)
- No training on real client documents
- Audit trail for all document access

### Estimated Cost

| Item | Cost |
|------|------|
| Synthetic data generation | $2,500 |
| LoRA fine-tuning (compute) | $3,200 (8× A100 × 40h × $2/hr) |
| Benchmark creation | $1,800 |
| Validation & QA | $2,000 |
| **Total** | **~$9,500** |

---

## 4. TinyDoc-Logistics

### Industry Overview & Pain Points

Global supply chains process billions of shipping documents annually:
- Bill of lading data entry errors cause customs delays costing $500+ per incident
- HS code misclassification results in penalties and incorrect duty assessment
- Multi-carrier shipping label formats vary dramatically
- Customs form processing (CBP 3461, 7501) requires near-perfect accuracy
- Real-time delivery confirmation processing at scale is labor-intensive

### Document Types & Formats

| Document | Format | Complexity |
|----------|--------|------------|
| Bill of Lading (BOL) | PDF, scanned | High |
| Shipping labels | PDF, thermal print scans | Medium |
| Customs forms (CBP 3461, 7501) | PDF | High |
| Packing slips | PDF, scanned | Medium |
| Delivery confirmations (POD) | PDF, mobile photo | Low-Medium |
| Certificates of Origin | PDF | Medium |
| Freight invoices | PDF | Medium |

### Special Tokens

```
Logistics: [TRACKING_NUM], [CONTAINER_NUM], [SEAL_NUM], [HOUSE_BL], [MASTER_BL]
Customs: [HS_CODE], [DUTY_RATE], [CUSTOMS_VALUE], [COO]
Locations: [PORT_CODE], [UNLOCODE], [WAREHOUSE], [SCAC]
Transport: [VESSEL], [VOYAGE], [CARRIER], [MODE]
Identifiers: [SSCC], [GTIN], [SSN]
```

### Output Schema

```json
{
  "bill_of_lading": {
    "bol_number": "string",
    "shipper": {
      "name": "string",
      "address": "string"
    },
    "consignee": {
      "name": "string",
      "address": "string"
    },
    "notify_party": {
      "name": "string"
    },
    "vessel": "string",
    "voyage": "string",
    "port_of_loading": "string",
    "port_of_discharge": "string",
    "cargo": [
      {
        "description": "string",
        "pieces": "integer",
        "weight_kg": "float",
        "volume_cbm": "float",
        "marks_and_numbers": "string"
      }
    ],
    "container_numbers": ["string"],
    "hs_codes": ["string"],
    "total_weight_kg": "float",
    "place_of_issue": "string",
    "date_of_issue": "date",
    "freight_terms": "string"
  }
}
```

### Training Data Strategy

**Volume**: 20,000 synthetic logistics documents

**Generation pipeline**:
1. **BOL templates**: 80+ templates from major carriers (Maersk, MSC, CMA CGM, COSCO)
2. **Customs form generator**: CBP Form 3461 and 7501 with realistic HS codes and values
3. **Shipping label generator**: Thermal print artifacts, barcode regions, carrier logos
4. **Packing slip generator**: Multi-item packing lists with varied layouts
5. **Augmentation**: Stamps, watermarks, multi-language text, handwritten annotations, fold marks

**Data distribution**:
- Bills of Lading: 6,000
- Customs forms: 5,000
- Shipping labels: 4,000
- Packing slips: 3,000
- Delivery confirmations: 2,000

### Fine-Tuning Recipe

| Parameter | Value |
|-----------|-------|
| Base model | TinyDoc-VLM |
| Method | LoRA |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Learning rate | 2e-4 (cosine schedule) |
| Epochs | 5 |
| Batch size | 32 (effective 128) |
| Max sequence length | 4096 |
| Warmup ratio | 0.03 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| FP precision | BF16 |
| Training hardware | 4× A100 80GB |
| Estimated training time | ~24 hours |

### Evaluation Plan

**Custom benchmark**: `TinyDoc-Bench-Logistics` (1,000 annotated documents)

**Metrics**:
- Shipper/Consignee extraction accuracy
- HS code accuracy (critical — 6-digit exact match)
- Container number accuracy
- Weight/value extraction accuracy
- End-to-end BOL accuracy

**Target accuracy**:
| Document Type | Clean | Noisy |
|--------------|-------|-------|
| BOL | >95% | >84% |
| Customs forms | >96% | >86% |
| Shipping labels | >94% | >82% |
| Packing slips | >93% | >80% |

### Deployment Considerations

- **Hardware target**: Edge device at warehouse/port + cloud API
- **Latency**: <1.5s per document (high-throughput scanning environments)
- **Integration**: API with pre-built connectors to major TMS platforms (SAP TM, Oracle Transportation, BluJay)
- **Model size**: ~3.5GB (INT8 quantized)

### Compliance & Privacy

- CBP data handling requirements
- No retention of customs data beyond processing
- GDPR compliance for EU-bound shipments (data minimization)

### Estimated Cost

| Item | Cost |
|------|------|
| Synthetic data generation | $1,800 |
| LoRA fine-tuning (compute) | $1,920 (4× A100 × 24h × $2/hr) |
| Benchmark creation | $1,200 |
| Validation & QA | $1,500 |
| **Total** | **~$6,420** |

---

## 5. TinyDoc-HR

### Industry Overview & Pain Points

HR departments spend significant time on document-intensive tasks:
- Resume screening takes 7.4 seconds per resume, yet 75% of resumes are never read fully
- I-9 verification errors result in fines of $234-$2,332 per violation
- Offer letter processing requires manual data entry into HRIS systems
- Job application data extraction for ATS (Applicant Tracking System) reporting is manual
- Skills gap analysis requires parsing thousands of job descriptions

### Document Types & Formats

| Document | Format | Complexity |
|----------|--------|------------|
| Resumes/CVs | PDF, DOCX | Very High (varied layouts) |
| Job postings | PDF, web scrape | Medium |
| I-9 forms | PDF, scanned | Medium |
| W-4 forms | PDF | Medium |
| Offer letters | PDF | Medium |
| Employee handbooks | PDF | High |
| Performance reviews | PDF | Medium |

### Special Tokens

```
HR: [CANDIDATE_NAME], [EMAIL], [PHONE], [LINKEDIN]
Skills: [SKILL], [PROFICIENCY], [CERTIFICATION]
Education: [DEGREE], [INSTITUTION], [GRAD_DATE], [GPA]
Employment: [JOB_TITLE], [COMPANY], [DATES], [RESPONSIBILITIES]
Compensation: [SALARY], [BONUS], [EQUITY], [BENEFITS]
Form fields: [FORM_FIELD], [SSN_LAST4], [SIGNATURE_DATE]
```

### Output Schema

```json
{
  "resume": {
    "candidate": {
      "name": "string",
      "email": "string",
      "phone": "string",
      "location": "string",
      "linkedin": "string"
    },
    "summary": "string",
    "experience": [
      {
        "title": "string",
        "company": "string",
        "location": "string",
        "start_date": "date",
        "end_date": "date",
        "current": "boolean",
        "description": "string",
        "skills_used": ["string"]
      }
    ],
    "education": [
      {
        "degree": "string",
        "field_of_study": "string",
        "institution": "string",
        "graduation_date": "date",
        "gpa": "float"
      }
    ],
    "skills": [
      {
        "name": "string",
        "category": "string",
        "proficiency": "beginner|intermediate|advanced|expert"
      }
    ],
    "certifications": [
      {
        "name": "string",
        "issuer": "string",
        "date": "date",
        "expiry": "date"
      }
    ],
    "languages": ["string"],
    "years_of_experience": "integer"
  }
}
```

### Training Data Strategy

**Volume**: 15,000 synthetic HR documents

**Generation pipeline**:
1. **Resume generator**: 200+ resume templates covering 20+ industries, multiple formats (chronological, functional, combination)
2. **Job posting generator**: Synthetic job descriptions with realistic requirements and responsibilities
3. **Form generator**: I-9, W-4, and common HR onboarding forms with realistic data
4. **Offer letter generator**: Varied offer letter templates from different company sizes
5. **Augmentation**: Watermarks, multi-column layouts, infographic elements, photo placeholders

**Data distribution**:
- Resumes: 8,000
- Job postings: 3,000
- I-9 forms: 1,500
- W-4 forms: 1,000
- Offer letters: 1,500

### Fine-Tuning Recipe

| Parameter | Value |
|-----------|-------|
| Base model | TinyDoc-VLM |
| Method | LoRA |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Learning rate | 2e-4 (cosine schedule) |
| Epochs | 4 |
| Batch size | 32 (effective 128) |
| Max sequence length | 4096 |
| Warmup ratio | 0.03 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| FP precision | BF16 |
| Training hardware | 4× A100 80GB |
| Estimated training time | ~18 hours |

### Evaluation Plan

**Custom benchmark**: `TinyDoc-Bench-HR` (800 annotated documents)

**Metrics**:
- Candidate info extraction accuracy
- Experience section accuracy (title, company, dates)
- Skills extraction F1
- Education accuracy
- End-to-end resume parsing accuracy

**Target accuracy**:
| Document Type | Clean | Noisy |
|--------------|-------|-------|
| Resumes | >92% | >78% |
| Job postings | >94% | >83% |
| I-9 forms | >96% | >88% |
| W-4 forms | >96% | >87% |
| Offer letters | >93% | >82% |

### Deployment Considerations

- **Hardware target**: Cloud API (SaaS HR platforms)
- **Latency**: <2s per resume (high-volume recruiting)
- **Integration**: REST API with connectors to Workday, Greenhouse, Lever, iCIMS
- **Model size**: ~3.5GB (INT8 quantized)

### Compliance & Privacy

- EEOC compliance: No extraction of protected characteristics (age, race, gender, etc.)
- GDPR/CCPA: Candidate data minimization and right-to-deletion support
- SOC 2 Type II for cloud deployment
- No retention of candidate data beyond processing

### Estimated Cost

| Item | Cost |
|------|------|
| Synthetic data generation | $1,500 |
| LoRA fine-tuning (compute) | $1,440 (4× A100 × 18h × $2/hr) |
| Benchmark creation | $1,000 |
| Validation & QA | $1,200 |
| **Total** | **~$5,140** |

---

## 6. TinyDoc-Government

### Industry Overview & Pain Points

Government agencies process identity documents and forms at massive scale:
- DMV offices process 100M+ ID documents annually
- Tax form processing backlogs cost billions in delayed refunds
- Voter registration verification requires manual cross-referencing
- Passport verification at borders demands <3-second processing
- Privacy regulations prohibit cloud processing of identity documents

### Document Types & Formats

| Document | Format | Complexity |
|----------|--------|------------|
| Passports (MRZ) | PDF, mobile photo | High |
| Driver's licenses (all 50 states) | JPEG, scan | High |
| Visas | PDF | High |
| Tax forms (W-2, 1040) | PDF | Medium |
| Voter registration forms | PDF, scanned | Medium |
| Permit applications | PDF | Medium-High |
| Birth certificates | PDF, scanned | Medium |

### Special Tokens

```
Identity: [DOC_TYPE], [ISSUING_COUNTRY], [ISSUING_STATE], [ID_NUMBER]
MRZ: [MRZ_LINE1], [MRZ_LINE2], [MRZ_CHECKSUM]
Dates: [DOB], [ISSUE_DATE], [EXPIRY_DATE]
Form fields: [FORM_ID], [FIELD_NAME], [FIELD_VALUE]
Agency: [AGENCY_CODE], [JURISDICTION]
Address: [STREET], [CITY], [STATE], [ZIP], [COUNTRY]
```

### Output Schemas

**ID Document Schema:**
```json
{
  "id_document": {
    "document_type": "passport|drivers_license|visa|state_id",
    "issuing_country": "string (ISO 3166-1)",
    "issuing_state": "string",
    "personal": {
      "surname": "string",
      "given_names": "string",
      "date_of_birth": "date",
      "sex": "M|F|X",
      "nationality": "string (ISO 3166-1)",
      "document_number": "string",
      "expiry_date": "date"
    },
    "mrz": {
      "line1": "string",
      "line2": "string",
      "valid": "boolean"
    },
    "address": {
      "street": "string",
      "city": "string",
      "state": "string",
      "zip": "string"
    },
    "photo_region": {
      "x": "integer",
      "y": "integer",
      "width": "integer",
      "height": "integer"
    },
    "confidence_score": "float"
  }
}
```

**Tax Form Schema:**
```json
{
  "tax_form": {
    "form_type": "W-2|1040|1099|W-4",
    "tax_year": "integer",
    "filer": {
      "name": "string",
      "ssn_last4": "string",
      "address": "string"
    },
    "employer": {
      "name": "string",
      "ein": "string"
    },
    "income": {
      "wages": "float",
      "federal_tax_withheld": "float",
      "social_security_wages": "float",
      "medicare_wages": "float"
    },
    "deductions": [
      {
        "type": "string",
        "amount": "float"
      }
    ],
    "filing_status": "string"
  }
}
```

### Training Data Strategy

**Volume**: 20,000 synthetic government documents

**Generation pipeline**:
1. **Passport generator**: Synthetic passports from 50+ countries with valid MRZ checksums
2. **Driver's license generator**: All 50 US states with realistic layouts and security features
3. **Tax form generator**: W-2, 1040, 1099 variants with realistic financial data
4. **Voter registration generator**: State-specific forms (10 state variants)
5. **Augmentation**: Laminate glare, lamination artifacts, UV-light simulation, worn/folded documents

**Data distribution**:
- Passports: 4,000
- Driver's licenses: 5,000
- Tax forms: 5,000
- Voter registration: 3,000
- Permit applications: 2,000
- Birth certificates: 1,000

### Fine-Tuning Recipe

| Parameter | Value |
|-----------|-------|
| Base model | TinyDoc-VLM |
| Method | QLoRA (4-bit quantized base) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj, gate_proj, up_proj, down_proj |
| Learning rate | 1e-4 (cosine schedule) |
| Epochs | 6 |
| Batch size | 16 (effective 64) |
| Max sequence length | 4096 |
| Warmup ratio | 0.05 |
| Optimizer | Paged AdamW 8-bit |
| Weight decay | 0.01 |
| FP precision | NF4 + BF16 compute |
| Training hardware | 4× A100 80GB |
| Estimated training time | ~30 hours |

### Evaluation Plan

**Custom benchmark**: `TinyDoc-Bench-Government` (1,000 annotated documents)

**Metrics**:
- MRZ extraction accuracy (character-level)
- Field extraction accuracy per document type
- Document classification accuracy
- ID number extraction accuracy (critical)
- End-to-end identity verification accuracy

**Target accuracy**:
| Document Type | Clean | Noisy |
|--------------|-------|-------|
| Passports | >98% | >92% |
| Driver's licenses | >96% | >86% |
| Tax forms | >96% | >88% |
| Voter registration | >94% | >83% |

### Deployment Considerations

- **Hardware target**: On-device / edge (mandatory for privacy)
- **Latency**: <3s per document (border control throughput requirement)
- **Integration**: Offline SDK for mobile verification apps
- **Model size**: ~2.5GB (QLoRA 4-bit quantized)
- **Key advantage**: Full offline capability — no identity data transmitted

### Compliance & Privacy

- **Privacy by design**: On-device processing — identity data never leaves device
- **NIST 800-76**: Biometric data handling for passport photos
- **REAL ID**: Supports REAL ID verification workflows
- **FedRAMP**: Architecture supports FedRAMP authorization for cloud-adjacent deployments
- **No PII retention**: Zero storage of extracted identity data

### Estimated Cost

| Item | Cost |
|------|------|
| Synthetic data generation | $2,200 |
| QLoRA fine-tuning (compute) | $2,400 (4× A100 × 30h × $2/hr) |
| Benchmark creation | $1,500 |
| Validation & QA | $2,000 |
| **Total** | **~$8,100** |

---

## Cross-Cutting Architecture

### Multi-Domain Training Without Catastrophic Forgetting

When training a single model on multiple domains, catastrophic forgetting is a primary concern. The following strategies mitigate this:

**Strategy 1: Sequential Training with Rehearsal Buffer**
1. Train on Domain A → save 5% of training data as rehearsal buffer
2. Train on Domain B with mixed buffer (80% Domain B + 20% rehearsal from A)
3. Repeat for each domain, accumulating rehearsal buffers
4. Final pass: Fine-tune on mixed data from all domains (10% each)

**Strategy 2: Domain-Specific LoRA Adapters (Recommended)**
1. Train base TinyDoc-VLM once
2. Train separate LoRA adapter per domain
3. At inference time, load the appropriate adapter based on document classification
4. Adapters are ~50-200MB each — hot-swappable in <100ms

**Strategy 3: Domain Prompt Tokens**
1. Add domain-specific system tokens: `[DOMAIN:FINANCE]`, `[DOMAIN:HEALTHCARE]`, etc.
2. Train single model with domain-prefixed prompts
3. Route documents to appropriate domain prompt at inference
4. Works well for domains with overlapping document structures

**Recommended approach**: Strategy 2 (domain-specific LoRA adapters) for production deployments where accuracy is critical. Strategy 3 for lightweight deployments where model size matters.

### Adapter Architecture

```
TinyDoc-VLM (frozen)
    │
    ├── LoRA-Finance (200MB)
    ├── LoRA-Healthcare (180MB)
    ├── LoRA-Legal (190MB)
    ├── LoRA-Logistics (170MB)
    ├── LoRA-HR (160MB)
    └── LoRA-Government (175MB)
```

**Adapter hot-swapping**:
```python
class TinyDocVariant:
    def __init__(self, base_model_path):
        self.model = load_model(base_model_path)
        self.current_adapter = None

    def load_adapter(self, domain: str):
        adapter_path = f"adapters/tinydoc-{domain}.safetensors"
        self.model.load_adapter(adapter_path)
        self.current_adapter = domain

    def predict(self, image, domain: str):
        if self.current_adapter != domain:
            self.load_adapter(domain)
        return self.model.generate(image, domain_token=domain)
```

**Adapter routing**: A lightweight document classifier (MobileNetV3, <5ms) determines the domain before routing to the appropriate LoRA adapter.

### Unified Benchmark Suite: TinyDoc-Bench

A comprehensive benchmark covering all verticals:

```
TinyDoc-Bench/
├── finance/
│   ├── invoices_clean/ (500 docs)
│   ├── invoices_noisy/ (500 docs)
│   ├── receipts/ (300 docs)
│   ├── tax_forms/ (200 docs)
│   └── financial_statements/ (200 docs)
├── healthcare/
│   ├── insurance_cards/ (300 docs)
│   ├── lab_reports/ (300 docs)
│   ├── prescriptions/ (300 docs)
│   ├── medical_forms/ (300 docs)
│   └── clinical_notes/ (300 docs)
├── legal/
│   ├── contracts/ (300 docs)
│   ├── ndas/ (200 docs)
│   ├── leases/ (200 docs)
│   └── court_filings/ (200 docs)
├── logistics/
│   ├── bills_of_lading/ (300 docs)
│   ├── customs_forms/ (200 docs)
│   ├── shipping_labels/ (200 docs)
│   └── packing_slips/ (200 docs)
├── hr/
│   ├── resumes/ (300 docs)
│   ├── job_postings/ (200 docs)
│   ├── i9_forms/ (100 docs)
│   └── offer_letters/ (100 docs)
└── government/
    ├── passports/ (200 docs)
    ├── drivers_licenses/ (300 docs)
    ├── tax_forms/ (200 docs)
    └── voter_registrations/ (200 docs)
```

**TinyDoc-Bench scoring**:
- Per-domain F1 score (weighted by field criticality)
- Cross-domain average (macro)
- Per-document-type accuracy
- End-to-end extraction accuracy
- Latency benchmark (p50, p95, p99)

**Leaderboard categories**:
1. **Best Overall**: Highest cross-domain average
2. **Best per Domain**: Highest score in each vertical
3. **Best Latency**: Lowest p95 latency at >90% accuracy
4. **Best Efficiency**: Highest accuracy per parameter count

### Release Roadmap

| Phase | Timeline | Deliverables |
|-------|----------|-------------|
| **Phase 1** | Month 1-2 | TinyDoc-Finance (highest ROI, largest TAM) |
| **Phase 2** | Month 2-3 | TinyDoc-Healthcare (strong compliance angle) |
| **Phase 3** | Month 3-4 | TinyDoc-HR (fastest to train, smallest model) |
| **Phase 4** | Month 4-5 | TinyDoc-Logistics (supply chain demand) |
| **Phase 5** | Month 5-6 | TinyDoc-Legal (complex documents, high value) |
| **Phase 6** | Month 6-7 | TinyDoc-Government (privacy-first positioning) |
| **Phase 7** | Month 7-8 | TinyDoc-Bench unified benchmark + leaderboard |
| **Phase 8** | Month 8-9 | Multi-domain adapter + routing model |

**Priority rationale**:
1. Finance first: Largest market ($4.5B OCR in finance), clear ROI, well-defined schemas
2. Healthcare second: Strongest differentiator (on-device + HIPAA), high compliance barrier to entry
3. HR third: Fastest to market, smallest data volume, SaaS distribution model
4. Logistics fourth: Supply chain digitization trend, international expansion potential
5. Legal fifth: High-value per document, but complex schemas require more R&D
6. Government sixth: Longest sales cycle, but highest volume and strategic value

### Total Estimated Program Cost

| Variant | Compute | Data | QA | Total |
|---------|---------|------|-----|-------|
| Finance | $3,840 | $2,000 | $3,500 | $9,340 |
| Healthcare | $2,880 | $3,000 | $4,500 | $10,380 |
| Legal | $3,200 | $2,500 | $3,800 | $9,500 |
| Logistics | $1,920 | $1,800 | $2,700 | $6,420 |
| HR | $1,440 | $1,500 | $2,200 | $5,140 |
| Government | $2,400 | $2,200 | $3,500 | $8,100 |
| **Multi-domain + Bench** | $1,500 | $2,000 | $3,000 | $6,500 |
| **Total** | **$17,180** | **$15,000** | **$23,200** | **$55,380** |

---

## Appendix A: LoRA Configuration Reference

```yaml
# Standard LoRA config (Finance, Legal, Logistics, HR)
lora:
  r: 32
  lora_alpha: 64
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM

# QLoRA config (Healthcare, Government)
qlora:
  r: 16
  lora_alpha: 32
  target_modules:
    - q_proj
    - v_proj
    - gate_proj
    - up_proj
    - down_proj
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4
```

## Appendix B: Synthetic Data Generation Tools

| Tool | Purpose | Used By |
|------|---------|---------|
| `docgen-finance` | Invoice/receipt generation | Finance |
| `docgen-medical` | Medical form/prescription generation | Healthcare |
| `docgen-legal` | Contract/clause generation | Legal |
| `docgen-logistics` | BOL/customs form generation | Logistics |
| `docgen-hr` | Resume/form generation | HR |
| `docgen-identity` | ID document generation | Government |
| `augment-pipeline` | Noise/augmentation pipeline | All |

## Appendix C: Hardware Reference

| Deployment | Hardware | Latency Target | Model Size |
|-----------|----------|---------------|------------|
| Cloud API | NVIDIA A10G | <1s | 4GB |
| Edge (premise) | NVIDIA Jetson AGX | <3s | 2.5-4GB |
| Edge (mobile) | Apple Neural Engine / Qualcomm NPU | <5s | 1.5-2GB |
| On-prem server | Intel Xeon + NVIDIA T4 | <2s | 4GB |
| CPU-only | Intel Xeon (INT8) | <5s | 2GB |
