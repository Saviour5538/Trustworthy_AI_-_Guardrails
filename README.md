# 🛡️ Trustworthy AI & Guardrails

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Free_API-orange)](https://console.groq.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference_API-yellow)](https://huggingface.co)
[![Qdrant](https://img.shields.io/badge/Qdrant-In--Memory-purple)](https://qdrant.tech)
[![Tests](https://img.shields.io/badge/Tests-130_passed-brightgreen)](#testing)

> **Prepared by:** Adarsh Kumar Singh
> **Topic:** Trustworthy AI & Guardrails — Research & Implementation

A production-grade demonstration of **multi-layer AI safety** using **GenAI-powered guardrails** — showing how to make AI systems that are honest, safe, fair, and accountable. Built with Groq (LLM API), Qdrant (vector database), and HuggingFace (embeddings API).

---

## 🎯 What This Demonstrates

| Concept | Implementation |
|---|---|
| **LLM Injection Detection** | LLM classifier understands intent — catches novel jailbreaks regex never saw |
| **LLM Toxicity Check** | Context-aware classification — allows education, blocks harmful intent |
| **PII Redaction** | 6 PII types (incl. Aadhaar, PAN) redacted before LLM sees them |
| **LLM Output Reviewer** | LLM reviews its own response before user sees it |
| **RAG Hallucination Check** | Qdrant + HuggingFace API verifies AI response against 59 facts |
| **Constitutional AI** | Anthropic's self-critique + revision loop — 3 Groq API calls |
| **Accountability** | Full audit trail with CSV export |
| **Red-teaming** | 130 automated tests covering attack vectors and edge cases |

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/Saviour5538/Trustworthy_AI_-_Guardrails.git
cd Trustworthy_AI_-_Guardrails

# 2. Install
pip install groq streamlit plotly pandas python-dotenv qdrant-client

# 3. Set API keys in .env file
GROQ_API_KEY=gsk_your_groq_key
HUGGINGFACE_API_KEY=hf_your_hf_key

# 4. Run
streamlit run app.py
```

App opens at **http://localhost:8501**

> **Get free API keys:**
> - Groq: https://console.groq.com
> - HuggingFace: https://huggingface.co/settings/tokens

---

## 🏗️ Architecture — 9-Step Pipeline

```
User Input
    ↓
[INPUT GUARDRAILS]
  ├── Step 2: Injection Detection   ← LLM Classifier (Groq API)
  ├── Step 3: Toxicity Check        ← LLM Classifier (Groq API)
  ├── Step 4: PII Detection         ← Regex (Aadhaar, PAN, SSN, email, phone, card)
  └── Input Length Validation       ← Python
    ↓ sanitised input only
[GROQ LLM — LLaMA 3.3 70B]
  └── Step 5: Response Generation   ← Core GenAI
    ↓ raw AI response
[OUTPUT GUARDRAILS]
  ├── Step 6: Output Harm Check     ← LLM Reviewer (Groq API)
  ├── Step 7: Hallucination Check   ← RAG: Qdrant + HuggingFace Embeddings
  └── Step 8: Constitutional AI     ← 3-step LLM self-critique (Groq API)
    ↓
✅ Step 9: Safe Final Response → User
```

**5 out of 9 steps are GenAI-powered.** PII detection stays as regex — precise format matching is the right tool for that job.

---

## 🤖 GenAI Components

### LLM Classifiers (Steps 2 & 3)
Instead of hardcoded regex patterns, we send the user input to LLaMA 3.3 with a structured prompt:
```
VERDICT: SAFE or INJECTION
REASON: one sentence
CONFIDENCE: HIGH / MEDIUM / LOW
```
This catches **novel jailbreaks** that regex patterns never saw — e.g. *"Disregard your earlier guidance"*.

### RAG Hallucination Check (Step 7)
Uses **Retrieval Augmented Generation** with a 59-fact knowledge base:
1. AI response → HuggingFace API → 384-dim vector embedding
2. Qdrant in-memory searches for 5 most relevant facts
3. LLM compares response against retrieved facts
4. Flags mismatches as hallucinations

```python
# Tools used — no C++ / no local PyTorch required
qdrant-client   # Pure Python vector database (in-memory)
HuggingFace     # Inference API for sentence-transformers/all-MiniLM-L6-v2
```

### Constitutional AI (Step 8)
Three Groq API calls per message:
```
Call 1 → Generate initial response
Call 2 → Critique against 6 ethical principles
Call 3 → Revise if violations found
```
Based on Bai et al. (2022) *Constitutional AI: Harmlessness from AI Feedback* — Anthropic.

---

## 🖥️ Application Tabs

| Tab | What It Shows |
|---|---|
| 💬 **Chat** | Live 9-step pipeline with confidence score bars |
| 🔍 **Inspector** | Test any input through all guardrails — see every score |
| 🧬 **Constitutional AI** | 3-column view: Original → Critique → Revised |
| 📊 **Analytics** | Block rates, PII detections, CAI revisions + CSV export |
| 📚 **Learn** | Built-in reference guide — all pillars explained |
| ⚖️ **Benchmark** | Before/after comparison + evaluation vs production systems |
| 🗺️ **Architecture** | Interactive diagram — hover nodes to see pillar + technology |

---

## 🔒 Guardrails

### Input Guardrails
| Step | Check | Technology | Catches |
|---|---|---|---|
| 2 | Prompt Injection | LLM Classifier | DAN, jailbreaks, novel overrides |
| 3 | Toxicity | LLM Classifier | Harmful intent — context-aware |
| 4 | PII Detection | Regex | Email, phone, SSN, Aadhaar, PAN, credit card |

### Output Guardrails
| Step | Check | Technology | Catches |
|---|---|---|---|
| 6 | Output Harm | LLM Reviewer | Harmful instructions in AI response |
| 7 | Hallucination | RAG + LLM | Factual errors verified against knowledge base |
| 8 | Constitutional AI | LLM Self-Critique | Bias, unfairness, ethical violations |

### Constitutional AI — 6 Principles
```
1. Be truthful      — only assert things believed to be true
2. Be harmless      — never assist with illegal or dangerous activities
3. Be helpful       — provide useful, accurate, complete information
4. Respect privacy  — never reveal personal data
5. Be fair          — avoid stereotypes and discriminatory language
6. Be transparent   — acknowledge uncertainty honestly
```

---

## 🧪 Testing

```bash
python test_guardrails.py
```

```
Running 136 tests across 14 categories:
  ▸ GuardrailResult Structure        (6 tests)
  ▸ Prompt Injection (Regex)         (22 tests)
  ▸ Toxicity Detection (Regex)       (20 tests)
  ▸ PII Detection & Redaction        (19 tests)
  ▸ Input Length Validation          (9 tests)
  ▸ Input Pipeline (run_all)         (6 tests)
  ▸ Output Harm Detection (Regex)    (9 tests)
  ▸ Hallucination Signals (Heuristic)(7 tests)
  ▸ Output Length/Quality            (7 tests)
  ▸ Output PII Leak                  (5 tests)
  ▸ Output Pipeline (run_all)        (3 tests)
  ▸ RAG Knowledge Base               (6 tests)
  ▸ End-to-End Integration           (8 tests)
  ▸ Edge Cases & Stress Tests        (9 tests)

✅ ALL 136 TESTS PASSED
```

> **Note on test scope:** Steps 2, 3, and 6 use live Groq API LLM classifiers which are tested through the live demo (integration testing). The test suite covers all deterministic components — regex fallbacks, PII redaction, pipeline structure, RAG knowledge base, and edge cases — without requiring API keys.

---

## 🗺️ Trustworthy AI Pillar Mapping

| Pillar | Component | Technology | Industry Equivalent |
|---|---|---|---|
| **Robustness** | Injection Detection | LLM Classifier | OWASP GenAI / LlamaGuard |
| **Safety** | Toxicity Check | LLM Classifier | OpenAI Moderation API |
| **Safety** | Output Harm Check | LLM Reviewer | Meta LlamaGuard |
| **Privacy** | PII Redaction | Regex | Amazon Bedrock Guardrails |
| **Transparency** | Hallucination Check | RAG + Qdrant + HF | RAG with large vector DBs |
| **Fairness** | Constitutional AI | LLM Self-Critique | Anthropic CAI (2022) |
| **Accountability** | Audit Log + CSV | Streamlit + Pandas | NIST AI RMF |

---

## 📁 Files

```
app.py               ← Main Streamlit application (2000+ lines)
test_guardrails.py   ← 136 automated tests (no API key needed)
requirements.txt     ← Python dependencies
README.md            ← This file
.gitignore           ← Excludes .env and __pycache__
```

---

## 📚 References

- Bai et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* Anthropic.
- Ouyang et al. (2022). *Training LMs to Follow Instructions with Human Feedback.* OpenAI.
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* Facebook AI.
- NIST (2023). *AI Risk Management Framework 1.0.*
- European Parliament (2024). *EU Artificial Intelligence Act.*
- Ministry of Electronics and IT (2023). *Digital Personal Data Protection Act.* India.

---

**Author:** Adarsh Kumar Singh | Topic: Trustworthy AI & Guardrails