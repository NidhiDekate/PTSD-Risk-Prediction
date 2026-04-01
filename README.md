# PTSD Risk Prediction — ML vs LLM on Tabular Clinical Data

A course project for AI in Healthcare (UT Austin MSAI) comparing traditional ML models against a large language model (Llama 3 70B) for predicting depression/PTSD risk from structured health survey data.

---

## The Idea

Can an LLM predict PTSD risk from patient health data as well as XGBoost? And if not — where exactly does it fail?

This project uses NHANES 2017-2018 survey data to build a binary risk classifier (PHQ-9 score ≥ 10 = high risk). Three ML models are trained as baselines, then Llama 3 70B is prompted in 8 different ways to make the same predictions. The main contribution is a detailed failure analysis showing *why* the LLM struggles — not just that it does.

---

## Dataset

**NHANES 2017-2018** (National Health and Nutrition Examination Survey)  
Source: [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017)  
Free and publicly available — no download needed, notebooks pull directly from CDC.

Four files used:
- `DPQ_J.XPT` — PHQ-9 depression screener (target label source)
- `DEMO_J.XPT` — Demographics (age, gender, income, education)
- `ALQ_J.XPT` — Alcohol use
- `SLQ_J.XPT` — Sleep habits

**Final dataset:** 5065 patients, 22 features, 8.3% high risk class.

---

## Project Structure

```
ptsd-risk-prediction/
├── README.md
├── requirements.txt
├── 01_data_setup.ipynb
├── 02_eda.ipynb
├── 03_preprocessing.ipynb
├── 04_baseline_models.ipynb
├── 05_llm_experiments.ipynb
└── 06_analysis.ipynb
├── 07_interpretability_debate.ipynb  # LLM interpretability + faithfulness + debate
```

---

## Models

**ML Baselines (trained on data):**
- Logistic Regression
- Random Forest  
- XGBoost + SHAP explainability

**LLM (zero training, general knowledge only):**
- Llama 3.3 70B via Groq API (free tier)
- 4 serialization formats × 2 prompting styles = 8 experiments

**The 4 prompt formats:**

| Format | Style |
|--------|-------|
| F1 — Basic | Key-value pairs |
| F2 — Sentence | Natural language description |
| F3 — Clinical | Compact clinical notation |
| F4 — SHAP-guided | Uses XGBoost SHAP values to highlight top features per patient |

Format 4 is the unique contribution — instead of blindly serializing all features, it uses per-patient SHAP values from XGBoost to tell the LLM which features matter most for that specific patient.

---

## Results

**ML Models (full test set, n=1013):**

| Model | F1 Macro | AUC |
|-------|----------|-----|
| Logistic Regression | 0.969 | 1.000 |
| XGBoost | 0.940 | 0.998 |
| Random Forest | 0.912 | 0.995 |

**LLM (sample n=200, reliable formats only):**

| Experiment | F1 Macro | Valid Predictions |
|------------|----------|-------------------|
| F1 zero-shot | 0.840 | 200/200 |
| F1 few-shot | 0.831 | 200/200 |
| F2 zero-shot | 0.831 | 200/200 |

More complex formats (F3, F4) caused the LLM to give verbose responses instead of clean 0/1 outputs, resulting in low valid prediction counts.

---

## Key Findings

1. **ML models outperform LLM** — best LLM F1 is 0.84 vs XGBoost 0.94, a gap of ~0.10

2. **LLM never missed a high risk patient** — 0 false negatives, 12 false positives

3. **All LLM errors cluster on borderline cases** — every false positive had PHQ score of 8 or 9, just below the threshold of 10

4. **Failures are format-independent** — 9 patients failed across all 3 reliable formats, confirming the issue is not about prompt design

5. **The LLM reasons clinically, not statistically** — it flags PHQ=9 as concerning (medically reasonable) but our label says low risk (PHQ < 10). This reveals a fundamental difference between LLM contextual reasoning and threshold-based classification

6. **Complex prompts hurt reliability** — clinical and SHAP-guided formats caused unparseable responses, showing LLMs become less predictable as prompt complexity increases

7. **LLM accuracy drops by severity** — 100% accurate on PHQ>12, only 40% on PHQ 10-12

8. **Right for wrong reasons** — LLM explanations moderately faithful to SHAP (avg 0.69) but 2/15 patients had correct predictions with wrong reasoning

9. **Debate did not help** — structured two-LLM debate on borderline cases showed no improvement, judge consistently sided with high risk

10. **Core limitation confirmed** — LLMs apply clinical conservatism and cannot learn hard numerical thresholds without fine-tuning on labeled data 

---

## How to Run

### Requirements
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap groq
```

Or install from requirements.txt:
```
pip install -r requirements.txt
```

### Setup
1. Clone this repo
2. Open notebooks in Google Colab (recommended) or Jupyter
3. Mount Google Drive and set your base path in Cell 1 of each notebook
4. For notebook 05, add your Groq API key (free at [console.groq.com](https://console.groq.com))
5. Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06

### Note on Data
Notebooks 01 downloads NHANES files automatically from the CDC website. No manual download needed.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Everything |
| Pandas, NumPy | Data handling |
| Scikit-learn | ML models, preprocessing, metrics |
| XGBoost | Gradient boosting baseline |
| SHAP | Explainability + SHAP-guided prompts |
| Groq API | Free access to Llama 3 70B |
| Matplotlib, Seaborn | Visualizations |
| Google Colab | Development environment |

---

## Limitations

- PHQ-9 is a depression proxy, not a direct PTSD diagnosis label
- LLM experiments used a sample of 200 patients due to API rate limits
- Some prompt formats produced unparseable LLM responses
- Results may vary with different LLM models or prompt designs

---

## Future Work

- Use actual PTSD diagnosis data instead of PHQ proxy
- Run LLM experiments on full test set with better rate limit handling
- Try medical-specific LLMs (Med-PaLM, BioMedLM)
- Explore chain-of-thought prompting for clinical reasoning
- Test whether LLMs perform better on narrative clinical notes vs structured data

---

## Author

Nidhi — MSAI Student, UT Austin  
AI in Healthcare — High Risk Project  
April 2026
