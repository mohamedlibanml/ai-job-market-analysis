# AI Job Market Analysis

**Predicting salary and opportunity in AI/ML roles across 5 countries using job posting data.**

## Why This Project?

Salary data is hidden — most employers don't publish compensation for 80-95% of roles outside the US and UK. This project uses machine learning on 2,000+ job postings to uncover:

- **Which career paths pay the most** (and which offer the most opportunities)
- **How much individual skills are worth** (LLM, Python, AWS, etc.)
- **How steep the career ladder is** in each role (Junior → Management)
- **Work-life trade-offs** (does remote work come with a pay discount?)

## Key Findings

### 🎯 Role Beats Location
- Choosing **role** (Data Scientist vs. MLOps) drives **12% more salary difference** than choosing remote vs. onsite
- Best-paying roles: **Computer Vision** (~$152k median), **NLP Engineer** (~$122k), **MLOps/Platform** (~$117k)
- Highest-volume roles: **AI/GenAI Engineer**, **Data Scientist**, **ML Engineer** (1,900+ postings combined)

### 💰 LLM Skills Command a Premium
- The model predicts **+$1,832 salary lift** for LLM proficiency
- LLM demand has permeated every role family — even Software Engineers see it in ~38% of postings
- Traditional skills (SQL, Python, Statistics) still matter but are now table stakes, not differentiators

### 📈 Experience Ladders Vary by Role
- **MLOps/Platform**: Steepest ladder (+$19.5k from Junior to Management)
- **Data Scientist**: Nearly flat ladder (+$9k total) — experience matters less here
- **Entry barrier**: Only 2 of 11 career paths have substantial Junior-level openings

### 🌍 Geography Matters (But Data Is Uneven)
- **US salaries**: ~71% higher than UK (median $141k vs. $82.5k)
- **Salary disclosure**: Nearly 100% in US/UK, but only 7-19% in Canada, Germany, Australia
- Findings are US/UK-focused; other countries need more data

### 🏠 Remote Work Trade-off
- ~85-100% of postings across all roles offer Remote or Hybrid options
- **Paradox**: Onsite jobs pay highest (~$127k median), remote lowest (~$108k)
  - May reflect that senior, high-paying roles are more often required onsite

---

## Methodology

### Data
- **Source**: Kaggle AI Jobs Dataset (~2,000 deduplicated postings)
- **Countries**: United States, United Kingdom, Canada, Germany, Australia
- **Time span**: ~18 months of postings
- **Role families**: 11 categories (Data Scientist, ML Engineer, AI/GenAI Engineer, etc.)

### Cleaning & Feature Engineering
- **Deduplication**: Identified 1,100+ duplicate postings using company + job title + description similarity
- **Salary standardization**: Converted to annual USD; removed outliers (1.5× IQR); corrected swapped min/max
- **Skill extraction**: Regex-based parser on 42-term vocabulary (Python, SQL, LLM, RAG, AWS, Kubernetes, etc.)
  - Validated against labeled data: **56% recall**, **87% precision**
  - Skill detection from truncated descriptions (see Limitations)

### Modeling
**Target**: `log(salary_mid)` — logarithmic because salary is right-skewed

**Features**: 
- Role family (11 categories)
- Country (5 categories)
- Experience level (5 tiers: Junior → Management)
- Remote type (Remote, Hybrid, Onsite)
- Individual skill flags (42 binary indicators)
- Total skill count

**Models Tested**:
| Model | CV R² | Test MAE | MAPE |
|-------|-------|----------|------|
| Ridge (Linear) | 0.417 | $24.6k | 18.0% |
| Random Forest | 0.432 | $23.8k | 17.4% |
| **Gradient Boosting** | **0.448** | **$23.0k** | **16.8%** |

**Interpretation**: All three models performed similarly (error bars overlap). The ~$23k mean error reflects predicting individual salaries from job postings alone — adding structured compensation data (Levels.fyi, H-1B, employer bands) would likely improve this significantly.

---

## What's Inside

- **`msis_502_project_predicting_ai_job_market.py`** — Complete analysis pipeline (data cleaning → EDA → modeling)
  - Data audit and deduplication
  - Salary and skill extraction
  - 6+ exploratory visualizations
  - Gradient Boosting model with feature importance
  - Scenario analysis: "What if I add this skill?" "Which country pays most?"

- **`requirements.txt`** — Python dependencies

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/mohamedlibanml/ai-job-market-analysis.git
cd ai-job-market-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the dataset
Download `ai_jobs_global.csv` from [Kaggle](https://www.kaggle.com/datasets/asaniczka/ai-jobs-dataset) and save to the repo directory.

### 4. Run the analysis
```bash
# Option A: Run as a Python script
python msis_502_project_predicting_ai_job_market.py

# Option B: Open in Jupyter Notebook (recommended for exploration)
jupyter notebook msis_502_project_predicting_ai_job_market.py
```

The script will:
- Load and clean the data
- Generate 10+ visualizations
- Train the salary prediction model
- Output scenario analyses and insights

---

## Limitations & Honest Talk

### Data Limitations
1. **Truncated descriptions (97.5%)** — Job descriptions cut off at ~500 characters, removing the requirements section
   - Result: Only 40% of postings have observable skills; undercounts later-mentioned tools
   
2. **Title-echo bias** — 51.6% of detected skills come from the job title itself
   - "Computer Vision Engineer" registers as 100% proficiency in Computer Vision
   - Inflates coverage for roles named after tools; underestimates demand for generalist roles
   
3. **Salary disclosure is uneven** — 100% in US/UK, but only 7-19% elsewhere
   - Medians for Canada, Germany, Australia are unreliable; based on small, unrepresentative samples
   
4. **Vocabulary-bound** — Skill parser knows only 42 terms
   - Soft skills (communication, project management), BI tools (Tableau, Power BI), and niche frameworks are invisible

5. **Snapshot, not trend** — Single time period; can't forecast how skills go in/out of favor

### Model Limitations
1. **Moderate explanatory power** — R² ≈ 0.45 means the model explains ~45% of salary variance
   - The other ~55% comes from employer, team size, equity, location city (not just country), domain expertise, and factors job postings don't capture
   
2. **Structural data missing** — No access to:
   - Employer size, funding stage, or profitability
   - Pay bands per title/level (H-1B visa data, Levels.fyi, Glassdoor internals)
   - Equity, sign-on bonus, or location cost-of-living adjustment

---

## For TPM / Product Hiring Managers

This project demonstrates:

✅ **Data-driven decision making** — Quantifies career tradeoffs (role vs. location, experience vs. specialization) rather than relying on opinions or anecdotes

✅ **Technical literacy** — Builds, validates, and interprets ML models; explains why one algorithm outperforms another (even slightly)

✅ **Product thinking** — Frames analysis around user decisions: *"If I'm a mid-level engineer choosing roles, what should I optimize for? If I'm switching to AI, where's the entry barrier?"*

✅ **Communication** — Translates model outputs into actionable insights; honest about limitations; uses visuals to clarify

✅ **Ownership** — Took a raw dataset and shipped a complete, reproducible analysis

---

## Next Steps to Improve This

1. **Untruncated descriptions** — Partner with Kaggle or job APIs to get full text; would double skill observability and eliminate title-echo bias

2. **Structured compensation data** — Join with Levels.fyi, H-1B filings, or Glassdoor salary data; would likely push R² from 0.45 to 0.60+

3. **Time series** — Extend to 2+ years; forecast which skills are rising/falling fastest (LLM adoption curve, decline of Spark, etc.)

4. **Broader geography** — Add more countries; weight by cost-of-living to test if US salary premium generalizes

5. **LLM-based parser** — Replace regex skill extraction with a fine-tuned NER model; capture terms outside our 42-word vocab

6. **Post-stratification weighting** — Correct for bias toward roles named after tools; weight by true hiring volume

---

## Tech Stack

- **Python 3.8+**
- **Data**: Pandas, NumPy
- **Modeling**: Scikit-learn (Ridge, Random Forest, Gradient Boosting)
- **Visualization**: Matplotlib, Seaborn
- **Notebook**: Jupyter

---

## Author

**Mohamed Liban** | MSIS Student @ University of Washington  
Aspiring Technical Product Manager

---

## License

This project uses publicly available Kaggle data. The analysis, visualizations, and insights are open for educational and non-commercial use.

---

## Questions or Feedback?

Open an issue or reach out. I'm happy to discuss the methodology, limitations, or specific findings!
