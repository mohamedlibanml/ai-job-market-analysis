# -*- coding: utf-8 -*-
"""
MSIS 502 Project - Predicting AI Job Market

Analyzes 2,000+ job postings from Kaggle to predict salary and understand
career paths in AI/ML roles across 5 countries.

Author: Mohamed Liban
Date: 2026
"""

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
    uploaded_filename = filename
    print(f'User uploaded file "{uploaded_filename}" ({len(uploaded[filename])} bytes)')

import pandas as pd, numpy as np, re
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 50); pd.set_option('display.width', 200)

RAW = pd.read_csv(uploaded_filename)
print(RAW.shape)
RAW.head(3)

# ============================================================================
# DATA CLEANING
# ============================================================================

audit = pd.DataFrame({'dtype':RAW.dtypes.astype(str),'n_missing':RAW.isna().sum(),
    'pct_missing':(RAW.isna().mean()*100).round(1),'n_unique':RAW.nunique()})
display(audit)
print("Exact duplicate rows:", RAW.duplicated().sum())
print("Duplicate descriptions:", RAW.job_description.duplicated().sum())
print("Truncated:", f"{RAW.job_description.fillna('').str.strip().str.endswith('…').mean()*100:.1f}%")
print("Currencies:", RAW.currency.unique())

df = RAW.copy()
for c in ['job_title','company','country','city','remote_type','experience_level','source']:
    df[c] = df[c].astype('string').str.strip().str.replace(r'\s+',' ',regex=True)
df['company'] = df.company.fillna('Unknown')
df['remote_type'] = df.remote_type.replace('Unspecified', pd.NA)
df['is_truncated'] = df.job_description.fillna('').str.strip().str.endswith('…')
print(df.remote_type.value_counts(dropna=False))

df['posted_date'] = pd.to_datetime(df.posted_date, errors='coerce')
df['posted_year'] = df.posted_date.dt.year
df['posted_month'] = df.posted_date.dt.to_period('M').astype('string')
df['date_missing'] = df.posted_date.isna()
df['is_recent'] = (df.posted_date >= pd.Timestamp('2025-01-01')).astype('boolean')
df.loc[df.posted_date.isna(), 'is_recent'] = pd.NA
print(df.groupby('source').posted_date.apply(lambda s: s.isna().mean()).round(3))
print(df.posted_year.value_counts().sort_index())
print(df.is_recent.value_counts(dropna=False))

for c in ['salary_min','salary_max']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df.loc[df[c] < 5000, c] = np.nan
sw = df.salary_min > df.salary_max
df.loc[sw, ['salary_min','salary_max']] = df.loc[sw, ['salary_max','salary_min']].values
df['salary_mid'] = df[['salary_min','salary_max']].mean(axis=1)
q1, q3 = df.salary_mid.quantile([.25,.75]); iqr = q3-q1
df['salary_outlier'] = df.salary_mid.notna() & ((df.salary_mid < q1-1.5*iqr)|(df.salary_mid > q3+1.5*iqr))
df['salary_missing'] = df.salary_mid.isna()
print(f"swapped {sw.sum()} | usable {df.salary_mid.notna().sum()} | outliers {df.salary_outlier.sum()}")
display(df.salary_mid.describe().round(0))

BOILER = re.compile(r"this jobot job is hosted by.*?a bit about us:", re.I|re.S)
df['desc_key'] = (df.job_description.fillna('')
                     .str.replace(BOILER, ' ', regex=True)
                                          .str.lower().str.replace(r'[^a-z0-9]', '', regex=True).str[:300])
df['title_key'] = df.job_title.fillna('').str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
df['posting_group'] = df.groupby(['company','title_key','desc_key'], dropna=False).ngroup()
df['group_size'] = df.groupby('posting_group').job_title.transform('size')
f = df.sort_values(['posting_group','posted_date'])
df['is_group_primary'] = ~f.duplicated('posting_group', keep='first')
print("groups:", df.posting_group.nunique(), "from", len(df))
display(df.nlargest(5,'group_size')[['company','job_title','country','group_size']])

ROLES = [('Data Scientist',r'data scientist'),('ML Engineer',r'machine learning engineer|\bml engineer'),
 ('MLOps / Platform',r'mlops|ml ops|platform engineer'),
 ('Research Scientist',r'research scientist|research engineer|applied scientist'),
 ('AI / GenAI Engineer',r'\bai engineer|genai|generative ai|llm engineer|prompt'),
 ('Data Engineer',r'data engineer'),
 ('Data / Business Analyst',r'data analyst|business analyst|business intelligence'),
 ('Computer Vision',r'computer vision|perception'),('NLP Engineer',r'\bnlp\b|natural language'),
 ('Software Engineer',r'software engineer|developer|full.?stack|backend|frontend'),
 ('Manager / Director',r'manager|director|head of|\bvp\b|vice president|lead\b')]

def role(t):
    t = (t or '').lower()
    for n,p in ROLES:
        if re.search(p,t): return n
    return 'Other / Specialist'

df['role_family'] = df.job_title.apply(role)
display(df.role_family.value_counts())

SKILL_VOCAB = {
 "Python":[r"\bpython\b",r"\bpandas\b",r"\bnumpy\b"],"SQL":[r"\bsql\b",r"\bpostgres",r"\bmysql\b"],
 "R":[r"(?:Python|SQL|SAS|MATLAB|Tableau|SPSS|TensorFlow)\s*[,/&]\s*R\b(?!&)",
      r"\bR\b(?!&)\s*[,/]\s*(?:Python|SQL|SAS|MATLAB)",
      r"\bR\b(?!&)\s*\(\s*(?:purrr|tidyr|dplyr|ggplot|shiny)",
      r"(?:using|utilizing|with)\s+R\b(?!&)",r"\bR programming\b",r"\bRStudio\b",r"\btidyverse\b|\bggplot2?\b"],
 "Java":[r"\bjava\b(?!script)"],"Scala":[r"\bscala\b"],"C++":[r"c\+\+"],
 "PyTorch":[r"\bpytorch\b"],"TensorFlow":[r"\btensorflow\b"],
 "Scikit-learn":[r"scikit[- ]?learn",r"\bsklearn\b"],"Keras":[r"\bkeras\b"],
 "NLP":[r"\bnlp\b",r"natural language processing"],"Computer Vision":[r"computer vision",r"\bopencv\b"],
 "LLM":[r"\bllms?\b",r"large language model"],"RAG":[r"\brag\b",r"retrieval[- ]augmented"],
 "LangChain":[r"\blangchain\b",r"\bllamaindex\b"],"OpenAI":[r"\bopenai\b",r"\bgpt-?\d",r"\bchatgpt\b"],
 "Fine-tuning":[r"fine[- ]?tun"],"Prompt Engineering":[r"prompt engineering"],
 "Transformers":[r"\btransformers?\b",r"\bbert\b",r"hugging ?face"],
 "Generative AI":[r"generative ai",r"\bgenai\b"],"Deep Learning":[r"deep learning",r"neural network"],
 "AWS":[r"\baws\b",r"amazon web services",r"\bsagemaker\b"],"Azure":[r"\bazure\b"],
 "GCP":[r"\bgcp\b",r"google cloud",r"\bbigquery\b",r"\bvertex ai\b"],
 "Docker":[r"\bdocker\b"],"Kubernetes":[r"\bkubernetes\b",r"\bk8s\b"],
 "Spark":[r"\bspark\b",r"\bpyspark\b"],"Hadoop":[r"\bhadoop\b"],"Airflow":[r"\bairflow\b"],
 "MLflow":[r"\bmlflow\b"],"MLOps":[r"\bmlops\b",r"model deployment",r"ml pipeline"],
 "CI/CD":[r"\bci/cd\b"],"Terraform":[r"\bterraform\b"],"Git":[r"\bgit\b",r"\bgithub\b"],
 "Tableau":[r"\btableau\b"],"Power BI":[r"power ?bi"],"Excel":[r"\bexcel\b"],
 "Statistics":[r"\bstatistics\b",r"\bstatistical\b",r"\bregression\b"],
 "A/B Testing":[r"a/b test",r"experimentation"],"ETL":[r"\betl\b",r"data pipeline"],
 "Snowflake":[r"\bsnowflake\b"],"Databricks":[r"\bdatabricks\b"]}

CASE_SENSITIVE = {"R"}
COMPILED = {k:[re.compile(p, 0 if k in CASE_SENSITIVE else re.I) for p in v]
            for k,v in SKILL_VOCAB.items()}

BOILER = re.compile(r"this jobot job is hosted by.*?a bit about us:", re.I|re.S)
df['clean_text'] = (df.job_title.fillna('')+' . '+df.job_description.fillna('')).str.replace(BOILER,' ',regex=True)

def extract(t):
    if not isinstance(t,str): return []
    return sorted([s for s,pats in COMPILED.items() if any(r.search(t) for r in pats)])

df['skills_text'] = df.clean_text.apply(extract)
print("vocab size:", len(SKILL_VOCAB))

_lab = df.required_skills.fillna('').apply(lambda x:[s.strip() for s in x.split(',') if s.strip()])
SUSPECT = {"R"}
df['skills_labeled'] = [[s for s in lst if not (s in SUSPECT and not any(r.search(t) for r in COMPILED[s]))]
                        for lst,t in zip(_lab, df.clean_text)]
df['skills'] = [sorted(set(a)|set(b)) for a,b in zip(df.skills_text, df.skills_labeled)]
df['n_skills'] = df.skills.str.len()
df['skills_observable'] = df.n_skills > 0

lab_R = df[_lab.apply(lambda x:'R' in x)]
print(f"'R' labels: {len(lab_R)} | contain 'R&D': {lab_R.clean_text.str.contains('R&D').sum()}"
      f" | no standalone R: {(~lab_R.clean_text.str.contains(r'\bR\b(?!&)')).sum()}")
print("observable:", f"{df.skills_observable.mean():.1%}")

lab = df[df.required_skills.notna()].copy()
lab['truth'] = lab.required_skills.str.split(',').apply(lambda x:{s.strip() for s in x if s.strip()})
lab['pred'] = lab.clean_text.apply(lambda t: set(extract(t)))
tp = lab.apply(lambda r: len(r.truth & r.pred),axis=1).sum()
fn = lab.apply(lambda r: len(r.truth - r.pred),axis=1).sum()
fp = lab.apply(lambda r: len(r.pred - r.truth),axis=1).sum()
print(f"validated on {len(lab)} | recall {tp/(tp+fn):.1%} | precision(lower bound) {tp/(tp+fp):.1%}")

clean = df.drop(columns=['desc_key','clean_text','skills_text','skills_labeled'])
analysis = clean[clean.is_group_primary].copy()
skill_frame = analysis[analysis.skills_observable].copy()
skills_long = skill_frame.explode('skills').rename(columns={'skills':'skill'})
for n,f in [('clean',clean),('analysis',analysis),('skill_frame',skill_frame),('skills_long',skills_long)]:
    print(f"{n:12s}{f.shape}")

raw_pct = clean[clean.skills_observable].explode('skills').skills.value_counts(normalize=True)*100
ded_pct = skills_long.skill.value_counts(normalize=True)*100
cmp = pd.DataFrame({'raw_pct':raw_pct.round(1),'dedup_pct':ded_pct.round(1)})
cmp['rank_raw'] = raw_pct.rank(ascending=False).astype(int)
cmp['rank_dedup'] = ded_pct.rank(ascending=False).astype(int)
cmp['rank_shift'] = cmp.rank_raw - cmp.rank_dedup
display(cmp.reindex(cmp.rank_shift.abs().sort_values(ascending=False).index).head(8))

MIN_COUNT = 20
keep = skills_long.skill.value_counts()
keep = keep[keep >= MIN_COUNT].index.tolist()
mh = pd.DataFrame({f'skill_{s}': skill_frame.skills.apply(lambda x: int(s in x)) for s in keep},
                  index=skill_frame.index)
model_df = pd.concat([skill_frame[['salary_mid','role_family','country','experience_level',
                                   'remote_type','n_skills','posted_year','is_recent']], mh], axis=1)
model_df['remote_type'] = model_df['remote_type'].fillna('Unspecified')
print("model frame:", model_df.shape, "| salary targets:", model_df.salary_mid.notna().sum())
model_df.head()

print(f"\nData cleaning complete!")
print(f"Raw postings: {len(RAW):,}")
print(f"Deduplicated: {len(analysis):,}")
print(f"With salary data: {model_df.salary_mid.notna().sum():,}")
print(f"With skill data: {skill_frame.shape[0]:,}")
