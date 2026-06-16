"""Cleaning and feature engineering for ESADE DSF26 — v2.

Differences vs prep.py:
- Z-score normalization of credit scores uses TRAIN-ONLY stats (no test peeking)
- Adds explicit missingness indicators for ann_income, other_income, prev_default
- Keeps the same dropped columns (internal_code, external_pd_score, religion, race)
"""
import pandas as pd
import numpy as np

DROP_TAMPERED = ['internal_code']
DROP_TEST_MISSING = ['external_pd_score']
DROP_PROTECTED = ['religion', 'race']


def load(data_dir):
    tr = pd.read_csv(data_dir + "train.csv")
    te = pd.read_csv(data_dir + "test.csv")
    return tr, te


def parse_dates(s):
    iso = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
    mon = pd.to_datetime(s, format='%b-%Y', errors='coerce') + pd.Timedelta(days=14)
    return iso.fillna(mon)


def engineer(df):
    out = pd.DataFrame(index=df.index)
    d = parse_dates(df['date'])
    out['year'] = d.dt.year
    out['month'] = d.dt.month

    inc = pd.to_numeric(df['ann_income'], errors='coerce')
    out['ann_income_missing'] = inc.isna().astype(int)
    out['ann_income'] = np.where(inc < 700, inc * 1000, inc)
    oth = pd.to_numeric(df['other_income'], errors='coerce')
    out['other_income_missing'] = oth.isna().astype(int)
    out['other_income'] = np.where((oth > 0) & (oth < 700), oth * 1000, oth)
    out['total_income'] = out['ann_income'].fillna(0) + out['other_income'].fillna(0)
    out['amount'] = df['amount']
    out['amt_to_income'] = df['amount'] / (out['total_income'] + 1)
    out['log_income'] = np.log1p(out['total_income'])

    age = df['age'].copy()
    age = age.fillna(out['year'] - df['birth_year'])
    out['age'] = age

    out['prev_default_missing'] = df['prev_default'].isna().astype(int)
    out['prev_default'] = df['prev_default'].map(
        {'0': 0, 'No': 0, '1': 1, 'Yes': 1, 0: 0, 1: 1}
    ).astype(float)
    out['highest_ed'] = df['highest_ed']
    out['kids'] = df['kids']
    out['vip'] = df['vip'].astype(str).map({'True': 1, 'False': 0})

    out['risk12'] = df['risk_indicator_1'].fillna(df['risk_indicator_2'])
    out['risk3'] = df['risk_indicator_3']
    out['risk12_dev'] = (out['risk12'] - 55).abs()
    out['risk3_dev'] = (out['risk3'] - 55).abs()
    out['risk_max'] = out[['risk12', 'risk3']].max(axis=1)
    out['risk_mean'] = out[['risk12', 'risk3']].mean(axis=1)

    out['has_score'] = df[['cr_scores_fico', 'cr_scores_vantage',
                           'cr_scores_schufa']].notna().any(axis=1).astype(int)
    return out, df[['cr_scores_fico', 'cr_scores_vantage', 'cr_scores_schufa']].copy()


def add_scores_fixed(out_tr, out_te, raw_tr, raw_te):
    """FIX vs prep.py: mean/std computed on TRAIN ONLY, then applied to test."""
    for c in ['cr_scores_fico', 'cr_scores_vantage', 'cr_scores_schufa']:
        mu, sd = raw_tr[c].mean(), raw_tr[c].std()
        raw_tr[c + '_z'] = (raw_tr[c] - mu) / sd
        raw_te[c + '_z'] = (raw_te[c] - mu) / sd
    for out, raw in ((out_tr, raw_tr), (out_te, raw_te)):
        out['credit_z'] = (raw['cr_scores_fico_z']
                           .fillna(raw['cr_scores_vantage_z'])
                           .fillna(raw['cr_scores_schufa_z']))
    return out_tr, out_te


CATS = ['job_category', 'status', 'analyst_opinion']


def build(data_dir):
    tr, te = load(data_dir)
    Xtr, str_tr = engineer(tr)
    Xte, str_te = engineer(te)
    Xtr, Xte = add_scores_fixed(Xtr, Xte, str_tr, str_te)
    for c in CATS:
        cats = sorted(set(tr[c].dropna()) | set(te[c].dropna()))
        Xtr[c] = pd.Categorical(tr[c], categories=cats)
        Xte[c] = pd.Categorical(te[c], categories=cats)
    y = tr['credit_decision']
    return Xtr, y, Xte, te['id'], tr
