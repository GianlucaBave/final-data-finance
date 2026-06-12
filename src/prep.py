"""Shared cleaning/feature engineering for the ESADE DSF26 credit challenge."""
import pandas as pd, numpy as np

DROP_TAMPERED = ['internal_code']          # scrambled in test (verified via opinion-probe)
DROP_TEST_MISSING = ['external_pd_score']  # 100% missing in test
DROP_PROTECTED = ['religion', 'race']      # legal/regulatory exclusion (and ~no risk signal)

def load(D):
    tr = pd.read_csv(D + "train.csv")
    te = pd.read_csv(D + "test.csv")
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

    # income: half the records are in thousands (bimodal with a clean gap 700..2000)
    inc = pd.to_numeric(df['ann_income'], errors='coerce')
    out['ann_income'] = np.where(inc < 700, inc * 1000, inc)
    oth = pd.to_numeric(df['other_income'], errors='coerce')
    out['other_income'] = np.where((oth > 0) & (oth < 700), oth * 1000, oth)
    out['total_income'] = out['ann_income'] + out['other_income']
    out['amount'] = df['amount']
    out['amt_to_income'] = df['amount'] / (out['total_income'] + 1)
    out['log_income'] = np.log1p(out['total_income'])

    # age: impute from birth_year + decision year
    age = df['age'].copy()
    age = age.fillna(out['year'] - df['birth_year'])
    out['age'] = age

    out['prev_default'] = df['prev_default'].map({'0':0,'No':0,'1':1,'Yes':1,0:0,1:1}).astype(float)
    out['highest_ed'] = df['highest_ed']
    out['kids'] = df['kids']
    out['vip'] = df['vip'].astype(str).map({'True':1,'False':0})

    # risk indicators: r1/r2 mutually exclusive -> coalesce; keep r3
    out['risk12'] = df['risk_indicator_1'].fillna(df['risk_indicator_2'])
    out['risk3'] = df['risk_indicator_3']

    # credit scores: different bureau scales -> z-score per bureau then coalesce
    out['has_score'] = df[['cr_scores_fico','cr_scores_vantage','cr_scores_schufa']].notna().any(axis=1).astype(int)
    return out, df[['cr_scores_fico','cr_scores_vantage','cr_scores_schufa']]

def add_scores(out_tr, out_te, raw_tr, raw_te):
    for c in ['cr_scores_fico','cr_scores_vantage','cr_scores_schufa']:
        allv = pd.concat([raw_tr[c], raw_te[c]])
        mu, sd = allv.mean(), allv.std()
        raw_tr[c + '_z'] = (raw_tr[c] - mu) / sd
        raw_te[c + '_z'] = (raw_te[c] - mu) / sd
    for out, raw in ((out_tr, raw_tr), (out_te, raw_te)):
        out['credit_z'] = raw['cr_scores_fico_z'].fillna(raw['cr_scores_vantage_z']).fillna(raw['cr_scores_schufa_z'])
    return out_tr, out_te

CATS = ['job_category', 'status', 'analyst_opinion']

def build(D):
    tr, te = load(D)
    Xtr, str_tr = engineer(tr)
    Xte, str_te = engineer(te)
    Xtr, Xte = add_scores(Xtr, Xte, str_tr.copy(), str_te.copy())
    for c in CATS:
        cats = sorted(set(tr[c].dropna()) | set(te[c].dropna()))
        Xtr[c] = pd.Categorical(tr[c], categories=cats)
        Xte[c] = pd.Categorical(te[c], categories=cats)
    y = tr['credit_decision']
    return Xtr, y, Xte, te['id'], tr

def provenance(df):
    """Source-system flags recovered from the legacy formats (computed on raw strings)."""
    out = pd.DataFrame(index=df.index)
    out['src_system'] = df['date'].astype(str).str.match(r'\d{4}-').astype(int)
    out['pd_word'] = df['prev_default'].isin(['Yes','No']).astype(int)
    return out
