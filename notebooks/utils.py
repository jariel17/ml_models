import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm  # optional progress bar

def _to_numeric_series(s, fillna=None, treat_boolean_as_binary=True):
    """
    Convert a pandas Series to a numeric score usable by roc_auc_score.
    - Maps common yes/no/true/false strings to 1/0.
    - If categorical or object, returns pd.Categorical.codes (WARNING: codes are arbitrary order).
    - If numeric, returns as-is.
    """
    if s.dtype.kind in 'biufc':   # boolean, integer, unsigned, float, complex (complex unlikely)
        out = s.astype(float)
    else:
        # uniform lowercase mapping
        s_str = s.astype(str).str.lower().str.strip()
        mapdict = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0,
                   'male': 1, 'female': 0}
        out = s_str.map(mapdict)
        # if some still NaN, fallback to categorical codes (keeps consistency)
        if out.isna().any():
            try:
                out = pd.Categorical(s).codes.astype(float)
            except Exception:
                out = s_str.factorize()[0].astype(float)
    if fillna is not None:
        out = pd.Series(out).fillna(fillna)
    return pd.Series(out)

def missingness_auc_summary(df, target_cols, ref_cols, fill_ref_na=None, return_df=True):
    """
    For each pair (target_col, ref_col) compute:
    - n_total, n_missing, pct_missing
    - auc (roc_auc_score of M vs numeric(ref))
    Returns a tidy DataFrame sorted by pct_missing desc then auc abs distance from 0.5.
    """
    rows = []
    for ref in ref_cols:
        if ref not in df.columns:
            continue
        y_ref_raw = df[ref]
        y_ref = _to_numeric_series(y_ref_raw, fillna=fill_ref_na)
        for target in target_cols:
            if target not in df.columns:
                continue
            M = df[target].isnull().astype(int)
            n_total = len(M)
            n_missing = M.sum()
            pct_missing = n_missing / n_total if n_total else np.nan
            # require both classes present
            if n_missing == 0 or n_missing == n_total:
                auc = np.nan
            else:
                try:
                    auc = roc_auc_score(M, y_ref)
                except Exception:
                    auc = np.nan
            rows.append({
                'ref_col': ref,
                'target_col': target,
                'n_total': n_total,
                'n_missing': int(n_missing),
                'pct_missing': pct_missing,
                'auc': auc
            })
    res = pd.DataFrame(rows)
    # helper metric: distance from 0.5 (how informative)
    res['auc_abs_dev_from_0_5'] = res['auc'].apply(lambda x: abs(x-0.5) if pd.notna(x) else np.nan)
    res = res.sort_values(['pct_missing', 'auc_abs_dev_from_0_5'], ascending=[False, False])
    if return_df:
        return res
    return None

def auc_permutation_pvalue(y_true_binary, score, n_permutations=1000, random_state=None):
    """
    Estimate a permutation p-value for the observed AUC.
    Null: score is unrelated to y_true. We permute scores relative to labels.
    Returns (observed_auc, p_value, permuted_aucs_array)
    """
    rng = np.random.default_rng(random_state)
    # check
    y = np.asarray(y_true_binary)
    s = np.asarray(score)
    # require both classes present
    if y.sum()==0 or y.sum()==len(y):
        raise ValueError("y_true must contain both classes")
    obs_auc = roc_auc_score(y, s)
    perm_aucs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(s)
        perm_aucs[i] = roc_auc_score(y, perm)
    # two-sided p-value: fraction of permuted AUCs as extreme or more than observed
    # but since AUC is bounded in [0,1] and expected 0.5, we can use deviation from 0.5
    obs_dev = abs(obs_auc - 0.5)
    perm_dev = np.abs(perm_aucs - 0.5)
    p_value = (np.sum(perm_dev >= obs_dev) + 1) / (n_permutations + 1)
    return obs_auc, p_value, perm_aucs

def get_top_missing_columns(df, top_n=10):
    return (
        df.isnull().sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'
    
def categorize_age(age):
    if age < 10:
        return 'Child'
    elif age < 14:
        return 'Preteen'
    else:
        return 'Teenager'