# ======================= MASTER CLEANING SCRIPT (run once) =======================
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"     # if this file does not exist, the script skips it

import os
import numpy as np, pandas as pd
from difflib import get_close_matches
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# ----------------------------- settings ------------------------------------
NUM = ["age", "bmi", "waist_cm", "fasting_glucose", "bp_systolic", "hba1c",
       "cholesterol_total", "hdl", "triglycerides"]
IMP_COLS = NUM + ["years_since_diagnosis"]
SIMPLE = ["visits_last_year", "length_of_stay_days"]
LIMITS = {"age": (18, 100), "bmi": (12, 70), "waist_cm": (40, 200),
          "bp_systolic": (70, 250), "fasting_glucose": (40, 500),
          "hba1c": (3.5, 15), "cholesterol_total": (100, 400), "hdl": (15, 100),
          "triglycerides": (30, 800), "years_since_diagnosis": (0, 60),
          "visits_last_year": (0, 60), "length_of_stay_days": (1, 60)}
VALID = {"medication": ["metformin", "insulin", "combination", "no medication"],
         "physical_activity": ["low", "medium", "high"],
         "smoking_status": ["never", "former", "current"],
         "readmission_ward": ["general medicine", "endocrinology", "icu"]}
GENDER = {"m": "M", "male": "M", "f": "F", "female": "F"}
TARGET_MAP = {"yes": 1, "y": 1, "1": 1, "true": 1, "no": 0, "n": 0, "0": 0, "false": 0}
DATE_FORMATS = ["%Y-%m-%d", "%d %b %Y", "%d-%m-%Y", "%d/%m/%Y", "%d %B %Y",
                "%Y/%m/%d", "%d.%m.%Y", "%b %d, %Y", "%d-%m-%y", "%d/%m/%y"]

LOG = []
def note(problem, rows, fix):
    LOG.append((problem, int(rows), fix))
    print(f"  [{int(rows):>5}] {problem} -> {fix}")

def fuzzy(s, valid):
    if pd.isna(s): return np.nan
    m = get_close_matches(s, valid, n=1, cutoff=0.6)
    return m[0] if m else np.nan

# ------------------- STEP A: row-level fixes (no learned values) -------------
def clean_rows(df, tag):
    print(f"\n--- cleaning {tag} ({len(df)} rows) ---")
    df = df.copy()
    # 1 text columns: strip, collapse spaces, lowercase
    for c in ["patient_id", "gender", "medication", "physical_activity", "smoking_status",
              "readmitted_30d", "readmission_ward"]:
        if c in df.columns:
            df[c] = (df[c].astype("string").str.strip()
                     .str.replace(r"\s+", " ", regex=True).str.lower())
    df["patient_id"] = df["patient_id"].str.upper()
    # 2 typos in categories -> fuzzy match
    for c, valid in VALID.items():
        if c in df.columns:
            before = df[c].copy()
            df[c] = df[c].apply(lambda s: fuzzy(s, valid)).astype(object)
            changed = (before.notna() & (before.astype(object) != df[c])).sum()
            note(f"{c}: typos / bad spellings", changed, "fuzzy-matched to valid values")
    # 3 gender
    g = df["gender"].map(GENDER).astype(object)
    note("gender: 12 spellings", (df["gender"].notna() & (df["gender"].str.upper() != g)).sum(), "standardised to M/F")
    df["gender"] = g
    # 4 target
    if "readmitted_30d" in df.columns:
        t = df["readmitted_30d"].map(TARGET_MAP)
        note("readmitted_30d: Yes/y/1/True/No/n/0/False", t.notna().sum(), "standardised to 0/1")
        df["readmitted_30d"] = t
    # 5 numeric conversion (hba1c "7.0%" etc.)
    for c in NUM + ["years_since_diagnosis"] + SIMPLE:
        s = df[c].astype("string").str.replace("%", "", regex=False).str.strip()
        df[c] = pd.to_numeric(s, errors="coerce")
    note("hba1c stored as text with % sign", df["hba1c"].notna().sum(), "converted to float")
    # 6 hba1c in mmol/mol (values > 20) -> %
    m = df["hba1c"] > 20
    df.loc[m, "hba1c"] = df.loc[m, "hba1c"] / 10.929 + 2.15
    note("hba1c given in mmol/mol", m.sum(), "converted to %")
    # 7 glucose in mmol/L -> mg/dL
    m = df["fasting_glucose"] < 30
    df.loc[m, "fasting_glucose"] *= 18.016
    note("fasting_glucose in mmol/L", m.sum(), "x18.016 to mg/dL")
    # 8 impossible / sentinel values -> missing
    for c, (lo, hi) in LIMITS.items():
        bad = df[c].notna() & ~df[c].between(lo, hi)
        df.loc[bad, c] = np.nan
        note(f"{c}: impossible/sentinel values outside {lo}-{hi}", bad.sum(), "set to missing")
    bad = df["years_since_diagnosis"] > df["age"]
    df.loc[bad, "years_since_diagnosis"] = np.nan
    note("years_since_diagnosis > age", bad.sum(), "set to missing")
    # 9 dates
    d = df["admission_date"].astype("string").str.strip()
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        out = out.fillna(pd.to_datetime(d, format=fmt, errors="coerce"))
    note("admission_date: mixed formats", (d.notna()).sum(), f"parsed to one date format ({out.isna().sum()} unparsed)")
    df["admission_date"] = out
    return df

# ------------------- STEP B: duplicates (training data only) -----------------
def dedupe(df):
    n = len(df)
    df = df.drop_duplicates()
    note("exact duplicate rows (after cleaning)", n - len(df), "dropped")
    n = len(df)
    df = df.sort_values("admission_date", na_position="first").drop_duplicates("patient_id", keep="last").sort_index()
    note("repeated patient_id", n - len(df), "kept latest admission")
    return df

# ------------------- STEP C: learn from TRAIN, apply to anything -------------
class Fit: pass
F = Fit()

def learn(df):
    F.flag_cols = [c for c in NUM if df[c].isna().sum() > 0]
    F.modes = {c: df[c].mode().iloc[0] for c in ["gender", "medication", "smoking_status"]}
    F.medians = {c: df[c].median() for c in SIMPLE}
    F.month_mode = df["admission_date"].dt.month.mode().iloc[0]
    F.dow_mode = df["admission_date"].dt.dayofweek.mode().iloc[0]
    F.imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
    F.imputer.fit(df[IMP_COLS])

def finish(df, tag):
    df = df.copy()
    for c in F.flag_cols:                              # keep a record of what was missing
        df[c + "_missing"] = df[c].isna().astype(int)
    for c, v in F.modes.items():                       # categorical gaps
        df[c] = df[c].fillna(v)
    df["physical_activity"] = df["physical_activity"].fillna("unknown")
    for c, v in F.medians.items():
        df[c] = df[c].fillna(v)
    df["admission_month"] = df["admission_date"].dt.month.fillna(F.month_mode).astype(int)
    df["admission_weekday"] = df["admission_date"].dt.dayofweek.fillna(F.dow_mode).astype(int)
    n_missing = int(df[IMP_COLS].isna().sum().sum())
    df[IMP_COLS] = F.imputer.transform(df[IMP_COLS])
    for c in IMP_COLS:
        lo, hi = LIMITS[c]; df[c] = df[c].clip(lo, hi)
    for c in ["age", "waist_cm", "fasting_glucose", "bp_systolic", "cholesterol_total",
              "hdl", "triglycerides", "years_since_diagnosis"] + SIMPLE:
        df[c] = df[c].round().astype(int)
    df["bmi"] = df["bmi"].round(1); df["hba1c"] = df["hba1c"].round(1)
    df["years_since_diagnosis"] = np.minimum(df["years_since_diagnosis"], df["age"])
    df["diag_before_18"] = (df["years_since_diagnosis"] > df["age"] - 18).astype(int)
    df["triglycerides_log"] = np.log1p(df["triglycerides"])
    if "readmission_ward" in df.columns:
        df["readmission_ward"] = df["readmission_ward"].fillna("not readmitted")
    note(f"{tag}: missing numeric cells", n_missing, "IterativeImputer (uses correlated columns), clipped, rounded")
    return df

def model_ready(df):
    m = df.drop(columns=[c for c in ["patient_id", "readmission_ward", "admission_date"] if c in df.columns])
    cats = {"gender": ["M", "F"], "medication": VALID["medication"],
            "physical_activity": VALID["physical_activity"] + ["unknown"],
            "smoking_status": VALID["smoking_status"]}
    for c, cat in cats.items():
        m[c] = pd.Categorical(m[c], categories=cat)
    return pd.get_dummies(m, columns=list(cats), drop_first=True, dtype=int)

# ------------------- STEP D: independent second check -----------------------
def audit(df, name, is_train):
    fails = []
    def chk(ok, msg):
        if not ok: fails.append(msg)
    non_ward = [c for c in df.columns if c != "readmission_ward"]
    chk(df[non_ward].isna().sum().sum() == 0, "NaN values remain")
    chk(not np.isinf(df.select_dtypes("number")).any().any(), "inf values present")
    for c, (lo, hi) in LIMITS.items():
        chk(df[c].between(lo, hi).all(), f"{c} outside {lo}-{hi}")
    chk(set(df.gender) <= {"M", "F"}, "gender has bad values")
    for c in ["medication", "smoking_status"]:
        chk(set(df[c]) <= set(VALID[c]), f"{c} has bad values")
    chk(set(df.physical_activity) <= set(VALID["physical_activity"] + ["unknown"]), "physical_activity bad values")
    chk(set(df.readmission_ward) <= set(VALID["readmission_ward"] + ["not readmitted"]), "ward bad values")
    chk((df.years_since_diagnosis <= df.age).all(), "years_since_diagnosis > age")
    chk((df.hdl <= df.cholesterol_total).all(), "hdl > total cholesterol")
    chk(df.fasting_glucose.min() >= 40, "glucose looks like mmol/L still")
    chk(df.admission_date.notna().all(), "unparsed dates")
    chk(all(set(df[c + "_missing"]) <= {0, 1} for c in F.flag_cols), "flag columns not 0/1")
    if is_train:
        chk(df.duplicated().sum() == 0, "duplicate rows")
        chk(df.patient_id.is_unique, "repeated patient_id")
    if "readmitted_30d" in df.columns:
        chk(set(df.readmitted_30d.dropna()) <= {0, 1}, "target not 0/1")
        chk(df.readmitted_30d.notna().all(), "target has NaN")
        if is_train:
            chk(((df.readmitted_30d == 1) == (df.readmission_ward != "not readmitted")).all(),
                "ward inconsistent with target")
    print(f"\n=== SECOND CHECK: {name} ===")
    print("ALL CHECKS PASSED" if not fails else "FAILED: " + "; ".join(fails))
    return fails

# ================================ RUN =========================================
raw = pd.read_csv(TRAIN_PATH)
print("RAW train:", raw.shape)
tr = clean_rows(raw, "TRAIN")
tr = tr.dropna(subset=["readmitted_30d"])
tr = dedupe(tr)
learn(tr)
tr = finish(tr, "TRAIN")
tr["readmitted_30d"] = tr["readmitted_30d"].astype(int)

fails = audit(tr, "TRAIN", True)

# before / after report
rep = pd.DataFrame({"raw_min": raw[["age", "bmi", "waist_cm", "bp_systolic"]].apply(pd.to_numeric, errors="coerce").min(),
                    "clean_min": tr[["age", "bmi", "waist_cm", "bp_systolic"]].min(),
                    "raw_max": raw[["age", "bmi", "waist_cm", "bp_systolic"]].apply(pd.to_numeric, errors="coerce").max(),
                    "clean_max": tr[["age", "bmi", "waist_cm", "bp_systolic"]].max()})
print("\nBefore vs after (impossible values gone):\n", rep)
print("\nRows: raw", len(raw), "-> clean", len(tr), "| readmission rate:", round(tr.readmitted_30d.mean(), 3))
drift = {c: round(float(tr[c].mean() / pd.to_numeric(raw[c], errors='coerce').where(lambda s: s.between(*LIMITS[c])).mean() - 1), 3)
         for c in ["bmi", "bp_systolic", "cholesterol_total", "hdl", "triglycerides"]}
print("Mean change vs raw valid values (should be small):", drift)

tr.to_csv("clean_train_final.csv", index=False)
mr = model_ready(tr); mr.to_csv("model_ready_train.csv", index=False)
outs = ["clean_train_final.csv", "model_ready_train.csv"]

if os.path.exists(TEST_PATH):
    te = clean_rows(pd.read_csv(TEST_PATH), "TEST")
    te = finish(te, "TEST")
    audit(te, "TEST", False)
    te.to_csv("clean_test_final.csv", index=False)
    model_ready(te).reindex(columns=mr.columns, fill_value=0).to_csv("model_ready_test.csv", index=False)
    outs += ["clean_test_final.csv", "model_ready_test.csv"]

with open("cleaning_log_final.md", "w") as f:
    f.write("| problem | rows affected | fix |\n|---|---|---|\n")
    for p, r, x in LOG: f.write(f"| {p} | {r} | {x} |\n")
outs.append("cleaning_log_final.md")
print("\nSaved:", outs)
try:
    from google.colab import files
    for fn in outs: files.download(fn)
except ImportError:
    pass
