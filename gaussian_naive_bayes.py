"""
Շաքարային դիաբետի կանխատեսումը միամիտ բայեսի (Gaussian Naive Bayes) ալգորիթմով.

Կոդի հերթականությունը.
    1. Տվյալների բեռնում
    2. Ուսումնասիրություն
    3. Մաքրում
    4. Train / test բաժանում
    5. Ուսուցում 
    6. Կանխատեսում նոր տվյալների վրա
    7. Արդյունքների ներկայացում 
    8. Գնահատում տարբեր չափորոշիչներով
    9. 5-fold Cross Validation (միայն Train տվյալներով)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 42          # արդյունքները կրկնելի դարձնելու համար
TEST_SIZE = 0.2    # տվյալների 20%-ը թողնում ենք թեստի համար
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


# =============================================================================
# 1. ՏՎՅԱԼՆԵՐԻ ԲԵՌՆՈՒՄ
# =============================================================================
print("*" * 70)
print("1. ՏՎՅԱԼՆԵՐԻ ԲԵՌՆՈՒՄ")
print("*" * 70)

# CSV-ն որոնում ենք նախ սքրիպտի կողքին, ապա ընթացիկ պանակում։
DATA_FILE = Path(__file__).resolve().parent / "diabetes.csv"
if not DATA_FILE.exists():
    DATA_FILE = Path("diabetes.csv")
if not DATA_FILE.exists():
    raise FileNotFoundError("Տեղադրեք diabetes.csv-ը սքրիպտի կողքին կամ ընթացիկ պանակում")
df = pd.read_csv(DATA_FILE)
TARGET = "Outcome"                                  # 1 = դիաբետ, 0 = առանց դիաբետի
# Նախնական 8 սյունակները կարող ենք ուսումնասիրել, բայց մոդելը օգտագործում է ՄԻԱՅՆ 4-ը։
ALL_FEATURES = [c for c in df.columns if c != TARGET]
FEATURES = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]
required = FEATURES + [TARGET]
missing_columns = sorted(set(required) - set(df.columns))
if missing_columns:
    raise ValueError(f"CSV-ում բացակայում են սյունակները՝ {missing_columns}")
print("Մոդելի 4 հատկանիշները:", ", ".join(FEATURES))

print(f"Տողերի քանակ: {df.shape[0]}, սյունակների քանակ: {df.shape[1]}")
print(df.head())


# =============================================================================
# 2. ՈՒՍՈՒՄՆԱՍԻՐՈՒԹՅՈՒՆ
# =============================================================================
print("\n" + "*" * 70)
print("2. ՈՒՍՈՒՄՆԱՍԻՐՈՒԹՅՈՒՆ")
print("*" * 70)

print("\nՏվյալների տիպերը և բացակայող (NaN) արժեքները:")
print(pd.DataFrame({"dtype": df.dtypes, "NaN": df.isna().sum()}))

print("\nՆկարագրական վիճակագրություն:")
print(df.describe().T.round(2))

print("\nԴասերի բաշխումը:")
class_counts = df[TARGET].value_counts().sort_index()
for cls, cnt in class_counts.items():
    print(f"  Outcome = {cls}: {cnt} հիվանդ ({100 * cnt / len(df):.1f}%)")

# Կարևոր դիտարկում. NaN-եր չկան, բայց կան անհնարին զրոներ
print("\nԶրո արժեքների քանակը յուրաքանչյուր սյունակում:")
print((df[ALL_FEATURES] == 0).sum())


# =============================================================================
# 3. ՄԱՔՐՈՒՄ (անվիճակագրական մաս)
# =============================================================================
print("\n" + "*" * 70)
print("3. ՄԱՔՐՈՒՄ")
print("*" * 70)

# --- 3.1 Կրկնօրինակներ ------------------------------------------------------
n_dup = df.duplicated().sum()
print(f"Կրկնվող տողեր: {n_dup}")
df = df.drop_duplicates().reset_index(drop=True)

# --- 3.2 Անհնարին զրոները դարձնում ենք բացակայող արժեք (NaN) -----------------
# ԻՆՉՈՒ ԱՅՍՊԵՍ.
# Glucose, BloodPressure, SkinThickness, Insulin և BMI ողջ մարդու մոտ չեն
# կարող լինել 0 (օրինակ, 0 գլյուկոզ կամ 0 ճնշում նշանակում է մահ): Այսինքն 0-ն
# իրական չափում չէ, այլ բացակայող տվյալի «կոդ»: Եթե զրոները թողնենք, ապա.
#   * միջինները և տարբերությունները կմեծանան (օր.՝ Insulin-ի 49%-ը զրո է),
#   * Gaussian Naive Bayes-ը կգնահատի ՍԽԱԼ միջին և դիսպերսիա, քանի որ զրոները
#     կստեղծեն արհեստական «երկրորդ գագաթ» բաշխման մեջ:
# Pregnancies = 0 ՃԻՇՏ արժեք է (կինը կարող է չծննդաբերած լինել), ուստի այն չենք
# փոխում: DiabetesPedigreeFunction և Age սյունակներում զրո չկա:
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[ZERO_IS_MISSING] = df[ZERO_IS_MISSING].replace(0, np.nan)

missing_report = pd.DataFrame({
    "Բացակայող": df[ALL_FEATURES].isna().sum(),
    "Տոկոս (%)": (100 * df[ALL_FEATURES].isna().mean()).round(1),
})
print("\nԶրոները NaN դարձնելուց հետո բացակայող արժեքները:")
print(missing_report[missing_report["Բացակայող"] > 0])

# --- 3.3 Հատկանիշների ընտրություն -------------------------------------------
# Նախորդ 5-fold CV համեմատությամբ ընտրվել են 4 հատկանիշները՝
# Glucose, BMI, Age և DiabetesPedigreeFunction։
# Insulin և SkinThickness սյունակների շատ մեծ բացակայությունը կարող էր
# վատացնել Gaussian Naive Bayes-ի պարամետրերի գնահատումը։
# Սյունակները ֆիզիկապես պարտադիր չէ ջնջել df-ից. FEATURES-ը հստակ
# որոշում է, թե որ սյունակներն են գնում մոդելին։
# Outlier-ները այստեղ չենք հեռացնում. դրանք առանձին հետազոտության թեմա են։


# =============================================================================
# 4. TRAIN / TEST ԲԱԺԱՆՈՒՄ + ՓՈԽԱՐԻՆՈՒՄ
# =============================================================================
print("\n" + "*" * 70)
print("4. TRAIN / TEST ԲԱԺԱՆՈՒՄ ԵՎ ՓՈԽԱՐԻՆՈՒՄ")
print("*" * 70)


def stratified_split(frame, target, test_size, seed):
    """Շերտավորված (stratified) բաժանում. երկու մասում էլ դասերի
    համամասնությունը (~65% / ~35%) մնում է նույնը, ինչ ամբողջ տվյալներում:"""
    rng = np.random.default_rng(seed)
    test_idx = []
    for cls in frame[target].unique():
        idx = frame.index[frame[target] == cls].to_numpy().copy()
        rng.shuffle(idx)
        test_idx.extend(idx[: int(round(len(idx) * test_size))])
    test_idx = np.array(sorted(test_idx))
    train_idx = np.setdiff1d(frame.index.to_numpy(), test_idx)
    return frame.loc[train_idx], frame.loc[test_idx]


train_df, test_df = stratified_split(df, TARGET, TEST_SIZE, SEED)
print(f"Train: {len(train_df)} տող ({100 * train_df[TARGET].mean():.1f}% դիաբետ)")
print(f"Test:  {len(test_df)} տող ({100 * test_df[TARGET].mean():.1f}% դիաբետ)")

# --- 4.1 Փոխարինում median-ով, հաշվված ՄԻԱՅՆ train-ից --------------
# ԻՆՉՈՒ ՄԻՋՆԱԳԻԾ (median), ԱՅԼ ՈՉ ԹԵ ՄԻՋԻՆ (mean).
#   * Insulin-ի բաշխումը շատ աջ-թեքությամբ է (skewness ≈ 2.2), DiabetesPedigree-ինը
#     և Age-ինը նույնպես: Միջինը ձգվում է մեծ արժեքներով, միջնագիծը՝ ոչ:
#     Միջնագիծը «տիպիկ հիվանդին» ավելի լավ է նկարագրում:
#   * Միջնագիծը կայուն է ծայրահեղ արժեքների նկատմամբ:
# ԻՆՉՈՒ ՄԻԱՅՆ TRAIN-ԻՑ.
#   * Եթե միջնագիծը հաշվեինք ամբողջ տվյալներից, թեստի տվյալների մասին
#     տեղեկատվությունը կհայտնվեր ուսուցման մեջ (data leakage), և
#     գնահատականը կլիներ չափազանց լավատեսական: Նոր հիվանդի համար էլ մենք
#     կարող ենք օգտագործել միայն ԱՐԴԵՆ ՀԱՅՏՆԻ (train-ի) արժեքները:
# ԻՆՉՈՒ ԸՆԴՀԱՆՈՒՐ, ԱՅԼ ՈՉ ԴԱՍԵՐԻ ՀԱՄԱՐ ՏԱՐԲԵՐ MEDIAN.
#   * Դասով (դիաբետ/չկա) մեդիանը օգտագործելու համար պետք է իմանալ պատասխանը,
#     իսկ նոր հիվանդի համար պատասխանը հենց այն է, ինչ փորձում ենք գտնել:
#     Ընդհանուր մեդիանը աշխատում է նաև նոր
#     տվյալների վրա:
# ՀՆԱՐԱՎՈՐ ԹԵՐՈՒԹՅՈՒՆ.
#   * Մեդիանով փոխարինումը փոքրացնում է դիսպերսիան (շատ արժեքներ դառնում են
#     նույնը), հատկապես Insulin-ի և SkinThickness-ի համար: Դրա պատճառով
#     մոդելը այդ նշանների նկատմամբ կարող է լինել «չափազանց վստահ»: 

train_medians = train_df[FEATURES].median()
print("\nTrain-ի միջնագծերը (օգտագործվում են փոխարինման համար):")
print(train_medians.round(2))


def impute(frame, medians):
    """Բացակայող արժեքները փոխարինում է տրված միջնագծերով."""
    out = frame.copy()
    out[FEATURES] = out[FEATURES].fillna(medians)
    return out


train_df = impute(train_df, train_medians)
test_df = impute(test_df, train_medians)      # ՆՈՒՅՆ train-ի միջնագծերը
assert train_df[FEATURES].isna().sum().sum() == 0
assert test_df[FEATURES].isna().sum().sum() == 0
print("Փոխարինումից հետո NaN-եր չեն մնացել:")

X_train = train_df[FEATURES].to_numpy(dtype=float)
y_train = train_df[TARGET].to_numpy(dtype=int)
X_test = test_df[FEATURES].to_numpy(dtype=float)
y_test = test_df[TARGET].to_numpy(dtype=int)


# =============================================================================
# 5. ՈՒՍՈՒՑՈՒՄ. GAUSSIAN NAIVE BAYES
# =============================================================================
print("\n" + "*" * 70)
print("5. ՈՒՍՈՒՑՈՒՄ")
print("*" * 70)


class GaussianNaiveBayes:
    """
    Գաուսյան միամիտ բայես.

    Բայեսի բանաձև:      P(c | x) ∝ P(c) · Π_j P(x_j | c)
    «Միամիտ» ենթադրություն. նշանները դասի ներսում անկախ են միմյանցից, այդ պատճառով
    համատեղ հավանականությունը դառնում է առանձին հավանականությունների արտադրյալ:
    «Գաուսյան» ենթադրություն. յուրաքանչյուր նշանը դասի ներսում ենթարկվում է
    նորմալ բաշխման N(μ_cj, σ²_cj):

    Ուսուցումը պարզապես հաշվելն է.
        P(c)    = դասի հիվանդների մասնաբաժինը
        μ_cj    = j-րդ փոփոխականի միջինը c դասում
        σ²_cj   = j-րդ փոփոխականի դիսպերսիան c դասում
    Թվային կայունության համար ամեն ինչ հաշվում ենք լոգարիթմներով.
        log P(c|x) ∝ log P(c) + Σ_j [ -½·log(2πσ²_cj) - (x_j - μ_cj)² / (2σ²_cj) ]
    """

    def __init__(self, var_smoothing=1e-9):
        # Շատ փոքր թիվ, որը ավելացվում է դիսպերսիային, որպեսզի 0-ի վրա բաժանում չլինի
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))

        for k, cls in enumerate(self.classes_):
            Xc = X[y == cls]
            self.priors_[k] = Xc.shape[0] / X.shape[0]
            self.means_[k] = Xc.mean(axis=0)
            self.vars_[k] = Xc.var(axis=0)          # ընդհանուր (ddof=0) դիսպերսիա

        # Smoothing. համամասնական է ամենամեծ դիսպերսիային
        self.vars_ += self.var_smoothing * X.var(axis=0).max()
        return self

    def _joint_log_likelihood(self, X):
        """log P(c) + Σ log P(x_j | c) յուրաքանչյուր օրինակի և դասի համար."""
        X = np.asarray(X, dtype=float)
        jll = np.zeros((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            log_prior = np.log(self.priors_[k])
            log_pdf = -0.5 * np.log(2.0 * np.pi * self.vars_[k]) \
                      - ((X - self.means_[k]) ** 2) / (2.0 * self.vars_[k])
            jll[:, k] = log_prior + log_pdf.sum(axis=1)
        return jll

    def predict_proba(self, X):
        """Հետին (posterior) հավանականություններ. log-sum-exp հնարքով նորմավորում ենք,
        որպեսզի շատ փոքր թվերը չդառնան 0:"""
        jll = self._joint_log_likelihood(X)
        jll_max = jll.max(axis=1, keepdims=True)
        log_norm = jll_max + np.log(np.exp(jll - jll_max).sum(axis=1, keepdims=True))
        return np.exp(jll - log_norm)

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


model = GaussianNaiveBayes().fit(X_train, y_train)

print("Prior հավանականություններ:")
for cls, p in zip(model.classes_, model.priors_):
    print(f"  P(Outcome={cls}) = {p:.3f}")

params = pd.DataFrame({
    "μ (առանց դիաբետի)": model.means_[0], "μ (դիաբետ)": model.means_[1],
    "σ (առանց դիաբետի)": np.sqrt(model.vars_[0]), "σ (դիաբետ)": np.sqrt(model.vars_[1]),
}, index=FEATURES).round(2)
print("\nՈւսուցված պարամետրերը (միջին և ստանդարտ շեղում դասերի ներսում):")
print(params)


# =============================================================================
# 6. ԿԱՆԽԱՏԵՍՈՒՄ ՆՈՐ ՏՎՅԱԼՆԵՐԻ ՎՐԱ
# =============================================================================
print("\n" + "*" * 70)
print("6. ԿԱՆԽԱՏԵՍՈՒՄ ՆՈՐ ՏՎՅԱԼՆԵՐԻ ՎՐԱ")
print("*" * 70)

# 4 հատկանիշով երեք ՊԱՅՄԱՆԱԿԱՆ օրինակ՝ ոչ կլինիկական ախտորոշումներ։
# Երրորդ հիվանդի BMI-ն բացակայում է. այն լրացվում է Train-ի միջնագծով։
new_patients = pd.DataFrame([
    # Glucose, BMI, DPF, Age
    [85,  26.6, 0.35, 25],
    [168, 38.5, 0.85, 52],
    [140, np.nan, 0.50, 45],
], columns=["Glucose", "BMI", "DiabetesPedigreeFunction", "Age"])
# Կազմակերպում ենք նույն հերթականությամբ, ինչ FEATURES-ում։
new_patients = new_patients[FEATURES]

new_ready = impute(new_patients, train_medians)     # ՆՈՒՅՆ մաքրումը, ինչ train/test-ում
new_proba = model.predict_proba(new_ready.to_numpy(dtype=float))
new_pred = model.predict(new_ready.to_numpy(dtype=float))

new_report = new_ready.copy()  # ցույց ենք տալիս նաև լրացված BMI-ն
new_report["P(դիաբետ)"] = new_proba[:, 1].round(3)
new_report["Կանխատեսում"] = np.where(new_pred == 1, "Դիաբետ", "Առողջ")
print(new_report)


# =============================================================================
# 7. ԱՐԴՅՈՒՆՔՆԵՐԻ ՆԵՐԿԱՅԱՑՈՒՄ
# =============================================================================
print("\n" + "*" * 70)
print("7. ԱՐԴՅՈՒՆՔՆԵՐԻ ՆԵՐԿԱՅԱՑՈՒՄ")
print("*" * 70)

y_pred = model.predict(X_test)
proba_test = model.predict_proba(X_test)[:, 1]

preview = test_df[FEATURES].head(10).copy()
preview["Իրական"] = np.where(y_test[:10] == 1, "Դիաբետ", "Առողջ")
preview["Կանխատեսված"] = np.where(y_pred[:10] == 1, "Դիաբետ", "Առողջ")
preview["P(դիաբետ)"] = proba_test[:10].round(3)
preview["Ճիշտ է"] = np.where(y_test[:10] == y_pred[:10], "✓", "✗")
print("Թեստի առաջին 10 հիվանդի արդյունքները:")
print(preview.to_string())


# --- Գնահատականների ֆունկցիաներ-----------------------------
def confusion_matrix(y_true, y_hat):
    """Վերադարձնում է TN, FP, FN, TP."""
    tn = int(np.sum((y_true == 0) & (y_hat == 0)))
    fp = int(np.sum((y_true == 0) & (y_hat == 1)))
    fn = int(np.sum((y_true == 1) & (y_hat == 0)))
    tp = int(np.sum((y_true == 1) & (y_hat == 1)))
    return tn, fp, fn, tp


def roc_curve(y_true, score):
    """ROC կորի կետերը (FPR, TPR). յուրաքանչյուր եզակի շեմի համար:"""
    thresholds = np.concatenate(([np.inf], np.unique(score)[::-1]))
    P, N = np.sum(y_true == 1), np.sum(y_true == 0)
    tpr = np.array([np.sum((score >= t) & (y_true == 1)) / P for t in thresholds])
    fpr = np.array([np.sum((score >= t) & (y_true == 0)) / N for t in thresholds])
    return fpr, tpr


def auc_score(fpr, tpr):
    """Տրապեցիաների մեթոդով մակերես ROC կորի տակ:"""
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def log_loss(y_true, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def all_metrics(y_true, y_hat, p1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat)
    div = lambda a, b: a / b if b > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)                       # sensitivity
    specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall)
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = div(tp * tn - fp * fn, mcc_den)
    fpr, tpr = roc_curve(y_true, p1)
    return {
        "Accuracy (ճշտություն)": accuracy,
        "Balanced accuracy": (recall + specificity) / 2,
        "Precision (դիպուկություն)": precision,
        "Recall / Sensitivity (լրիվություն)": recall,
        "Specificity (յուրահատկություն)": specificity,
        "F1": f1,
        "MCC": mcc,
        "ROC AUC": auc_score(fpr, tpr),
        "Log loss": log_loss(y_true, p1),
    }


# --- Գրաֆիկներ --------------------------------------------------------------
tn, fp, fn, tp = confusion_matrix(y_test, y_pred)
cm = np.array([[tn, fp], [fn, tp]])
fpr, tpr = roc_curve(y_test, proba_test)
auc_val = auc_score(fpr, tpr)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

# (ա) Confusion matrix
ax = axes[0]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["No diabetes", "Diabetes"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No diabetes", "Diabetes"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black")

# (բ) ROC
ax = axes[1]
ax.plot(fpr, tpr, color="#d73027", lw=2, label=f"Naive Bayes (AUC = {auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set_xlabel("False positive rate (1 - specificity)")
ax.set_ylabel("True positive rate (recall)")
ax.set_title("ROC curve (test)"); ax.legend(loc="lower right")

# (գ) Կանխատեսված հավանականությունների բաշխումը դասերի համար
ax = axes[2]
bins = np.linspace(0, 1, 21)
ax.hist(proba_test[y_test == 0], bins=bins, alpha=0.6, color="#4575b4", label="Actual: no diabetes")
ax.hist(proba_test[y_test == 1], bins=bins, alpha=0.6, color="#d73027", label="Actual: diabetes")
ax.axvline(0.5, color="k", ls="--", lw=1)
ax.set_xlabel("Predicted P(diabetes)"); ax.set_ylabel("Patients")
ax.set_title("Predicted probabilities (test)"); ax.legend()

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "results.png", dpi=150)
print("\nԳրաֆիկները պահպանվել են results.png ֆայլում:")

# Նշանների դասակարգման ազդեցությունը. միջինների տարբերություն (σ-ներով)
sep = (np.abs(model.means_[1] - model.means_[0])
       / np.sqrt((model.vars_[0] + model.vars_[1]) / 2))
order = np.argsort(sep)
plt.figure(figsize=(7, 4))
plt.barh(np.array(FEATURES)[order], sep[order], color="steelblue")
plt.xlabel("|μ_diabetes − μ_healthy| / pooled σ")
plt.title("How strongly each feature separates the classes\n(estimated by the trained model)")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "feature_separation.png", dpi=150)

# =============================================================================
# 8. ԳՆԱՀԱՏՈՒՄ ՏԱՐԲԵՐ ՉԱՓՈՐՈՇԻՉՆԵՐՈՎ
# =============================================================================
print("\n" + "=" * 70)
print("8. ԳՆԱՀԱՏՈՒՄ")
print("=" * 70)

proba_train = model.predict_proba(X_train)[:, 1]
m_test = all_metrics(y_test, y_pred, proba_test)
m_train = all_metrics(y_train, model.predict(X_train), proba_train)

results = pd.DataFrame({"Train": m_train, "Test": m_test}).round(3)
print(results)

# Baseline. մոդել, որը միշտ ասում է «առողջ»
baseline_acc = np.mean(y_test == 0)
print(f"\nBaseline (միշտ «առողջ») accuracy = {baseline_acc:.3f}; "
      f"մոդելինը = {m_test['Accuracy (ճշտություն)']:.3f}")

print(f"\nConfusion matrix (test): TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"  FN = {fn} հիվանդ, որոնց դիաբետը մոդելը բաց թողեց (բժշկական առումով ամենավտանգավորը)")
print(f"  FP = {fp} առողջ մարդ, որոնց մոդելը սխալմամբ համարեց հիվանդ")

# --- Շեմի ազդեցությունը (0.5-ից տարբեր) -------------------------------------
print("\nՇեմի ազդեցությունը (դիաբետ կանխատեսել, երբ P(դիաբետ) ≥ շեմ):")
rows = []
for thr in [0.3, 0.4, 0.5, 0.6]:
    pred_thr = (proba_test >= thr).astype(int)
    m = all_metrics(y_test, pred_thr, proba_test)
    rows.append({"Շեմ": thr,
                 "Accuracy": m["Accuracy (ճշտություն)"],
                 "Recall": m["Recall / Sensitivity (լրիվություն)"],
                 "Specificity": m["Specificity (յուրահատկություն)"],
                 "F1": m["F1"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# --- Արդյունքի կայունությունը. նույն գործընթացը 30 տարբեր բաժանումով ------------
# Մեկ train/test բաժանումը (154 թեստային հիվանդ) կարող է պատահաբար «բախտավոր»
# կամ «անբախտ» լինել, ուստի ամբողջ գործընթացը (բաժանում -> փոխարինում -> ուսուցում
# -> գնահատում) կրկնում ենք տարբեր seed-երով և նայում միջինը ու շեղումը:
rep_rows = []
for rep_seed in range(30):
    tr, te = stratified_split(df, TARGET, TEST_SIZE, rep_seed)
    med = tr[FEATURES].median()
    tr, te = impute(tr, med), impute(te, med)
    mdl = GaussianNaiveBayes().fit(tr[FEATURES].to_numpy(float), tr[TARGET].to_numpy(int))
    Xe, ye = te[FEATURES].to_numpy(float), te[TARGET].to_numpy(int)
    mm = all_metrics(ye, mdl.predict(Xe), mdl.predict_proba(Xe)[:, 1])
    rep_rows.append(mm)
rep = pd.DataFrame(rep_rows)[["Accuracy (ճշտություն)", "Recall / Sensitivity (լրիվություն)",
                              "Specificity (յուրահատկություն)", "F1", "ROC AUC"]]
print("\n30 կրկնվող բաժանումների արդյունքները (միջին ± ստանդարտ շեղում):")
for col in rep.columns:
    print(f"  {col:38s} {rep[col].mean():.3f} ± {rep[col].std():.3f}")

# --- Ստուգում պատրաստի գրադարանի հետ (միայն ստուգման համար) -------------------
try:
    from sklearn.naive_bayes import GaussianNB
    sk = GaussianNB().fit(X_train, y_train)
    same = np.mean(sk.predict(X_test) == y_pred)
    print(f"\nՍտուգում sklearn.GaussianNB-ի հետ. կանխատեսումների համընկնում = {100 * same:.1f}%")
    print(f"  sklearn accuracy = {np.mean(sk.predict(X_test) == y_test):.3f}, "
          f"մեր իրականացումը = {m_test['Accuracy (ճշտություն)']:.3f}")
except ImportError:
    print("\n(scikit-learn-ը տեղադրված չէ. ստուգումը բաց թողնվեց)")


# =============================================================================
# 9. 5-FOLD CROSS VALIDATION՝ ԱՌԱՆՑ DATA LEAKAGE
# =============================================================================
print("\n" + "=" * 70)
print("9. 5-FOLD CROSS VALIDATION (4 ՀԱՏԿԱՆԻՇ)")
print("=" * 70)

# Կարևոր է. train_df-ն վերևում արդեն լրացվել է median-ներով,
# ուստի CV-ի համար վերցնում ենք ԱՆՓՈԽԱՐԻՆՎԱԾ սկզբնական Train-ի տողերը՝ df-ից։
cv_source = df.loc[train_df.index].copy()

try:
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics = []
    cv_labels = cv_source[TARGET].to_numpy(dtype=int)

    for fold_no, (tr_idx, val_idx) in enumerate(cv.split(cv_source[FEATURES], cv_labels), 1):
        fold_train = cv_source.iloc[tr_idx].copy()
        fold_val = cv_source.iloc[val_idx].copy()
        fold_medians = fold_train[FEATURES].median()  # ՄԻԱՅՆ այս fold-ի train-ից
        fold_train = impute(fold_train, fold_medians)
        fold_val = impute(fold_val, fold_medians)
        xf_tr = fold_train[FEATURES].to_numpy(dtype=float)
        yf_tr = fold_train[TARGET].to_numpy(dtype=int)
        xf_val = fold_val[FEATURES].to_numpy(dtype=float)
        yf_val = fold_val[TARGET].to_numpy(dtype=int)
        fold_model = GaussianNaiveBayes().fit(xf_tr, yf_tr)
        yf_pred = fold_model.predict(xf_val)
        yf_proba = fold_model.predict_proba(xf_val)[:, 1]
        fm = all_metrics(yf_val, yf_pred, yf_proba)
        fold_metrics.append(fm)
        print(f"Fold {fold_no}: Accuracy={fm['Accuracy (ճշտություն)']:.3f}, "
              f"Recall={fm['Recall / Sensitivity (լրիվություն)']:.3f}, "
              f"ROC AUC={fm['ROC AUC']:.3f}, Log Loss={fm['Log loss']:.3f}")

    cv_table = pd.DataFrame(fold_metrics)
    cv_summary = pd.DataFrame({
        "Միջին": cv_table.mean(axis=0),
        "Ստանդարտ շեղում": cv_table.std(axis=0)
    }).round(3)
    print("\n5 fold-երի ամփոփում (միջին ± ստանդարտ շեղում):")
    print(cv_summary)
    cv_table.to_csv(Path(__file__).resolve().parent / "cv_4_features_folds.csv", index=False)
    cv_summary.to_csv(Path(__file__).resolve().parent / "cv_4_features_summary.csv")
except ImportError:
    print("scikit-learn տեղադրված չէ. 5-fold CV հատվածի համար տեղադրեք scikit-learn-ը")

print("\nՆշում. ծրագիրը ցուցադրում է վիճակագրական գնահատականներ, ոչ բժշկական ախտորոշում։")

plt.show()
