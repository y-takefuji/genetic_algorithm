import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import f1_score
from scipy import stats
import xgboost as xgb
import shap
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SHAP version-safe extraction helper
# ============================================================
def extract_shap_values(shap_values):
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values

    if isinstance(shap_values, list):
        arr = np.abs(shap_values[1]).mean(axis=0)
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            arr = np.abs(shap_values[:, :, -1]).mean(axis=0)
        elif shap_values.ndim == 2:
            arr = np.abs(shap_values).mean(axis=0)
        else:
            arr = np.abs(shap_values)
    else:
        raise TypeError(f"Unexpected SHAP values type: {type(shap_values)}")

    return arr  # always 1D, shape (n_features,)

# ============================================================
# 1. Load Dataset
# ============================================================
print("=" * 60)
print("Loading Dataset: Heart_Failure_Prediction (ID=45950)")
print("=" * 60)

dataset = fetch_openml(data_id=45950, as_frame=True, parser='auto')
df = dataset.frame

# Convert all columns to numeric; non-numeric entries become NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Remove instances (rows) with any missing data
n_before = len(df)
df = df.dropna(axis=0, how='any').reset_index(drop=True)
n_after = len(df)
print(f"\nInstances before removing missing data : {n_before}")
print(f"Instances after removing missing data  : {n_after}")
print(f"Instances removed                      : {n_before - n_after}")

df['DEATH_EVENT'] = df['DEATH_EVENT'].astype(int)

X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

print(f"\nDataset Shape      : {df.shape}")
print(f"Features Shape     : {X.shape}")
print(f"\nTarget Distribution:")
print(y.value_counts())
print(f"\nTarget Distribution (%):")
print(y.value_counts(normalize=True).round(4) * 100)

feature_names = list(X.columns)
print(f"\nFeatures: {feature_names}")

# ============================================================
# Helper: Cross-validation (accuracy, 95% CI, class-wise F1)
# ============================================================
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def _ci95(scores):
    """95% CI of the CV mean (t-distribution over the 5 fold accuracies)."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    mean = scores.mean()
    sem = scores.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, df=n - 1) * sem
    return mean, mean - half, mean + half

def _cv_full(clf, features, X, y):
    """Per-fold accuracy and class-wise F1; returns mean, 95% CI, F1(0), F1(1)."""
    Xs = X[features]
    acc_scores, f1_0, f1_1 = [], [], []
    for tr, te in cv5.split(Xs, y):
        clf.fit(Xs.iloc[tr], y.iloc[tr])
        pred = clf.predict(Xs.iloc[te])
        acc_scores.append((pred == y.iloc[te].values).mean())
        f = f1_score(y.iloc[te], pred, labels=[0, 1], average=None, zero_division=0)
        f1_0.append(f[0])
        f1_1.append(f[1])
    mean, lo, hi = _ci95(acc_scores)
    return {
        'cv_mean': mean, 'ci_lo': lo, 'ci_hi': hi,
        'f1_0': float(np.mean(f1_0)), 'f1_1': float(np.mean(f1_1))
    }

def cv_rf(features, X, y):
    clf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(clf, X[features], y, cv=cv5, scoring='accuracy')
    return round(scores.mean(), 4)

def cv_xgb(features, X, y):
    clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    scores = cross_val_score(clf, X[features], y, cv=cv5, scoring='accuracy')
    return round(scores.mean(), 4)

# Top-4 (reduced) accuracy: printed only, not included in Table 1
def cv_rf4(features, X, y):
    return cv_rf(features, X, y)

def cv_xgb4(features, X, y):
    return cv_xgb(features, X, y)

def cv_rf_full(features, X, y):
    return _cv_full(RandomForestClassifier(random_state=42), features, X, y)

def cv_xgb_full(features, X, y):
    return _cv_full(xgb.XGBClassifier(random_state=42, eval_metric='logloss',
                                      verbosity=0), features, X, y)

# ============================================================
# 2. RANDOM FOREST Feature Selection
# ============================================================
print("\n" + "=" * 60)
print("METHOD 1: Random Forest Feature Importance")
print("=" * 60)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
rf_importances = pd.Series(rf_model.feature_importances_, index=feature_names)
rf_ranked = rf_importances.sort_values(ascending=False)

print("\nRF Feature Importances (ranked):")
print(rf_ranked.round(4))

rf_top5     = list(rf_ranked.index[:5])
rf_top5_acc = cv_rf(rf_top5, X, y)
print(f"\nRF Top 5: {rf_top5}")
print(f"RF CV5 Accuracy (top 5): {rf_top5_acc}")

rf_highest   = rf_ranked.index[0]
X_rf_reduced = X.drop(columns=[rf_highest])
rf_model2    = RandomForestClassifier(random_state=42)
rf_model2.fit(X_rf_reduced, y)
rf_imp2      = pd.Series(rf_model2.feature_importances_, index=X_rf_reduced.columns)
rf_ranked2   = rf_imp2.sort_values(ascending=False)
rf_top4      = list(rf_ranked2.index[:4])
rf_top4_acc  = cv_rf4(rf_top4, X, y)
print(f"RF Highest Feature Removed: {rf_highest}")
print(f"RF Top 4 (reduced): {rf_top4}")
print(f"RF CV5 Accuracy (top 4): {rf_top4_acc}")

# ============================================================
# 3. XGBOOST Feature Selection
# ============================================================
print("\n" + "=" * 60)
print("METHOD 2: XGBoost Feature Importance")
print("=" * 60)

xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
xgb_model.fit(X, y)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
xgb_ranked = xgb_importances.sort_values(ascending=False)

print("\nXGB Feature Importances (ranked):")
print(xgb_ranked.round(4))

xgb_top5     = list(xgb_ranked.index[:5])
xgb_top5_acc = cv_xgb(xgb_top5, X, y)
print(f"\nXGB Top 5: {xgb_top5}")
print(f"XGB CV5 Accuracy (top 5): {xgb_top5_acc}")

xgb_highest   = xgb_ranked.index[0]
X_xgb_reduced = X.drop(columns=[xgb_highest])
xgb_model2    = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
xgb_model2.fit(X_xgb_reduced, y)
xgb_imp2      = pd.Series(xgb_model2.feature_importances_, index=X_xgb_reduced.columns)
xgb_ranked2   = xgb_imp2.sort_values(ascending=False)
xgb_top4      = list(xgb_ranked2.index[:4])
xgb_top4_acc  = cv_xgb4(xgb_top4, X, y)
print(f"XGB Highest Feature Removed: {xgb_highest}")
print(f"XGB Top 4 (reduced): {xgb_top4}")
print(f"XGB CV5 Accuracy (top 4): {xgb_top4_acc}")

# ============================================================
# 4. GENETIC ALGORITHM Feature Selection (Pure GA)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 3: Genetic Algorithm Feature Selection (Pure GA)")
print("=" * 60)

POP_SIZE       = 10
N_GENERATIONS  = 10
MUTATION_RATE  = 0.01
CROSSOVER_RATE = 0.8
N_FEATURES     = X.shape[1]

np.random.seed(42)

X_arr = X.values
y_arr = y.values

def ga_fitness_pure(chromosome, X_arr, y_arr, cv=3):
    selected_idx = np.where(chromosome == 1)[0]
    if len(selected_idx) == 0:
        return 0.0
    X_sel  = X_arr[:, selected_idx]
    clf    = RandomForestClassifier(random_state=42)
    kf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X_sel, y_arr):
        clf.fit(X_sel[train_idx], y_arr[train_idx])
        scores.append(clf.score(X_sel[val_idx], y_arr[val_idx]))
    return float(np.mean(scores))

def ga_tournament_select(population, fitnesses):
    i1, i2 = np.random.randint(0, len(population), size=2)
    return population[i1].copy() if fitnesses[i1] >= fitnesses[i2] else population[i2].copy()

def ga_crossover(p1, p2):
    if np.random.rand() < CROSSOVER_RATE:
        pt = np.random.randint(1, N_FEATURES)
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])
        return c1, c2
    return p1.copy(), p2.copy()

def ga_mutate(chromosome):
    chrom = chromosome.copy()
    for i in range(N_FEATURES):
        if np.random.rand() < MUTATION_RATE:
            chrom[i] = 1 - chrom[i]
    return chrom

population         = [np.random.randint(0, 2, N_FEATURES) for _ in range(POP_SIZE)]
best_fitness_ga    = 0.0
best_chromosome_ga = None

for gen in range(N_GENERATIONS):
    fitnesses    = [ga_fitness_pure(c, X_arr, y_arr) for c in population]
    gen_best_idx = int(np.argmax(fitnesses))

    if fitnesses[gen_best_idx] > best_fitness_ga:
        best_fitness_ga    = fitnesses[gen_best_idx]
        best_chromosome_ga = population[gen_best_idx].copy()

    new_pop = [population[gen_best_idx].copy()]  # elitism
    while len(new_pop) < POP_SIZE:
        p1 = ga_tournament_select(population, fitnesses)
        p2 = ga_tournament_select(population, fitnesses)
        c1, c2 = ga_crossover(p1, p2)
        new_pop.append(ga_mutate(c1))
        if len(new_pop) < POP_SIZE:
            new_pop.append(ga_mutate(c2))

    population = new_pop
    if (gen + 1) % 10 == 0:
        print(f"  Gen {gen+1:3d}/{N_GENERATIONS} | Best Fitness: {best_fitness_ga:.4f}")

fitnesses_final = [ga_fitness_pure(c, X_arr, y_arr) for c in population]
top_half_idx    = np.argsort(fitnesses_final)[-(POP_SIZE // 2):]

feature_freq = np.zeros(N_FEATURES)
for idx in top_half_idx:
    feature_freq += population[idx]

ga_ranked = pd.Series(feature_freq, index=feature_names).sort_values(ascending=False)
print("\nGA Feature Selection Frequencies (top-50% of final population):")
print(ga_ranked)

ga_top5     = list(ga_ranked.index[:5])
ga_top5_acc = cv_rf(ga_top5, X, y)
print(f"\nGA Top 5: {ga_top5}")
print(f"GA CV5 Accuracy (top 5): {ga_top5_acc}")

ga_highest         = ga_ranked.index[0]
remaining_features = [f for f in feature_names if f != ga_highest]
remaining_idx_map  = [feature_names.index(f) for f in remaining_features]

feature_freq2 = np.zeros(len(remaining_features))
for idx in top_half_idx:
    for j, fi in enumerate(remaining_idx_map):
        feature_freq2[j] += population[idx][fi]

ga_ranked2  = pd.Series(feature_freq2, index=remaining_features).sort_values(ascending=False)
ga_top4     = list(ga_ranked2.index[:4])
ga_top4_acc = cv_rf4(ga_top4, X, y)
print(f"GA Highest Feature Removed: {ga_highest}")
print(f"GA Top 4 (reduced): {ga_top4}")
print(f"GA CV5 Accuracy (top 4): {ga_top4_acc}")

# ============================================================
# 4B. GA-SHAP Feature Selection (Hybrid GA + SHAP)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 4: GA-SHAP Feature Selection (Hybrid GA + SHAP)")
print("=" * 60)

# First, compute SHAP importance scores to guide GA initialization
np.random.seed(42)
shap_sample_idx_ga = np.random.choice(len(X), min(100, len(X)), replace=False)
X_shap_sample_ga   = X.iloc[shap_sample_idx_ga]

rf_shap_ga_model  = RandomForestClassifier(random_state=42)
rf_shap_ga_model.fit(X, y)
rf_explainer_ga   = shap.TreeExplainer(rf_shap_ga_model)
rf_shap_values_ga = rf_explainer_ga.shap_values(X_shap_sample_ga)
shap_importance   = extract_shap_values(rf_shap_values_ga)
shap_importance   = shap_importance / (shap_importance.sum() + 1e-10)  # normalize

print("\nSHAP-based importance scores (normalized):")
for fname, score in zip(feature_names, shap_importance):
    print(f"  {fname}: {score:.4f}")

# GA-SHAP fitness: combines CV accuracy with SHAP guidance
SHAP_WEIGHT = 0.3  # weight for SHAP guidance component

def ga_fitness_shap(chromosome, X_arr, y_arr, shap_scores, cv=3):
    selected_idx = np.where(chromosome == 1)[0]
    if len(selected_idx) == 0:
        return 0.0

    # CV accuracy component
    X_sel  = X_arr[:, selected_idx]
    clf    = RandomForestClassifier(random_state=42, n_estimators=50)
    kf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X_sel, y_arr):
        clf.fit(X_sel[train_idx], y_arr[train_idx])
        scores.append(clf.score(X_sel[val_idx], y_arr[val_idx]))
    cv_acc = float(np.mean(scores))

    # SHAP guidance component: reward selecting high-SHAP features
    shap_score = np.sum(shap_scores[selected_idx]) / len(selected_idx)

    # Combined fitness
    fitness = (1 - SHAP_WEIGHT) * cv_acc + SHAP_WEIGHT * shap_score
    return fitness

# Initialize population with SHAP-guided bias
population_gashap = []
for _ in range(POP_SIZE):
    # Sample features with probability proportional to SHAP importance
    probs = shap_importance ** 2  # square to emphasize high-SHAP features
    probs = probs / probs.sum()
    chrom = np.random.binomial(1, probs)
    # Ensure at least one feature is selected
    if chrom.sum() == 0:
        chrom[np.argmax(shap_importance)] = 1
    population_gashap.append(chrom)

best_fitness_gashap    = 0.0
best_chromosome_gashap = None

print("\nRunning GA-SHAP optimization...")
for gen in range(N_GENERATIONS):
    fitnesses = [ga_fitness_shap(c, X_arr, y_arr, shap_importance) for c in population_gashap]
    gen_best_idx = int(np.argmax(fitnesses))

    if fitnesses[gen_best_idx] > best_fitness_gashap:
        best_fitness_gashap    = fitnesses[gen_best_idx]
        best_chromosome_gashap = population_gashap[gen_best_idx].copy()

    new_pop = [population_gashap[gen_best_idx].copy()]  # elitism
    while len(new_pop) < POP_SIZE:
        p1 = ga_tournament_select(population_gashap, fitnesses)
        p2 = ga_tournament_select(population_gashap, fitnesses)
        c1, c2 = ga_crossover(p1, p2)
        new_pop.append(ga_mutate(c1))
        if len(new_pop) < POP_SIZE:
            new_pop.append(ga_mutate(c2))

    population_gashap = new_pop
    if (gen + 1) % 10 == 0:
        print(f"  Gen {gen+1:3d}/{N_GENERATIONS} | Best Fitness: {best_fitness_gashap:.4f}")

# Extract feature importance from top chromosomes
fitnesses_final_gashap = [ga_fitness_shap(c, X_arr, y_arr, shap_importance) for c in population_gashap]
top_half_idx_gashap    = np.argsort(fitnesses_final_gashap)[-(POP_SIZE // 2):]

feature_freq_gashap = np.zeros(N_FEATURES)
for idx in top_half_idx_gashap:
    feature_freq_gashap += population_gashap[idx]

gashap_ranked = pd.Series(feature_freq_gashap, index=feature_names).sort_values(ascending=False)
print("\nGA-SHAP Feature Selection Frequencies (top-50% of final population):")
print(gashap_ranked)

gashap_top5     = list(gashap_ranked.index[:5])
gashap_top5_acc = cv_rf(gashap_top5, X, y)
print(f"\nGA-SHAP Top 5: {gashap_top5}")
print(f"GA-SHAP CV5 Accuracy (top 5): {gashap_top5_acc}")

# Remove highest and compute top 4
gashap_highest            = gashap_ranked.index[0]
remaining_features_gashap = [f for f in feature_names if f != gashap_highest]
remaining_idx_map_gashap  = [feature_names.index(f) for f in remaining_features_gashap]

feature_freq2_gashap = np.zeros(len(remaining_features_gashap))
for idx in top_half_idx_gashap:
    for j, fi in enumerate(remaining_idx_map_gashap):
        feature_freq2_gashap[j] += population_gashap[idx][fi]

gashap_ranked2  = pd.Series(feature_freq2_gashap, index=remaining_features_gashap).sort_values(ascending=False)
gashap_top4     = list(gashap_ranked2.index[:4])
gashap_top4_acc = cv_rf4(gashap_top4, X, y)
print(f"GA-SHAP Highest Feature Removed: {gashap_highest}")
print(f"GA-SHAP Top 4 (reduced): {gashap_top4}")
print(f"GA-SHAP CV5 Accuracy (top 4): {gashap_top4_acc}")

# ============================================================
# 5. RF-SHAP Feature Selection
# ============================================================
print("\n" + "=" * 60)
print("METHOD 5: RF-SHAP Feature Selection")
print("=" * 60)

np.random.seed(42)
shap_sample_idx = np.random.choice(len(X), min(100, len(X)), replace=False)
X_shap_sample   = X.iloc[shap_sample_idx]

rf_shap_model  = RandomForestClassifier(random_state=42)
rf_shap_model.fit(X, y)

rf_explainer   = shap.TreeExplainer(rf_shap_model)
rf_shap_values = rf_explainer.shap_values(X_shap_sample)
rf_shap_abs    = extract_shap_values(rf_shap_values)

rfshap_ranked  = pd.Series(rf_shap_abs, index=feature_names).sort_values(ascending=False)
print("\nRF-SHAP Mean |SHAP| (ranked):")
print(rfshap_ranked.round(4))

rfshap_top5     = list(rfshap_ranked.index[:5])
rfshap_top5_acc = cv_rf(rfshap_top5, X, y)
print(f"\nRF-SHAP Top 5: {rfshap_top5}")
print(f"RF-SHAP CV5 Accuracy (top 5): {rfshap_top5_acc}")

rfshap_highest   = rfshap_ranked.index[0]
X_rfshap_reduced = X.drop(columns=[rfshap_highest])
rf_shap_model2   = RandomForestClassifier(random_state=42)
rf_shap_model2.fit(X_rfshap_reduced, y)
rf_explainer2    = shap.TreeExplainer(rf_shap_model2)
X_shap_sample2   = X_rfshap_reduced.iloc[shap_sample_idx]
rf_shap_values2  = rf_explainer2.shap_values(X_shap_sample2)
rf_shap_abs2     = extract_shap_values(rf_shap_values2)
rfshap_ranked2   = pd.Series(rf_shap_abs2, index=X_rfshap_reduced.columns).sort_values(ascending=False)
rfshap_top4      = list(rfshap_ranked2.index[:4])
rfshap_top4_acc  = cv_rf4(rfshap_top4, X, y)
print(f"RF-SHAP Highest Feature Removed: {rfshap_highest}")
print(f"RF-SHAP Top 4 (reduced): {rfshap_top4}")
print(f"RF-SHAP CV5 Accuracy (top 4): {rfshap_top4_acc}")

# ============================================================
# 6. XGB-SHAP Feature Selection
# ============================================================
print("\n" + "=" * 60)
print("METHOD 6: XGB-SHAP Feature Selection")
print("=" * 60)

xgb_shap_model  = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
xgb_shap_model.fit(X, y)

xgb_explainer   = shap.TreeExplainer(xgb_shap_model)
xgb_shap_values = xgb_explainer.shap_values(X_shap_sample)
xgb_shap_abs    = extract_shap_values(xgb_shap_values)

xgbshap_ranked  = pd.Series(xgb_shap_abs, index=feature_names).sort_values(ascending=False)
print("\nXGB-SHAP Mean |SHAP| (ranked):")
print(xgbshap_ranked.round(4))

xgbshap_top5     = list(xgbshap_ranked.index[:5])
xgbshap_top5_acc = cv_xgb(xgbshap_top5, X, y)
print(f"\nXGB-SHAP Top 5: {xgbshap_top5}")
print(f"XGB-SHAP CV5 Accuracy (top 5): {xgbshap_top5_acc}")

xgbshap_highest   = xgbshap_ranked.index[0]
X_xgbshap_reduced = X.drop(columns=[xgbshap_highest])
xgb_shap_model2   = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
xgb_shap_model2.fit(X_xgbshap_reduced, y)
xgb_explainer2    = shap.TreeExplainer(xgb_shap_model2)
X_shap_sample3    = X_xgbshap_reduced.iloc[shap_sample_idx]
xgb_shap_values2  = xgb_explainer2.shap_values(X_shap_sample3)
xgb_shap_abs2     = extract_shap_values(xgb_shap_values2)
xgbshap_ranked2   = pd.Series(xgb_shap_abs2, index=X_xgbshap_reduced.columns).sort_values(ascending=False)
xgbshap_top4      = list(xgbshap_ranked2.index[:4])
xgbshap_top4_acc  = cv_xgb4(xgbshap_top4, X, y)
print(f"XGB-SHAP Highest Feature Removed: {xgbshap_highest}")
print(f"XGB-SHAP Top 4 (reduced): {xgbshap_top4}")
print(f"XGB-SHAP CV5 Accuracy (top 4): {xgbshap_top4_acc}")

# ============================================================
# 7. FEATURE AGGLOMERATION (FA) - pure FA, no RF, no target
#    Score = combination of (a) closeness to the cluster centroid and
#    (b) feature variance. Features are ranked ACROSS ALL clusters.
# ============================================================
print("\n" + "=" * 60)
print("METHOD 7: Feature Agglomeration (FA)")
print("=" * 60)

FA_WEIGHT = 0.5   # weight of the centroid component (1 - FA_WEIGHT = variance component)

def _minmax(s):
    """Min-max normalize a pd.Series to [0, 1] (constant -> zeros)."""
    rng = s.max() - s.min()
    if rng == 0:
        return s * 0.0
    return (s - s.min()) / rng

def fa_score_features(X_df, n_clusters, w=FA_WEIGHT):
    """
    Pure FA scoring (unsupervised: no target, no RF).
    1) Cluster the features with FeatureAgglomeration.
    2) Cluster centroid = mean of the member features (per sample).
       Centroid score = |corr(feature, its cluster centroid)|
       (a singleton cluster is its own centroid -> score 1.0).
    3) Variance score = min-max normalized feature variance.
    4) FA score = w * centroid score + (1 - w) * variance score.
    All features are ranked together across clusters.
    Returns cluster labels, ranked FA scores, centroid scores, variance scores.
    """
    fa_model = FeatureAgglomeration(n_clusters=n_clusters)
    fa_model.fit(X_df)
    labels_ = fa_model.labels_

    centroid_score = {}
    for c in np.unique(labels_):
        members  = [f for f, l in zip(X_df.columns, labels_) if l == c]
        centroid = X_df[members].mean(axis=1)
        for f in members:
            if len(members) == 1:
                centroid_score[f] = 1.0
            else:
                r = np.corrcoef(X_df[f].values, centroid.values)[0, 1]
                centroid_score[f] = float(abs(r)) if np.isfinite(r) else 0.0

    centroid_score = pd.Series(centroid_score)[X_df.columns]
    var_score      = _minmax(X_df.var())[X_df.columns]

    fa_score = w * centroid_score + (1 - w) * var_score
    return labels_, fa_score.sort_values(ascending=False), centroid_score, var_score

n_clusters = 5
fa_labels, fa_ranked, fa_centroid, fa_var = fa_score_features(X, n_clusters)

print(f"\nFeature Cluster Assignments:")
for feat, lbl in zip(feature_names, fa_labels):
    print(f"  {feat}: Cluster {lbl}")

print("\nFA Components (centroid score, normalized variance):")
print(pd.DataFrame({'centroid_score': fa_centroid, 'variance_score': fa_var}).round(4))

print(f"\nFA Combined Score (w={FA_WEIGHT} centroid + {1-FA_WEIGHT} variance, ranked):")
print(fa_ranked.round(4))

fa_top5     = list(fa_ranked.index[:5])      # top 5 across all clusters
fa_top5_acc = cv_xgb(fa_top5, X, y)
print(f"\nFA Top 5 (across all clusters): {fa_top5}")
print(f"FA CV5 Accuracy (top 5): {fa_top5_acc}")

# Remove the highest-ranked feature, re-run FA with 4 clusters on the rest
fa_highest   = fa_ranked.index[0]
X_fa_reduced = X.drop(columns=[fa_highest])
fa_labels2, fa_ranked2, _, _ = fa_score_features(
    X_fa_reduced, min(4, X_fa_reduced.shape[1])
)

fa_top4     = list(fa_ranked2.index[:4])     # top 4 across all clusters
fa_top4_acc = cv_xgb4(fa_top4, X, y)
print(f"FA Highest Feature Removed: {fa_highest}")
print(f"FA Top 4 (reduced, across all clusters): {fa_top4}")
print(f"FA CV5 Accuracy (top 4): {fa_top4_acc}")

# ============================================================
# 8. HIGHLY VARIABLE GENE SELECTION (HVGS)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 8: Highly Variable Gene Selection (HVGS) - Variance Based")
print("=" * 60)

feature_variance = X.var()
hvgs_ranked      = feature_variance.sort_values(ascending=False)

print("\nHVGS Feature Variance (ranked):")
print(hvgs_ranked.round(4))

hvgs_top5     = list(hvgs_ranked.index[:5])
hvgs_top5_acc = cv_xgb(hvgs_top5, X, y)
print(f"\nHVGS Top 5: {hvgs_top5}")
print(f"HVGS CV5 Accuracy (top 5): {hvgs_top5_acc}")

hvgs_highest   = hvgs_ranked.index[0]
X_hvgs_reduced = X.drop(columns=[hvgs_highest])
hvgs_variance2 = X_hvgs_reduced.var().sort_values(ascending=False)
hvgs_top4      = list(hvgs_variance2.index[:4])
hvgs_top4_acc  = cv_xgb4(hvgs_top4, X, y)
print(f"HVGS Highest Feature Removed (highest variance): {hvgs_highest}")
print(f"HVGS Top 4 (reduced): {hvgs_top4}")
print(f"HVGS CV5 Accuracy (top 4): {hvgs_top4_acc}")

# ============================================================
# 9. SPEARMAN Correlation Feature Selection
# ============================================================
print("\n" + "=" * 60)
print("METHOD 9: Spearman Correlation Feature Selection")
print("=" * 60)

spearman_scores = {}
for feat in feature_names:
    corr, _ = spearmanr(X[feat], y)
    spearman_scores[feat] = abs(corr)

spearman_ranked = pd.Series(spearman_scores).sort_values(ascending=False)

print("\nSpearman |Correlation| with target (ranked):")
print(spearman_ranked.round(4))

spearman_top5     = list(spearman_ranked.index[:5])
spearman_top5_acc = cv_xgb(spearman_top5, X, y)
print(f"\nSpearman Top 5: {spearman_top5}")
print(f"Spearman CV5 Accuracy (top 5): {spearman_top5_acc}")

spearman_highest = spearman_ranked.index[0]
X_sp_reduced     = X.drop(columns=[spearman_highest])
spearman_scores2 = {}
for feat in X_sp_reduced.columns:
    corr2, _ = spearmanr(X_sp_reduced[feat], y)
    spearman_scores2[feat] = abs(corr2)
spearman_ranked2  = pd.Series(spearman_scores2).sort_values(ascending=False)
spearman_top4     = list(spearman_ranked2.index[:4])
spearman_top4_acc = cv_xgb4(spearman_top4, X, y)
print(f"Spearman Highest Feature Removed: {spearman_highest}")
print(f"Spearman Top 4 (reduced): {spearman_top4}")
print(f"Spearman CV5 Accuracy (top 4): {spearman_top4_acc}")

# ============================================================
# 10. SUMMARY TABLE (Table 1: with 95% CI and class-wise F1)
#     Note: top-4 accuracy is printed above but NOT included here.
# ============================================================
print("\n" + "=" * 60)
print("TABLE 1")
print("=" * 60)

# (method name, evaluation function, top 5 features, top 4 features)
# Classifier per method:
#   RF / GA / GA-SHAP / RF-SHAP -> Random Forest
#   XGB / XGB-SHAP / FA / HVGS / Spearman -> XGBoost
method_info = [
    ('RF',       cv_rf_full,  rf_top5,       rf_top4),
    ('XGB',      cv_xgb_full, xgb_top5,      xgb_top4),
    ('GA',       cv_rf_full,  ga_top5,       ga_top4),
    ('GA-SHAP',  cv_rf_full,  gashap_top5,   gashap_top4),
    ('RF-SHAP',  cv_rf_full,  rfshap_top5,   rfshap_top4),
    ('XGB-SHAP', cv_xgb_full, xgbshap_top5,  xgbshap_top4),
    ('FA',       cv_xgb_full, fa_top5,       fa_top4),
    ('HVGS',     cv_xgb_full, hvgs_top5,     hvgs_top4),
    ('Spearman', cv_xgb_full, spearman_top5, spearman_top4),
]

rows = []
for method, cv_func, top5, top4 in method_info:
    r = cv_func(top5, X, y)
    rows.append({
        'Method': method,
        'CV_Mean_Top5': round(r['cv_mean'], 4),
        'CI95_Lower': round(r['ci_lo'], 4),
        'CI95_Upper': round(r['ci_hi'], 4),
        'CV_Mean_95CI': f"{r['cv_mean']:.3f} ({r['ci_lo']:.3f}-{r['ci_hi']:.3f})",
        'F1_0': round(r['f1_0'], 4),
        'F1_1': round(r['f1_1'], 4),
        'Top_5_Features': ', '.join(top5),
        'Top_4_Features_Reduced': ', '.join(top4),
    })

result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))

result_df.to_csv('result.csv', index=False)
print("\nSaved to result.csv")