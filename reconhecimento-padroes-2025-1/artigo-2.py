import numpy as np
from scipy.optimize import minimize

# PDF multivariada
def pdfnvar(x, m, K, n=2):
    x = np.asarray(x).reshape(-1)
    m = np.asarray(m).reshape(-1)
    diff = x - m
    exponent = -0.5 * np.dot(diff.T, np.linalg.solve(K, diff))
    denominator = np.sqrt((2 * np.pi) ** n * np.linalg.det(K))
    return (1 / denominator) * np.exp(exponent)

def project_data(X_source, X_ref, y_ref, k_proj, h_proj):
    dist = np.sum((X_source[:, None, :] - X_ref[None, :, :])**2, axis=2)
    k_proj = min(k_proj, X_ref.shape[0])
    idx = np.argsort(dist, axis=1)[:, :k_proj]
    n_features = X_ref.shape[1]
    K_mat = h_proj * np.eye(n_features)
    proj = []
    for i, indices_for_xt in enumerate(idx):
        xt = X_source[i]
        sum_pos = sum(pdfnvar(X_ref[j], xt, K_mat, n_features) for j in indices_for_xt if y_ref[j] == 1)
        sum_neg = sum(pdfnvar(X_ref[j], xt, K_mat, n_features) for j in indices_for_xt if y_ref[j] == -1)
        proj.append([sum_pos, sum_neg])
    return np.array(proj)

# Distância entre centróides no espaço Q1, Q2
def centroid_distance(X, y, k, h):
    X_pos = X[y == 1]
    X_neg = X[y == -1]

    if X_pos.shape[0] == 0 or X_neg.shape[0] == 0:
        return np.inf

    proj_pos = project_data(X_pos, X, y, k, h)
    proj_neg = project_data(X_neg, X, y, k, h)

    if proj_pos.size == 0 or proj_neg.size == 0:
        return np.inf

    c1 = proj_pos.mean(axis=0)
    c2 = proj_neg.mean(axis=0)
    return np.linalg.norm(c1 - c2)

# Otimiza k e h para minimizar distância entre centróides
def optimize_centroid_distance(X, y):
    def objective(params):
        k = int(np.clip(round(params[0]), 1, 400))
        h = 10 ** np.clip(params[1], -5, 2)
        return centroid_distance(X, y, k, h)

    result = minimize(objective, x0=[10, -1], bounds=[(1, 200), (-5, 1)], method='L-BFGS-B')
    best_k = int(round(result.x[0]))
    best_h = 10 ** result.x[1]
    return best_k, best_h



from sklearn.model_selection import StratifiedKFold,train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def evaluate_knn_centroid_distance_gridsearch_cv(dataset_df, n_cv_splits=2, frac=1, n_k=20, n_h=20):
    df = dataset_df.sample(frac=frac, random_state=42).reset_index(drop=True)
    X = df.drop('target', axis=1).values
    y = df['target'].values

    skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
    accs, f1s, aucs, mccs = [], [], [], []
    train_times = []

    k_values = np.unique(np.round(np.linspace(1, 201, n_k)).astype(int))
    h_values = np.logspace(-5, 2, n_h)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        start_time = time.time()
        # Grid search só no treino!
        best_score = -np.inf
        best_k = None
        best_h = None
        for k in k_values:
            for h in h_values:
                proj = project_data(X_train, X_train, y_train, k, h)
                y_pred = np.where(proj[:, 0] > proj[:, 1], 1, -1)
                score = np.mean(y_train == y_pred)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_h = h
        train_time = time.time() - start_time
        train_times.append(train_time)
        # Avalia no teste
        proj = project_data(X_test, X_train, y_train, best_k, best_h)
        y_pred = np.where(proj[:, 0] > proj[:, 1], 1, -1)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        if len(np.unique(y_test)) == 2:
            aucs.append(roc_auc_score(y_test, proj[:, 0] - proj[:, 1]))
        else:
            aucs.append(np.nan)
        mccs.append(matthews_corrcoef(y_test, y_pred))

    results = {
        "accuracy": accs,
        "f1": f1s,
        "roc_auc": aucs,
        "mcc": mccs,
        "train_time": train_times,
    }
    return results


def evaluate_knn_centroid_distance(dataset_df, n_cv_splits=5, frac=1):
    # Embaralha o dataset antes de tudo
    df = dataset_df.sample(frac=frac, random_state=42).reset_index(drop=True)
    X = df.drop('target', axis=1).values
    y = df['target'].values

    skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=False)
    accs, f1s, aucs, mccs = [], [], [], []
    train_times = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        start_time = time.time()
        best_k, best_h = optimize_centroid_distance(X_train, y_train)
        train_time = time.time() - start_time
        train_times.append(train_time)
        proj = project_data(X_test, X_train, y_train, best_k, best_h)
        y_pred = np.where(proj[:, 0] > proj[:, 1], 1, -1)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        if len(np.unique(y_test)) == 2:
            aucs.append(roc_auc_score(y_test, proj[:, 0] - proj[:, 1]))
        else:
            aucs.append(np.nan)
        mccs.append(matthews_corrcoef(y_test, y_pred))

    results = {
        "accuracy": accs,
        "f1": f1s,
        "roc_auc": aucs,
        "mcc": mccs,
        "train_time": train_times,
    }
    return results

def evaluate_random_forest(dataset_df, n_cv_splits=5, rf_kwargs=None, frac=1):
    # Embaralha o dataset antes de tudo
    df = dataset_df.sample(frac=frac, random_state=42).reset_index(drop=True)
    X = df.drop('target', axis=1).values
    y = df['target'].values

    if rf_kwargs is None:
        rf_kwargs = {}

    skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=False)
    accs, f1s, aucs, mccs = [], [], [], []
    train_times = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(random_state=42, **rf_kwargs)
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        train_times.append(train_time)
        y_pred = clf.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        # Para ROC AUC, precisa de ambas as classes no y_test
        if len(np.unique(y_test)) == 2:
            # Usa a probabilidade da classe positiva como score contínuo
            y_score = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_score))
        else:
            aucs.append(np.nan)
        mccs.append(matthews_corrcoef(y_test, y_pred))

    results = {
        "accuracy": accs,
        "f1": f1s,
        "roc_auc": aucs,
        "mcc": mccs,
        "train_time": train_times,
    }
    return results

def evaluate_svm(dataset_df, n_cv_splits=5, svm_kwargs=None, frac=1):
    # Embaralha o dataset antes de tudo
    df = dataset_df.sample(frac=frac, random_state=42).reset_index(drop=True)
    X = df.drop('target', axis=1).values
    y = df['target'].values

    if svm_kwargs is None:
        svm_kwargs = {}

    skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=False)
    accs, f1s, aucs, mccs = [], [], [], []
    train_times = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # probability=True é necessário para predict_proba (ROC AUC)
        clf = SVC(probability=True, random_state=42, **svm_kwargs)
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        train_times.append(train_time)
        y_pred = clf.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        # Para ROC AUC, precisa de ambas as classes no y_test
        if len(np.unique(y_test)) == 2:
            y_score = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_score))
        else:
            aucs.append(np.nan)
        mccs.append(matthews_corrcoef(y_test, y_pred))

    results = {
        "accuracy": accs,
        "f1": f1s,
        "roc_auc": aucs,
        "mcc": mccs,
        "train_time": train_times,
    }
    return results

def evaluate_logistic_regression(dataset_df, n_cv_splits=5, lr_kwargs=None, frac=1):
    # Embaralha o dataset antes de tudo
    df = dataset_df.sample(frac=frac, random_state=42).reset_index(drop=True)
    X = df.drop('target', axis=1).values
    y = df['target'].values

    if lr_kwargs is None:
        lr_kwargs = {}

    skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=False)
    accs, f1s, aucs, mccs = [], [], [], []
    train_times = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = LogisticRegression(random_state=42, max_iter=1000, **lr_kwargs)
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        train_times.append(train_time)
        y_pred = clf.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        # Para ROC AUC, precisa de ambas as classes no y_test
        if len(np.unique(y_test)) == 2:
            y_score = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_score))
        else:
            aucs.append(np.nan)
        mccs.append(matthews_corrcoef(y_test, y_pred))

    results = {
        "accuracy": accs,
        "f1": f1s,
        "roc_auc": aucs,
        "mcc": mccs,
        "train_time": train_times,
    }
    return results

def evaluate_xgboost(dataset_df, n_cv_splits=5, xgb_kwargs=None, frac=1):
    # Embaralha o dataset antes de tudo
    df = dataset_df.sample(frac=frac, random_state=42).reset_index(drop=True)
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Converte -1/1 para 0/1 para o XGBoost
    y_xgb = np.where(y == -1, 0, 1)

    if xgb_kwargs is None:
        xgb_kwargs = {}

    skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=False)
    accs, f1s, aucs, mccs = [], [], [], []
    train_times = []

    for train_index, test_index in skf.split(X, y_xgb):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_xgb[train_index], y_xgb[test_index]
        clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **xgb_kwargs)
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        train_times.append(train_time)
        y_pred = clf.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        # Para ROC AUC, precisa de ambas as classes no y_test
        if len(np.unique(y_test)) == 2:
            y_score = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_score))
        else:
            aucs.append(np.nan)
        mccs.append(matthews_corrcoef(y_test, y_pred))

    results = {
        "accuracy": accs,
        "f1": f1s,
        "roc_auc": aucs,
        "mcc": mccs,
        "train_time": train_times,
    }
    return results

import os
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

pasta = 'data/dados'
arquivos_mat = [f for f in os.listdir(pasta) if f.endswith('_1.mat')]
datasets_mat = {}

for arquivo in tqdm(arquivos_mat, desc="Carregando datasets"):
    mat_data = scipy.io.loadmat(os.path.join(pasta, arquivo))
    nome_dataset = '_'.join(arquivo.split('_')[:2])
    X, y = mat_data['data'][0][0][0], mat_data['data'][0][0][1]
    
    # Normaliza os dados (sem PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Garante que y seja 1D e valores sejam -1 ou 1
    y = y.flatten()
    y = np.where(y == 0, -1, y)  # Converter possíveis 0s para -1

    # Cria DataFrame igual aos outros datasets
    colunas = [f'feat_{i}' for i in range(X_scaled.shape[1])]
    df = pd.DataFrame(X_scaled, columns=colunas)
    df['target'] = y

    datasets_mat[nome_dataset] = df


import pandas as pd
from tqdm import tqdm

# Defina aqui a fração dos dados que deseja usar (ex: 1.0 para tudo, 0.2 para 20%)
frac = 1  # Altere para testar mais rápido

model_funcs = {
    "KNN_Gridsearch": evaluate_knn_centroid_distance_gridsearch_cv,
    "KNN_Centroid": evaluate_knn_centroid_distance,
    "RandomForest": evaluate_random_forest,
    "SVM": evaluate_svm,
    "LogisticRegression": evaluate_logistic_regression,
    "XGBoost": evaluate_xgboost,
}

all_results = []

for dataset_name, dataset_df in tqdm(datasets_mat.items(), desc="Datasets"):
    print(f"\n=== Rodando para dataset: {dataset_name} ===")
    for model_name, model_func in tqdm(model_funcs.items(), desc=f"Modelos ({dataset_name})", leave=False, position=1):
        print(f"  -> Rodando modelo: {model_name}")
        try:
            results = model_func(dataset_df, frac=frac)
            n_folds = len(results['accuracy'])
            for i in range(n_folds):
                all_results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "run": i+1,
                    "accuracy": results['accuracy'][i],
                    "f1": results['f1'][i],
                    "roc_auc": results['roc_auc'][i],
                    "mcc": results['mcc'][i],
                    "train_time": results['train_time'][i],
                    "frac": frac,
                })
        except Exception as e:
            print(f"Erro inesperado para {model_name} em {dataset_name}: {e}")
            continue

df_results = pd.DataFrame(all_results)
df_results.to_csv("data/resultados_modelos_cv_mat_cv.csv", index=False)
