import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN, AffinityPropagation
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# === Константы ===
PARQUET_PATH      = "data15.parquet"
PLOTS_DIR         = "analysis_plots"
RANDOM_STATE      = 42
TSNE_SAMPLE_SIZE  = 3000
CORR_THRESHOLD    = 0.8

MS_DBSCAN_ATTEMPTS = 7   # число попыток для MeanShift/DBSCAN
AP_ATTEMPTS        = 10  # число шагов бинарного поиска для AP
DAMPING_AP         = 0.9

# === Функции окружения и утилиты ===
def setup_environment():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.style.use('seaborn-v0_8')
    sns.set_theme(style="whitegrid")

def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def save_plot(fig, filename: str):
    safe = sanitize_filename(filename)
    path = os.path.join(PLOTS_DIR, safe)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Сохранен график: {safe}")

# === Загрузка и препроцессинг данных ===
def remove_highly_correlated_features(X: pd.DataFrame) -> pd.DataFrame:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
    if to_drop:
        print(f"Удалены высококоррелированные признаки: {len(to_drop)}")
        return X.drop(columns=to_drop)
    return X

def load_and_preprocess(path: str):
    df = pd.read_parquet(path)
    y = df["Label"]
    X = df.drop(columns=["Label"])
    X = remove_highly_correlated_features(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\nИспользовано признаков: {X.shape[1]}")
    return X_scaled, y, X

# === EDA ===
def explore_data(X: pd.DataFrame, y: pd.Series):
    print("\n=== Анализ данных (EDA) ===")
    print(f"Образцов: {X.shape[0]}, Признаков: {X.shape[1]}")
    print("Классы:", list(np.unique(y)))
    # распределение классов
    fig, ax = plt.subplots(figsize=(8,4))
    y.value_counts(normalize=True).plot(kind='bar', ax=ax)
    ax.set_title("Распределение классов")
    save_plot(fig, "class_distribution.png")
    # корреляционная матрица
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(X.corr(), cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Корреляционная матрица признаков")
    save_plot(fig, "correlation_matrix.png")

# === Визуализация t-SNE ===
def plot_tsne(X: np.ndarray, labels, title: str, filename: str):
    labels_arr = labels.values if hasattr(labels, 'values') else np.array(labels)
    # подвыборка для ускорения
    if X.shape[0] > TSNE_SAMPLE_SIZE:
        idx = np.random.choice(X.shape[0], TSNE_SAMPLE_SIZE, replace=False)
        X_s, lbl_s = X[idx], labels_arr[idx]
    else:
        X_s, lbl_s = X, labels_arr
    emb = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        perplexity=30,
        max_iter=1000
    ).fit_transform(X_s)
    df2 = pd.DataFrame({"C1": emb[:,0], "C2": emb[:,1], "label": lbl_s})
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df2, x="C1", y="C2", hue="label", palette="deep",
                    s=40, alpha=0.8, edgecolor="k", ax=ax)
    ax.set_title(title)
    save_plot(fig, filename)

# === DBSCAN ===
def run_dbscan(X: np.ndarray, y: pd.Series) -> np.ndarray:
    print("\n=== DBSCAN ===")
    # подбор eps через KneeLocator
    nbrs = NearestNeighbors(n_neighbors=5).fit(X)
    dists = np.sort(nbrs.kneighbors(X)[0][:, -1])
    kl = KneeLocator(range(len(dists)), dists, curve='convex', direction='increasing')
    # eps = dists[kl.knee] if kl.knee else np.median(dists)
    eps = 0.5
    print(f"Используем eps = {eps:.4f}")

    best_score, best_labels, best_params = -1, None, None
    for i in range(MS_DBSCAN_ATTEMPTS):
        ms = 5 + 2*i
        model = DBSCAN(eps=eps, min_samples=ms).fit(X)
        lbl = model.labels_
        mask = lbl != -1
        score = adjusted_rand_score(y[mask], lbl[mask])
        print(f"Попытка {i+1}: min_samples={ms}, eps={eps:.4f}\n  Rand Index={score:.4f}")
        if score > best_score:
            best_score, best_labels, best_params = score, lbl, (ms, eps)
        eps *= 1.5

    print(f"Лучший Rand Index={best_score:.4f} при min_samples={best_params[0]} и eps={best_params[1]:.4f}")
    plot_tsne(X, best_labels, "DBSCAN Clusters", "tsne_dbscan.png")
    return best_labels

# === MeanShift ===
def run_meanshift(X: np.ndarray, y: pd.Series) -> np.ndarray:
    print("\n=== MeanShift ===")
    best_score, best_labels, best_bw = -1, None, None
    bw = estimate_bandwidth(X, quantile=0.2)

    for i in range(MS_DBSCAN_ATTEMPTS):
        model = MeanShift(bandwidth=bw).fit(X)
        lbl = model.labels_
        mask = lbl != -1
        score = adjusted_rand_score(y[mask], lbl[mask])
        print(f"Попытка {i+1}: bandwidth={bw:.4f}\n  Rand Index={score:.4f}")
        if score > best_score:
            best_score, best_labels, best_bw = score, lbl, bw
        bw *= 1.5

    print(f"Лучший Rand Index={best_score:.4f} при bandwidth={best_bw:.4f}")
    plot_tsne(X, best_labels, "MeanShift Clusters", "tsne_meanshift.png")
    return best_labels

# === Affinity Propagation ===
def run_affinity(X: np.ndarray, y: pd.Series) -> np.ndarray:
    print("\n=== Affinity Propagation ===")
    sim = -euclidean_distances(X)
    low, high = sim.min()*2, np.median(sim)
    best_score, best_labels, best_pref = -1, None, None

    for i in range(AP_ATTEMPTS):
        pref = (low + high) / 2
        model = AffinityPropagation(
            damping=DAMPING_AP,
            preference=pref,
            max_iter=500,
            convergence_iter=30,
            random_state=RANDOM_STATE
        ).fit(X)
        lbl = model.labels_
        mask = lbl != -1
        ncls = len(np.unique(lbl[mask]))

        score = adjusted_rand_score(y[mask], lbl[mask])
        print(f"Попытка {i+1}: preference={pref:.2f}\n  Rand Index={score:.4f}")
        if score > best_score:
            best_score, best_labels, best_pref = score, lbl, pref
        high = pref

    print(f"Rand Index={best_score:.4f} при preference={best_pref:.2f}")
    plot_tsne(X, best_labels, "AffinityClusters", "tsne_affinity.png")
    return best_labels

# === Основной запуск ===
def main():
    setup_environment()
    X_scaled, y, X_raw = load_and_preprocess(PARQUET_PATH)

    explore_data(X_raw, y)
    plot_tsne(X_scaled, y, "t-SNE: Истинные метки", "tsne_true_labels.png")

    lbl_ms = run_meanshift(X_scaled, y)
    lbl_db = run_dbscan(X_scaled, y)
    lbl_ap = run_affinity(X_scaled, y)

    print("\n=== Финальные метрики ===")
    
    # Собираем результаты всех алгоритмов
    results = []
    for name, lbl in [("MeanShift", lbl_ms), ("DBSCAN", lbl_db), ("Affinity Propagation", lbl_ap)]:
        mask = lbl != -1
        if len(np.unique(lbl[mask])) < 2:
            continue
            
        metrics = {
            "Algorithm": name,
            "Silhouette": silhouette_score(X_scaled[mask], lbl[mask]),
            "Davies-Bouldin": davies_bouldin_score(X_scaled[mask], lbl[mask]),
            "Calinski-Harabasz": calinski_harabasz_score(X_scaled[mask], lbl[mask]),
            "Rand Index": adjusted_rand_score(y[mask], lbl[mask]),
        }
        results.append(metrics)
    
    # Создаем DataFrame для красивого вывода
    df_results = pd.DataFrame(results)
    
    # Выводим таблицу с метриками
    print("\nСравнительная таблица метрик кластеризации:")
    print(df_results.to_string(index=False, float_format="{:.4f}".format))
    
    # Анализ лучших алгоритмов
    if not df_results.empty:
        best_rand = df_results.loc[df_results['Rand Index'].idxmax()]
        best_silhouette = df_results.loc[df_results['Silhouette'].idxmax()]
        
        print("\n=== Сравнение лучших алгоритмов ===")
        print(f"Лучший по Rand Index ({best_rand['Rand Index']:.4f}): {best_rand['Algorithm']}")
        print(f"\nЛучший по Silhouette Score ({best_silhouette['Silhouette']:.4f}): {best_silhouette['Algorithm']}")
        
        if best_rand['Algorithm'] == best_silhouette['Algorithm']:
            print("\nВывод: Один и тот же алгоритм показал лучшие результаты по обеим метрикам")
        else:
            print("\nВывод: Разные алгоритмы показали лучшие результаты по разным метрикам")

if __name__ == "__main__":
    main()

