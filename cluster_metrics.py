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
import os
import re

# Константы
PARQUET_PATH = "data15.parquet"
RANDOM_STATE = 42
TSNE_SAMPLE_SIZE = 3000
PLOTS_DIR = "analysis_plots"
CORR_THRESHOLD = 0.8
MS_DBSCAN_ATTEMPTS = 7
AP_ATTEMPTS = 15
DAMPING_AP = 0.9

def setup_environment():
    """Создает директории для графиков"""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.style.use('seaborn-v0_8')
    sns.set_theme(style="whitegrid")

def sanitize_filename(filename: str) -> str:
    """Очищает имя файла от спецсимволов"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def remove_highly_correlated_features(X: pd.DataFrame) -> pd.DataFrame:
    """Удаляет высококоррелированные признаки"""
    corr_matrix = X.corr().abs()
    upper_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper = corr_matrix.where(upper_mask)
    to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
    
    if to_drop:
        print(f"Удалены высококоррелированные признаки ({len(to_drop)}): {to_drop}")
        return X.drop(columns=to_drop)
    return X

def load_and_preprocess_data(path: str):
    """Загрузка и подготовка данных"""
    df = pd.read_parquet(path)
    y = df["Label"]
    X = df.drop(columns=["Label"])
    
    X_clean = remove_highly_correlated_features(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    print(f"\nИспользовано признаков: {X_clean.shape[1]}")
    return X_scaled, y, X_clean

def save_plot(fig, filename: str):
    """Сохраняет график с обработкой имени файла"""
    safe_name = sanitize_filename(filename)
    path = os.path.join(PLOTS_DIR, safe_name)
    fig.savefig(path)
    plt.close(fig)
    print(f"Сохранен график: {safe_name}")

def explore_data(X: pd.DataFrame, y: pd.Series):
    """Анализ и визуализация данных"""
    print("\n=== Анализ данных ===")
    print(f"Образцов: {len(X)}, Признаков: {len(X.columns)}")
    print(f"Классы: {np.unique(y)}")
    
    # Распределение классов
    fig, ax = plt.subplots(figsize=(8,4))
    y.value_counts(normalize=True).plot(kind='bar', ax=ax)
    ax.set_title('Распределение классов')
    save_plot(fig, 'class_distribution.png')
    
    # Корреляционная матрица
    if len(X.columns) > 1:
        fig, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(X.corr(), cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Корреляционная матрица")
        save_plot(fig, 'correlation_matrix.png')

def plot_tsne(X: np.ndarray, labels: np.ndarray, title: str, filename: str):
    """Визуализация t-SNE"""
    if len(X) > TSNE_SAMPLE_SIZE:
        idx = np.random.choice(len(X), TSNE_SAMPLE_SIZE, replace=False)
        X_sample, labels_sample = X[idx], labels[idx]
    else:
        X_sample, labels_sample = X, labels
        
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
    emb = tsne.fit_transform(X_sample)
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=labels_sample, 
                   palette="deep", s=40, alpha=0.8, ax=ax)
    ax.set_title(title)
    save_plot(fig, filename)

def run_clustering(X: np.ndarray, y: pd.Series) -> dict:
    """Запуск и оценка алгоритмов кластеризации"""
    results = {}
    
    # DBSCAN
    print("\n=== DBSCAN ===")
    nn = NearestNeighbors(n_neighbors=5).fit(X)
    distances = np.sort(nn.kneighbors(X)[0][:, -1])
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    eps = distances[kneedle.knee] if kneedle.knee else np.median(distances)
    
    dbscan = DBSCAN(eps=eps, min_samples=5).fit(X)
    dbscan_labels = dbscan.labels_
    results['DBSCAN'] = evaluate_clustering(X, y, dbscan_labels)
    plot_tsne(X, dbscan_labels, "DBSCAN Clusters", "tsne_dbscan.png")
    
    # MeanShift
    print("\n=== MeanShift ===")
    bandwidth = estimate_bandwidth(X, quantile=0.2)
    meanshift = MeanShift(bandwidth=bandwidth).fit(X)
    meanshift_labels = meanshift.labels_
    results['MeanShift'] = evaluate_clustering(X, y, meanshift_labels)
    plot_tsne(X, meanshift_labels, "MeanShift Clusters", "tsne_meanshift.png")
    
    # Affinity Propagation
    print("\n=== Affinity Propagation ===")
    sim = -euclidean_distances(X)
    pref_values = np.linspace(np.median(sim), sim.min(), AP_ATTEMPTS)
    
    best_ap_score = -1
    for i, pref in enumerate(pref_values):
        ap = AffinityPropagation(damping=DAMPING_AP, preference=pref, random_state=RANDOM_STATE).fit(X)
        labels = ap.labels_
        mask = labels != -1
        if len(np.unique(labels[mask])) >= 2:
            score = adjusted_rand_score(y[mask], labels[mask])
            if score > best_ap_score:
                best_ap_score = score
                best_ap_labels = labels
                print(f"Попытка {i+1}: preference={pref:.2f}, Rand Index={score:.4f}")
    
    if best_ap_score > -1:
        results['AffinityProp'] = evaluate_clustering(X, y, best_ap_labels)
        plot_tsne(X, best_ap_labels, "Affinity Propagation Clusters", "tsne_affinity.png")
    else:
        print("Не удалось найти хорошие кластеры для Affinity Propagation")
    
    return results

def evaluate_clustering(X: np.ndarray, y: pd.Series, labels: np.ndarray) -> dict:
    """Вычисление метрик качества кластеризации"""
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    
    if len(unique_labels) < 2:
        print("Обнаружено менее 2 кластеров")
        return {}
    
    metrics = {
        'Silhouette': silhouette_score(X[mask], labels[mask]),
        'Davies-Bouldin': davies_bouldin_score(X[mask], labels[mask]),
        'Calinski-Harabasz': calinski_harabasz_score(X[mask], labels[mask]),
        'Rand Index': adjusted_rand_score(y[mask], labels[mask]),
        'Clusters': len(unique_labels),
        'Noise Points': np.sum(labels == -1)
    }
    
    print("Метрики кластеризации:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    return metrics

def main():
    setup_environment()
    
    # Загрузка и подготовка данных
    X_scaled, y, X_clean = load_and_preprocess_data(PARQUET_PATH)
    
    # Анализ данных
    explore_data(X_clean, y)
    plot_tsne(X_scaled, y, "Истинные метки", "tsne_true_labels.png")
    
    # Кластеризация
    results = run_clustering(X_scaled, y)
    
    # Вывод результатов
    print("\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
    for algo, metrics in results.items():
        print(f"\n{algo}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    main()
