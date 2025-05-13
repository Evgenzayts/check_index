import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN, AffinityPropagation
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

PARQUET_PATH = "data15.parquet"
MS_DBSCAN_ATTEMPTS = 7
AP_ATTEMPTS = 15
DAMPING_AP = 0.9

def load_data(path):
    df = pd.read_parquet(path)
    X = df.drop(columns=["Label"])
    y = df["Label"]
    Xs = StandardScaler().fit_transform(X)
    return Xs, y

def tsne_plot(X_scaled, labels, title, filename):
    print(f"Сохраняем t-SNE визуализацию: {title} → {filename}")
    tsne = TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        perplexity=30,
        max_iter=1000,
        random_state=42
    )
    emb = tsne.fit_transform(X_scaled)
    df = pd.DataFrame({"Component 1": emb[:,0], "Component 2": emb[:,1], "label": labels})
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df,
        x="Component 1", y="Component 2",
        hue="label",
        palette="deep",
        s=40,
        alpha=0.8,
        edgecolor="k"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_model_best(name, X_scaled, y_true, params):
    best_score = -1.0
    best_labels = None

    if name in ("MeanShift", "DBSCAN"):
        for i in range(MS_DBSCAN_ATTEMPTS):
            print(f"\n{name}: попытка #{i+1}")
            if name == "MeanShift":
                if params["bandwidth"] is None:
                    params["bandwidth"] = estimate_bandwidth(X_scaled, quantile=0.2)
                print(f"  bandwidth = {params['bandwidth']:.4f}")
                model = MeanShift(bandwidth=params["bandwidth"])
                params["bandwidth"] *= 1.5
            else:
                print(f"  eps = {params['eps']:.4f}, min_samples = {params['min_samples']}")
                model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
                params["eps"] *= 1.5
                params["min_samples"] += 2

            model.fit(X_scaled)
            labels = model.labels_
            mask = labels != -1
            if len(np.unique(labels[mask])) < 2:
                continue
            score = adjusted_rand_score(y_true[mask], labels[mask])
            print(f"  Rand Index = {score:.4f}")
            if score > best_score:
                best_score, best_labels = score, labels

    else:
        sim = -euclidean_distances(X_scaled)
        median_pref = np.median(sim)
        min_pref = sim.min()
        step = (median_pref - min_pref) / AP_ATTEMPTS
        pref = median_pref

        for i in range(AP_ATTEMPTS):
            print(f"\nAffinityPropagation: попытка #{i+1}\n  preference = {pref:.4f}")
            model = AffinityPropagation(
                damping=DAMPING_AP,
                preference=pref,
                max_iter=500,
                convergence_iter=30,
                random_state=0
            )
            model.fit(X_scaled)
            labels = model.labels_
            mask = labels != -1
            if len(np.unique(labels[mask])) < 2:
                print("  - меньше 2 кластеров")
            else:
                score = adjusted_rand_score(y_true[mask], labels[mask])
                print(f"  Rand Index = {score:.4f}")
                if score > best_score:
                    best_score, best_labels = score, labels
            pref -= step

    return best_labels

def evaluate_metrics(X_scaled, y_true, labels):
    mask = labels != -1
    unique = np.unique(labels[mask])
    if len(unique) < 2:
        return {}
    return {
        "Silhouette": silhouette_score(X_scaled[mask], labels[mask]),
        "Davies-Bouldin": davies_bouldin_score(X_scaled[mask], labels[mask]),
        "Calinski-Harabasz": calinski_harabasz_score(X_scaled[mask], labels[mask]),
        "Rand Index": adjusted_rand_score(y_true[mask], labels[mask]),
    }

def main():
    X_scaled, y_true = load_data(PARQUET_PATH)

    # 1) t-SNE для истинных меток
    tsne_plot(X_scaled, y_true, "t-SNE: Истинные метки", "tsne_true_labels.png")

    params = {
        "MeanShift": {"bandwidth": None},
        "DBSCAN": {"eps": 0.5, "min_samples": 5},
        "AffinityPropagation": {}
    }
    results = {}

    for name in ("MeanShift", "DBSCAN", "AffinityPropagation"):
        print(f"\n===== {name} =====")
        labels = run_model_best(name, X_scaled, y_true, copy.deepcopy(params[name]))
        # 2) t-SNE после кластеризации
        tsne_plot(X_scaled, labels, f"t-SNE: {name}", f"tsne_{name}.png")
        # 3) метрики
        metrics = evaluate_metrics(X_scaled, y_true, labels)
        if metrics:
            results[name] = metrics

    df = pd.DataFrame(results).T
    print("\n=== Итоговые метрики ===")
    print(df)
    print("\nЛучший по Rand Index:", df["Rand Index"].idxmax())
    print("Лучший по Silhouette Score:", df["Silhouette"].idxmax())

if __name__ == "__main__":
    main()

