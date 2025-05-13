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

MIN_CLUSTERS = 3    # минимум кластеров
MAX_CLUSTERS = 6    # максимум кластеров

# MeanShift / DBSCAN: итерации для подбора параметров
MS_DBSCAN_ATTEMPTS = 7

# AffinityPropagation: параметры бинарного поиска
DAMPING_AP   = 0.9
SEARCH_ITERS = 10

def load_data(path):
    df = pd.read_parquet(path)
    X = df.drop(columns=["Label"])
    y = df["Label"]
    Xs = StandardScaler().fit_transform(X)
    return Xs, y

def tsne_plot(X_scaled, y_true):
    print("Выполняется t-SNE визуализация...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_embedded = tsne.fit_transform(X_scaled)
    df_tsne = pd.DataFrame({
        "Component 1": X_embedded[:, 0],
        "Component 2": X_embedded[:, 1],
        "Label": y_true
    })
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_tsne,
        x="Component 1", y="Component 2",
        hue="Label",
        palette="deep",
        s=50,
        alpha=0.85,
        edgecolor="k"
    )
    plt.title("t-SNE Visualization of Clusters for df15", fontsize=14)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig("tsne_visualization.png")
    plt.show()

def find_optimal_preference(X_scaled, damping, min_c, max_c, max_iter):
    sim = -euclidean_distances(X_scaled)
    low  = sim.min() * 2       # стартовое вдвое ниже минимума
    high = np.median(sim)      # медиана сходств
    best = {"diff": float("inf")}

    for i in range(1, max_iter+1):
        pref = (low + high) / 2
        ap = AffinityPropagation(
            damping=pref and damping or DAMPING_AP,
            preference=pref,
            max_iter=500,
            convergence_iter=30,
            random_state=0
        )
        ap.fit(X_scaled)
        labels = ap.labels_
        mask   = labels != -1
        n_cls  = len(np.unique(labels[mask]))
        diff   = 0 if min_c <= n_cls <= max_c else abs(n_cls - max_c if n_cls>max_c else n_cls-min_c)

        print(f"[AP #{i:02d}] pref={pref:.4f}, clusters={n_cls}, diff_to_range={diff}")

        # записывается лучший
        if diff < best["diff"]:
            best.update({"pref": pref, "model": ap, "labels": labels, "diff": diff})

        # если уже в интервале, то заканчивается
        if min_c <= n_cls <= max_c:
            break

        # определяется, увеличивать или уменьшать
        if n_cls > max_c:
            high = pref
        else:
            low = pref

    model, labels = best["model"], best["labels"]
    print(f"AP окончательно: pref={best['pref']:.4f}, clusters={len(np.unique(labels[labels!=-1]))}")
    return model, labels

def run_model(name, X_scaled, params):
    params = copy.deepcopy(params)
    max_attempts = MS_DBSCAN_ATTEMPTS if name in ("MeanShift", "DBSCAN") else 1
    attempt = 0

    while True:
        attempt += 1
        print(f"\nЗапуск {name}, попытка #{attempt}")

        if name == "MeanShift":
            if params["bandwidth"] is None:
                params["bandwidth"] = estimate_bandwidth(X_scaled, quantile=0.2)
            model = MeanShift(bandwidth=params["bandwidth"])
            model.fit(X_scaled)
            labels = model.labels_

        elif name == "DBSCAN":
            model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
            model.fit(X_scaled)
            labels = model.labels_

        elif name == "AffinityPropagation":
            if params["preference"] is None:
                model, labels = find_optimal_preference(
                    X_scaled, DAMPING_AP, MIN_CLUSTERS, MAX_CLUSTERS, SEARCH_ITERS
                )
                params["preference"] = model.preference
            else:
                model = AffinityPropagation(
                    damping=DAMPING_AP,
                    preference=params["preference"],
                    max_iter=500,
                    convergence_iter=30,
                    random_state=0
                )
                model.fit(X_scaled)
                labels = model.labels_

        else:
            raise ValueError(f"Неизвестная модель {name}")

        # оценка числа кластеров
        mask  = labels != -1
        n_cls = len(np.unique(labels[mask]))
        print(f"{name}: {n_cls} кластеров")

        if MIN_CLUSTERS <= n_cls <= MAX_CLUSTERS:
            return model, labels

        if attempt < max_attempts and name in ("MeanShift", "DBSCAN"):
            print(f"{name}: вне диапазона кластеров, подбор параметров...")
            if name == "MeanShift":
                params["bandwidth"] *= 1.5
                print(f"   bandwidth → {params['bandwidth']:.3f}")
            else:
                params["eps"] *= 1.5
                params["min_samples"] += 2
                print(f"   eps → {params['eps']:.3f}, min_samples → {params['min_samples']}")
            continue

        # если попытки исчерпаны — принимается текущий результат
        print(f"{name}: попытки закончились, принимаем текущее ({n_cls} кластеров)")
        return model, labels

def evaluate(X_scaled, y_true, labels):
    mask = labels != -1
    if len(np.unique(labels[mask])) < 2:
        return None
    return {
        "Silhouette": silhouette_score(X_scaled[mask], labels[mask]),
        "Davies-Bouldin": davies_bouldin_score(X_scaled[mask], labels[mask]),
        "Calinski-Harabasz": calinski_harabasz_score(X_scaled[mask], labels[mask]),
        "Rand Index":     adjusted_rand_score(y_true[mask], labels[mask]),
    }

def main():
    X_scaled, y_true = load_data(PARQUET_PATH)
    # исходные параметры
    global params
    params = {
        "MeanShift": {"bandwidth": None},
        "DBSCAN": {"eps": 0.5, "min_samples": 5},
        "AffinityPropagation": {"preference": None}
    }

    results = {}
    for name in ("MeanShift", "DBSCAN", "AffinityPropagation"):
        print(f"\n\n===== Модель: {name} =====")
        try:
            model, labels = run_model(name, X_scaled, params[name])
            metrics = evaluate(X_scaled, y_true, labels)
            if metrics:
                results[name] = metrics
                print(f"{name}: метрики рассчитаны")
            else:
                print(f"{name}: недостаточно кластеров для оценки")
        except Exception as e:
            print(f"Ошибка в {name}: {e}")

    if results:
        df_res = pd.DataFrame(results).T
        print("\n=== Результаты ===")
        print(df_res, "\n")
        best_rand = df_res["Rand Index"].idxmax()
        best_sil  = df_res["Silhouette"].idxmax()
        print(f"Лучший по Rand Index:       {best_rand}")
        print(f"Лучший по Silhouette Score: {best_sil}")
    else:
        print("Нет успешных результатов")
    tsne_plot(X_scaled, y_true)

if __name__ == "__main__":
    main()

