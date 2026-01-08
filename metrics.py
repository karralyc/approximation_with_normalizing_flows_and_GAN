import torch
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import gaussian_kde, norm
import ot
from sklearn.metrics.pairwise import rbf_kernel
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore")


class Metrics:
    def __init__(self, device="cpu", seed=None):
        self.device = device
        self.rng = np.random.RandomState(seed)

    def _to_numpy(self, x):
        return x.cpu().detach().numpy() if torch.is_tensor(x) else np.asarray(x)

    # ===================== bandwidth =====================
    def _estimate_bandwidth(self, X, method="median", n_samples=1000):
        X = X[self.rng.choice(len(X), min(len(X), n_samples), replace=False)]
        if method == "median":
            d = pdist(X)
            return np.median(d[d > 0]) if len(d) else 1.0
        d, n = X.shape[1], len(X)
        if method == "scott":
            return n ** (-1 / (d + 4))
        if method == "silverman":
            return (n * (d + 2) / 4) ** (-1 / (d + 4))
        return float(method) if isinstance(method, (int, float)) else 1.0

    # ===================== W1 =====================
    def compute_w1_distance(self, X_true, X_gen, method="sinkhorn", reg=0.1,
                            n_samples=None):

        X, Y = map(self._to_numpy, (X_true, X_gen))
        if n_samples:
            idx = lambda Z: self.rng.choice(len(Z), min(len(Z), n_samples), False)
            X, Y = X[idx(X)], Y[idx(Y)]

        if len(X) < 10 or len(Y) < 10:
            return float("inf")

        M = cdist(X, Y)
        a, b = np.ones(len(X)) / len(X), np.ones(len(Y)) / len(Y)

        P = ot.sinkhorn(a, b, M, reg) if method == "sinkhorn" else ot.emd(a, b, M)
        return float(np.sum(P * M))

    # ===================== MMD =====================
    def compute_mmd(self, X_true, X_gen, kernel="rbf", bandwidth="median",
                    return_p_value=False):

        X, Y = map(self._to_numpy, (X_true, X_gen))
        n, m = len(X), len(Y)

        if n < 2 or m < 2:
            return (1.0, 1.0, 0.0) if return_p_value else 1.0

        if kernel == "rbf":
            bw = self._estimate_bandwidth(np.vstack([X, Y]), bandwidth)
            gamma = 1.0 / (2 * bw ** 2)

            Kxx = rbf_kernel(X, X, gamma)
            Kyy = rbf_kernel(Y, Y, gamma)
            Kxy = rbf_kernel(X, Y, gamma)

        elif kernel == "linear":
            Kxx = X @ X.T
            Kyy = Y @ Y.T
            Kxy = X @ Y.T

        else:  # polynomial
            Kxx = (X @ X.T + 1) ** 3
            Kyy = (Y @ Y.T + 1) ** 3
            Kxy = (X @ Y.T + 1) ** 3

        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)

        mmd2 = (
                Kxx.sum() / (n * (n - 1)) +
                Kyy.sum() / (m * (m - 1)) -
                2.0 * Kxy.mean()
        )

        mmd = float(np.sqrt(max(mmd2, 1e-12)))

        if not return_p_value:
            return mmd
        var = np.var(Kxy)
        thr = np.sqrt(var / min(n, m)) if var > 0 else 0.0
        p = 1 - norm.cdf(mmd / thr) if thr > 0 else 1.0

        return mmd, float(p), float(thr)

    def compute_mmd_multiple_kernels(self, X_true, X_gen,
                                     kernels=("rbf", "linear", "polynomial"),
                                     bandwidths=("median",)):

        results = {}
        for k in kernels:
            for bw in bandwidths:
                key = f"{k}_{bw}"
                try:
                    v, p, _ = self.compute_mmd(X_true, X_gen, k, bw, True)
                except:
                    v, p = 1.0, 1.0
                results[key] = dict(value=v, p_value=p, kernel=k, bandwidth=bw)

        best = min(results.values(), key=lambda x: x["value"])
        return {"best": best, "all": results,
                "selected_kernel": best["kernel"],
                "selected_bandwidth": best["bandwidth"]}

    # ===================== KSD =====================
    @lru_cache(10)
    def _get_cached_score_function(self, name, _=""):
        if name == "gaussian_mixture":
            return lambda x: -x
        return None

    def get_target_score_function(self, name, params=None):
        return self._get_cached_score_function(name, str(params))

    def compute_ksd(self, X_gen, score_function=None,
                    bandwidth="median", true_distribution=None, n_samples=1000):

        X = self._to_numpy(X_gen)
        if len(X) < 10:
            return 1.0
        X = X[self.rng.choice(len(X), min(len(X), n_samples), False)]

        if score_function is None and true_distribution:
            score_function = self.get_target_score_function(true_distribution)
        score = score_function(X) if score_function else self._estimate_score_kde(X)

        bw = self._estimate_bandwidth(X, bandwidth)
        return self._ksd_rbf(X, score, bw)

    def _ksd_rbf(self, X, score, bw, block=512):
        n, d = X.shape
        inv_bw2 = 1.0 / (bw ** 2)
        inv_bw4 = inv_bw2 ** 2

        total = 0.0
        count = 0

        for i in range(0, n, block):
            Xi = X[i:i + block]
            si = score[i:i + block]

            diff = Xi[:, None] - X[None]
            r2 = np.sum(diff ** 2, axis=-1)

            k = np.exp(-0.5 * r2 * inv_bw2)
            g = -diff * inv_bw2 * k[..., None]
            h = (r2 * inv_bw4 - d * inv_bw2) * k

            term = (
                    (si @ score.T) * k +
                    (si[:, None] * g).sum(-1) -
                    (score[None] * g).sum(-1) +
                    h
            )

            total += term.sum()
            count += term.size

        return float(np.sqrt(max(total / count, 1e-12)))

    def _estimate_score_kde(self, X):
        kde = gaussian_kde(X.T)
        p = kde(X.T) + 1e-12

        grad = np.zeros_like(X)
        eps = 1e-4

        for i in range(X.shape[1]):
            Xp = X.copy()
            Xm = X.copy()
            Xp[:, i] += eps
            Xm[:, i] -= eps
            grad[:, i] = (kde(Xp.T) - kde(Xm.T)) / (2 * eps * p)

        return grad

    # ===================== SUMMARY =====================
    def compute_all_metrics(self, X_true, X_gen, true_distribution=None, **_):
        X, Y = map(self._to_numpy, (X_true, X_gen))
        if len(X) < 20 or len(Y) < 20:
            return dict(W1=float("inf"), MMD_best=1.0, KSD=1.0)

        mmd = self.compute_mmd_multiple_kernels(X, Y)
        return dict(
            W1=self.compute_w1_distance(X, Y),
            MMD_best=mmd["best"]["value"],
            MMD_selected_kernel=mmd["best"]["kernel"],
            KSD=self.compute_ksd(Y, true_distribution=true_distribution),
            true_mean=X.mean(0).tolist(),
            gen_mean=Y.mean(0).tolist(),
            mean_distance=float(np.linalg.norm(X.mean(0) - Y.mean(0)))
        )
