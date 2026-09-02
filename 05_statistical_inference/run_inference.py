"""Hypothesis tests, p-values, and a Central Limit Theorem simulation."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path(__file__).parent / "outputs"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42); alpha = .05
# Simulated A/B experiment: null hypotheses are equal means and equal variances.
control = rng.lognormal(mean=3.0, sigma=.35, size=90)
treatment = rng.lognormal(mean=3.10, sigma=.35, size=95)
t_stat, t_p = stats.ttest_ind(control, treatment, equal_var=False)
f_stat = np.var(control, ddof=1) / np.var(treatment, ddof=1)
df1, df2 = len(control)-1, len(treatment)-1
f_p = 2 * min(stats.f.cdf(f_stat, df1, df2), stats.f.sf(f_stat, df1, df2))
result = {"significance_level": alpha, "t_statistic": float(t_stat), "t_test_p_value": float(t_p),
          "t_test_conclusion": "reject equal means" if t_p < alpha else "fail to reject equal means",
          "f_statistic": float(f_stat), "f_test_p_value": float(f_p),
          "f_test_conclusion": "reject equal variances" if f_p < alpha else "fail to reject equal variances"}
(OUT / "test_results.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
# Repeated sample means from a skewed distribution become approximately normal.
sample_means = np.array([rng.exponential(scale=1, size=40).mean() for _ in range(4000)])
fig, axes = plt.subplots(1,2, figsize=(10,4)); axes[0].hist(rng.exponential(size=4000), bins=45, density=True)
axes[0].set(title="Skewed population", xlabel="Value"); axes[1].hist(sample_means, bins=45, density=True)
x = np.linspace(sample_means.min(), sample_means.max(), 250); axes[1].plot(x, stats.norm.pdf(x, sample_means.mean(), sample_means.std()), "r--")
axes[1].set(title="Sample means: CLT", xlabel="Mean of n=40 samples"); fig.tight_layout(); fig.savefig(OUT / "central_limit_theorem.png", dpi=160)
