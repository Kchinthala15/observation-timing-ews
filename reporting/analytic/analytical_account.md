---
title: "Supporting Information S1. Analytical account: lag-1 autocorrelation and effective sample size under even versus random observation times"
---

Supporting Information for: Early-warning statistics under escalating observation density: a simulation study with evidence from hospital laboratory data. Lakshmi Kalyani Chinthala.

# Setting

Let $X(t)$ be a stationary Ornstein–Uhlenbeck (OU) process with mean 0, stationary variance $V$ and mean-reversion rate $\theta > 0$, so that $\operatorname{Corr}\{X(s), X(s+\Delta)\} = \exp(-\theta\Delta)$ for $\Delta \ge 0$. An analysis window is defined as $n$ consecutive observations made at mean rate $\lambda$ (observations per unit time). Measurement error and the drift of $\theta$ within a window are ignored (see "Scope"). Two sampling designs with the same rate are compared:

- Even: every gap $t_{i+1} - t_i$ equals $1/\lambda$.
- Random: observation times follow a homogeneous Poisson process, so the $n - 1$ gaps within a window of $n$ consecutive observations are independent Exponential($\lambda$) variables.

The random design is an idealised analytical comparison. The simulated random-time arm instead places a fixed number of observations at independent uniform times over the post-baseline period (equivalent to a Poisson process conditioned on its count), so its gaps are exchangeable but not independent. The results below are therefore not an exact derivation for that arm; to first order they agree for large $n$.

# Result 1: lag-1 autocorrelation in observation order

Even times: the lag-1 correlation of consecutive observations is $\rho_E = \exp(-\theta/\lambda)$.

Random times: given its gap $\Delta$, a consecutive pair has correlation $\exp(-\theta\Delta)$. Because $\operatorname{Var}(X(t_i)) = V$ for every $i$, the correlation of consecutive observations over the random times is exactly $\rho_P = E[\exp(-\theta\Delta)] = \lambda/(\lambda+\theta)$, the Laplace transform of the exponential distribution. With $x = \theta/\lambda > 0$, $\rho_P = 1/(1+x) > \exp(-x) = \rho_E$, since $\exp(x) > 1 + x$ (equivalently, Jensen's inequality for the strictly convex function $\exp(-\theta\Delta)$).

These are population (limiting) values. Under random times the observed sequence is a stationary, ergodic random-coefficient AR(1) process, $Y_{i+1} = a_i Y_i + e_i$ with $a_i = \exp(-\theta\Delta_i)$, whose autocorrelation function is exactly $\rho_P^{\,k}$; the sample lag-1 autocorrelation therefore converges to $\lambda/(\lambda+\theta)$ as the number of observations grows. In windows as short as those used here ($n = 60$ in the checks below) the sample autocorrelation is biased downward under both designs.

Consequences:

(a) Surrogate specification. Because the randomly timed sequence has autocorrelation function $\rho_P^{\,k}$, it is second-order identical to an AR(1) series with coefficient $\rho_P$. An AR(1) surrogate whose coefficient is fitted to the observed lag-1 autocorrelation, as in the timestamp-blind test used here, therefore reproduces the second-order structure but not the higher-order dependence: given the gaps, the series is conditionally heteroscedastic, so window statistics of the observed series vary differently from those of the AR(1) surrogate (for the window variance, the relevant quantity is $s = \rho_P^{2}$ for the surrogate versus $s_P = \lambda/(\lambda+2\theta)$ for the observed series; Result 2). The reference distribution of the timestamp-blind test therefore need not match the data-generating timing; whether the resulting test is conservative or anti-conservative in a given design is an empirical question, answered here by the calibration blocks.

(b) Compression of the response to slowing. Across a decline in $\theta$ from $\theta_1$ to $\theta_2 < \theta_1$, observation-order autocorrelation changes by $\exp(-\theta_2/\lambda) - \exp(-\theta_1/\lambda)$ under even timing and by $\lambda/(\lambda+\theta_2) - \lambda/(\lambda+\theta_1)$ under random timing. The derivative of $\rho$ with respect to $\theta$ is smaller in magnitude under random timing whenever $(1 + x)^2 < \exp(x)$, that is, for $\theta/\lambda$ below about 2.51; above that value the ordering reverses. The simulated designs lie in the compressed range ($\theta \le 1.2$ and $\lambda \ge 0.6$, so $\theta/\lambda \le 2$). For example, with $\lambda = 0.6$ and $\theta$ falling from 1.2 to 0.05, the change is $0.920 - 0.135 = 0.785$ under even timing and $0.923 - 0.333 = 0.590$ under random timing.

(c) Extra dispersion. Under random timing the conditional correlation of a consecutive pair, given its gap, varies from pair to pair, with variance $\lambda/(\lambda+2\theta) - [\lambda/(\lambda+\theta)]^2 > 0$; no such variation exists under even timing.

# Result 2: effective number of observations for the window variance

For a Gaussian process whose mean is known to be 0, the estimator $S = n^{-1}\sum_i X_i^2$ (the mean squared observation, without subtracting a sample mean) satisfies, by Isserlis' theorem,
$$\operatorname{Var}(S \mid \text{times}) = \frac{2V^2}{n^2}\sum_i\sum_j \operatorname{Corr}(X_i, X_j)^2 .$$
Define $n_{\mathrm{eff}} = 2V^2/\operatorname{Var}(S)$, so that $n_{\mathrm{eff}} = n$ for independent observations.

Even times: $\operatorname{Corr}(X_i, X_{i+k})^2 = s_E^{\,k}$ with $s_E = \exp(-2\theta/\lambda)$.

Random times: $t_{i+k} - t_i$ is a sum of $k$ independent Exponential($\lambda$) gaps (a Gamma($k$, $\lambda$) variable), so $E[\exp\{-2\theta(t_{i+k} - t_i)\}] = s_P^{\,k}$ with $s_P = \lambda/(\lambda+2\theta)$. Because $E[S \mid \text{times}] = V$ for every set of times, the law of total variance gives $\operatorname{Var}(S) = E[\operatorname{Var}(S \mid \text{times})]$, and the expectation passes through the sum term by term.

Hence, for either design,
$$n_{\mathrm{eff}} = \frac{n^2}{n + 2\sum_{k=1}^{n-1}(n-k)\,s^k} \approx n\,\frac{1-s}{1+s} \quad \text{for large } n,$$
with $s = s_E$ or $s_P$. Since $s_P > s_E$ (the inequality of Result 1 with $x = 2\theta/\lambda$), random timing gives fewer effective observations for $S$ at the same $n$ and $\lambda$. The large-$n$ approximation is loose when $n_{\mathrm{eff}}$ is small, so the finite-$n$ formula is used for numerical values. The same argument for the window mean, with $n_{\mathrm{eff}}$ defined as $V/\operatorname{Var}(\text{mean})$, gives $s = \rho_E$ or $\rho_P$ in place of $s_E$ or $s_P$.

# Result 3: the best achievable precision when the decay rate must be estimated

Result 2 concerns one estimator, $S$, which ignores the timestamps. If $\theta$ were known and the times used, the maximum-likelihood estimator of $V$ would have variance $2V^2/n$ under any timing, so there would be no loss. In practice $\theta$ must be estimated from the same window. The Cramér–Rao bound for $V$ with $\theta$ unknown and the times known, averaged over 400 random-time draws ($n = 60$), gives the following effective numbers of observations. These are information-based effective sample sizes for estimating the window variance ($2V^2$ divided by the variance bound, or by the variance of $S$), not detection rates:

| $(\lambda, \theta)$ | Even: Fisher-information $n_{\mathrm{eff}}$ (bound) | Random: Fisher-information $n_{\mathrm{eff}}$ (bound) | Random: $n_{\mathrm{eff}}$ of $S$ |
|---|---|---|---|
| (0.6, 1.2) | 57.9 | 44.8 | 40.3 |
| (0.6, 0.2) | 20.0 | 18.0 | 15.5 |
| (0.3, 0.2) | 35.4 | 28.5 | 24.4 |

Under even timing the bound equals $n(1 - s_E)/(1 + s_E)$, so $S$ is efficient there. Under random timing the bound is lower than under even timing, so clustered sampling reduces the best achievable precision even for methods that use the timestamps, but timing-aware estimation recovers part of the loss that $S$ shows.

# Numerical check

Monte Carlo with $n = 60$ per window, seed 20261101, 20,000 replicates per setting (primary check), confirmed independently with 40,000 replicates (seed 12345, independent code).

| $\lambda$ | $\theta$ | Even $\rho$: MC / theory | Random $\rho$: MC / theory | Even $n_{\mathrm{eff}}$: MC / finite-$n$ formula | Random $n_{\mathrm{eff}}$: MC / finite-$n$ formula |
|---|---|---|---|---|---|
| 0.6 | 1.2 | 0.136 / 0.135 | 0.334 / 0.333 | 57.0 / 57.9 | 40.4 / 40.3 |
| 0.6 | 0.2 | 0.715 / 0.717 | 0.751 / 0.750 | 20.2 / 19.7 | 15.4 / 15.5 |
| 0.6 | 0.05 | 0.918 / 0.920 | 0.921 / 0.923 | 5.5 / 5.5 | 5.2 / 5.2 |
| 1.2 | 0.2 | 0.841 / 0.846 | 0.861 / 0.857 | 10.5 / 10.4 | 8.9 / 9.1 |
| 0.3 | 0.2 | 0.511 / 0.513 | 0.599 / 0.600 | 35.7 / 35.3 | 24.3 / 24.4 |

$\rho$ columns are mean lag-1 product correlations with known mean and variance (population values); sample autocorrelations at $n = 60$ are lower under both designs.

# Scope and what this does not show

- The results concern estimation precision and the population autocorrelation for a stationary OU process within a window, without measurement error, under homogeneous Poisson timing. The simulations add measurement error, a slowly changing $\theta$, Kendall-$\tau$ trend tests across overlapping windows, and escalating rather than homogeneous observation rates. Power of the trend test is not derived. The account therefore explains the direction of the simulated findings and does not predict their size.
- It is consistent with the simulation finding that irregular timing at a matched count costs detection even for a timing-aware test, because the best achievable precision of window statistics is lower under clustered timing (Result 3), and with the finding that most of the loss at a twofold escalation was attributable to irregularity. It is not offered as a demonstrated mechanism for the escalation-specific component.
