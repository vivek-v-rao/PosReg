"""
fitpos.py  --  Fit linear models for positive continuous outcomes.

Compares five models on the same data:
  OLS         normal errors, constant variance
  WLS         normal errors, variance proportional to fitted value
  Gamma       gamma GLM with identity link, variance proportional to mean^2
  LN-linear   conditionally lognormal, linear mean E[Y]=Xb, variance proportional to mean^2
  LN-skewnorm log-skew-normal, linear mean E[Y]=Xb, one extra shape parameter

Outputs a coefficient table, a model-comparison summary (log-likelihood, BIC,
R^2), and a skewness diagnostic comparing empirical OLS residual skew to the
skew implied by the fitted lognormal.

Usage
-----
  python fitpos.py --file data.csv --target y --predictors x1 x2 x3
  python fitpos.py --file data.parquet --target claims --predictors age tenure region
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, log_ndtr as _log_ndtr
from scipy.stats import skew as _skew
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--file", required=True,
                   help="Input data file (.csv or .parquet)")
    p.add_argument("--target", nargs="+", required=True,
                   help="Dependent variable column(s) (must be positive); fitted separately")
    p.add_argument("--predictors", nargs="+", required=True,
                   help="Predictor column names")
    p.add_argument("--no-intercept", action="store_true",
                   help="Omit intercept (default: include)")
    p.add_argument("--filter", nargs="*", metavar="COL=VAL",
                   help="Row filter(s) applied before fitting, e.g. --filter symbol=SPY")
    p.add_argument("--transform-powers", nargs="+", type=int, default=[1],
                   metavar="P",
                   help="Power(s) applied to target and predictors before fitting. "
                        "0 means log, 1 means identity (default), 2 means square. "
                        "Each power produces a separate set of regressions.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _ols_fit(y, X):
    """OLS. Returns (coef, sigma, loglik, yhat, r2)."""
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    resid = y - yhat
    n = len(y)
    ss_res = float(resid @ resid)
    sigma2 = ss_res / n
    sigma = np.sqrt(sigma2)
    ll = -n / 2 * np.log(2 * np.pi * sigma2) - n / 2
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, sigma, ll, yhat, r2


def _wls_fit(y, X, yhat_ols):
    """WLS with weights 1/yhat_ols. Returns (coef, sigma_w, loglik, yhat, r2)."""
    w = 1.0 / np.clip(yhat_ols, 1e-8, None)
    sw = np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    yhat = X @ coef
    resid = y - yhat
    n = len(y)
    sigma2_w = float((w * resid ** 2).sum()) / n
    sigma_w = np.sqrt(sigma2_w)
    ll = (-n / 2 * np.log(2 * np.pi * sigma2_w)
          + 0.5 * np.log(w).sum()
          - n / 2)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, sigma_w, ll, yhat, r2


def _ln_linear_nll(params, y, X, log_y_sum):
    """NLL for conditionally lognormal model with linear mean E[Y]=Xb."""
    b = params[:-1]
    sigma = np.exp(params[-1])
    mu = X @ b
    if np.any(mu <= 0):
        return 1e15
    n = len(y)
    z = (np.log(y) - np.log(mu) + sigma ** 2 / 2) / sigma
    return (log_y_sum + n * np.log(sigma) + n / 2 * np.log(2 * np.pi)
            + 0.5 * float(z @ z))


def _ln_linear_fit(y, X, coef_ols):
    """Lognormal MLE with linear mean. Returns (coef, sigma, loglik, yhat, r2)."""
    log_y_sum = float(np.log(y).sum())
    b0_init = max(coef_ols[0], 1e-4)
    params0 = np.concatenate([[b0_init], coef_ols[1:], [np.log(0.1)]])
    bounds = [(1e-8, None)] + [(None, None)] * (len(coef_ols) - 1) + [(None, None)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(_ln_linear_nll, params0, args=(y, X, log_y_sum),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-12})
        if not res.success:
            res = minimize(_ln_linear_nll, params0, args=(y, X, log_y_sum),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 5000, "ftol": 1e-14})
    coef = res.x[:-1]
    sigma = np.exp(res.x[-1])
    ll = -res.fun
    yhat = X @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, sigma, ll, yhat, r2


def _gamma_fit(y, X):
    """Gamma GLM with identity link. Returns (coef, shape_k, loglik, yhat, r2)."""
    family = sm.families.Gamma(link=sm.families.links.Identity())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            glm_res = sm.GLM(y, X, family=family).fit(maxiter=200, tol=1e-8)
        except Exception:
            return None, np.nan, np.nan, None, np.nan
    coef = glm_res.params
    yhat = glm_res.fittedvalues
    if np.any(yhat <= 0):
        return coef, np.nan, np.nan, yhat, np.nan
    k = 1.0 / glm_res.scale
    n = len(y)
    ll = float((k - 1) * np.log(y).sum()
               - k * (y / yhat).sum()
               + n * k * np.log(k)
               - n * gammaln(k)
               - k * np.log(yhat).sum())
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, k, ll, yhat, r2


_LN_HS_MAX_OMEGA = np.pi / 2 - 1e-4   # omega must be < pi/2 for finite E[Y]


def _ln_hs_nll(params, y, X, log_y_sum):
    """NLL for log-hyperbolic-secant with linear mean E[Y]=Xb.

    log(Y_i) ~ HS(xi_i, omega)
    MGF of HS(xi, omega) at t=1: exp(xi) * sec(omega)  [valid for |omega| < pi/2]
    Mean constraint: xi_i = log(mu_i) + log(cos(omega))
    PDF: f(x) = 1 / (2*omega * cosh(pi*(x-xi)/(2*omega)))
    log f_Y(y) = -log(2*omega) - log(cosh(z)) - log(y)
    where z = pi*(log(y) - xi) / (2*omega)
    """
    b = params[:-1]
    omega = np.exp(params[-1])
    if omega >= _LN_HS_MAX_OMEGA:
        return 1e15
    mu = X @ b
    if np.any(mu <= 0):
        return 1e15
    xi = np.log(mu) + np.log(np.cos(omega))
    z = np.pi * (np.log(y) - xi) / (2 * omega)
    log_cosh_z = np.logaddexp(z, -z) - np.log(2)   # numerically stable
    n = len(y)
    return log_y_sum + n * np.log(2 * omega) + float(log_cosh_z.sum())


def _ln_hs_fit(y, X, ln_coef, ln_sigma):
    """Log-hyperbolic-secant MLE with linear mean. Returns (coef, omega, loglik, yhat, r2)."""
    log_y_sum = float(np.log(y).sum())
    b0_init = max(ln_coef[0], 1e-4)
    omega_init = min(ln_sigma * 0.8, _LN_HS_MAX_OMEGA * 0.9)
    params0 = np.concatenate([[b0_init], ln_coef[1:], [np.log(omega_init)]])
    bounds = ([(1e-8, None)]
              + [(None, None)] * (len(ln_coef) - 1)
              + [(None, np.log(_LN_HS_MAX_OMEGA))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(_ln_hs_nll, params0, args=(y, X, log_y_sum),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-12})
        if not res.success:
            res = minimize(_ln_hs_nll, params0, args=(y, X, log_y_sum),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 5000, "ftol": 1e-14})
    coef = res.x[:-1]
    omega = np.exp(res.x[-1])
    ll = -res.fun
    yhat = X @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, omega, ll, yhat, r2


def _ln_logistic_nll(params, y, X, log_y_sum):
    """NLL for log-logistic with linear mean E[Y]=Xb.

    log(Y_i) ~ Logistic(xi_i, s),  s < 1 required for finite E[Y].
    MGF of Logistic(xi, s) at t=1: exp(xi) * s*pi/sin(s*pi)
    Mean constraint: xi_i = log(mu_i) + log(sin(s*pi)) - log(s*pi)
    log f_Y(y) = -log(s) - z - 2*log(1+exp(-z)) - log(y)
    where z = (log(y) - xi) / s
    """
    b = params[:-1]
    s = np.exp(params[-1])
    if s >= 1.0:
        return 1e15
    mu = X @ b
    if np.any(mu <= 0):
        return 1e15
    xi = np.log(mu) + np.log(np.sin(s * np.pi)) - np.log(s * np.pi)
    z = (np.log(y) - xi) / s
    log1p_exp_neg_z = np.logaddexp(0.0, -z)   # log(1 + exp(-z)), stable
    n = len(y)
    return (log_y_sum + n * np.log(s)
            + float(z.sum())
            + 2 * float(log1p_exp_neg_z.sum()))


def _ln_logistic_fit(y, X, ln_coef, ln_sigma):
    """Log-logistic MLE with linear mean. Returns (coef, s, loglik, yhat, r2)."""
    log_y_sum = float(np.log(y).sum())
    b0_init = max(ln_coef[0], 1e-4)
    # Match log-space variance: logistic has var = pi^2*s^2/3, normal has sigma^2
    s_init = min(ln_sigma * np.sqrt(3) / np.pi, 0.9)
    params0 = np.concatenate([[b0_init], ln_coef[1:], [np.log(s_init)]])
    bounds = ([(1e-8, None)]
              + [(None, None)] * (len(ln_coef) - 1)
              + [(None, -1e-6)])   # log_s < 0 enforces s < 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(_ln_logistic_nll, params0, args=(y, X, log_y_sum),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-12})
        if not res.success:
            res = minimize(_ln_logistic_nll, params0, args=(y, X, log_y_sum),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 5000, "ftol": 1e-14})
    coef = res.x[:-1]
    s = np.exp(res.x[-1])
    ll = -res.fun
    yhat = X @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, s, ll, yhat, r2


def _ln_skewnorm_nll(params, y, X, log_y_sum):
    """NLL for log-skew-normal with linear mean E[Y]=Xb.

    log(Y_i) ~ SN(xi_i, omega^2, alpha)
    Mean constraint: xi_i = log(mu_i) - omega^2/2 - log(2) - log_ndtr(omega*delta)
    where delta = alpha / sqrt(1 + alpha^2).
    """
    b = params[:-2]
    omega = np.exp(params[-2])
    alpha = params[-1]
    delta = alpha / np.sqrt(1.0 + alpha ** 2)
    mu = X @ b
    if np.any(mu <= 0):
        return 1e15
    xi = np.log(mu) - omega ** 2 / 2.0 - np.log(2.0) - _log_ndtr(omega * delta)
    z = (np.log(y) - xi) / omega
    n = len(y)
    return (log_y_sum + n * np.log(omega) + n / 2.0 * np.log(2.0 * np.pi)
            + 0.5 * float(z @ z)
            - float(_log_ndtr(alpha * z).sum())
            - n * np.log(2.0))


def _ln_skewnorm_fit(y, X, ln_coef, ln_sigma):
    """Log-skew-normal MLE with linear mean. Returns (coef, omega, alpha, loglik, yhat, r2).

    The NLL gradient w.r.t. alpha is identically zero at alpha=0 (Azzalini
    parameterization), so we initialize from the log-space residual skewness
    and also try the negated value, keeping the lower NLL.
    """
    log_y_sum = float(np.log(y).sum())
    b0_init = max(ln_coef[0], 1e-4)
    bounds = ([(1e-8, None)]
              + [(None, None)] * (len(ln_coef) - 1)
              + [(None, None), (None, None)])

    ln_mu = np.clip(X @ ln_coef, 1e-8, None)
    log_z = (np.log(y) - np.log(ln_mu) + ln_sigma ** 2 / 2) / ln_sigma
    log_resid_skew = float(_skew(log_z, bias=False))
    _C = (4 - np.pi) / 2 * (2 / np.pi) ** 1.5
    alpha_init = float(np.clip(
        np.sign(log_resid_skew) * abs(log_resid_skew / _C) ** (1 / 3),
        -10, 10))

    best = None
    for a0 in [alpha_init, -alpha_init]:
        params0 = np.concatenate([[b0_init], ln_coef[1:], [np.log(ln_sigma)], [a0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(_ln_skewnorm_nll, params0, args=(y, X, log_y_sum),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 3000, "ftol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res

    coef = best.x[:-2]
    omega = np.exp(best.x[-2])
    alpha = best.x[-1]
    ll = -best.fun
    yhat = X @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, omega, alpha, ll, yhat, r2


def _ln_implied_skewness(sigma):
    """Skewness of a lognormal Y around its mean, given log-scale sigma."""
    s2 = sigma ** 2
    return (np.exp(s2) + 2) * np.sqrt(np.exp(s2) - 1)

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_section(title):
    print(f"\n=== {title} ===")


def _fmt_float(v, fmt="{:.4f}"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    return fmt.format(v)

# ---------------------------------------------------------------------------
# Per-target fitting and output
# ---------------------------------------------------------------------------

def _power_label(name, p):
    """Human-readable label for a power-transformed variable."""
    if p == 0:
        return f"log({name})"
    if p == 1:
        return name
    ip = int(p) if p == int(p) else p
    return f"{name}^{ip}"


def _fit_one_target(df, target, predictors, use_intercept, power=1.0):
    sub = df[[target] + predictors].dropna()
    sub = sub[sub[target] > 0]
    min_rows = len(predictors) + (1 if use_intercept else 0) + 5
    if len(sub) < min_rows:
        print(f"\n  [skipping {target}: only {len(sub)} valid rows]")
        return

    y_raw = sub[target].to_numpy(dtype=float)
    Xraw  = sub[predictors].to_numpy(dtype=float)

    # --- transform target ---
    y = np.log(y_raw) if power == 0 else y_raw ** power
    target_label = _power_label(target, power)

    # --- transform predictors ---
    Xcols, col_labels = [], []
    for j, pred in enumerate(predictors):
        col = Xraw[:, j]
        if power == 0:
            if np.all(col > 0):
                Xcols.append(np.log(col))
                col_labels.append(f"log({pred})")
            else:
                print(f"  Note: '{pred}' has non-positive values; left untransformed.")
                Xcols.append(col)
                col_labels.append(pred)
        else:  # integer power: x^p is always real for any real x
            Xcols.append(col ** power)
            col_labels.append(_power_label(pred, power))

    Xraw_t = np.column_stack(Xcols)
    X = np.column_stack([np.ones(len(y)), Xraw_t]) if use_intercept else Xraw_t
    coef_labels = (["intercept"] if use_intercept else []) + col_labels

    n = len(y)
    n_coef = X.shape[1]
    k_base = n_coef + 1

    # positive-distribution models require transformed target to be all positive
    y_positive = bool(np.all(y > 0))

    print(f"\n{'='*60}")
    print(f"Target: {target_label}  |  n={n}")
    if not y_positive:
        print("  (transformed target has non-positive values; "
              "only OLS and WLS will be fitted)")
    print(f"{'='*60}")

    rows = []

    ols_coef, ols_sigma, ols_ll, ols_yhat, ols_r2 = _ols_fit(y, X)
    rows.append(dict(model="OLS", coef=ols_coef, spread=ols_sigma,
                     alpha=np.nan, ll=ols_ll,
                     bic=k_base * np.log(n) - 2 * ols_ll, r2=ols_r2))

    wls_coef, wls_sigma, wls_ll, wls_yhat, wls_r2 = _wls_fit(y, X, ols_yhat)
    rows.append(dict(model="WLS", coef=wls_coef, spread=wls_sigma,
                     alpha=np.nan, ll=wls_ll,
                     bic=k_base * np.log(n) - 2 * wls_ll, r2=wls_r2))

    if y_positive:
        gam_coef, gam_k, gam_ll, gam_yhat, gam_r2 = _gamma_fit(y, X)
        if gam_coef is not None and np.isfinite(gam_ll):
            rows.append(dict(model="Gamma", coef=gam_coef, spread=gam_k,
                             alpha=np.nan, ll=gam_ll,
                             bic=k_base * np.log(n) - 2 * gam_ll, r2=gam_r2))
        else:
            print("  Warning: Gamma GLM did not converge.")

        ln_coef, ln_sigma, ln_ll, ln_yhat, ln_r2 = _ln_linear_fit(y, X, ols_coef)
        rows.append(dict(model="LN-linear", coef=ln_coef, spread=ln_sigma,
                         alpha=np.nan, ll=ln_ll,
                         bic=k_base * np.log(n) - 2 * ln_ll, r2=ln_r2))

        llog_coef, llog_s, llog_ll, llog_yhat, llog_r2 = _ln_logistic_fit(y, X, ln_coef, ln_sigma)
        rows.append(dict(model="LN-logistic", coef=llog_coef, spread=llog_s,
                         alpha=np.nan, ll=llog_ll,
                         bic=k_base * np.log(n) - 2 * llog_ll, r2=llog_r2))

        lhs_coef, lhs_omega, lhs_ll, lhs_yhat, lhs_r2 = _ln_hs_fit(y, X, ln_coef, ln_sigma)
        rows.append(dict(model="LN-HS", coef=lhs_coef, spread=lhs_omega,
                         alpha=np.nan, ll=lhs_ll,
                         bic=k_base * np.log(n) - 2 * lhs_ll, r2=lhs_r2))

        lsn_coef, lsn_omega, lsn_alpha, lsn_ll, lsn_yhat, lsn_r2 = \
            _ln_skewnorm_fit(y, X, ln_coef, ln_sigma)
        lsn_k = k_base + 1
        rows.append(dict(model="LN-skewnorm", coef=lsn_coef, spread=lsn_omega,
                         alpha=lsn_alpha, ll=lsn_ll,
                         bic=lsn_k * np.log(n) - 2 * lsn_ll, r2=lsn_r2))

    # --- Coefficients ---
    _print_section("Coefficients")
    print("(spread: sigma=OLS/WLS/LN-linear, k=Gamma, s=LN-logistic, omega=LN-HS/LN-skewnorm)")
    print("(alpha: log-space skewness for LN-skewnorm only)\n")
    col_w = max(12, max(len(lb) for lb in coef_labels) + 2)
    header = f"{'model':<14}" + "".join(f"{lb:>{col_w}}" for lb in coef_labels)
    header += f"{'spread':>10}{'alpha':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['model']:<14}"
        for c in row["coef"]:
            line += f"{c:>{col_w}.4f}"
        line += f"{row['spread']:>10.4f}"
        line += f"{row['alpha']:>10.4f}" if np.isfinite(row["alpha"]) else f"{'':>10}"
        print(line)

    # --- Model comparison ---
    _print_section("Model comparison")
    ols_ll_val = next(r["ll"] for r in rows if r["model"] == "OLS")
    bic_rank = pd.Series([r["bic"] for r in rows]).rank().astype(int).tolist()
    comp_header = f"{'model':<14}{'loglik':>12}{'bic':>12}{'r2':>10}{'dloglik':>10}{'bic_rank':>10}"
    print(comp_header)
    print("-" * len(comp_header))
    for row, rank in zip(rows, bic_rank):
        print(f"{row['model']:<14}"
              f"{row['ll']:>12.1f}"
              f"{row['bic']:>12.1f}"
              f"{row['r2']:>10.4f}"
              f"{row['ll'] - ols_ll_val:>+10.1f}"
              f"{rank:>10}")

    # --- Skewness diagnostic (only when positive-distribution models were fitted) ---
    if y_positive:
        _print_section("Skewness diagnostic")
        ols_resid = y - ols_yhat
        emp_skew = float(_skew(ols_resid, bias=False))
        model_skew = _ln_implied_skewness(ln_sigma)
        ratio = emp_skew / model_skew if model_skew > 0 else np.nan
        print(f"  empirical skew (OLS resid) : {emp_skew:.3f}")
        print(f"  LN-linear implied skew     : {model_skew:.3f}")
        print(f"  ratio                      : {ratio:.3f}")
        print(f"  LN-skewnorm alpha          : {lsn_alpha:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    use_intercept = not args.no_intercept

    # --- load data ---
    if args.file.endswith(".parquet"):
        df = pd.read_parquet(args.file)
    else:
        df = pd.read_csv(args.file)

    missing = [c for c in args.target + args.predictors if c not in df.columns]
    if missing:
        sys.exit(f"Columns not found in {args.file}: {missing}")

    # apply --filter COL=VAL conditions
    if args.filter:
        for filt in args.filter:
            if "=" not in filt:
                sys.exit(f"--filter entries must be COL=VAL, got: {filt!r}")
            col, val = filt.split("=", 1)
            if col not in df.columns:
                sys.exit(f"Filter column {col!r} not found in {args.file}")
            try:
                df = df[df[col] == type(df[col].iloc[0])(val)]
            except (ValueError, TypeError):
                df = df[df[col].astype(str) == val]

    filter_str = "  |  filter: " + ", ".join(args.filter) if args.filter else ""
    print(f"\nData: {args.file}{filter_str}")
    print(f"Predictors: {args.predictors}"
          + ("  |  intercept" if use_intercept else "  |  no intercept"))

    for target in args.target:
        for p in args.transform_powers:
            _fit_one_target(df, target, args.predictors, use_intercept, power=p)


if __name__ == "__main__":
    main()
