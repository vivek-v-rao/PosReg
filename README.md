# posreg

Fit linear models for positive continuous outcomes and compare them by log-likelihood and BIC.

## Motivation

Standard OLS assumes normally distributed errors, which is a poor fit when the dependent variable is positive and right-skewed — as with insurance claims, volatility forecasts, income, or survival times. `xfit_pos.py` fits seven models on the same data and reports a side-by-side comparison, making it easy to see whether heavier-tailed or skewed distributions improve fit.

## Models

| Model | Distribution of Y | Extra parameters |
|---|---|---|
| OLS | Normal, constant variance | — |
| WLS | Normal, variance ∝ fitted value | — |
| Gamma | Gamma GLM, identity link | — |
| LN-linear | Lognormal, linear mean E[Y]=Xβ | — |
| LN-logistic | Log-logistic, linear mean E[Y]=Xβ | — |
| LN-HS | Log-hyperbolic-secant, linear mean E[Y]=Xβ | — |
| LN-skewnorm | Log-skew-normal, linear mean E[Y]=Xβ | α (log-space skewness) |

LN-logistic and LN-HS are heavier-tailed than lognormal. LN-skewnorm adds one shape parameter to capture asymmetry in log space; its BIC count reflects the extra parameter.

All models with "linear mean" share the same parameterization: coefficients β define E[Y]=Xβ directly, so they are comparable across models.

## Output

For each target and power transform:

- **Coefficients table** — one row per model, with spread parameter (σ, k, s, or ω) and skewness shape α for LN-skewnorm
- **Model comparison table** — log-likelihood, BIC, R², Δlog-lik vs OLS, BIC rank
- **Skewness diagnostic** — empirical OLS residual skewness vs lognormal-implied skewness, plus LN-skewnorm α (only when target is positive)

## Usage

```
python xfit_pos.py --file DATA --target COL [COL ...] --predictors COL [COL ...]
                   [--no-intercept]
                   [--filter COL=VAL [COL=VAL ...]]
                   [--transform-powers P [P ...]]
```

### Arguments

| Argument | Description |
|---|---|
| `--file` | Input CSV or Parquet file |
| `--target` | Dependent variable column(s); each is fitted separately |
| `--predictors` | Predictor column names |
| `--no-intercept` | Omit the intercept term |
| `--filter COL=VAL` | Keep only rows where COL equals VAL (repeatable) |
| `--transform-powers P ...` | Integer power(s) applied to target and all predictors. `0`=log, `1`=identity (default), `2`=square. Each power produces a separate set of regressions. |

### Examples

Fit claims on age, bmi, and smoker status:

```
python xfit_pos.py --file claims.csv --target claims --predictors age bmi smoker
```

Compare log, identity, and square transforms in one run:

```
python xfit_pos.py --file claims.csv --target claims --predictors age bmi smoker \
    --transform-powers 0 1 2
```

Fit 1-day and 21-day forward volatility for SPY using lagged close-to-close volatility:

```
python xfit_pos.py --file vol_data.csv \
    --target y_1 y_21 \
    --predictors MA1.close-to-close MA5.close-to-close MA20.close-to-close \
    --filter symbol=SPY \
    --transform-powers 0 1
```

## Dependencies

- Python 3.9+
- numpy
- pandas
- scipy
- statsmodels

## Sample data

### claims.csv

Synthetic insurance dataset (n=600) with a gamma-distributed dependent variable.

| Column | Description |
|---|---|
| `claims` | Annual medical claim amount (positive, right-skewed) |
| `age` | Age in years |
| `bmi` | Body mass index |
| `smoker` | Binary indicator (1=smoker, 0=non-smoker) |

### vol_data.csv

Daily volatility forecasting dataset covering five ETFs (GLD, HYG, QQQ, SPY, USO) from 2010-01-05 to 2026-04-15 (20,470 rows).

| Column | Description |
|---|---|
| `date` | Trading date |
| `symbol` | ETF ticker |
| `y_1` | Realized close-to-close volatility (annualized %) over the next 1 trading day |
| `y_5` | Realized volatility over the next 5 trading days |
| `y_21` | Realized volatility over the next 21 trading days |
| `MA1.close-to-close` | 1-day (most recent) close-to-close volatility |
| `MA5.close-to-close` | 5-day moving average of close-to-close volatility |
| `MA20.close-to-close` | 20-day moving average of close-to-close volatility |
| `MA1.neg_ret_cc` | Most recent negative-return indicator (captures asymmetric vol response) |
| `MA5.neg_ret_cc` | 5-day moving average of negative-return indicator |
| `MA1.SPY.close-to-close` | Most recent SPY close-to-close volatility (market predictor) |
| `MA5.SPY.close-to-close` | 5-day SPY volatility moving average |
| `MA20.SPY.close-to-close` | 20-day SPY volatility moving average |

Volatility is measured in annualized percentage points. The dataset is suitable for demonstrating that lognormal and heavier-tailed models outperform OLS for forecasting positive, right-skewed quantities.
