"""
VAR / SVAR / Granger Causality / Cointegration Analysis
US 10-Year Treasury Yield vs. German 10-Year Bund Yield
=========================================================

This script follows the standard time-series course sequence for multivariate
shock analysis:

    1. Data preparation & stationarity testing        (ADF, KPSS)
    2. Reduced-form VAR estimation & lag-order selection
    3. Granger causality testing
    4. Cointegration testing (Johansen trace / max-eigenvalue) and VECM
    5. Structural identification via Cholesky (recursive) decomposition
    6. Structural impulse response functions (IRFs)
    7. Forecast error variance decomposition (FEVD)

Variables (11, block-recursive ordering; same set used in the prior
us_spillover_vs_ecb analysis in this project):

  US block  (ordered first -> contemporaneously exogenous to the DE/ECB block)
    us_ip        INDPRO            US industrial production (real activity)
    us_cpi       CPIAUCSL          US CPI (prices)
    fed_funds    FEDFUNDS          Federal funds rate (policy instrument)
    fed_bs       WALCL (log)       Fed balance sheet (unconventional policy)
    vix          VIXCLS            CBOE VIX (global risk sentiment)
    us_10y       GS10              US 10Y Treasury yield

  DE/ECB block (ordered second -> may react contemporaneously to the US block)
    de_ip        DEUPROINDMISMEI   Germany industrial production
    de_cpi       CP0000DEM086NEST  Germany CPI (HICP)
    ecb_rate     ECBDFR            ECB deposit facility rate (policy instrument)
    ecb_bs       ECBASSETSW (log)  ECB balance sheet (unconventional policy)
    de_10y       IRLTLT01DEM156N   German 10Y Bund yield

Ordering rationale within each block follows the standard recursive monetary
VAR convention: slow-moving real/price variables -> policy instrument ->
fast-moving, market-priced variables. The US block is ordered ahead of the
DE/ECB block on the "small open financial area" assumption that US Treasury
and dollar funding markets are large relative to any single Euro Area
sovereign bond market (Rey, 2013; Miranda-Agrippino & Rey, 2020).

All series are monthly, sourced from FRED (cached under data/fred_data/).
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Academic (journal-style) plotting defaults: serif fonts to match the LaTeX
# report body, muted/desaturated palette, minimal chart junk (no top/right
# spines, light horizontal gridlines only).
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.edgecolor": "0.3",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "xtick.color": "0.3",
    "ytick.color": "0.3",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "fred_data")
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

SERIES = {
    "us_ip": ("INDPRO.csv", False),
    "us_cpi": ("CPIAUCSL.csv", False),
    "fed_funds": ("FEDFUNDS.csv", False),
    "fed_bs": ("WALCL.csv", True),      # log level
    "vix": ("VIXCLS.csv", False),
    "us_10y": ("GS10.csv", False),
    "de_ip": ("DEUPROINDMISMEI.csv", False),
    "de_cpi": ("CP0000DEM086NEST.csv", False),
    "ecb_rate": ("ECBDFR.csv", False),
    "ecb_bs": ("ECBASSETSW.csv", True),  # log level
    "de_10y": ("IRLTLT01DEM156N.csv", False),
}

US_BLOCK = ["us_ip", "us_cpi", "fed_funds", "fed_bs", "vix", "us_10y"]
DE_BLOCK = ["de_ip", "de_cpi", "ecb_rate", "ecb_bs", "de_10y"]
ORDER = US_BLOCK + DE_BLOCK

# Human-readable series names for plot titles/labels (never show FRED codes
# or internal variable names in a figure).
LABELS = {
    "us_ip": "US Industrial Production",
    "us_cpi": "US CPI",
    "fed_funds": "Fed Funds Rate",
    "fed_bs": "Fed Balance Sheet (log)",
    "vix": "VIX",
    "us_10y": "US 10Y Treasury Yield",
    "de_ip": "German Industrial Production",
    "de_cpi": "German CPI",
    "ecb_rate": "ECB Deposit Rate",
    "ecb_bs": "ECB Balance Sheet (log)",
    "de_10y": "German 10Y Bund Yield",
}

FIXED_LAG = 2          # deliberately conservative given ~250 monthly obs, 11 vars
IRF_HORIZON = 24        # months
SIG = 0.05

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def load_series(name, fname, log_transform):
    df = pd.read_csv(os.path.join(DATA_DIR, fname), parse_dates=["date"])
    df = df.rename(columns={"value": name}).set_index("date")
    df.index = df.index.to_period("M").to_timestamp("M")
    if log_transform:
        df[name] = np.log(df[name])
    return df[[name]]


def build_panel():
    frames = [load_series(k, f, lt) for k, (f, lt) in SERIES.items()]
    panel = pd.concat(frames, axis=1, join="inner").sort_index()
    panel = panel.dropna()
    panel = panel[ORDER]
    return panel


panel_levels = build_panel()
panel_levels.to_csv(os.path.join(RES_DIR, "panel_levels.csv"))
print(f"Panel sample: {panel_levels.index.min().date()} to "
      f"{panel_levels.index.max().date()}  (n={len(panel_levels)})")

# ---------------------------------------------------------------------------
# Figure: raw series
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(4, 3, figsize=(16, 12))
for ax, col in zip(axes.flat, ORDER):
    ax.plot(panel_levels.index, panel_levels[col], lw=1, color="#1f3b57")
    ax.set_title(LABELS[col])
    ax.tick_params(axis="x", rotation=45)
axes.flat[-1].axis("off")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "raw_series.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(panel_levels.index, panel_levels["us_10y"], label=LABELS["us_10y"], color="#1f3b57")
ax.plot(panel_levels.index, panel_levels["de_10y"], label=LABELS["de_10y"], color="#b5651d")
ax.set_ylabel("Percent")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "us_de_10y_levels.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Stationarity testing (ADF, KPSS) — levels and first differences
# ---------------------------------------------------------------------------


def adf_test(series):
    stat, pval, *_ = adfuller(series.dropna(), autolag="AIC")
    return stat, pval


def kpss_test(series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, *_ = kpss(series.dropna(), regression="c", nlags="auto")
    return stat, pval


rows = []
for col in ORDER:
    lvl = panel_levels[col]
    dif = panel_levels[col].diff().dropna()
    adf_l, adfp_l = adf_test(lvl)
    kp_l, kpp_l = kpss_test(lvl)
    adf_d, adfp_d = adf_test(dif)
    kp_d, kpp_d = kpss_test(dif)
    rows.append({
        "variable": col,
        "adf_stat_level": adf_l, "adf_pval_level": adfp_l,
        "kpss_stat_level": kp_l, "kpss_pval_level": kpp_l,
        "adf_stat_diff": adf_d, "adf_pval_diff": adfp_d,
        "kpss_stat_diff": kp_d, "kpss_pval_diff": kpp_d,
        "likely_I1": (adfp_l > SIG) and (adfp_d <= SIG),
    })
stationarity = pd.DataFrame(rows).set_index("variable")
stationarity.to_csv(os.path.join(RES_DIR, "stationarity_report.csv"))
print("\n=== Stationarity (ADF/KPSS) ===")
print(stationarity[["adf_pval_level", "kpss_pval_level", "adf_pval_diff", "likely_I1"]])

panel_diff = panel_levels.diff().dropna()

# ---------------------------------------------------------------------------
# 3. Reduced-form VAR: lag-order selection + estimation (on differences)
# ---------------------------------------------------------------------------

var_model = VAR(panel_diff)
lag_order_results = var_model.select_order(maxlags=8)
with open(os.path.join(RES_DIR, "lag_order_selection.txt"), "w") as f:
    f.write(str(lag_order_results.summary()))
print("\n=== Lag order selection (info criteria) ===")
print(lag_order_results.summary())

var_fitted = var_model.fit(FIXED_LAG)
with open(os.path.join(RES_DIR, "var_summary.txt"), "w") as f:
    f.write(f"Fixed lag used: {FIXED_LAG} (see lag_order_selection.txt for AIC/BIC/HQIC picks)\n\n")
    f.write(str(var_fitted.summary()))
print(f"\nVAR estimated with fixed lag = {FIXED_LAG} "
      f"(AIC/BIC-selected lags reported separately; a low fixed lag is used "
      f"deliberately to avoid overfitting an 11-variable system on "
      f"~{len(panel_diff)} monthly observations).")

# Portmanteau / residual whiteness check (diagnostic only)
try:
    port = var_fitted.test_whiteness(nlags=12)
    with open(os.path.join(RES_DIR, "var_whiteness_test.txt"), "w") as f:
        f.write(str(port.summary()))
except Exception as e:
    print("Whiteness test failed:", e)

# ---------------------------------------------------------------------------
# 4. Granger causality
# ---------------------------------------------------------------------------

pairs_of_interest = [
    ("us_10y", "de_10y"), ("de_10y", "us_10y"),
    ("fed_funds", "us_10y"), ("ecb_rate", "de_10y"),
    ("fed_funds", "de_10y"), ("ecb_rate", "us_10y"),
    ("vix", "us_10y"), ("vix", "de_10y"),
    ("us_cpi", "us_10y"), ("de_cpi", "de_10y"),
    ("fed_bs", "de_10y"), ("ecb_bs", "de_10y"),
]

gc_rows = []
for cause, effect in pairs_of_interest:
    sub = panel_diff[[effect, cause]]
    try:
        res = grangercausalitytests(sub, maxlag=FIXED_LAG, verbose=False)
        pval = res[FIXED_LAG][0]["ssr_ftest"][1]
    except Exception:
        pval = np.nan
    gc_rows.append({"cause": cause, "effect": effect, "lag": FIXED_LAG, "p_value": pval,
                     "significant_5pct": pval < SIG if pd.notna(pval) else False})

granger_df = pd.DataFrame(gc_rows)
granger_df.to_csv(os.path.join(RES_DIR, "granger_causality_pvalues.csv"), index=False)
print("\n=== Granger causality (selected pairs, VAR-in-differences) ===")
print(granger_df)

# Full pairwise matrix among the two yields and both policy rates / VIX
full_vars = ["us_10y", "de_10y", "fed_funds", "ecb_rate", "vix"]
mat = pd.DataFrame(index=full_vars, columns=full_vars, dtype=float)
for cause in full_vars:
    for effect in full_vars:
        if cause == effect:
            continue
        sub = panel_diff[[effect, cause]]
        try:
            res = grangercausalitytests(sub, maxlag=FIXED_LAG, verbose=False)
            mat.loc[cause, effect] = res[FIXED_LAG][0]["ssr_ftest"][1]
        except Exception:
            mat.loc[cause, effect] = np.nan
mat.to_csv(os.path.join(RES_DIR, "granger_causality_matrix_core5.csv"))

# ---------------------------------------------------------------------------
# 5. Cointegration testing (Johansen) and VECM
# ---------------------------------------------------------------------------

# Focus Johansen test on a tractable subsystem to keep asymptotics sane
# (yields + policy rates + balance sheets — the theoretically cointegrated core)
coint_vars = ["us_10y", "de_10y", "fed_funds", "ecb_rate", "fed_bs", "ecb_bs"]
coint_data = panel_levels[coint_vars]

johansen = coint_johansen(coint_data, det_order=0, k_ar_diff=FIXED_LAG)

trace_stat = johansen.lr1
trace_crit = johansen.cvt[:, 1]  # 5% critical values
maxeig_stat = johansen.lr2
maxeig_crit = johansen.cvm[:, 1]

coint_report = pd.DataFrame({
    "rank_null_h0_r<=": range(len(trace_stat)),
    "trace_stat": trace_stat,
    "trace_crit_5pct": trace_crit,
    "trace_reject_5pct": trace_stat > trace_crit,
    "maxeig_stat": maxeig_stat,
    "maxeig_crit_5pct": maxeig_crit,
    "maxeig_reject_5pct": maxeig_stat > maxeig_crit,
})
coint_report.to_csv(os.path.join(RES_DIR, "johansen_cointegration_test.csv"), index=False)
print("\n=== Johansen cointegration test (trace & max-eigenvalue, subsystem) ===")
print(coint_report)

coint_rank = int(coint_report["trace_reject_5pct"].sum())
coint_rank = max(coint_rank, 1)  # at least 1 for VECM estimation below
print(f"\nSelected cointegration rank (trace test, 5%): {coint_rank}")

vecm_model = VECM(coint_data, k_ar_diff=FIXED_LAG, coint_rank=coint_rank, deterministic="co")
vecm_fitted = vecm_model.fit()

with open(os.path.join(RES_DIR, "vecm_summary.txt"), "w") as f:
    f.write(str(vecm_fitted.summary()))

beta_df = pd.DataFrame(vecm_fitted.beta, index=coint_vars,
                        columns=[f"coint_vec_{i+1}" for i in range(coint_rank)])
alpha_df = pd.DataFrame(vecm_fitted.alpha, index=coint_vars,
                         columns=[f"coint_vec_{i+1}" for i in range(coint_rank)])
beta_df.to_csv(os.path.join(RES_DIR, "vecm_beta_cointegrating_vectors.csv"))
alpha_df.to_csv(os.path.join(RES_DIR, "vecm_alpha_adjustment_speeds.csv"))
print("\n=== VECM cointegrating vector(s) (beta) ===")
print(beta_df)
print("\n=== VECM adjustment speeds (alpha) ===")
print(alpha_df)

# ---------------------------------------------------------------------------
# 6. Structural identification: Cholesky (recursive) decomposition
# ---------------------------------------------------------------------------
# Reduced-form VAR-in-differences residual covariance -> lower-triangular
# Cholesky factor P such that P P' = Sigma_u. Column j of the orthogonalized
# IRF traces the effect of a one-std-dev structural shock to variable j,
# under the block-recursive ordering: US block variables cannot react
# contemporaneously to DE/ECB shocks, but DE/ECB variables can react
# contemporaneously to US shocks (US block ordered first).

sigma_u = var_fitted.sigma_u
P = np.linalg.cholesky(sigma_u.values)
chol_df = pd.DataFrame(P, index=ORDER, columns=ORDER)
chol_df.to_csv(os.path.join(RES_DIR, "cholesky_factor.csv"))

irf = var_fitted.irf(IRF_HORIZON)
orth_irfs = irf.orth_irfs  # shape (horizon+1, n_vars_response, n_vars_shock)

# Save full IRF array for reproducibility
np.save(os.path.join(RES_DIR, "orthogonalized_irfs.npy"), orth_irfs)

# ---------------------------------------------------------------------------
# 7. Structural IRFs — key plots
# ---------------------------------------------------------------------------

response_vars = ["us_10y", "de_10y"]
response_labels = {"us_10y": "US 10-year Treasury yield", "de_10y": "German 10-year Bund yield"}
shock_vars_of_interest = ["fed_funds", "ecb_rate", "vix", "us_cpi", "de_cpi", "fed_bs", "ecb_bs"]
horizons = np.arange(IRF_HORIZON + 1)


def _plot_irf_row(rv, shock_list, fname, ncols_wrap=None):
    """One response variable's IRF to each shock in shock_list, one panel per shock."""
    n = len(shock_list)
    ncols = ncols_wrap if ncols_wrap else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    ri = ORDER.index(rv)
    for k, sv in enumerate(shock_list):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        si = ORDER.index(sv)
        ax.plot(horizons, orth_irfs[:, ri, si], color="#1f3b57", lw=1.3)
        ax.axhline(0, color="0.4", lw=0.6, ls="--")
        ax.set_title(LABELS[sv], fontsize=9)
        if c == 0:
            ax.set_ylabel("Response")
        if r == nrows - 1:
            ax.set_xlabel("Horizon (months)")
    # blank any unused trailing panels
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)


# Two separate, uncluttered figures (one per yield) for the key-shock IRFs
_plot_irf_row("us_10y", shock_vars_of_interest, "structural_irf_us10y_key_shocks.png")
_plot_irf_row("de_10y", shock_vars_of_interest, "structural_irf_de10y_key_shocks.png")

# Two separate figures (one per yield) for the response to ALL 11 shocks,
# replacing the illegible 11x11 full grid with two readable panels.
_plot_irf_row("us_10y", ORDER, "structural_irf_us10y_all_shocks.png", ncols_wrap=4)
_plot_irf_row("de_10y", ORDER, "structural_irf_de10y_all_shocks.png", ncols_wrap=4)

# Save IRF paths for the two headline responses as CSVs
for rv in response_vars:
    ri = ORDER.index(rv)
    df = pd.DataFrame({sv: orth_irfs[:, ri, ORDER.index(sv)] for sv in ORDER}, index=horizons)
    df.index.name = "horizon_months"
    df.to_csv(os.path.join(RES_DIR, f"irf_{rv}_to_all_shocks.csv"))

# ---------------------------------------------------------------------------
# 8. Forecast error variance decomposition (FEVD)
# ---------------------------------------------------------------------------

fevd = var_fitted.fevd(IRF_HORIZON)
fevd_de10y = pd.DataFrame(fevd.decomp[ORDER.index("de_10y")], columns=ORDER)
fevd_us10y = pd.DataFrame(fevd.decomp[ORDER.index("us_10y")], columns=ORDER)
fevd_de10y.index.name = "horizon_months"
fevd_us10y.index.name = "horizon_months"
fevd_de10y.to_csv(os.path.join(RES_DIR, "fevd_de10y_full_decomp.csv"))
fevd_us10y.to_csv(os.path.join(RES_DIR, "fevd_us10y_full_decomp.csv"))

us_share_de10y = fevd_de10y[US_BLOCK].sum(axis=1)
de_share_de10y = fevd_de10y[DE_BLOCK].sum(axis=1)
fevd_de10y_grouped = pd.DataFrame({
    "us_block_share": us_share_de10y,
    "de_ecb_block_share": de_share_de10y,
})
fevd_de10y_grouped.to_csv(os.path.join(RES_DIR, "fevd_us_vs_ecb_de10y.csv"))

fevd_horizons = np.arange(len(fevd_de10y_grouped))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.stackplot(
    fevd_horizons,
    fevd_de10y_grouped["us_block_share"], fevd_de10y_grouped["de_ecb_block_share"],
    labels=["US / global block", "DE / ECB block"],
    colors=["#1f3b57", "#b5651d"], alpha=0.85, edgecolor="white", linewidth=0.4,
)
ax.set_xlabel("Horizon (months)")
ax.set_ylabel("Share of forecast error variance")
ax.set_xlim(0, fevd_horizons.max())
ax.set_ylim(0, 1)
ax.grid(axis="y")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fevd_us_vs_ecb_de10y.png"), dpi=150)
plt.close(fig)

print("\n=== FEVD: share of DE 10Y forecast-error variance, final horizon ===")
print(fevd_de10y_grouped.iloc[-1])

print("\n=== FEVD: share of US 10Y forecast-error variance from US-block shocks, final horizon ===")
print(fevd_us10y[US_BLOCK].sum(axis=1).iloc[-1])

# ---------------------------------------------------------------------------
# 9. Generate LaTeX table snippets for report/report.tex (\input'd there)
# ---------------------------------------------------------------------------

REPORT_DIR = os.path.join(HERE, "report")
os.makedirs(REPORT_DIR, exist_ok=True)
BS = "\\_"  # literal backslash-underscore for LaTeX escaping


def _write_table(fname, caption, label, col_spec, header, rows):
    body = "\n".join(rows)
    tbl = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"
        f"{header} \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(REPORT_DIR, fname), "w") as f:
        f.write(tbl)


# Stationarity table
sel_vars = ["us_10y", "de_10y", "fed_funds", "ecb_rate", "us_cpi", "de_cpi", "vix", "fed_bs", "ecb_bs"]
rows = []
for v in sel_vars:
    row = stationarity.loc[v]
    vname = v.replace("_", BS)
    likely = "Yes" if row["likely_I1"] else "No"
    rows.append(f"{vname} & {row['adf_pval_level']:.3f} & {row['kpss_pval_level']:.3f} & "
                f"{row['adf_pval_diff']:.3f} & {likely} \\\\")
_write_table("table_stationarity.tex", "Stationarity tests (selected variables)",
             "tab:stationarity", "lcccc",
             "Variable & ADF $p$ (level) & KPSS $p$ (level) & ADF $p$ (diff) & Likely $I(1)$",
             rows)

# Granger table
rows = []
for _, row in granger_df.iterrows():
    cause = row["cause"].replace("_", BS)
    effect = row["effect"].replace("_", BS)
    sig = "Yes" if row["significant_5pct"] else "No"
    rows.append(f"{cause} & {effect} & {row['p_value']:.4f} & {sig} \\\\")
_write_table("table_granger.tex", "Granger causality tests, selected pairs",
             "tab:granger", "llcc", "Cause & Effect & $p$-value & Significant (5\\%)", rows)

# Johansen table
rows = []
for _, row in coint_report.iterrows():
    r = int(row["rank_null_h0_r<="])
    rej = "Yes" if row["trace_reject_5pct"] else "No"
    rows.append(f"{r} & {row['trace_stat']:.2f} & {row['trace_crit_5pct']:.2f} & {rej} & "
                f"{row['maxeig_stat']:.2f} & {row['maxeig_crit_5pct']:.2f} \\\\")
_write_table("table_johansen.tex", "Johansen cointegration test", "tab:johansen", "cccccc",
             "$H_0$: rank $\\le$ & Trace stat. & 5\\% crit. & Reject? & Max-eig stat. & 5\\% crit.", rows)

# VECM beta table
rows = []
for v, row in beta_df.iterrows():
    vname = v.replace("_", BS)
    rows.append(f"{vname} & {row.iloc[0]:.4f} \\\\")
_write_table("table_vecm_beta.tex", "VECM cointegrating vector (normalized on \\texttt{us\\_10y})",
             "tab:vecm_beta", "lc", "Variable & Coefficient", rows)

# Sample period string (\input'd inline in report.tex prose)
with open(os.path.join(REPORT_DIR, "sample_period.tex"), "w") as f:
    start = panel_levels.index.min().strftime("%B %Y")
    end = panel_levels.index.max().strftime("%B %Y")
    f.write(f"{start} to {end} ({len(panel_levels)} monthly observations)")

print("\nLaTeX table snippets written to report/ (table_*.tex, sample_period.tex).")
print("All results written to results/, all figures written to figures/.")
print("Done.")
