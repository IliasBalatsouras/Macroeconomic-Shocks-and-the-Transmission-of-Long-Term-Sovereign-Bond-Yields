# VAR / SVAR / Cointegration Analysis of U.S. and German 10-Year Government Bond Yields

A structural time-series analysis of the **U.S. 10-Year Treasury yield** and
the **German 10-Year Bund yield**, following the standard multivariate
course sequence: stationarity testing → reduced-form VAR → Granger
causality → Johansen cointegration / VECM → structural (Cholesky)
identification → impulse response functions → forecast error variance
decomposition (FEVD).

**Core question:** how much of each yield's variation comes from domestic
monetary policy versus foreign/global shocks, and through what channel —
lag-predictable (Granger-causal), long-run equilibrium (cointegration), or
purely contemporaneous (structural/Cholesky) — does that transmission run?

A full write-up, with literature review and discussion, is in
[`report/report.pdf`](report/report.pdf) (source: `report/report.tex`,
compiles standalone with `pdflatex`, no bibtex/biber needed).

## Headline findings

- **U.S./global shocks explain ~66% of the German 10-year Bund yield's**
  **forecast error variance** at a 24-month horizon, versus ~34% for
  domestic ECB-block shocks — a "global financial cycle" result (Rey 2013;
  Miranda-Agrippino & Rey 2020).
- **The two yields do not Granger-cause each other** (US→DE $p=0.89$;
  DE→US $p=0.18$): cross-border transmission is not lag-predictable
  through the yields' own past values — it is largely contemporaneous.
- **VIX and each country's own CPI Granger-cause that country's yield**
  (VIX→US $p=0.003$; US CPI→US $p<0.001$; DE CPI→DE $p=0.002$); neither
  policy rate Granger-causes either yield at the 5% level.
- **A single cointegrating relationship** links the two yields, the two
  policy rates, and the two central bank balance sheets (Johansen trace
  test rejects $r=0$, fails to reject $r\le1$). Neither yield's VECM
  adjustment speed ($\alpha$) is large — the yields don't rapidly
  self-correct back to the estimated long-run relationship.

See `results/` for the full numerical output and `figures/` for all plots.

## Model

**11 variables, block-recursive Cholesky ordering** (same variable set as
the companion "U.S. spillover vs. ECB" analysis in the broader project this
was built from, for direct comparability):

| Block | Variable | FRED series | Role |
|---|---|---|---|
| US (ordered first) | `us_ip` | INDPRO | real activity |
| | `us_cpi` | CPIAUCSL | prices |
| | `fed_funds` | FEDFUNDS | policy instrument |
| | `fed_bs` | WALCL (log) | unconventional policy |
| | `vix` | VIXCLS | global risk sentiment |
| | `us_10y` | GS10 | **US 10Y Treasury yield** |
| DE/ECB (ordered second) | `de_ip` | DEUPROINDMISMEI | real activity |
| | `de_cpi` | CP0000DEM086NEST | prices (HICP) |
| | `ecb_rate` | ECBDFR | policy instrument |
| | `ecb_bs` | ECBASSETSW (log) | unconventional policy |
| | `de_10y` | IRLTLT01DEM156N | **German 10Y Bund yield** |

The US block is ordered first — contemporaneously exogenous to the DE/ECB
block — on the "small open financial area" assumption that US Treasury and
dollar funding markets are large relative to any single Euro Area sovereign
bond market. Within each block, variables are ordered slow (real/price) →
policy instrument → fast (market-priced), the standard recursive-monetary-VAR
convention (Bernanke & Blinder 1992; Christiano, Eichenbaum & Evans 1999).

Sample: monthly, January 2003 – March 2024 (255 observations); the end date
is set by `DEUPROINDMISMEI` (German industrial production), which stops
updating on FRED after March 2024.

## Methodology sequence

1. **Stationarity** — ADF and KPSS tests on levels and first differences
   (`results/stationarity_report.csv`). Most series behave as I(1).
2. **Reduced-form VAR** — estimated in first differences at a fixed lag of
   2 (deliberately conservative for an 11-variable system on ~250 monthly
   observations); AIC/BIC/HQIC lag-order picks reported separately
   (`results/lag_order_selection.txt`).
3. **Granger causality** — pairwise F-tests for the economically relevant
   pairs (`results/granger_causality_pvalues.csv`).
4. **Cointegration / VECM** — Johansen trace and max-eigenvalue tests on the
   theoretically cointegrated core (both yields, both policy rates, both
   balance sheets); VECM estimation of the cointegrating vector(s) and
   adjustment speeds (`results/johansen_cointegration_test.csv`,
   `results/vecm_beta_cointegrating_vectors.csv`,
   `results/vecm_alpha_adjustment_speeds.csv`).
5. **Structural (Cholesky) identification** — recursive decomposition of
   the reduced-form VAR residual covariance under the block-recursive
   ordering above (`results/cholesky_factor.csv`).
6. **Structural IRFs** — orthogonalized impulse responses of both yields to
   all 11 shocks, 24-month horizon
   (`figures/structural_irf_yields_to_key_shocks.png`,
   `figures/structural_irf_full_grid.png`).
7. **FEVD** — forecast error variance decomposition of both yields, and a
   grouped US-block-vs-DE/ECB-block share for the German yield
   (`figures/fevd_us_vs_ecb_de10y.png`, `results/fevd_us_vs_ecb_de10y.csv`).

## Repo structure

```
us-de-bond-yield-var-svar/
├── README.md                              (this file)
├── requirements.txt
├── .gitignore
├── var_svar_us_de_bond_yields.py          (single analysis script, all 7 steps)
├── data/
│   └── fred_data/                         (cached FRED series, CSV)
├── results/                               (generated: stationarity, Granger,
│                                            Johansen, VECM, IRF, FEVD CSVs/summaries)
├── figures/                               (generated: raw series, IRF grids,
│                                            FEVD stacked-area chart)
└── report/
    ├── report.tex                         (source, journal-style, no bibtex)
    ├── report.pdf                         (compiled write-up)
    └── table_*.tex                        (auto-generated table snippets, \input by report.tex)
```

## Running

```bash
git clone <this-repo>
cd us-de-bond-yield-var-svar
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python var_svar_us_de_bond_yields.py
```

All FRED series are already cached under `data/fred_data/`, so the script
runs immediately after cloning with **no FRED API key and no network access
required**. This also means the data reflects whatever was current when it
was fetched (2003-01 to 2024-03) — refresh the CSVs with a fresh FRED API
key if you want a more current sample.

The script's last step (re)generates the LaTeX table snippets under
`report/` (`table_*.tex`, `sample_period.tex`) directly from the
`results/*.csv` output, so the report always reflects the latest run.
Recompile it with:

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # run twice for cross-references
```

## Extending this

- Since the table snippets are regenerated by the main script on every
  run, changing the sample, lag order, or variable set and re-running
  `var_svar_us_de_bond_yields.py` followed by a report recompile is enough
  to keep everything in sync.
- A natural robustness check: replace the Cholesky/recursive identification
  with the high-frequency, event-based residual-restriction approach of
  Badinger & Schiman (2023) for ECB shocks.
- The Portmanteau residual-whiteness test rejects at every lag
  specification tried (see `results/var_whiteness_test.txt`), most likely
  reflecting volatility clustering (GARCH-type effects) in the financial
  variables. An ARCH-LM test on the VAR residuals, and/or HAC standard
  errors, would be the natural next diagnostic step.

## Data source and license

All series are from [FRED](https://fred.stlouisfed.org/) (Federal Reserve
Bank of St. Louis). Get a free API key at
https://fred.stlouisfed.org/docs/api/api_key.html if you want to refresh
the cached data — **do not commit your API key to git** (already covered by
`.gitignore`).

Code and write-up are shared for educational/research purposes. If you reuse
substantial portions, a citation back to this repo is appreciated.
