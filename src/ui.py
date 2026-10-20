"""Presentation helpers for the Streamlit dashboard: theme, formatting, cards and the
read-only views of saved robustness results. Nothing here computes model metrics."""
import html
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ACCENT = "#34D399"          # emerald
ACCENT_2 = "#A3E635"        # lime
MUTED = "#8B949E"
TEXT = "#E6EDF3"
GRID = "rgba(255,255,255,0.06)"
WARN = "#F5B942"
DANGER = "#F87171"
MODEL_COLORS = {"Rule-based": "#8B949E", "Logistic Regression": "#38BDF8",
                "Random Forest": ACCENT, "Fraud-rate baseline": WARN}
SEED_DASH = ["solid", "dash", "dot", "dashdot"]

METRIC_HELP = {
    "pr_auc": "Area under the precision–recall curve: how well the model ranks fraud above "
              "legitimate transactions, across all thresholds. A random scorer gets the fraud "
              "rate (the baseline), so always compare with it.",
    "baseline": "Fraud prevalence in the evaluated period. It is the PR-AUC a random scorer would "
                "get and the precision of flagging everything.",
    "precision": "Of the transactions the model alerted on, the share that were really fraud.",
    "recall": "Of all fraud in the period, the share the model alerted on.",
    "f1": "Harmonic mean of precision and recall; 'max F1' picks the threshold that maximises it "
          "on validation data.",
    "false_alerts": "Legitimate transactions that were alerted (false positives): reviewer "
                    "workload with no fraud found.",
    "missed_fraud": "Fraudulent transactions that were not alerted (false negatives).",
    "alerts_per_1000": "Alerts raised per 1,000 transactions: the review workload the threshold "
                       "creates.",
    "threshold": "Score cut-off chosen on the VALIDATION period only; the TEST period never "
                 "influences it.",
    "lift": "PR-AUC divided by the baseline: how many times better than random ranking.",
    "psi": "Population stability index of model scores between validation and test (label-free "
           "drift signal).",
}

CSS = """
<style>
:root { --fs-accent: #34D399; --fs-accent2: #A3E635; --fs-border: rgba(255,255,255,0.08);
        --fs-muted: #8B949E; --fs-card: rgba(255,255,255,0.025); }
html, body, [class*="css"] { font-feature-settings: "tnum" 1, "cv11" 1; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0E1412 0%, #0A0D0C 100%);
                            border-right: 1px solid var(--fs-border); }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    width: 100%; padding: 7px 10px; border-radius: 8px; border: 1px solid transparent;
    transition: background .15s ease, border-color .15s ease; margin: 0; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,0.04); }
[data-testid="stSidebar"] [role="radiogroup"] label > div > div:first-child:not([data-testid]) {
    display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: rgba(52,211,153,0.10); border-color: rgba(52,211,153,0.35);
    box-shadow: inset 3px 0 0 var(--fs-accent); }
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color: var(--fs-accent); font-weight: 600; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 0.92rem; }
.fs-brand { display: flex; align-items: center; gap: 14px; padding: 18px 22px; margin-bottom: 14px;
    border: 1px solid var(--fs-border); border-radius: 14px;
    background: radial-gradient(1200px 200px at 0% 0%, rgba(52,211,153,0.10), transparent 60%),
                linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); }
.fs-mark { width: 40px; height: 44px; flex: none;
    background: linear-gradient(160deg, var(--fs-accent2), var(--fs-accent) 55%, #0F766E);
    clip-path: polygon(50% 0, 100% 16%, 100% 55%, 50% 100%, 0 55%, 0 16%); }
.fs-brand h1 { font-size: 1.55rem; margin: 0; padding: 0; letter-spacing: -0.02em; line-height: 1.1; }
.fs-brand p { margin: 2px 0 0 0; color: var(--fs-muted); font-size: 0.92rem; }
.fs-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.fs-badge { font-size: 0.70rem; letter-spacing: .06em; text-transform: uppercase; font-weight: 600;
    padding: 3px 8px; border-radius: 999px; border: 1px solid var(--fs-border); color: #C9D1D9;
    background: rgba(255,255,255,0.03); white-space: nowrap; }
.fs-badge.accent { color: var(--fs-accent); border-color: rgba(52,211,153,0.35);
    background: rgba(52,211,153,0.08); }
.fs-badge.warn { color: #F5B942; border-color: rgba(245,185,66,0.35); background: rgba(245,185,66,0.08); }
.fs-section { margin: 26px 0 10px 0; }
.fs-section h3 { font-size: 1.08rem; margin: 0; padding: 0; letter-spacing: -0.01em; }
.fs-section p { margin: 2px 0 0 0; color: var(--fs-muted); font-size: 0.86rem; }
.fs-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 10px;
    margin: 6px 0 4px 0; }
.fs-kpi { border: 1px solid var(--fs-border); border-radius: 12px; padding: 12px 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
    transition: border-color .15s ease; cursor: default; min-width: 0; }
.fs-kpi:hover { border-color: rgba(52,211,153,0.40); }
.fs-kpi .lbl { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em;
    color: var(--fs-muted); display: flex; justify-content: space-between; gap: 6px; }
.fs-kpi .lbl i { font-style: normal; opacity: .6; border: 1px solid var(--fs-border);
    border-radius: 999px; width: 15px; height: 15px; line-height: 13px; text-align: center;
    font-size: 0.62rem; flex: none; }
.fs-kpi .val { font-size: 1.45rem; font-weight: 650; margin-top: 4px; letter-spacing: -0.02em;
    color: #F0F6FC; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fs-kpi .val.accent { color: var(--fs-accent); }
.fs-kpi .sub { font-size: 0.76rem; color: var(--fs-muted); margin-top: 2px; }
.fs-note { border-left: 3px solid var(--fs-accent); background: rgba(52,211,153,0.06);
    padding: 9px 12px; border-radius: 0 8px 8px 0; font-size: 0.88rem; color: #C9D1D9; margin: 8px 0; }
.fs-note.warn { border-left-color: #F5B942; background: rgba(245,185,66,0.07); }
.fs-ctx { font-size: 0.80rem; color: var(--fs-muted); line-height: 1.55; border: 1px solid var(--fs-border);
    border-radius: 10px; padding: 10px 12px; background: rgba(255,255,255,0.02); }
.fs-ctx b { color: #C9D1D9; font-weight: 600; }
.fs-ctx code { font-size: 0.74rem; }
[data-testid="stExpander"] details { border-color: var(--fs-border); border-radius: 10px; }
[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stMetric"] { border: 1px solid var(--fs-border); border-radius: 12px; padding: 10px 14px;
    background: rgba(255,255,255,0.02); }
.st-key-fs-report h1 { font-size: 1.5rem; }
.st-key-fs-report h2 { font-size: 1.2rem; }
.st-key-fs-report h3 { font-size: 1.02rem; }
.st-key-fs-report table { display: block; overflow-x: auto; }
@media (max-width: 760px) {
  .fs-brand { padding: 14px; } .fs-brand h1 { font-size: 1.25rem; }
  .fs-kpi .val { font-size: 1.2rem; }
}
</style>
"""


def register_plotly_template():
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, -apple-system, 'SF Pro Text', 'Segoe UI', sans-serif",
              "color": "#C9D1D9", "size": 12},
        colorway=[ACCENT, "#38BDF8", ACCENT_2, WARN, "#C084FC", MUTED],
        xaxis={"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": "rgba(255,255,255,0.12)"},
        yaxis={"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": "rgba(255,255,255,0.12)"},
        legend={"bgcolor": "rgba(0,0,0,0)", "orientation": "h", "y": -0.22, "yanchor": "top", "x": 0,
                "title": {"text": ""}},
        hoverlabel={"bgcolor": "#161B22", "bordercolor": "#30363D", "font": {"color": TEXT}},
        margin={"t": 64, "b": 10, "l": 10, "r": 10},
    )
    tpl.data.bar = [go.Bar(marker_line_width=0, opacity=0.88)]
    pio.templates["finsentry"] = tpl
    pio.templates.default = "plotly_dark+finsentry"


def style_fig(fig, title, subtitle=None, height=360):
    text = f"<b>{html.escape(title, quote=False)}</b>"
    if subtitle:
        text += (f"<br><span style='font-size:12px;color:{MUTED}'>"
                 f"{html.escape(subtitle, quote=False)}</span>")
    fig.update_layout(title={"text": text, "x": 0, "xanchor": "left", "font": {"size": 15}},
                      height=height, legend_title_text="")
    fig.for_each_yaxis(lambda a: a.update(dtick=1) if a.type == "log" else None)
    fig.for_each_xaxis(lambda a: a.update(dtick=1) if a.type == "log" else None)
    return fig


# ---------- formatting (None means undefined and is never shown as zero) ----------
def fmt_pct(v, digits=1, undefined="undefined"):
    return undefined if v is None or pd.isna(v) else f"{v * 100:.{digits}f}%"


def fmt_num(v, digits=3, undefined="undefined"):
    return undefined if v is None or pd.isna(v) else f"{v:.{digits}f}"


def fmt_count(v, undefined="undefined"):
    return undefined if v is None or pd.isna(v) else f"{int(round(v)):,}"


def fmt_rate(v, undefined="undefined"):
    """Alerts per 1,000 transactions."""
    return undefined if v is None or pd.isna(v) else f"{v:.2f}"


def fmt_range(lo, hi, fn):
    return "undefined" if lo is None or hi is None else f"{fn(lo)} – {fn(hi)}"


FORMATTERS = {"pct": fmt_pct, "pct2": lambda v: fmt_pct(v, 2), "dec3": fmt_num,
              "dec4": lambda v: fmt_num(v, 4), "int": fmt_count, "rate": fmt_rate,
              "lift": lambda v: "undefined" if v is None or pd.isna(v) else f"{v:.1f}×"}


def format_frame(df, spec):
    """Return a display copy with columns formatted as strings; spec maps column -> kind."""
    out = df.copy()
    for col, kind in spec.items():
        if col in out:
            out[col] = [FORMATTERS[kind](v) if not isinstance(v, str) else v for v in out[col]]
    return out


# ---------- HTML snippets ----------
def badges(items):
    parts = []
    for text, kind in items:
        cls = f"fs-badge {kind}" if kind else "fs-badge"
        parts.append(f"<span class='{cls}'>{html.escape(str(text))}</span>")
    return f"<div class='fs-badges'>{''.join(parts)}</div>"


def brand_header(tagline, badge_items):
    return ("<div class='fs-brand'><div class='fs-mark'></div><div>"
            "<h1>FinSentry</h1>"
            f"<p>{html.escape(tagline)}</p>{badges(badge_items)}</div></div>")


def section(title, subtitle=None):
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    return f"<div class='fs-section'><h3>{html.escape(title)}</h3>{sub}</div>"


def note(text, kind=""):
    return f"<div class='fs-note {kind}'>{html.escape(text)}</div>"


def kpi_cards(cards):
    """cards: dicts with label, value, optional sub, help, accent."""
    out = []
    for c in cards:
        tip = html.escape(c.get("help", ""), quote=True)
        icon = "<i>?</i>" if tip else ""
        val_cls = "val accent" if c.get("accent") else "val"
        sub = f"<div class='sub'>{html.escape(c['sub'])}</div>" if c.get("sub") else ""
        out.append(f"<div class='fs-kpi' title='{tip}'><div class='lbl'>"
                   f"<span>{html.escape(c['label'])}</span>{icon}</div>"
                   f"<div class='{val_cls}'>{html.escape(str(c['value']))}</div>{sub}</div>")
    return f"<div class='fs-kpis'>{''.join(out)}</div>"


# ---------- saved robustness study (read-only) ----------
def load_robustness(output_dir):
    """(study, problems). study is None when no study exists or it cannot be read."""
    path = Path(output_dir) / "robustness" / "robustness.json"
    if not path.exists():
        return None, []
    try:
        study = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        return None, [f"robustness.json could not be read: {err}"]
    problems = []
    if study.get("status") != "complete":
        problems.append(f"Study status is '{study.get('status')}', so some seeds are missing or failed.")
    seeds = study.get("config", {}).get("seeds", [])
    if study.get("n_aggregated") != len(seeds):
        problems.append(f"Only {study.get('n_aggregated')} of {len(seeds)} seeds were aggregated.")
    if not study.get("aggregate"):
        problems.append("No aggregated results in robustness.json.")
    for r in study.get("runs", []):
        if not r.get("usable"):
            problems.append(f"Seed {r.get('seed')}: " + ("; ".join(r.get("problems", [])) or "not usable"))
    return study, problems


def _seeds(study):
    return [str(s) for s in study["aggregate"]["seeds"]]


def robustness_single_frame(study, model, keys):
    """Long frame of per-seed single-split values for one model."""
    rows = []
    entry = study["aggregate"]["single_split"][model]
    for key in keys:
        if key not in entry:
            continue
        for seed in _seeds(study):
            rows.append({"Seed": seed, "metric": key, "value": entry[key]["values"].get(seed)})
    return pd.DataFrame(rows)


def robustness_summary_table(study, model, keys):
    """Per-seed values plus between-seed mean, SD and range (descriptive)."""
    entry, rows = study["aggregate"]["single_split"][model], []
    for key, label, kind in keys:
        if key not in entry:
            continue
        e, d = entry[key], entry[key]["between_seed"]
        f = FORMATTERS[kind]
        spread = (lambda v: "undefined" if v is None else f"{v:,.1f}") if kind == "int" else f
        mean = (lambda v: "undefined" if v is None else f"{v:,.1f}") if kind == "int" else f
        rows.append({"Metric": label,
                     **{f"Seed {s}": f(e["values"].get(s)) for s in _seeds(study)},
                     "Mean": mean(d["mean"]), "SD": spread(d["sd"]),
                     "Range": fmt_range(d["min"], d["max"], f)})
    return pd.DataFrame(rows)


def robustness_window_frame(study):
    rows = []
    for seed in _seeds(study):
        for w in study["aggregate"]["per_seed_windows"][seed]:
            base = {"Seed": seed, "Window": w["window"],
                    "Test steps": "–".join(map(str, w["test_steps"])),
                    "Fraud": w["test_fraud"], "Baseline": w["prevalence"]}
            for model in ("Logistic Regression", "Random Forest"):
                m = w["models"].get(model, {})
                rows.append({**base, "Model": model, "PR-AUC": m.get("pr_auc"),
                             "Lift": m.get("pr_auc_lift")})
    return pd.DataFrame(rows)


def robustness_variation_frame(study, metric="pr_auc"):
    rows = []
    for model in ("Logistic Regression", "Random Forest"):
        t = study["aggregate"]["temporal"].get(model, {}).get(metric)
        if not t:
            continue
        rows.append({"Model": model,
                     "Temporal (typical SD across windows)": t["mean_within_seed_temporal_sd"]["mean"],
                     "Between seeds (SD of window mean)": t["between_seed_of_window_mean"]["sd"]})
    return pd.DataFrame(rows)


def strategy_points(study):
    """Per-seed precision/recall at both validation-chosen thresholds, for each model."""
    rows = []
    for model in ("Logistic Regression", "Random Forest"):
        entry = study["aggregate"]["single_split"][model]
        for strat, label in (("max_f1", "Max F1"), ("alert_budget", "Alert budget")):
            for seed in _seeds(study):
                rows.append({"Model": model, "Strategy": label, "Seed": seed,
                             "Precision": entry[f"{strat}.precision"]["values"].get(seed),
                             "Recall": entry[f"{strat}.recall"]["values"].get(seed),
                             "False alerts": entry[f"{strat}.false_positive"]["values"].get(seed),
                             "Alerts per 1,000": entry[f"{strat}.alerts_per_1000"]["values"].get(seed)})
    return pd.DataFrame(rows)
