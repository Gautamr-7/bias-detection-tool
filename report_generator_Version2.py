from __future__ import annotations
from datetime import datetime
from typing import Dict, List


def _badge(text: str, severity: str) -> str:
    colours = {
        "critical": ("#fecaca", "#7f1d1d"),
        "warning": ("#fde68a", "#78350f"),
        "good": ("#bbf7d0", "#14532d"),
        "pass": ("#bbf7d0", "#14532d"),
        "fail": ("#fecaca", "#7f1d1d"),
    }
    fg, bg = colours.get(severity, ("#cbd5e1", "#334155"))
    return f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;">{text}</span>'


def _score_colour(score: float) -> str:
    if score >= 90:
        return "#10B981"
    if score >= 70:
        return "#F59E0B"
    return "#EF4444"


def generate_html_report(
    dataset_name: str,
    overall_score: float,
    proxy_findings,
    cf_results,
    fairness_results,
    recommendations: List[Dict],
    model_accuracy: float = 0.0,
    model_auc: float = 0.0,
) -> str:
    now = datetime.now().strftime("%B %d, %Y at %H:%M")
    score_col = _score_colour(overall_score)

    proxy_rows = ""
    for f in proxy_findings[:12]:
        bar_w = int(f.suspicion_score * 100)
        sev_badge = _badge(f.severity.upper(), f.severity if f.severity != "ok" else "good")
        proxy_rows += f"""
        <tr>
          <td>{f.feature}</td>
          <td>
            <div class="bar"><div class="bar-fill risk" style="width:{bar_w}%"></div></div>
            <small>{f.suspicion_score:.3f} (CI: {f.confidence_interval[0]:.3f}, {f.confidence_interval[1]:.3f})</small>
          </td>
          <td>{sev_badge}</td>
          <td><small>{'; '.join(f'{k}: {v:.1%}' for k, v in list(f.approval_by_bin.items())[:3])}</small></td>
        </tr>"""

    cf_rows = ""
    avg_csi = 1.0
    if cf_results:
        avg_csi = sum(res.csi for res in cf_results.values()) / len(cf_results)

    for var, res in cf_results.items():
        bar_w = int(res.bias_rate * 100)
        sig_badge = _badge("SIGNIFICANT" if res.is_significant else "NOT SIGNIFICANT", "critical" if res.is_significant else "good")
        cf_rows += f"""
        <tr>
          <td>{var}</td>
          <td>{res.n_biased} / {res.n_tested}</td>
          <td>
            <div class="bar"><div class="bar-fill danger" style="width:{bar_w}%"></div></div>
            <small>{res.bias_rate:.1%}</small>
          </td>
          <td>{res.p_value:.5f}</td>
          <td>{res.csi:.3f}</td>
          <td>{sig_badge}</td>
        </tr>"""

    metric_rows = ""
    for _, r in fairness_results.items():
        status_badge = _badge("PASS" if r.passed else "FAIL", "good" if r.passed else "critical")
        metric_rows += f"""
        <tr>
          <td>{r.name}</td>
          <td>{r.value:.4f}</td>
          <td>{r.tolerance}</td>
          <td>{status_badge}</td>
          <td><small>{r.description}</small></td>
        </tr>"""

    rec_html = ""
    for rec in recommendations:
        rec_html += f"""
        <div class="rec-card">
          <div class="meta">{rec['priority']} · Effort: {rec['effort']}</div>
          <div class="title">{rec['action']}</div>
          <div class="body">{rec['rationale']}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Fairness Audit Report — {dataset_name}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: Inter, Arial, sans-serif; background:#0B1220; color:#E2E8F0; }}
.container {{ max-width: 1120px; margin: 0 auto; padding: 28px; }}
.header {{
  background: linear-gradient(135deg,#0f172a,#1e293b,#0f172a);
  border: 1px solid #334155;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 24px;
}}
h1,h2 {{ margin:0; }}
h2 {{ margin-bottom:12px; font-size:18px; }}
.section {{ margin-bottom:24px; }}
.grid {{ display:grid; gap:12px; grid-template-columns: repeat(4, minmax(0,1fr)); }}
.card {{ background:#111827; border:1px solid #334155; border-radius:12px; padding:16px; }}
.val {{ font-size:28px; font-weight:700; }}
.lbl {{ font-size:12px; color:#94A3B8; margin-top:4px; }}
table {{ width:100%; border-collapse: collapse; background:#111827; border:1px solid #334155; border-radius:12px; overflow:hidden; }}
th,td {{ padding:10px 12px; border-bottom:1px solid #1f2937; text-align:left; vertical-align:top; }}
th {{ font-size:12px; text-transform:uppercase; color:#94A3B8; }}
.bar {{ width:180px; height:8px; border-radius:999px; background:#1f2937; overflow:hidden; }}
.bar-fill {{ height:8px; }}
.risk {{ background:#f59e0b; }}
.danger {{ background:#ef4444; }}
.rec-card {{ background:#111827; border:1px solid #334155; border-radius:10px; padding:14px; margin-bottom:10px; }}
.rec-card .meta {{ color:#94A3B8; font-size:12px; margin-bottom:4px; }}
.rec-card .title {{ font-weight:600; margin-bottom:4px; }}
.rec-card .body {{ color:#cbd5e1; font-size:14px; }}
.disclaimer {{ background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.35); border-radius:10px; padding:14px; color:#fde68a; font-size:13px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div style="display:flex;justify-content:space-between;gap:18px;align-items:flex-start;flex-wrap:wrap">
      <div>
        <div style="color:#94A3B8;font-size:13px;">⚖️ AI FAIRNESS AUDITOR</div>
        <h1 style="margin-top:6px;">Audit Report</h1>
        <div style="margin-top:8px;color:#94A3B8;">Dataset: {dataset_name}</div>
        <div style="color:#64748b;font-size:13px;">Generated: {now}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:56px;font-weight:800;color:{score_col};line-height:1;">{overall_score:.0f}</div>
        <div style="font-size:12px;color:#94A3B8;">OVERALL SCORE / 100</div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Executive Summary</h2>
    <div class="grid">
      <div class="card"><div class="val">{len([f for f in proxy_findings if f.is_high_risk])}</div><div class="lbl">Critical Proxy Variables</div></div>
      <div class="card"><div class="val">{sum(r.n_biased for r in cf_results.values()) if cf_results else 0}</div><div class="lbl">Biased Decisions Found</div></div>
      <div class="card"><div class="val">{model_auc:.3f}</div><div class="lbl">Model AUC</div></div>
      <div class="card"><div class="val">{avg_csi:.3f}</div><div class="lbl">Counterfactual Stability Index (CSI)</div></div>
    </div>
  </div>

  <div class="section">
    <h2>Proxy Variable Analysis</h2>
    <table>
      <thead><tr><th>Feature</th><th>Suspicion Score</th><th>Severity</th><th>Approval Pattern</th></tr></thead>
      <tbody>{proxy_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>Counterfactual Testing</h2>
    <table>
      <thead><tr><th>Variable</th><th>Biased/Tested</th><th>Bias Rate</th><th>p-value</th><th>CSI</th><th>Significance</th></tr></thead>
      <tbody>{cf_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>Fairness Metrics Scorecard</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th><th>Tolerance</th><th>Status</th><th>Description</th></tr></thead>
      <tbody>{metric_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>Recommendations</h2>
    {rec_html}
  </div>

  <div class="section">
    <div class="disclaimer">
      ⚠️ <b>Disclaimer:</b> This is an automated risk-detection tool and does not constitute legal advice.
      Validate results with domain experts and legal counsel before production decisions.
    </div>
  </div>
</div>
</body>
</html>"""
    return html