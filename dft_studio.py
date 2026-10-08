"""
DFT STUDIO - interactive front-end for dft_sim.py
Stack: Streamlit, pandas, NumPy, SciPy, Plotly, PySCF

Run (same folder as dft_sim.py and sweep_results.csv):
    pip install streamlit pandas numpy scipy plotly pyscf
    streamlit run dft_studio.py
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

try:
    from dft_sim import EXPT, run_one, optimize, scf_energy
    LIVE_DFT = True
except ImportError:  # PySCF missing (e.g. native Windows) -> replay engine
    LIVE_DFT = False
    from scipy.optimize import minimize
    import time as _time

    EXPT = {"H2O": dict(r=0.9572, theta=104.52, dipole=1.855, center="O", ligand="H")}
    _CSV = "sweep_results.csv"
    _REF = pd.read_csv(_CSV) if os.path.exists(_CSV) else None
    if _REF is None:
        st.error("PySCF isn't installed and sweep_results.csv wasn't found next to this script. "
                 "Put the CSV in the same folder to use replay mode.")
        st.stop()

    def _row(fn, bs):
        m = _REF[(_REF.functional == fn) & (_REF.basis == bs)]
        return m.iloc[0]

    def scf_energy(mol, r, theta, fn, bs):
        """Surrogate surface anchored to the real DFT minimum for (fn, bs), in Hartree.
        Morse stretch (both O-H bonds) + (cos t - cos t0)^2 bend; the bend prefactor (~61 kcal/mol)
        was fitted to the b3lyp/6-31g bending curve."""
        w = _row(fn, bs)
        D, a, A = 120.0, 2.2, 61.0
        stretch = 2 * D * (1 - np.exp(-a * (r - w.r_A))) ** 2
        bend = A * (np.cos(np.radians(theta)) - np.cos(np.radians(w.theta_deg))) ** 2
        return w.energy_Eh + (stretch + bend) / 627.509

    def optimize(mol, fn, bs, r0=1.0, theta0=100.0):
        res = minimize(lambda x: scf_energy(mol, x[0], x[1], fn, bs), [r0, theta0],
                       method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-12))
        return res.x[0], res.x[1], res.fun

    def run_one(mol, fn, bs, r0=1.0, theta0=100.0):
        r, th, e = optimize(mol, fn, bs, r0, theta0)
        w, ex = _row(fn, bs), EXPT[mol]
        return dict(molecule=mol, functional=fn, basis=bs, r_A=r, theta_deg=th, energy_Eh=e,
                    dipole_D=w.dipole_D, gap_eV=w.gap_eV, time_s=w.time_s,
                    err_r_pct=100 * (r - ex["r"]) / ex["r"], err_theta_deg=th - ex["theta"],
                    err_dipole_pct=100 * (w.dipole_D - ex["dipole"]) / ex["dipole"])

st.set_page_config(page_title="DFT STUDIO", page_icon="⚛️", layout="wide")

# ------------------------------------------------------------------ style
ACCENT, HOT, ICE, AMBER = "#5eead4", "#fb7185", "#818cf8", "#fbbf24"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.5px; }
.stApp {
  background:
    radial-gradient(900px 520px at 6% -6%, rgba(99,102,241,.30), transparent 60%),
    radial-gradient(800px 520px at 100% 8%, rgba(244,114,182,.18), transparent 60%),
    radial-gradient(800px 600px at 50% 115%, rgba(45,212,191,.15), transparent 60%),
    #0a0a18;
}
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:2.2rem; max-width:1400px; padding-left:2.2rem; padding-right:2.2rem; }
.eyebrow { font-size:.8rem; letter-spacing:5px; color:#5eead4; font-weight:600; margin-bottom:.2rem; }
h1.hero { font-size:3.8rem; line-height:1.05; font-weight:700; margin:0;
  background:linear-gradient(90deg,#5eead4 0%,#818cf8 50%,#fb7185 100%);
  -webkit-background-clip:text; color:transparent; }
.sub { color:#94a3b8; font-size:1.1rem; margin:.5rem 0 1.4rem; max-width:720px; }
.card { border:1px solid rgba(255,255,255,.09); border-radius:22px; padding:18px 22px;
  background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02));
  backdrop-filter:blur(14px); box-shadow:0 10px 40px rgba(0,0,0,.35); height:100%; }
.card h3 { margin:0 0 8px; font-size:1.5rem; color:#f1f5f9; }
.tag { font-size:.68rem; letter-spacing:1.5px; padding:3px 11px; border-radius:99px;
  border:1px solid; margin-left:10px; vertical-align:middle; font-weight:600; }
.big { font-family:'Space Grotesk',sans-serif; font-size:2.3rem; font-weight:700; line-height:1.1; color:#e2e8f0; }
.lbl { color:#94a3b8; font-size:.72rem; text-transform:uppercase; letter-spacing:2.5px; margin-bottom:6px; font-weight:600; }
.good { color:#5eead4; } .bad { color:#fb7185; }
div[data-baseweb="tab-list"] { gap:10px; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display:none; }
button[data-baseweb="tab"] { font-size:1rem; font-weight:600; border-radius:99px; padding:.5rem 1.3rem;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); color:#94a3b8; }
button[data-baseweb="tab"][aria-selected="true"] { color:#fff; border-color:rgba(255,255,255,.25);
  background:linear-gradient(90deg,rgba(94,234,212,.22),rgba(129,140,248,.28)); }
div.stButton > button { background:linear-gradient(90deg,#5eead4,#818cf8); color:#08111a; font-weight:700;
  border:none; border-radius:14px; padding:.55rem 1.5rem; transition:all .2s; box-shadow:0 6px 24px rgba(94,234,212,.25); }
div.stButton > button:hover { transform:translateY(-2px); box-shadow:0 10px 30px rgba(129,140,248,.4); color:#08111a; }
div[data-baseweb="select"] > div { background:rgba(255,255,255,.05); border-radius:12px; border-color:rgba(255,255,255,.1); }
div[data-testid="stAlert"] { border-radius:16px; background:rgba(255,255,255,.05); }
</style>
""", unsafe_allow_html=True)

def style(fig, h=420):
    ax = dict(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor="rgba(255,255,255,.12)",
              tickfont=dict(color="#94a3b8"))
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=13, color="#cbd5e1"),
                      title=dict(font=dict(size=16, color="#e2e8f0")), margin=dict(l=10, r=10, t=50, b=10),
                      colorway=[ACCENT, ICE, HOT, AMBER], hoverlabel=dict(bgcolor="#14122e", font_size=13))
    fig.update_xaxes(**ax); fig.update_yaxes(**ax)
    return fig

def stat(label, value, sub="", good=None):
    cls = "" if good is None else ("good" if good else "bad")
    st.markdown(f"<div class='card'><div class='lbl'>{label}</div><div class='big {cls}'>{value}</div>"
                f"<div style='color:#a9a9c8'>{sub}</div></div>", unsafe_allow_html=True)

st.markdown("<div class='eyebrow'>COMPUTATIONAL CHEMISTRY</div><h1 class='hero'>DFT Studio</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Pick the cheapest level of theory that meets your accuracy target, "
            "export a ready-to-run input, and plan the compute.</div>", unsafe_allow_html=True)

if not LIVE_DFT:
    st.warning("REPLAY MODE · PySCF not found. Optimizer and bending scan run SciPy on a surface anchored "
               "to your real sweep_results.csv (H₂O only). Dipole, gap and runtime are your recorded DFT values.")

tabA, tab0, tab1, tab2, tab3, tab4 = st.tabs(["🎯 METHOD ADVISOR", "🌀 LIVE LAB", "⚡ LIVE OPTIMIZER", "📊 SWEEP LAB", "📐 BENDING CURVE", "🧰 SOFTWARE"])

# ------------------------------------------------------------------ live
with tab1:
    c = st.columns([1, 1, 1, 1, 1])
    mol = c[0].selectbox("Molecule", list(EXPT))
    fn = c[1].selectbox("Functional", ["lda,vwn", "pbe", "b3lyp"], index=2)
    bs = c[2].selectbox("Basis", ["sto-3g", "6-31g", "6-31g*", "cc-pvdz"], index=2)
    r0 = c[3].slider("Start r (Å)", 0.8, 1.6, 1.0, 0.01)
    t0 = c[4].slider("Start θ (°)", 80, 140, 100)

    if st.button("RUN DFT ⚛️"):
        with st.spinner("Converging SCF and minimizing with Nelder-Mead…"):
            st.session_state["live"] = run_one(mol, fn, bs, r0, t0)
    d = st.session_state.get("live")
    if d:
        ex = EXPT[d["molecule"]]
        k = st.columns(4)
        with k[0]: stat("Bond length", f"{d['r_A']:.3f} Å", f"expt {ex['r']} · {d['err_r_pct']:+.2f}%", abs(d["err_r_pct"]) < 1.5)
        with k[1]: stat("Bond angle", f"{d['theta_deg']:.1f}°", f"expt {ex['theta']} · {d['err_theta_deg']:+.2f}°", abs(d["err_theta_deg"]) < 2)
        with k[2]: stat("Dipole", f"{d['dipole_D']:.2f} D", f"expt {ex['dipole']} · {d['err_dipole_pct']:+.1f}%", abs(d["err_dipole_pct"]) < 10)
        with k[3]: stat("HOMO–LUMO", f"{d['gap_eV']:.2f} eV", f"E = {d['energy_Eh']:.5f} Eh · {d['time_s']:.0f}s")

        t = np.radians(d["theta_deg"]) / 2
        xs = [-d["r_A"] * np.sin(t), 0, d["r_A"] * np.sin(t)]
        ys = [d["r_A"] * np.cos(t), 0, d["r_A"] * np.cos(t)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#555", width=10), showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=[ex["ligand"], ex["center"], ex["ligand"]],
                                 textfont=dict(size=20, color="black"), showlegend=False,
                                 marker=dict(size=[50, 80, 50], color=[ICE, HOT, ICE], line=dict(width=3, color="white"))))
        fig.update_xaxes(visible=False, range=[-1.8, 1.8]); fig.update_yaxes(visible=False, scaleanchor="x", range=[-0.6, 1.5])
        st.plotly_chart(style(fig, 380), use_container_width=True)
    else:
        st.info("Pick a setup and hit RUN. Larger bases take 20–30 s.")

# ------------------------------------------------------------------ sweep
with tab2:
    up = st.file_uploader("Sweep CSV (defaults to sweep_results.csv)", type="csv")
    src = up if up else ("sweep_results.csv" if os.path.exists("sweep_results.csv") else None)
    if src is None:
        st.warning("Put sweep_results.csv next to this script or upload it.")
    else:
        df = pd.read_csv(src)
        order = ["sto-3g", "6-31g", "6-31g*", "cc-pvdz"]
        df["basis"] = pd.Categorical(df["basis"], order, ordered=True)
        df = df.sort_values(["functional", "basis"])
        df["abs_err_r"] = df["err_r_pct"].abs()
        df["abs_err_theta"] = df["err_theta_deg"].abs()

        best = df.loc[df["abs_err_r"].idxmin()]
        k = st.columns(3)
        with k[0]: stat("Best bond length", f"{best.abs_err_r:.2f}%", f"{best.functional} / {best.basis}", True)
        with k[1]: stat("Best angle", f"{df.abs_err_theta.min():.2f}°", f"{df.loc[df.abs_err_theta.idxmin(),'functional']} / {df.loc[df.abs_err_theta.idxmin(),'basis']}", True)
        with k[2]: stat("Cheapest run", f"{df.time_s.min():.0f}s", f"{df.loc[df.time_s.idxmin(),'functional']} / {df.loc[df.time_s.idxmin(),'basis']}")

        metric = st.radio("Metric", ["r_A", "theta_deg", "time_s", "dipole_D", "gap_eV", "energy_Eh"], horizontal=True)
        fig = px.line(df, x="basis", y=metric, color="functional", markers=True)
        fig.update_traces(line=dict(width=5), marker=dict(size=13))
        if metric in ("r_A", "theta_deg"):
            fig.add_hline(y=EXPT["H2O"]["r" if metric == "r_A" else "theta"], line_dash="dash",
                          line_color="white", annotation_text="experiment")
        st.plotly_chart(style(fig), use_container_width=True)

        a, b = st.columns(2)
        piv = df.pivot(index="functional", columns="basis", values="abs_err_r")
        hm = px.imshow(piv, text_auto=".2f", color_continuous_scale=[[0, ACCENT], [1, HOT]], aspect="auto",
                       title="|bond-length error| %")
        a.plotly_chart(style(hm, 330), use_container_width=True)

        res = stats.linregress(df["time_s"], df["abs_err_r"])
        r_p, p_p = stats.pearsonr(df["time_s"], df["abs_err_r"])
        sc = px.scatter(df, x="time_s", y="abs_err_r", color="functional", symbol="basis", size_max=18,
                        title=f"Cost vs accuracy · Pearson r = {r_p:.2f} (p={p_p:.2f})")
        xx = np.linspace(df.time_s.min(), df.time_s.max(), 50)
        sc.add_trace(go.Scatter(x=xx, y=res.intercept + res.slope * xx, mode="lines",
                                line=dict(dash="dot", color="white"), name="linear fit"))
        sc.update_traces(marker=dict(size=14), selector=dict(mode="markers"))
        b.plotly_chart(style(sc, 330), use_container_width=True)
        st.caption("Reading it: error drops sharply from sto-3g to 6-31g* then plateaus, "
                   "while runtime keeps climbing. Biggest basis ≠ best value.")
        with st.expander("Raw data"):
            st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------------ bending
with tab3:
    c = st.columns(5)
    mol_b = c[0].selectbox("Molecule ", list(EXPT), key="mb")
    fn_b = c[1].selectbox("Functional ", ["lda,vwn", "pbe", "b3lyp"], index=2, key="fb")
    bs_b = c[2].selectbox("Basis ", ["sto-3g", "6-31g", "6-31g*"], index=1, key="bb")
    lo, hi = c[3].slider("Angle range (°)", 60, 180, (70, 180))
    n = c[4].slider("Points", 6, 20, 12)

    if st.button("SCAN THE BEND 📐"):
        with st.spinner("Optimizing r, then scanning θ…"):
            r_opt, _, _ = optimize(mol_b, fn_b, bs_b)
            ang = np.linspace(lo, hi, n)
            E = np.array([scf_energy(mol_b, r_opt, a, fn_b, bs_b) for a in ang])
            st.session_state["bend"] = (mol_b, ang, (E - E.min()) * 627.509, r_opt)
    if "bend" in st.session_state:
        m, ang, E, r_opt = st.session_state["bend"]
        cs = CubicSpline(ang, E)
        mn = minimize_scalar(cs, bounds=(ang.min(), ang.max()), method="bounded")
        dense = np.linspace(ang.min(), ang.max(), 400)
        # harmonic fit near the minimum
        win = np.abs(ang - mn.x) <= 25
        p = np.polyfit(ang[win], E[win], 2) if win.sum() >= 3 else None
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dense, y=cs(dense), mode="lines", line=dict(width=5, color=ICE), name="cubic spline"))
        fig.add_trace(go.Scatter(x=ang, y=E, mode="markers", marker=dict(size=13, color=ACCENT), name="DFT points"))
        if p is not None:
            fig.add_trace(go.Scatter(x=dense, y=np.polyval(p, dense), mode="lines",
                                     line=dict(dash="dot", color=HOT), name="harmonic fit"))
        fig.add_vline(x=EXPT[m]["theta"], line_dash="dash", line_color="white", annotation_text="experiment")
        fig.update_yaxes(range=[-1, min(E.max() * 1.1, 60)], title="Relative energy (kcal/mol)")
        fig.update_xaxes(title="H–X–H angle (deg)")
        st.plotly_chart(style(fig, 480), use_container_width=True)
        k = st.columns(3)
        with k[0]: stat("Spline minimum", f"{mn.x:.1f}°", f"expt {EXPT[m]['theta']}°")
        with k[1]: stat("Barrier to linear", f"{E[-1] - E.min():.1f}", "kcal/mol at max angle")
        with k[2]:
            if p is not None:
                stat("Curvature", f"{2 * p[0]:.4f}", "kcal/mol/deg² (stiffness of the bend)")
    else:
        st.info("Scan the potential energy surface and let SciPy find the true minimum.")

# ------------------------------------------------------------------ software
with tab4:
    st.markdown("### The software chemists actually open")
    tools = [("Gaussian", "PAID", HOT, "Descendant of Pople's original program. Still the most widely cited quantum chemistry software in the world."),
             ("ORCA", "FREE · ACADEMIC", "#ffb300", "Full-featured program from the Max Planck Institute, free for academic use."),
             ("Psi4", "OPEN SOURCE", ACCENT, "Open-source, scriptable from Python. Popular for teaching and building new methods."),
             ("NWChem", "OPEN SOURCE", ACCENT, "Built at Pacific Northwest National Lab to scale to huge supercomputers."),
             ("PySCF", "OPEN SOURCE · THIS APP", ICE, "Python-native DFT engine. It powers every number you just saw.")]
    cols = st.columns(3)
    for i, (nm, tg, col, tx) in enumerate(tools):
        with cols[i % 3]:
            st.markdown(f"<div class='card' style='border-color:{col}'><h3>{nm}<span class='tag' style='color:{col};border-color:{col}'>{tg}</span></h3>"
                        f"<div style='color:#c9c9e0'>{tx}</div></div><br>", unsafe_allow_html=True)





# ================================================================== LIVE LAB (three.js, client-side)
import json
import streamlit.components.v1 as components
from types import SimpleNamespace

_VCSV = "sweep_results.csv"
_VREF = pd.read_csv(_VCSV) if os.path.exists(_VCSV) else None

def _anchor(fn, bs):
    """Real DFT minimum for H2O at (fn, bs) from the sweep CSV; sensible default otherwise."""
    if _VREF is not None:
        m = _VREF[(_VREF.functional == fn) & (_VREF.basis == bs) & (_VREF.molecule == "H2O")]
        if len(m):
            return m.iloc[0]
    return SimpleNamespace(r_A=0.969, theta_deg=103.9, dipole_D=2.1, gap_eV=9.0, time_s=20.0)

_FUNCS = ["lda,vwn", "pbe", "b3lyp"]
_BASES = ["sto-3g", "6-31g", "6-31g*", "cc-pvdz"]

_LAB = r"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
*{box-sizing:border-box}html,body{margin:0;background:transparent;font-family:Inter,sans-serif;color:#e2e8f0}
.ctl{display:grid;grid-template-columns:1fr 1fr 2fr 2fr auto;gap:18px;align-items:end;margin-bottom:18px}
label{display:block;font-size:.7rem;letter-spacing:2.2px;text-transform:uppercase;color:#94a3b8;font-weight:600;margin-bottom:9px}
label b{float:right;color:#5eead4;font-size:.85rem;letter-spacing:0}
select{width:100%;padding:10px 12px;border-radius:12px;background:rgba(255,255,255,.06);color:#e2e8f0;border:1px solid rgba(255,255,255,.12);font:inherit}
select option{background:#14122e}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:99px;outline:none;
 background:linear-gradient(90deg,#5eead4 var(--p),rgba(255,255,255,.12) var(--p))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;background:#fff;border:5px solid #5eead4;
 box-shadow:0 0 0 6px rgba(94,234,212,.18),0 0 18px rgba(94,234,212,.6);cursor:grab}
input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:#fff;border:5px solid #5eead4;cursor:grab}
button{background:linear-gradient(90deg,#5eead4,#818cf8);color:#08111a;font:600 .95rem Inter;border:none;border-radius:14px;padding:11px 20px;
 cursor:pointer;box-shadow:0 6px 24px rgba(94,234,212,.25);transition:.2s}
button:hover{transform:translateY(-2px);box-shadow:0 10px 30px rgba(129,140,248,.45)}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:18px}
.card,.panel{position:relative;border:1px solid rgba(255,255,255,.09);border-radius:24px;box-shadow:0 14px 50px rgba(0,0,0,.4);
 background:linear-gradient(160deg,rgba(255,255,255,.08),rgba(255,255,255,.015))}
.card{padding:16px 22px;overflow:hidden}.card:before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--a,#818cf8)}
.lbl{color:#94a3b8;font-size:.68rem;text-transform:uppercase;letter-spacing:2.5px;font-weight:600;margin-bottom:6px}
.big{font:700 2.1rem 'Space Grotesk',sans-serif;line-height:1.1;transition:color .3s}.sm{color:#94a3b8;font-size:.82rem;margin-top:2px}
.good{color:#5eead4}.bad{color:#fb7185}
.row{display:grid;gap:18px;margin-bottom:18px}.r1{grid-template-columns:1fr 1fr}.r2{grid-template-columns:1fr 1fr}
.panel{overflow:hidden;background:radial-gradient(circle at 50% 45%,rgba(99,102,241,.28),rgba(10,10,24,.55) 70%)}
.gl{position:absolute;inset:0}.gl canvas{display:block}
.chip{position:absolute;left:18px;top:16px;z-index:2;font:600 .72rem Inter;letter-spacing:2.5px;text-transform:uppercase;color:#cbd5e1;
 padding:6px 14px;border-radius:99px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);backdrop-filter:blur(8px)}
.hint{position:absolute;right:18px;top:20px;z-index:2;font-size:.78rem;color:#64748b}
.leg{position:absolute;left:18px;bottom:14px;z-index:2;display:flex;gap:16px;font-size:.8rem;color:#94a3b8}
.leg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
canvas.sl{width:100%;height:100%;display:block}.cap{color:#64748b;font-size:.82rem}
</style>
<div class="ctl">
 <div><label>Functional</label><select id="fn"></select></div>
 <div><label>Basis</label><select id="bs"></select></div>
 <div><label>O–H bond length <b id="rv"></b></label><input id="R" type="range" min="0.8" max="1.3" step="0.005"></div>
 <div><label>H–O–H angle <b id="tv"></b></label><input id="T" type="range" min="60" max="180" step="1"></div>
 <button id="snap">Snap to minimum ✦</button>
</div>
<div class="cards">
 <div class="card" style="--a:#5eead4"><div class="lbl">Energy above minimum</div><div class="big" id="c1"></div><div class="sm">kcal/mol</div></div>
 <div class="card" style="--a:#818cf8"><div class="lbl">Δ bond vs expt</div><div class="big" id="c2"></div><div class="sm">experiment 0.957 Å</div></div>
 <div class="card" style="--a:#fb7185"><div class="lbl">Δ angle vs expt</div><div class="big" id="c3"></div><div class="sm">experiment 104.5°</div></div>
 <div class="card" style="--a:#fbbf24"><div class="lbl">Bend stiffness</div><div class="big" id="c4" style="color:#e2e8f0"></div><div class="sm">kcal/mol/rad² at the minimum</div></div>
</div>
<div class="row r1">
 <div class="panel" style="height:560px"><div class="chip">Molecule · 3D</div><div class="hint">drag · scroll · it drifts when idle</div><div class="gl" id="mol"></div></div>
 <div class="panel" style="height:560px"><div class="chip">Energy surface</div><div class="hint">drag to rotate · scroll to zoom</div><div class="gl" id="pes"></div>
  <div class="leg"><span><i style="background:#fff;box-shadow:0 0 8px #fb7185"></i>you</span><span><i style="background:#fbbf24;border-radius:2px"></i>DFT minimum</span><span><i style="background:transparent;border:2px solid #fb7185"></i>experiment</span><span><i style="background:#fff;width:18px;height:3px;border-radius:2px"></i>descent</span></div></div>
</div>
<div class="row r2"><div class="panel" style="height:290px"><canvas class="sl" id="s1"></canvas></div><div class="panel" style="height:290px"><canvas class="sl" id="s2"></canvas></div></div>
<div class="cap">Computed live in your browser from your real DFT minimum for the chosen functional and basis; the bend term is fitted to your bending curve. Lone-pair lobes are schematic.</div>
<script>
if(typeof THREE==='undefined'||!THREE.OrbitControls){document.body.innerHTML='<p style="color:#fb7185">three.js could not load (needs an internet connection).</p>';throw new Error('no three');}
const D=__DATA__,$=id=>document.getElementById(id),fnS=$('fn'),bsS=$('bs'),R=$('R'),T=$('T'),PI=Math.PI;
D.funcs.forEach(f=>fnS.add(new Option(f,f)));D.bases.forEach(b=>bsS.add(new Option(b,b)));fnS.value='b3lyp';bsS.value='6-31g*';
const a0=D.a['b3lyp']['6-31g*'];R.value=(Math.round((a0.r+.09)/.005)*.005).toFixed(3);T.value=Math.min(180,Math.round(a0.th+22));
const E=(r,th,a)=>2*120*Math.pow(1-Math.exp(-2.2*(r-a.r)),2)+61*Math.pow(Math.cos(th*PI/180)-Math.cos(a.th*PI/180),2);
function nm(f,x0){let S=[x0.slice(),[x0[0]+.05,x0[1]],[x0[0],x0[1]+5]],F=S.map(f),path=[x0.slice()];
 for(let k=0;k<120;k++){const o=[0,1,2].sort((a,b)=>F[a]-F[b]);S=o.map(i=>S[i]);F=o.map(i=>F[i]);path.push(S[0].slice());if(Math.abs(F[2]-F[0])<1e-6)break;
  const c=[(S[0][0]+S[1][0])/2,(S[0][1]+S[1][1])/2],Rf=[2*c[0]-S[2][0],2*c[1]-S[2][1]],fr=f(Rf);
  if(fr<F[0]){const Ex=[3*c[0]-2*S[2][0],3*c[1]-2*S[2][1]],fe=f(Ex);if(fe<fr){S[2]=Ex;F[2]=fe}else{S[2]=Rf;F[2]=fr}}
  else if(fr<F[1]){S[2]=Rf;F[2]=fr}
  else{const Cc=fr<F[2]?[(c[0]+Rf[0])/2,(c[1]+Rf[1])/2]:[(c[0]+S[2][0])/2,(c[1]+S[2][1])/2],fc=f(Cc);
   if(fc<Math.min(fr,F[2])){S[2]=Cc;F[2]=fc}else{for(let i=1;i<3;i++){S[i]=[(S[0][0]+S[i][0])/2,(S[0][1]+S[i][1])/2];F[i]=f(S[i]);}}}}
 path.push(S[0].slice());return path;}
/* ---------- three helpers ---------- */
const hex=n=>'#'+n.toString(16).padStart(6,'0'),V=(x,y,z)=>new THREE.Vector3(x,y,z),hc={};
function halo(col,size,op=.55){if(!hc[col]){const c=document.createElement('canvas');c.width=c.height=128;const g=c.getContext('2d'),gr=g.createRadialGradient(64,64,0,64,64,64);
 gr.addColorStop(0,col+'ff');gr.addColorStop(.3,col+'66');gr.addColorStop(1,col+'00');g.fillStyle=gr;g.fillRect(0,0,128,128);hc[col]=new THREE.CanvasTexture(c);}
 const s=new THREE.Sprite(new THREE.SpriteMaterial({map:hc[col],blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:op}));s.scale.set(size,size,1);return s;}
function label(txt,col,px=34){const c=document.createElement('canvas');c.width=256;c.height=96;const t=new THREE.CanvasTexture(c);
 const s=new THREE.Sprite(new THREE.SpriteMaterial({map:t,transparent:true,depthTest:false}));s.scale.set(1,.375,1);s.userData={c,t,col,px,txt:''};setLabel(s,txt);return s;}
function setLabel(s,txt){if(s.userData.txt===txt)return;const{c,t,col,px}=s.userData,g=c.getContext('2d');s.userData.txt=txt;g.clearRect(0,0,256,96);
 g.font='600 '+px+'px Inter, sans-serif';g.textAlign='center';g.textBaseline='middle';g.shadowColor='rgba(0,0,0,.85)';g.shadowBlur=8;g.fillStyle=col;g.fillText(txt,128,48);t.needsUpdate=true;}
function mk(el,auto){const r=new THREE.WebGLRenderer({antialias:true,alpha:true});r.setPixelRatio(Math.min(devicePixelRatio||1,2));el.appendChild(r.domElement);
 const s=new THREE.Scene(),cam=new THREE.PerspectiveCamera(40,1,.1,100),ctl=new THREE.OrbitControls(cam,r.domElement);ctl.enableDamping=true;ctl.dampingFactor=.07;
 if(auto){ctl.autoRotate=true;ctl.autoRotateSpeed=1.1;let tm;ctl.addEventListener('start',()=>{ctl.autoRotate=false;clearTimeout(tm)});
  ctl.addEventListener('end',()=>{tm=setTimeout(()=>ctl.autoRotate=true,3500)});}
 const fit=()=>{const w=el.clientWidth,h=el.clientHeight;r.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix();};new ResizeObserver(fit).observe(el);fit();
 s.add(new THREE.HemisphereLight(0xc7d2fe,0x1e1b4b,.95));const k=new THREE.DirectionalLight(0xffffff,1.1);k.position.set(3,5,4);s.add(k);
 const p1=new THREE.PointLight(0x5eead4,1.3,14);p1.position.set(-4,2,3);s.add(p1);const p2=new THREE.PointLight(0xfb7185,1.1,14);p2.position.set(4,-1,4);s.add(p2);
 return{r,s,cam,ctl};}
function stars(s){const n=360,p=new Float32Array(n*3);for(let i=0;i<n;i++){const u=Math.random()*2*PI,v=Math.acos(2*Math.random()-1),d=9+Math.random()*5;
 p.set([d*Math.sin(v)*Math.cos(u),d*Math.cos(v),d*Math.sin(v)*Math.sin(u)],i*3);}
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(p,3));
 s.add(new THREE.Points(g,new THREE.PointsMaterial({color:0xc7d2fe,size:.04,transparent:true,opacity:.55,depthWrite:false})));}
function cloud(sig,col,n=420){const p=new Float32Array(n*3),g=new THREE.BufferGeometry();
 for(let i=0;i<n*3;i++){p[i]=(Math.random()+Math.random()+Math.random()-1.5)*sig*2.4;}g.setAttribute('position',new THREE.BufferAttribute(p,3));
 return new THREE.Points(g,new THREE.PointsMaterial({color:col,size:.03,transparent:true,opacity:.75,blending:THREE.AdditiveBlending,depthWrite:false}));}
function atom(rad,col,emi){const m=new THREE.Mesh(new THREE.SphereGeometry(rad,56,56),new THREE.MeshStandardMaterial({color:col,roughness:.2,metalness:.12,emissive:emi,emissiveIntensity:.4}));
 m.add(halo(hex(col),rad*6));m.add(cloud(rad*1.1,col));return m;}
const YUP=V(0,1,0),ZUP=V(0,0,1);
/* ---------- molecule scene ---------- */
const M=mk($('mol'),true);M.cam.position.set(2.5,1.4,4.3);M.ctl.target.set(0,.4,0);M.ctl.minDistance=2.5;M.ctl.maxDistance=9;stars(M.s);
const O=atom(.34,0xfb7185,0x7f1d3a),H1=atom(.22,0xc7d2fe,0x3730a3),H2=atom(.22,0xc7d2fe,0x3730a3);M.s.add(O,H1,H2);
const bondMat=new THREE.MeshStandardMaterial({color:0xe0e7ff,roughness:.18,metalness:.2,emissive:0x5eead4,emissiveIntensity:.35,transparent:true,opacity:.92});
const B1=new THREE.Mesh(new THREE.CylinderGeometry(.06,.06,1,24),bondMat),B2=B1.clone();M.s.add(B1,B2);
function bond(m,a,b){const d=b.clone().sub(a),l=d.length();m.position.copy(a).add(b).multiplyScalar(.5);m.scale.set(1,l,1);m.quaternion.setFromUnitVectors(YUP,d.normalize());}
[1,-1].forEach(sg=>{const dir=V(0,-Math.cos(54.75*PI/180),sg*Math.sin(54.75*PI/180));
 for(const [sc,op] of [[1,.34],[1.5,.1]]){const l=new THREE.Mesh(new THREE.SphereGeometry(1,40,40),new THREE.MeshBasicMaterial({color:0xfbbf24,transparent:true,opacity:op,
  blending:THREE.AdditiveBlending,depthWrite:false}));l.scale.set(.2*sc,.2*sc,.42*sc);l.quaternion.setFromUnitVectors(ZUP,dir);l.position.copy(dir).multiplyScalar(.52);M.s.add(l);}});
const triG=new THREE.BufferGeometry();triG.setAttribute('position',new THREE.BufferAttribute(new Float32Array(9),3));
M.s.add(new THREE.Mesh(triG,new THREE.MeshBasicMaterial({color:0x5eead4,transparent:true,opacity:.09,side:THREE.DoubleSide,depthWrite:false})));
const ring=new THREE.Mesh(new THREE.RingGeometry(1.55,1.58,128),new THREE.MeshBasicMaterial({color:0x5eead4,transparent:true,opacity:.4,side:THREE.DoubleSide}));
ring.rotation.x=-PI/2;ring.position.y=-.95;M.s.add(ring);
const disc=new THREE.Mesh(new THREE.CircleGeometry(1.55,96),new THREE.MeshBasicMaterial({color:0x818cf8,transparent:true,opacity:.07,side:THREE.DoubleSide}));
disc.rotation.x=-PI/2;disc.position.y=-.95;M.s.add(disc);
const arrow=new THREE.Group(),amb=new THREE.MeshStandardMaterial({color:0xfbbf24,emissive:0xfbbf24,emissiveIntensity:.6,roughness:.3});
const shaft=new THREE.Mesh(new THREE.CylinderGeometry(.035,.035,1,16),amb),head=new THREE.Mesh(new THREE.ConeGeometry(.1,.24,24),amb);arrow.add(shaft,head,halo('#fbbf24',.9,.35));M.s.add(arrow);
const L1=label('', '#5eead4',32),L2=label('','#5eead4',32),LA=label('','#ffffff',40),LD=label('','#fbbf24',34);M.s.add(L1,L2,LA,LD);
let arcM=null,lastArc=-1;
function placeMol(r,th,dip){const t=th*PI/360,hx_=r*Math.sin(t),hy_=r*Math.cos(t);H1.position.set(-hx_,hy_,0);H2.position.set(hx_,hy_,0);
 bond(B1,O.position,H1.position);bond(B2,O.position,H2.position);
 triG.attributes.position.set([0,0,0,-hx_,hy_,0,hx_,hy_,0]);triG.attributes.position.needsUpdate=true;
 L1.position.set(-hx_*.5-.3,hy_*.5+.1,.2);L2.position.set(hx_*.5+.3,hy_*.5+.1,.2);setLabel(L1,r.toFixed(3)+' Å');setLabel(L2,r.toFixed(3)+' Å');
 if(Math.abs(th-lastArc)>.01){lastArc=th;if(arcM){M.s.remove(arcM);arcM.geometry.dispose();}const pts=[];
  for(let i=0;i<=36;i++){const a=(90-th/2+th*i/36)*PI/180;pts.push(V(.46*Math.cos(a),.46*Math.sin(a),0));}
  arcM=new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts),48,.016,8),new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.85}));M.s.add(arcM);}
 LA.position.set(0,.46+.24,.0);setLabel(LA,th.toFixed(1)+'°');
 const len=dip*.3,y0=Math.max(hy_,.5)+.35;arrow.position.set(0,y0,0);shaft.scale.y=len;shaft.position.y=len/2;head.position.y=len+.12;
 LD.position.set(.55,y0+len*.5,0);setLabel(LD,'μ '+dip.toFixed(2)+' D');}
/* ---------- surface scene ---------- */
const N=90,sx=r=>(r-.8)/.5*4-2,sz=th=>(th-60)/120*4-2,sy=e=>Math.min(e,90)/90*2.2;
const S=mk($('pes'),false);S.cam.position.set(5.3,4.1,5.6);S.ctl.target.set(0,.7,0);S.ctl.minDistance=4;S.ctl.maxDistance=16;S.ctl.maxPolarAngle=PI*.495;stars(S.s);
const sg=new THREE.PlaneGeometry(4,4,N-1,N-1),pos=sg.attributes.position.array,col=new Float32Array(N*N*3),TY=new Float32Array(N*N);
for(let i=0;i<N;i++)for(let j=0;j<N;j++){const k=i*N+j;pos[3*k]=-2+4*j/(N-1);pos[3*k+2]=-2+4*i/(N-1);}
sg.setAttribute('color',new THREE.BufferAttribute(col,3));
const surfM=new THREE.Mesh(sg,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.32,metalness:.15,side:THREE.DoubleSide,emissive:0x1e1b4b,emissiveIntensity:.35}));S.s.add(surfM);
S.s.add(new THREE.Mesh(sg,new THREE.MeshBasicMaterial({color:0x5eead4,wireframe:true,transparent:true,opacity:.055,depthWrite:false})));
const grid=new THREE.GridHelper(4.6,12,0x5eead4,0x3b3b78);grid.material.transparent=true;grid.material.opacity=.35;grid.position.y=-.01;S.s.add(grid);
const stops=[[0,'#99f6e4'],[.08,'#2dd4bf'],[.25,'#6366f1'],[.55,'#4c1d95'],[1,'#312e81']].map(([t,c])=>[t,new THREE.Color(c)]),tc=new THREE.Color();
function ramp(t){for(let i=1;i<stops.length;i++)if(t<=stops[i][0]){const a=stops[i-1],b=stops[i];return tc.copy(a[1]).lerp(b[1],(t-a[0])/(b[0]-a[0]));}return tc.copy(stops[4][1]);}
let moving=true,anchor=a0,key='';
function setTargets(a){for(let i=0;i<N;i++)for(let j=0;j<N;j++)TY[i*N+j]=sy(E(.8+.5*j/(N-1),60+120*i/(N-1),a));moving=true;}
function stepSurf(init){let md=0;for(let k=0;k<N*N;k++){const y=pos[3*k+1],d=TY[k]-y;pos[3*k+1]=init?TY[k]:y+d*.18;md=Math.max(md,Math.abs(d));
 ramp(pos[3*k+1]/2.2);col[3*k]=tc.r;col[3*k+1]=tc.g;col[3*k+2]=tc.b;}
 sg.attributes.position.needsUpdate=true;sg.attributes.color.needsUpdate=true;sg.computeVertexNormals();if(md<1e-3)moving=false;}
function axisLine(a,b,c=0x64748b){const g=new THREE.BufferGeometry().setFromPoints([a,b]);S.s.add(new THREE.Line(g,new THREE.LineBasicMaterial({color:c,transparent:true,opacity:.6})));}
axisLine(V(-2.3,0,2.3),V(2.3,0,2.3));axisLine(V(2.3,0,-2.3),V(2.3,0,2.3));axisLine(V(-2.3,0,2.3),V(-2.3,2.4,2.3));
[[.8,'0.8'],[1.05,'1.05'],[1.3,'1.30']].forEach(([v,t])=>{const l=label(t,'#94a3b8',26);l.position.set(sx(v),-.12,2.65);l.scale.set(.7,.26,1);S.s.add(l);});
[[60,'60°'],[120,'120°'],[180,'180°']].forEach(([v,t])=>{const l=label(t,'#94a3b8',26);l.position.set(2.75,-.12,sz(v));l.scale.set(.7,.26,1);S.s.add(l);});
[[0,'0'],[45,'45'],[90,'90']].forEach(([v,t])=>{const l=label(t,'#94a3b8',26);l.position.set(-2.65,sy(v),2.3);l.scale.set(.7,.26,1);S.s.add(l);});
const lr=label('r  (Å)','#5eead4',34),lt=label('θ  (°)','#818cf8',34),le=label('kcal/mol','#fbbf24',30);lr.position.set(0,-.3,3.2);lt.position.set(3.35,-.3,0);le.position.set(-2.65,2.65,2.3);
[lr,lt,le].forEach(l=>{l.scale.set(1.2,.45,1);S.s.add(l);});
const ball=new THREE.Mesh(new THREE.SphereGeometry(.11,40,40),new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xfb7185,emissiveIntensity:.8,roughness:.2}));
ball.add(halo('#fb7185',1.0,.8));const bl=new THREE.PointLight(0xfb7185,1.2,2.2);ball.add(bl);S.s.add(ball);
const dropG=new THREE.BufferGeometry().setFromPoints([V(),V()]),drop=new THREE.Line(dropG,new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:.45}));S.s.add(drop);
const minM=new THREE.Mesh(new THREE.OctahedronGeometry(.13),new THREE.MeshStandardMaterial({color:0xfbbf24,emissive:0xfbbf24,emissiveIntensity:.7,roughness:.25}));minM.add(halo('#fbbf24',.9,.6));S.s.add(minM);
const expM=new THREE.Mesh(new THREE.TorusGeometry(.1,.022,16,40),new THREE.MeshBasicMaterial({color:0xfb7185}));expM.rotation.x=PI/2;S.s.add(expM);
const pathG=new THREE.Group();S.s.add(pathG);
function buildPath(a,r,th){while(pathG.children.length){const c=pathG.children.pop();c.geometry&&c.geometry.dispose();}
 const P=nm(p=>E(p[0],p[1],a),[r,th]).map(p=>V(sx(p[0]),sy(E(p[0],p[1],a))+.08,sz(p[1]))),pts=[P[0]];
 for(const p of P)if(p.distanceTo(pts[pts.length-1])>.03)pts.push(p);
 if(pts.length>=2)pathG.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts),pts.length*6,.022,8),new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.9})));
 const g=new THREE.BufferGeometry().setFromPoints(pts);pathG.add(new THREE.Points(g,new THREE.PointsMaterial({color:0xffffff,size:.07,sizeAttenuation:true,transparent:true,opacity:.9})));}
/* ---------- slices (canvas 2D) ---------- */
const sl={};
function drawSlice(id){const q=sl[id];if(!q)return;const cv=$(id),dpr=Math.min(devicePixelRatio||1,2),w=cv.clientWidth,h=cv.clientHeight;
 if(cv.width!==Math.round(w*dpr)){cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);}
 const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
 const L=60,Rr=24,Tt=60,B=48,X=v=>L+(v-q.x0)/(q.x1-q.x0)*(w-L-Rr),Y=v=>h-B-Math.min(v,95)/95*(h-Tt-B);
 g.font='12px Inter,sans-serif';g.fillStyle='#94a3b8';g.strokeStyle='rgba(255,255,255,.07)';g.lineWidth=1;g.textAlign='right';
 for(let k=0;k<=4;k++){const v=k*95/4,y=Y(v);g.beginPath();g.moveTo(L,y);g.lineTo(w-Rr,y);g.stroke();g.fillText(v.toFixed(0),L-10,y+4);}
 g.textAlign='center';for(let k=0;k<=4;k++){const v=q.x0+(q.x1-q.x0)*k/4;g.fillText(q.dec?v.toFixed(2):v.toFixed(0),X(v),h-B+20);}
 g.fillText(q.xl,(L+w-Rr)/2,h-10);g.save();g.translate(16,(Tt+h-B)/2);g.rotate(-PI/2);g.fillText('kcal/mol',0,0);g.restore();
 const gr=g.createLinearGradient(0,Tt,0,h-B);gr.addColorStop(0,q.col+'66');gr.addColorStop(1,q.col+'00');
 g.beginPath();q.xs.forEach((x,i)=>{i?g.lineTo(X(x),Y(q.ys[i])):g.moveTo(X(x),Y(q.ys[i]));});
 g.lineTo(X(q.x1),h-B);g.lineTo(X(q.x0),h-B);g.closePath();g.fillStyle=gr;g.fill();
 g.beginPath();q.xs.forEach((x,i)=>{i?g.lineTo(X(x),Y(q.ys[i])):g.moveTo(X(x),Y(q.ys[i]));});
 g.shadowColor=q.col;g.shadowBlur=14;g.strokeStyle=q.col;g.lineWidth=3.5;g.lineJoin='round';g.stroke();g.shadowBlur=0;
 const px=X(q.px),py=Y(q.py);g.setLineDash([4,5]);g.strokeStyle='rgba(255,255,255,.35)';g.lineWidth=1.5;g.beginPath();g.moveTo(px,py);g.lineTo(px,h-B);g.stroke();g.setLineDash([]);
 const gg=g.createRadialGradient(px,py,0,px,py,22);gg.addColorStop(0,'rgba(255,255,255,.85)');gg.addColorStop(1,'rgba(255,255,255,0)');g.fillStyle=gg;g.beginPath();g.arc(px,py,22,0,7);g.fill();
 g.fillStyle='#fff';g.strokeStyle=q.col;g.lineWidth=4;g.beginPath();g.arc(px,py,7,0,7);g.fill();g.stroke();
 g.textAlign='left';g.font='600 16px Space Grotesk,sans-serif';g.fillStyle='#e2e8f0';g.fillText(q.title,22,34);}
['s1','s2'].forEach(id=>new ResizeObserver(()=>drawSlice(id)).observe($(id)));
/* ---------- state / update ---------- */
const cur={r:+R.value,th:+T.value},tgt={r:cur.r,th:cur.th};let dip=a0.dip;
function setP(el,inp,v){inp.style.setProperty('--p',((inp.value-inp.min)/(inp.max-inp.min)*100)+'%');el.textContent=v;}
function update(){const fn=fnS.value,bs=bsS.value,a=D.a[fn][bs],r=+R.value,th=+T.value,e=E(r,th,a);
 if(key!==fn+bs){key=fn+bs;anchor=a;dip=a.dip;setTargets(a);minM.position.set(sx(a.r),.13,sz(a.th));expM.position.set(sx(D.ex.r),sy(E(D.ex.r,D.ex.th,a))+.1,sz(D.ex.th));}
 tgt.r=r;tgt.th=th;setP($('rv'),R,r.toFixed(3)+' Å');setP($('tv'),T,th+'°');
 const dr=r-D.ex.r,dt=th-D.ex.th,kb=2*61*Math.pow(Math.sin(a.th*PI/180),2);
 const set=(id,t,g)=>{const el=$(id);el.textContent=t;el.className='big '+(g?'good':'bad');};
 set('c1',e.toFixed(1),e<3);set('c2',(dr>=0?'+':'')+dr.toFixed(3)+' Å',Math.abs(dr)<.015);set('c3',(dt>=0?'+':'')+dt.toFixed(1)+'°',Math.abs(dt)<2);$('c4').textContent=kb.toFixed(0);
 buildPath(a,r,th);
 const tl=[],rl=[];for(let i=0;i<=120;i++){tl.push(60+i);rl.push(.8+.5*i/120);}
 sl.s1={xs:tl,ys:tl.map(x=>E(r,x,a)),px:th,py:e,col:'#818cf8',x0:60,x1:180,xl:'H–O–H angle (°)',chip:'',title:'Bend slice · r = '+r.toFixed(3)+' Å'};
 sl.s2={xs:rl,ys:rl.map(x=>E(x,th,a)),px:r,py:e,col:'#fb7185',x0:.8,x1:1.3,dec:1,xl:'O–H bond length (Å)',chip:'',title:'Stretch slice · θ = '+th+'°'};
 drawSlice('s1');drawSlice('s2');}
let pend=false;const sched=()=>{if(!pend){pend=true;requestAnimationFrame(()=>{pend=false;update();});}};
[fnS,bsS].forEach(e=>e.addEventListener('change',sched));[R,T].forEach(e=>e.addEventListener('input',sched));
$('snap').onclick=()=>{const a=D.a[fnS.value][bsS.value];R.value=(Math.round(a.r/.005)*.005).toFixed(3);T.value=Math.round(a.th);sched();};
update();stepSurf(true);placeMol(cur.r,cur.th,dip);
let tm0=0;
function frame(t){requestAnimationFrame(frame);
 cur.r+=(tgt.r-cur.r)*.3;cur.th+=(tgt.th-cur.th)*.3;placeMol(cur.r,cur.th,dip);
 if(moving)stepSurf(false);
 const bx=sx(cur.r),bz=sz(cur.th),by=sy(E(cur.r,cur.th,anchor))+.11;ball.position.set(bx,by,bz);
 dropG.attributes.position.set([bx,by,bz,bx,0,bz]);dropG.attributes.position.needsUpdate=true;minM.rotation.y=t*.002;ball.scale.setScalar(1+.08*Math.sin(t*.005));
 M.ctl.update();S.ctl.update();M.r.render(M.s,M.cam);S.r.render(S.s,S.cam);}
requestAnimationFrame(frame);
</script>
"""

_data = dict(funcs=_FUNCS, bases=_BASES, ex=dict(r=0.9572, th=104.52),
             a={f: {b: dict(r=float(_anchor(f, b).r_A), th=float(_anchor(f, b).theta_deg),
                            dip=float(_anchor(f, b).dipole_D), gap=float(_anchor(f, b).gap_eV)) for b in _BASES} for f in _FUNCS})

with tab0:
    components.html(_LAB.replace("__DATA__", json.dumps(_data)), height=1230, scrolling=False)


# ================================================================== METHOD ADVISOR
_MOLS = {"H2O": ("O", "H"), "H2S": ("S", "H")}
# functional -> (gaussian, orca, psi4, nwchem xc, pyscf)
_XC = {"lda,vwn": ("SVWN5", "LDA", "svwn", "slater vwn_5", "lda,vwn"),
       "pbe": ("PBEPBE", "PBE", "pbe", "xpbe96 cpbe96", "pbe"),
       "b3lyp": ("B3LYP", "B3LYP", "b3lyp", "b3lyp", "b3lyp")}
# basis -> (gaussian, orca, psi4, nwchem)
_BS = {"sto-3g": ("STO-3G",) * 4, "6-31g": ("6-31G",) * 4,
       "6-31g*": ("6-31G(d)", "6-31G*", "6-31G*", "6-31G*"), "cc-pvdz": ("cc-pVDZ",) * 4}

def _atoms(mol, r, th):
    c, l = _MOLS.get(mol, ("O", "H")); t = np.radians(th) / 2
    xyz = [(c, 0.0, 0.0, 0.0), (l, r * np.sin(t), r * np.cos(t), 0.0), (l, -r * np.sin(t), r * np.cos(t), 0.0)]
    return "\n".join(f"{s:<2} {x:12.6f} {y:12.6f} {z:12.6f}" for s, x, y, z in xyz)

def _deck(prog, mol, fn, bs, r, th):
    """Input file for `prog` using the method's own optimized geometry as the starting point."""
    xc = _XC.get(fn, (fn.upper(), fn, fn, fn, fn)); b = _BS.get(bs, (bs.upper(), bs, bs, bs))
    at, m = _atoms(mol, r, th), mol.lower()
    if prog == "Gaussian":
        return f"%chk={m}.chk\n%nprocshared=4\n%mem=4GB\n#p {xc[0]}/{b[0]} opt freq\n\n{mol} {fn}/{bs}\n\n0 1\n{at}\n\n"
    if prog == "ORCA":
        return f"! {xc[1]} {b[1]} Opt Freq TightSCF\n%pal nprocs 4 end\n* xyz 0 1\n{at}\n*\n"
    if prog == "Psi4":
        return (f"molecule {m} {{\n0 1\n{at}\n}}\n\nset basis {b[2]}\nset reference rks\n"
                f"energy, wfn = optimize('{xc[2]}', return_wfn=True)\n")
    if prog == "NWChem":
        return (f"start {m}\ntitle \"{mol} {fn}/{bs}\"\ngeometry units angstrom\n{at}\nend\n\nbasis\n  * library {b[3]}\nend\n\n"
                f"dft\n  xc {xc[3]}\nend\n\ntask dft optimize\ntask dft freq\n")
    return (f"# pip install pyscf geometric\nfrom pyscf import gto, dft\nfrom pyscf.geomopt.geometric_solver import optimize\n\n"
            f"mol = gto.M(atom='''\n{at}\n''', basis='{bs}')\nmf = dft.RKS(mol)\nmf.xc = '{xc[4]}'\nmol_eq = optimize(mf)\n"
            f"print(mol_eq.atom_coords(unit='Angstrom'))\n")

def _pareto(t, s):
    keep, best = [], np.inf
    for i in np.argsort(t, kind="stable"):
        if s[i] < best - 1e-12:
            keep.append(i); best = s[i]
    return keep

_NEED = {"molecule", "functional", "basis", "r_A", "theta_deg", "err_r_pct", "err_theta_deg", "err_dipole_pct", "time_s"}

@(getattr(st, "fragment", lambda f: f))
def advisor():
    up = st.file_uploader("Benchmark CSV in the sweep_results.csv format (optional, defaults to your sweep)", type="csv", key="advup")
    df = pd.read_csv(up) if up else _VREF
    if df is None:
        st.info("Put sweep_results.csv next to this script or upload a benchmark CSV."); return
    if not _NEED <= set(df.columns):
        st.error(f"CSV is missing columns: {sorted(_NEED - set(df.columns))}"); return

    c = st.columns([1, 1.3, 1.3, 1.3, 1.3])
    mol = c[0].selectbox("Molecule", sorted(df.molecule.unique()), key="am")
    tol_r = c[1].slider("Bond length within (%)", 0.1, 5.0, 1.5, 0.1, key="atr")
    tol_t = c[2].slider("Angle within (°)", 0.25, 6.0, 2.0, 0.25, key="att")
    tol_d = c[3].slider("Dipole within (%)", 1.0, 40.0, 10.0, 1.0, key="atd")
    d = df[df.molecule == mol].copy().reset_index(drop=True)
    budget = c[4].slider("Time budget (s)", float(np.floor(d.time_s.min())), float(np.ceil(d.time_s.max())),
                         float(np.ceil(d.time_s.max())), 1.0, key="abud")

    d["label"] = d.functional + " / " + d.basis
    d["Bond"] = d.err_r_pct.abs() / tol_r; d["Angle"] = d.err_theta_deg.abs() / tol_t; d["Dipole"] = d.err_dipole_pct.abs() / tol_d
    d["score"] = d[["Bond", "Angle", "Dipole"]].max(axis=1)
    d["ok"] = (d.score <= 1) & (d.time_s <= budget)
    if d.ok.any():
        rec, met = d[d.ok].sort_values(["time_s", "score"]).iloc[0], True
    else:
        pool = d[d.time_s <= budget]; pool = pool if len(pool) else d
        rec, met = pool.sort_values(["score", "time_s"]).iloc[0], False
    fails = [k for k in ("Bond", "Angle", "Dipole") if rec[k] > 1]

    k = st.columns([2, 1, 1, 1])
    with k[0]:
        stat("Recommended level of theory" if met else "No method meets every target — closest",
             f"{rec.functional} / {rec.basis}",
             f"{rec.time_s:.0f} s per optimization" + ("" if met else f" · misses: {', '.join(fails)}"), met)
    with k[1]: stat("Bond error", f"{rec.err_r_pct:+.2f}%", f"target ±{tol_r}%", rec.Bond <= 1)
    with k[2]: stat("Angle error", f"{rec.err_theta_deg:+.2f}°", f"target ±{tol_t}°", rec.Angle <= 1)
    with k[3]: stat("Dipole error", f"{rec.err_dipole_pct:+.1f}%", f"target ±{tol_d}%", rec.Dipole <= 1)
    dominated = int(((d.time_s >= rec.time_s) & (d.score >= rec.score)).sum() - 1)
    rho, pv = stats.spearmanr(d.time_s, d.score) if d.score.nunique() > 1 else (np.nan, np.nan)
    st.caption(f"{dominated} of {len(d) - 1} other methods cost at least as much and are no more accurate. "
               f"Cost vs accuracy rank correlation ρ = {rho:.2f} (p = {pv:.2f}); a weak or positive ρ means paying more is not buying accuracy.")
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    A, B = st.columns([3, 2])
    fig = px.scatter(d, x="time_s", y="score", color="functional", symbol="basis", hover_name="label")
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
    fd = d.iloc[_pareto(d.time_s.values, d.score.values)].sort_values("time_s")
    fig.add_hrect(y0=0, y1=1, fillcolor="rgba(94,234,212,.08)", line_width=0)
    fig.add_hline(y=1, line_dash="dash", line_color=ACCENT, annotation_text="all targets met")
    fig.add_vline(x=budget, line_dash="dash", line_color=AMBER, annotation_text="time budget")
    fig.add_trace(go.Scatter(x=fd.time_s, y=fd.score, mode="lines", name="Pareto front",
                             line=dict(color="rgba(255,255,255,.55)", dash="dot", shape="hv")))
    fig.add_trace(go.Scatter(x=[rec.time_s], y=[rec.score], mode="markers", name="recommended",
                             marker=dict(size=28, color="rgba(0,0,0,0)", line=dict(width=3, color=AMBER))))
    fig.update_layout(title="Cost vs accuracy · below the green line a method meets every target",
                      xaxis_title="runtime per optimization (s)", yaxis_title="worst error ÷ tolerance")
    A.plotly_chart(style(fig, 470), use_container_width=True)
    ds = d.sort_values("time_s")
    hm = go.Figure(go.Heatmap(z=ds[["Bond", "Angle", "Dipole"]].values, x=["Bond", "Angle", "Dipole"], y=ds.label,
                              text=ds[["Bond", "Angle", "Dipole"]].values.round(2), texttemplate="%{text}", zmin=0, zmax=2,
                              colorscale=[[0, "#5eead4"], [.5, "#312e81"], [1, "#fb7185"]], showscale=False, xgap=3, ygap=3))
    hm.update_yaxes(autorange="reversed")
    hm.update_layout(title="Error ÷ tolerance (≤ 1 passes) · cheapest at top")
    B.plotly_chart(style(hm, 470), use_container_width=True)

    st.markdown("### Export a ready-to-run input")
    st.caption("Uses the recommended method and its own optimized geometry as the starting point. "
               "Keywords are mapped from PySCF names; check them against your program version before a production run.")
    progs = {"Gaussian": "gjf", "ORCA": "inp", "Psi4": "dat", "NWChem": "nw", "PySCF": "py"}
    for tab, (pg, ext) in zip(st.tabs(list(progs)), progs.items()):
        with tab:
            txt = _deck(pg, mol, rec.functional, rec.basis, float(rec.r_A), float(rec.theta_deg))
            st.code(txt, language="python" if pg == "PySCF" else "text")
            st.download_button(f"Download {mol.lower()}.{ext}", txt, file_name=f"{mol.lower()}.{ext}", key=f"dl{pg}")

    st.markdown("### Plan a screening campaign")
    p = st.columns(4)
    n = p[0].number_input("Molecules to run", 1, 1_000_000, 500, key="pn")
    size = p[1].slider("System size vs this molecule (×)", 1.0, 50.0, 1.0, 0.5, key="ps")
    expo = p[2].slider("Assumed cost scaling (size^p)", 1.5, 4.0, 3.0, 0.1, key="pe")
    wk = p[3].number_input("Parallel workers", 1, 512, 8, key="pw")
    picks = {"Recommended": rec, "Cheapest tested": d.loc[d.time_s.idxmin()], "Most accurate tested": d.loc[d.score.idxmin()]}
    plan = pd.DataFrame([dict(plan=k_, method=v.label, per_job_s=v.time_s * size ** expo, worst_error_ratio=v.score,
                              wall_hours=n * v.time_s * size ** expo / wk / 3600) for k_, v in picks.items()])
    q1, q2 = st.columns([2, 3])
    q1.dataframe(plan.round(2), use_container_width=True, hide_index=True)
    bar = px.bar(plan, x="wall_hours", y="plan", orientation="h", color="plan", text=plan.wall_hours.round(1),
                 color_discrete_sequence=[AMBER, ICE, HOT])
    bar.update_layout(showlegend=False, xaxis_title="wall-clock hours", yaxis_title="", title="Campaign wall time")
    q2.plotly_chart(style(bar, 260), use_container_width=True)
    st.caption("Runtimes are your recorded values for this molecule; scaling to bigger systems is a planning assumption, not a measurement.")
    st.download_button("Download ranked methods (CSV)", d.sort_values(["score", "time_s"]).drop(columns=["ok"]).to_csv(index=False),
                       file_name="ranked_methods.csv", key="dlcsv")

with tabA:
    advisor()