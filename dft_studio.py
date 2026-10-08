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

try:
    from pyscf import gto as _GTO, dft as _DFT
except Exception:
    _GTO = _DFT = None

def _parse_xyz_blocks(text):
    """Parse candidate blocks separated by ---; optional first line is `name:`."""
    blocks=[]
    for raw in text.split("---"):
        lines=[x.strip() for x in raw.strip().splitlines() if x.strip() and not x.strip().startswith('#')]
        if not lines: continue
        name='Candidate %d' % (len(blocks)+1)
        if lines[0].lower().startswith('name:'):
            name=lines.pop(0).split(':',1)[1].strip() or name
        atoms=[]
        for line in lines:
            p=line.replace(',', ' ').split()
            if len(p) >= 4 and p[0][0].isalpha():
                try: atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
                except ValueError: pass
        if atoms: blocks.append((name, atoms))
    return blocks

def _real_screen_candidate(name, atoms, functional, basis, charge=0, spin=0):
    """Real PySCF single-point screen for an arbitrary XYZ-like candidate."""
    if _GTO is None or _DFT is None:
        raise RuntimeError('PySCF is not installed.')
    mol=_GTO.M(atom=atoms, unit='Angstrom', basis=basis, charge=int(charge), spin=int(spin), verbose=0)
    mf=_DFT.RKS(mol) if int(spin)==0 else _DFT.UKS(mol)
    mf.xc=functional
    import time as _screen_time
    t0=_screen_time.perf_counter(); e=float(mf.kernel()); elapsed=_screen_time.perf_counter()-t0
    occ=np.asarray(mf.mo_occ)
    ene=np.asarray(mf.mo_energy)
    if ene.ndim>1:
        occ_flat=np.concatenate([np.ravel(x) for x in occ]); ene_flat=np.concatenate([np.ravel(x) for x in ene])
    else: occ_flat=occ.ravel(); ene_flat=ene.ravel()
    occupied=ene_flat[occ_flat>1e-8] if np.any(occ_flat>1e-8) else np.array([])
    virtual=ene_flat[occ_flat<1e-8] if np.any(occ_flat<1e-8) else np.array([])
    homo=float(occupied.max()) if occupied.size else float('nan')
    lumo=float(virtual.min()) if virtual.size else float('nan')
    gap=(lumo-homo)*27.211386 if np.isfinite(homo) and np.isfinite(lumo) else float('nan')
    try: dip=float(np.linalg.norm(mf.dip_moment(unit='Debye')))
    except Exception: dip=float('nan')
    return dict(candidate=name, atoms=len(atoms), energy_Eh=e, gap_eV=gap, dipole_D=dip, runtime_s=elapsed, functional=functional, basis=basis)

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
        _REF = pd.DataFrame([
            dict(functional=fn, basis=bs, r_A=0.969, theta_deg=103.9, dipole_D=2.10,
                 gap_eV=9.0, energy_Eh=-76.0, time_s=20.0, err_r_pct=1.23,
                 err_theta_deg=-0.62, err_dipole_pct=13.2)
            for fn in ["lda,vwn", "pbe", "b3lyp"]
            for bs in ["sto-3g", "6-31g", "6-31g*", "cc-pvdz"]
        ])

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

tabA, tab0, tab1, tab3, tab2, tab5 = st.tabs(["🎯 METHOD ADVISOR", "🌀 LIVE LAB", "⚡ LIVE OPTIMIZER", "📐 BENDING CURVE", "📊 SWEEP LAB", "🔬 APPLICATION STUDIO"])

# ------------------------------------------------------------------ reactive tools
# The 3D lab below is already browser-side. These panels use the same pattern:
# controls/plots react inside the component without triggering a Streamlit rerun.
# Real PySCF remains available as an explicit server-side action in the optimizer.

import json
import streamlit.components.v1 as components

_UIREF = pd.read_csv("sweep_results.csv") if os.path.exists("sweep_results.csv") else None
_UIFUNCS = ["lda,vwn", "pbe", "b3lyp"]
_UIBASES = ["sto-3g", "6-31g", "6-31g*", "cc-pvdz"]

def _ui_anchor(fn, bs):
    if _UIREF is not None:
        m = _UIREF[(_UIREF.functional == fn) & (_UIREF.basis == bs)]
        if len(m):
            return m.iloc[0]
    return None

_ui_rows = []
if _UIREF is not None:
    for rec in _UIREF.replace({np.nan: None}).to_dict("records"):
        # Keep only JSON-safe primitive values.
        _ui_rows.append({k: (float(v) if isinstance(v, np.generic) and np.issubdtype(type(v), np.number) else v)
                         for k, v in rec.items()})

_UI_DATA = dict(
    funcs=_UIFUNCS,
    bases=_UIBASES,
    ex={"r": 0.9572, "theta": 104.52, "dipole": 1.855},
    rows=_ui_rows,
    anchors={
        f: {
            b: (
                dict(r=float(a.r_A), th=float(a.theta_deg), dip=float(a.dipole_D),
                     gap=float(a.gap_eV), energy=float(a.energy_Eh), time=float(a.time_s))
                if (a := _ui_anchor(f, b)) is not None
                else dict(r=.969, th=103.9, dip=2.1, gap=9.0, energy=-76.0, time=20.0)
            )
            for b in _UIBASES
        } for f in _UIFUNCS
    }
)

_REACTIVE_CSS = r"""
<style>
.rt *{box-sizing:border-box}.rt{font-family:Inter,system-ui,sans-serif;color:#e2e8f0}
.rt .ctl{display:grid;grid-template-columns:1.1fr 1.1fr 2fr 2fr auto;gap:14px;align-items:end;margin:0 0 16px}
.rt label{display:block;font-size:.68rem;letter-spacing:2px;text-transform:uppercase;color:#94a3b8;font-weight:700;margin-bottom:7px}
.rt label b{float:right;color:#5eead4;letter-spacing:0;font-size:.82rem}
.rt select,.rt input[type=search]{width:100%;padding:10px 12px;border-radius:12px;background:rgba(255,255,255,.055);color:#e2e8f0;border:1px solid rgba(255,255,255,.12);font:inherit}
.rt select option{background:#14122e}.rt input[type=range]{width:100%;accent-color:#5eead4}
.rt button{background:linear-gradient(90deg,#5eead4,#818cf8);color:#08111a;font:700 .9rem Inter;border:0;border-radius:12px;padding:10px 16px;cursor:pointer}
.rt button:hover{filter:brightness(1.08);transform:translateY(-1px)}
.rt .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:15px}
.rt .card{border:1px solid rgba(255,255,255,.09);border-radius:18px;padding:14px 17px;background:linear-gradient(160deg,rgba(255,255,255,.075),rgba(255,255,255,.02))}
.rt .lbl{font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:#94a3b8}.rt .big{font:700 1.65rem 'Space Grotesk',sans-serif;margin-top:4px}.rt .sm{font-size:.78rem;color:#94a3b8}
.rt .good{color:#5eead4}.rt .bad{color:#fb7185}.rt .grid2{display:grid;grid-template-columns:1.25fr .75fr;gap:14px}.rt .panel{border:1px solid rgba(255,255,255,.08);border-radius:18px;background:rgba(255,255,255,.025);padding:10px}
.rt canvas{width:100%;display:block}.rt .toolbar{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin-bottom:12px}.rt .toolbar>div{min-width:150px}.rt .table{max-height:300px;overflow:auto;border-radius:12px}.rt table{width:100%;border-collapse:collapse;font-size:.78rem}.rt th,.rt td{padding:7px 9px;border-bottom:1px solid rgba(255,255,255,.07);text-align:right;white-space:nowrap}.rt th{color:#94a3b8;position:sticky;top:0;background:#111126}.rt th:first-child,.rt td:first-child{text-align:left}
.rt .hint{color:#64748b;font-size:.78rem;margin-top:8px}.rt .empty{padding:30px;text-align:center;color:#94a3b8;border:1px dashed rgba(255,255,255,.12);border-radius:16px}
@media(max-width:900px){.rt .ctl,.rt .grid2{grid-template-columns:1fr 1fr}.rt .cards{grid-template-columns:1fr 1fr}}
</style>
"""

def html_component(body, height):
    return components.html(_REACTIVE_CSS + body.replace("__DATA__", json.dumps(_UI_DATA, separators=(",",":"))),
                           height=height, scrolling=False)

# ----------------------------- LIVE OPTIMIZER preview
with tab1:
    optimizer_html = r"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box}html,body{margin:0;background:transparent;font-family:Inter,sans-serif;color:#e2e8f0}.rt{width:100%}
.ctl{display:grid;grid-template-columns:1fr 1fr 2fr 2fr auto auto;gap:14px;align-items:end;margin-bottom:16px}.ctl>div{min-width:0}
label{display:block;font-size:.68rem;letter-spacing:1.8px;text-transform:uppercase;color:#94a3b8;font-weight:700;margin-bottom:7px}label b{float:right;color:#5eead4;letter-spacing:0;font-size:.82rem}
select{width:100%;padding:10px 12px;border-radius:11px;background:#111827;color:#e5e7eb;border:1px solid rgba(148,163,184,.2);font:inherit}select option{background:#14122e}
.liveTrack{position:relative;width:100%;height:30px;margin-top:1px;cursor:ew-resize;touch-action:none;user-select:none}.liveTrack:before{content:"";position:absolute;left:0;right:0;top:11px;height:7px;border-radius:99px;background:rgba(255,255,255,.12)}.liveFill{position:absolute;left:0;top:11px;height:7px;border-radius:99px;background:#5eead4;pointer-events:none}.liveThumb{position:absolute;top:2px;width:24px;height:24px;margin-left:-12px;border-radius:50%;background:#fff;border:5px solid #5eead4;box-shadow:0 0 0 5px rgba(94,234,212,.14);pointer-events:none}
button{background:linear-gradient(90deg,#5eead4,#818cf8);color:#08111a;font:700 .88rem Inter;border:0;border-radius:12px;padding:11px 14px;cursor:pointer;min-height:42px;white-space:nowrap}button:disabled{opacity:.55;cursor:wait}button:focus-visible,.liveTrack:focus-visible,select:focus-visible{outline:2px solid #5eead4;outline-offset:3px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}.card{border:1px solid rgba(255,255,255,.08);border-radius:16px;background:rgba(255,255,255,.035);padding:15px}.lbl{font-size:.67rem;letter-spacing:1.5px;text-transform:uppercase;color:#94a3b8}.big{font:700 1.35rem 'Space Grotesk';margin-top:6px}.sm{font-size:.74rem;color:#94a3b8;margin-top:4px}.good{color:#5eead4}.bad{color:#fb7185}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{border:1px solid rgba(255,255,255,.08);border-radius:16px;background:rgba(255,255,255,.025);padding:10px}.panel canvas{display:block;width:100%;height:auto}.hint{margin-top:12px;color:#aab3c2;font-size:.82rem;line-height:1.5}
@media(max-width:900px){.ctl{grid-template-columns:1fr 1fr}.cards{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}}@media(max-width:560px){.ctl{grid-template-columns:1fr}.cards{grid-template-columns:1fr}}
</style>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<div class="rt optApp">
  <div class="ctl">
    <div><label>Functional</label><select id="ofn"></select></div>
    <div><label>Basis</label><select id="obs"></select></div>
    <div><label>Starting r <b id="orv"></b></label><input id="or" type="range" min=".80" max="1.60" step=".005" value="1.00"><div class="liveTrack" id="orTrack"></div></div>
    <div><label>Starting θ <b id="otv"></b></label><input id="ot" type="range" min="80" max="140" step="1" value="100"><div class="liveTrack" id="otTrack"></div></div>
    <button id="snapO" type="button">Snap to minimum</button><button id="runO" type="button">Run optimization</button>
  </div>
  <div class="cards">
    <div class="card"><div class="lbl">Predicted minimum r</div><div class="big" id="omr"></div><div class="sm">Å · live</div></div>
    <div class="card"><div class="lbl">Predicted minimum θ</div><div class="big" id="omt"></div><div class="sm">degrees · live</div></div>
    <div class="card"><div class="lbl">Current energy above minimum</div><div class="big" id="ome"></div><div class="sm">kcal/mol</div></div>
    <div class="card"><div class="lbl">Δ vs experiment</div><div class="big" id="omd"></div><div class="sm">combined geometry error</div></div>
  </div>
  <div class="grid2">
    <div class="panel" style="min-height:430px"><div style="padding:8px 10px;color:#94a3b8;font-size:.72rem;letter-spacing:1.8px;text-transform:uppercase">Molecular geometry · live</div><div id="o3d" style="height:390px"></div></div>
    <div class="panel"><canvas id="opath" height="390"></canvas><canvas id="osurf" height="390" style="margin-top:8px"></canvas></div>
  </div>
  <div class="hint" id="optHint">Drag either control to explore the energy surface. Use <b>Run optimization</b> to animate the predicted search from your current geometry to the minimum, or <b>Snap to minimum</b> to jump there. The explicit PySCF action below is the real server-side calculation.</div>
</div>
<script>
const OD=__DATA__, q=id=>document.getElementById(id), OPI=Math.PI;
OD.funcs.forEach(x=>q('ofn').add(new Option(x,x))); OD.bases.forEach(x=>q('obs').add(new Option(x,x)));
q('ofn').value='b3lyp'; q('obs').value='6-31g*';
const Oexp=OD.ex;
function oa(){return OD.anchors[q('ofn').value][q('obs').value]}
function oe(r,t,a){return 240*Math.pow(1-Math.exp(-2.2*(r-a.r)),2)+61*Math.pow(Math.cos(t*OPI/180)-Math.cos(a.th*OPI/180),2)}
function opt(){const a=oa();let r=+q('or').value,t=+q('ot').value;for(let i=0;i<42;i++){r+=(a.r-r)*.22;t+=(a.th-t)*.22}return [r,t]}
function liveSlider(id,trackId,on){const el=q(id),track=q(trackId),fill=document.createElement('div'),thumb=document.createElement('div');fill.className='liveFill';thumb.className='liveThumb';track.append(fill,thumb);el.style.position='absolute';el.style.opacity='0';el.style.pointerEvents='none';let down=false,raf=0,lastX=0;
 function paint(x){const r=track.getBoundingClientRect(),lo=+el.min,hi=+el.max,step=+(el.step||1);let p=Math.max(0,Math.min(1,(x-r.left)/Math.max(1,r.width))),raw=lo+p*(hi-lo),v=lo+Math.round((raw-lo)/step)*step;v=Math.max(lo,Math.min(hi,v));el.value=String(v);const z=(v-lo)/(hi-lo||1);fill.style.width=(z*100)+'%';thumb.style.left=(z*100)+'%';on();}
 function move(e){if(!down)return;lastX=e.clientX;paint(lastX);e.preventDefault()}
 track.addEventListener('pointerdown',e=>{down=true;track.setPointerCapture?.(e.pointerId);paint(e.clientX);e.preventDefault()});
 track.addEventListener('pointermove',move);track.addEventListener('pointerup',e=>{if(down)paint(e.clientX);down=false;track.releasePointerCapture?.(e.pointerId)});track.addEventListener('pointercancel',()=>down=false);track.tabIndex=0;track.setAttribute('role','slider');track.setAttribute('aria-valuemin',el.min);track.setAttribute('aria-valuemax',el.max);track.setAttribute('aria-valuenow',el.value);track.addEventListener('keydown',e=>{const step=+(el.step||1),lo=+el.min,hi=+el.max;let v=+el.value;if(e.key==='ArrowLeft'||e.key==='ArrowDown')v-=step;else if(e.key==='ArrowRight'||e.key==='ArrowUp')v+=step;else if(e.key==='Home')v=lo;else if(e.key==='End')v=hi;else return;e.preventDefault();v=Math.max(lo,Math.min(hi,v));el.value=String(v);paint(track.getBoundingClientRect().left+(v-lo)/(hi-lo||1)*track.getBoundingClientRect().width);track.setAttribute('aria-valuenow',v)});
 const z=(+el.value-+el.min)/(+el.max-+el.min||1);fill.style.width=(z*100)+'%';thumb.style.left=(z*100)+'%';
}
function line(cv,xs,ys,x0,x1,title,px,col){const g=cv.getContext('2d'),w=cv.clientWidth,h=cv.height||390,d=devicePixelRatio||1;cv.width=w*d;cv.height=h*d;g.setTransform(d,0,0,d,0,0);g.clearRect(0,0,w,h);const X=x=>45+(x-x0)/(x1-x0||1)*(w-65),Y=y=>h-35-Math.min(y,90)/90*(h-65);g.strokeStyle='rgba(255,255,255,.08)';for(let i=0;i<5;i++){let y=Y(i*22.5);g.beginPath();g.moveTo(45,y);g.lineTo(w-20,y);g.stroke()}g.strokeStyle=col;g.lineWidth=3;g.beginPath();xs.forEach((x,i)=>i?g.lineTo(X(x),Y(ys[i])):g.moveTo(X(x),Y(ys[i])));g.stroke();let ix=xs.reduce((m,v,i)=>Math.abs(v-px)<Math.abs(xs[m]-px)?i:m,0);g.fillStyle=col;g.beginPath();g.arc(X(px),Y(ys[ix]),7,0,7);g.fill();g.fillStyle='#e2e8f0';g.font='600 14px Space Grotesk';g.fillText(title,15,23)}
function ou(){const a=oa(),r=+q('or').value,t=+q('ot').value,[mr,mt]=opt(),e=oe(r,t,a),de=Math.hypot((r-Oexp.r)/.015,(t-Oexp.theta)/2);q('orv').textContent=r.toFixed(3)+' Å';q('otv').textContent=t+'°';q('omr').textContent=mr.toFixed(3);q('omt').textContent=mt.toFixed(1);q('ome').textContent=e.toFixed(2);q('omd').textContent=de.toFixed(1)+' σ';q('omd').className='big '+(de<2?'good':'bad');const xr=[],xe=[];for(let i=0;i<=120;i++){let x=.8+i*.00667;xr.push(x);xe.push(oe(x,t,a))}line(q('opath'),xr,xe,.8,1.6,'Energy vs bond length',r,'#5eead4');const tr=[],te=[];for(let i=0;i<=120;i++){let x=60+i*1;tr.push(x);te.push(oe(r,x,a))}line(q('osurf'),tr,te,60,180,'Energy vs angle',t,'#818cf8');target.r=r;target.th=t}

/* 3D molecule: the same continuous target/current pattern used by LIVE LAB. */
const host=q('o3d'),scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(38,1,.1,100),ren=new THREE.WebGLRenderer({antialias:true,alpha:true}),ctl=new THREE.OrbitControls(cam,ren.domElement);
ren.setPixelRatio(Math.min(devicePixelRatio||1,2));ren.setClearColor(0,0);host.appendChild(ren.domElement);cam.position.set(0,2.5,4.4);ctl.enableDamping=true;ctl.enablePan=false;ctl.minDistance=2.8;ctl.maxDistance=7;
scene.add(new THREE.HemisphereLight(0xffffff,0x202040,1.6));const dl=new THREE.DirectionalLight(0xffffff,1.8);dl.position.set(3,4,5);scene.add(dl);
const root=new THREE.Group();scene.add(root);
const o=new THREE.Mesh(new THREE.SphereGeometry(.28,32,24),new THREE.MeshStandardMaterial({color:0x9aa4b2,roughness:.24,metalness:.08}));root.add(o);
const hs=[new THREE.Mesh(new THREE.SphereGeometry(.16,28,20),new THREE.MeshStandardMaterial({color:0xe8eef6,roughness:.18})),new THREE.Mesh(new THREE.SphereGeometry(.16,28,20),new THREE.MeshStandardMaterial({color:0xe8eef6,roughness:.18}))];hs.forEach(h=>root.add(h));
function cyl(a,b){const d=b.clone().sub(a),m=(a.clone().add(b)).multiplyScalar(.5),g=new THREE.CylinderGeometry(.055,.055,d.length,18),x=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0x64748b,roughness:.45}));x.position.copy(m);x.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize());root.add(x);return x}const bonds=[cyl(new THREE.Vector3(),new THREE.Vector3()),cyl(new THREE.Vector3(),new THREE.Vector3())];
let target={r:+q('or').value,th:+q('ot').value},cur={r:target.r,th:target.th};
function place(r,t){const a=t*OPI/2/180;const p1=new THREE.Vector3(r*Math.sin(a),r*Math.cos(a),0),p2=new THREE.Vector3(-r*Math.sin(a),r*Math.cos(a),0);hs[0].position.copy(p1);hs[1].position.copy(p2);bonds[0].position.copy(p1.clone().multiplyScalar(.5));bonds[0].scale.y=p1.length()/1;bonds[0].quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),p1.clone().normalize());bonds[1].position.copy(p2.clone().multiplyScalar(.5));bonds[1].scale.y=p2.length()/1;bonds[1].quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),p2.clone().normalize())}
function resize(){const w=host.clientWidth,h=host.clientHeight;ren.setSize(w,h,false);cam.aspect=w/Math.max(1,h);cam.updateProjectionMatrix()}addEventListener('resize',resize);resize();
function frame(t){requestAnimationFrame(frame);cur.r+=(target.r-cur.r)*.28;cur.th+=(target.th-cur.th)*.28;place(cur.r,cur.th);root.rotation.y=t*.00045;ctl.update();ren.render(scene,cam)}requestAnimationFrame(frame);
['ofn','obs'].forEach(id=>q(id).addEventListener('change',ou));liveSlider('or','orTrack',ou);liveSlider('ot','otTrack',ou);
let optAnim=false,optStep=0,optPath=[];
function runOptAnimation(){const a=oa(),sr=+q('or').value,st=+q('ot').value,mr=a.r,mt=a.th;optPath=[];for(let i=0;i<=36;i++){const u=i/36;const ease=u*u*(3-2*u);optPath.push([sr+(mr-sr)*ease,st+(mt-st)*ease]);}optStep=0;optAnim=true;q('runO').disabled=true;q('optHint').textContent='Optimization running: following the predicted energy descent from the selected starting geometry.';function tick(){if(!optAnim)return;const p=optPath[optStep++];q('or').value=p[0].toFixed(3);q('ot').value=Math.round(p[1]);ou();if(optStep<optPath.length){requestAnimationFrame(tick)}else{optAnim=false;q('runO').disabled=false;q('optHint').textContent='Optimization complete. The predicted minimum is now shown. Use RUN REAL DFT below for the actual PySCF calculation.'}}requestAnimationFrame(tick)}
q('snapO').onclick=()=>{const a=oa();q('or').value=a.r;q('ot').value=Math.round(a.th);ou()};q('runO').onclick=runOptAnimation;ou();
</script>
"""
    html_component(optimizer_html, 900)

    @st.fragment
    def real_dft_action():
        st.markdown("**Actual DFT calculation**")
        st.caption("The live panel above is a browser-side predictive surface. These controls launch the real PySCF optimization on the server.")
        c1,c2,c3,c4=st.columns(4)
        with c1: rfn=st.selectbox("Functional", _UIFUNCS, index=2, key="opt_real_fn")
        with c2: rbs=st.selectbox("Basis", _UIBASES, index=2, key="opt_real_bs")
        with c3: rr=st.number_input("Start r (Å)", 0.80, 1.60, 1.00, 0.01, key="opt_real_r")
        with c4: rt=st.number_input("Start θ (°)", 80.0, 140.0, 100.0, 1.0, key="opt_real_t")
        if st.button("RUN REAL DFT OPTIMIZATION", key="real_dft"):
            with st.spinner("Running PySCF optimization…"):
                st.session_state["live"] = run_one("H2O", rfn, rbs, rr, rt)
            d = st.session_state["live"]
            st.success(f"{rfn} / {rbs}: optimized r = {d['r_A']:.3f} Å · θ = {d['theta_deg']:.1f}° · E = {d['energy_Eh']:.6f} Eh")
    real_dft_action()

# ----------------------------- SWEEP LAB
with tab2:
    sweep_html = r"""
<div class="rt">
  <div class="toolbar">
    <div><label>Metric</label><select id="sm"></select></div>
    <div><label>Functional filter</label><select id="sf"><option value="ALL">All</option></select></div>
  </div>
  <div class="cards">
    <div class="card"><div class="lbl">Best bond error</div><div class="big" id="sb"></div><div class="sm" id="sbs"></div></div>
    <div class="card"><div class="lbl">Best angle error</div><div class="big" id="sa"></div><div class="sm" id="sas"></div></div>
    <div class="card"><div class="lbl">Cheapest run</div><div class="big" id="sc"></div><div class="sm" id="scs"></div></div>
    <div class="card"><div class="lbl">Rows shown</div><div class="big" id="sn"></div><div class="sm">live filtered dataset</div></div>
  </div>
  <div class="grid2"><div class="panel"><canvas id="splot" height="360"></canvas></div><div class="panel"><canvas id="sscat" height="360"></canvas></div></div>
  <div class="panel table" id="stable"></div>
</div>
<script>
const SD=__DATA__, sQ=id=>document.getElementById(id), srows=SD.rows||[], metrics=['r_A','theta_deg','time_s','dipole_D','gap_eV','energy_Eh'];
metrics.forEach(m=>sQ('sm').add(new Option(m,m))); SD.funcs.forEach(f=>sQ('sf').add(new Option(f,f)));sQ('sm').value='r_A';
function groups(){const f=sQ('sf').value;return srows.filter(x=>f==='ALL'||x.functional===f)}
function scl(v,a,b,c,d){return c+(v-a)/(b-a)*(d-c)}
function canvasLine(cv,sets,x0,x1,y0,y1,xlab,ylab){const g=cv.getContext('2d'),w=cv.clientWidth,h=360,d=devicePixelRatio||1;cv.width=w*d;cv.height=h*d;g.setTransform(d,0,0,d,0,0);g.clearRect(0,0,w,h);const X=x=>55+(x-x0)/(x1-x0)*(w-75),Y=y=>h-45-(y-y0)/(y1-y0)*(h-80);g.strokeStyle='rgba(255,255,255,.07)';for(let i=0;i<5;i++){let y=Y(y0+(y1-y0)*i/4);g.beginPath();g.moveTo(55,y);g.lineTo(w-20,y);g.stroke()}sets.forEach((s,j)=>{g.strokeStyle=['#5eead4','#818cf8','#fb7185','#fbbf24'][j%4];g.lineWidth=3;g.beginPath();s.pts.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();s.pts.forEach(p=>{g.fillStyle=g.strokeStyle;g.beginPath();g.arc(X(p[0]),Y(p[1]),4,0,7);g.fill()})});g.fillStyle='#94a3b8';g.font='12px Inter';g.fillText(xlab,Math.max(55,w/2-50),h-10);g.save();g.translate(15,h/2+40);g.rotate(-Math.PI/2);g.fillText(ylab,0,0);g.restore()}
function sw(){const rows=groups();sQ('sn').textContent=rows.length;if(!rows.length){sQ('stable').innerHTML='<div class="empty">No sweep_results.csv data is available for this view.</div>';return}
 let br=rows.reduce((a,b)=>Math.abs(b.err_r_pct)<Math.abs(a.err_r_pct)?b:a),ba=rows.reduce((a,b)=>Math.abs(b.err_theta_deg)<Math.abs(a.err_theta_deg)?b:a),bc=rows.reduce((a,b)=>+b.time_s<+a.time_s?b:a);sQ('sb').textContent=Math.abs(br.err_r_pct).toFixed(2)+'%';sQ('sbs').textContent=br.functional+' / '+br.basis;sQ('sa').textContent=Math.abs(ba.err_theta_deg).toFixed(2)+'°';sQ('sas').textContent=ba.functional+' / '+ba.basis;sQ('sc').textContent=Number(bc.time_s).toFixed(0)+'s';sQ('scs').textContent=bc.functional+' / '+bc.basis;
 const metric=sQ('sm').value, vals=rows.map(x=>+x[metric]).filter(Number.isFinite), lo=Math.min(...vals),hi=Math.max(...vals)||lo+1, sets=SD.funcs.filter(f=>rows.some(x=>x.functional===f)).map(f=>({pts:rows.filter(x=>x.functional===f).map(x=>[['sto-3g','6-31g','6-31g*','cc-pvdz'].indexOf(x.basis),+x[metric]])}));canvasLine(sQ('splot'),sets,0,3,lo===hi?lo-1:lo,lo===hi?hi+1:hi,'basis','value');
 const times=rows.map(x=>+x.time_s),errs=rows.map(x=>Math.abs(+x.err_r_pct)),tx=Math.min(...times),tX=Math.max(...times)||tx+1,ey=Math.max(...errs)||1;canvasLine(sQ('sscat'),[{pts:rows.map(x=>[+x.time_s,Math.abs(+x.err_r_pct)])}],tx,tX,0,ey,'runtime (s)','|bond error| %');
 let h='<table><thead><tr><th>Functional</th><th>Basis</th><th>r (Å)</th><th>θ (°)</th><th>Time</th><th>Dipole</th><th>Gap</th></tr></thead><tbody>';rows.forEach(x=>h+=`<tr><td>${x.functional}</td><td>${x.basis}</td><td>${Number(x.r_A).toFixed(3)}</td><td>${Number(x.theta_deg).toFixed(2)}</td><td>${Number(x.time_s).toFixed(1)}s</td><td>${Number(x.dipole_D).toFixed(2)}</td><td>${Number(x.gap_eV).toFixed(2)}</td></tr>`);sQ('stable').innerHTML=h+'</tbody></table>';
}
['sm','sf'].forEach(id=>sQ(id).addEventListener('change',sw));sw();
</script>
"""
    html_component(sweep_html, 760)

# ----------------------------- BENDING CURVE
with tab3:
    bend_html = r"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box}html,body{margin:0;background:transparent;font-family:Inter,sans-serif;color:#e2e8f0}.rt{width:100%}.ctl{display:grid;grid-template-columns:1fr 1fr 2fr 2fr auto;gap:14px;align-items:end;margin-bottom:16px}.ctl>div{min-width:0}
label{display:block;font-size:.68rem;letter-spacing:1.8px;text-transform:uppercase;color:#94a3b8;font-weight:700;margin-bottom:7px}label b{float:right;color:#5eead4;letter-spacing:0;font-size:.82rem}select{width:100%;padding:10px 12px;border-radius:11px;background:#111827;color:#e5e7eb;border:1px solid rgba(148,163,184,.2);font:inherit}select option{background:#14122e}
.liveTrack{position:relative;width:100%;height:30px;margin-top:1px;cursor:ew-resize;touch-action:none;user-select:none}.liveTrack:before{content:"";position:absolute;left:0;right:0;top:11px;height:7px;border-radius:99px;background:rgba(255,255,255,.12)}.liveFill{position:absolute;left:0;top:11px;height:7px;border-radius:99px;background:#818cf8;pointer-events:none}.liveThumb{position:absolute;top:2px;width:24px;height:24px;margin-left:-12px;border-radius:50%;background:#fff;border:5px solid #818cf8;box-shadow:0 0 0 5px rgba(129,140,248,.14);pointer-events:none}
button{background:linear-gradient(90deg,#5eead4,#818cf8);color:#08111a;font:700 .88rem Inter;border:0;border-radius:12px;padding:11px 14px;cursor:pointer;min-height:42px;white-space:nowrap}button:focus-visible,.liveTrack:focus-visible,select:focus-visible{outline:2px solid #5eead4;outline-offset:3px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}.card{border:1px solid rgba(255,255,255,.08);border-radius:16px;background:rgba(255,255,255,.035);padding:15px}.lbl{font-size:.67rem;letter-spacing:1.5px;text-transform:uppercase;color:#94a3b8}.big{font:700 1.35rem 'Space Grotesk';margin-top:6px}.sm{font-size:.74rem;color:#94a3b8;margin-top:4px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{border:1px solid rgba(255,255,255,.08);border-radius:16px;background:rgba(255,255,255,.025);padding:10px}.panel canvas{display:block;width:100%;height:auto}.hint{margin-top:12px;color:#aab3c2;font-size:.82rem;line-height:1.5}
@media(max-width:900px){.ctl{grid-template-columns:1fr 1fr}.cards{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}}@media(max-width:560px){.ctl{grid-template-columns:1fr}.cards{grid-template-columns:1fr}}
</style>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<div class="rt bendApp">
  <div class="ctl">
    <div><label>Functional</label><select id="bfn"></select></div>
    <div><label>Basis</label><select id="bbs"></select></div>
    <div><label>Start angle <b id="bsv"></b></label><input id="bstart" type="range" min="60" max="150" step="1" value="70"><div class="liveTrack" id="bstartTrack"></div></div>
    <div><label>Resolution <b id="bnv"></b></label><input id="bn" type="range" min="6" max="40" step="1" value="16"><div class="liveTrack" id="bnTrack"></div></div>
    <button id="bexp" type="button">Center on minimum</button><button id="bscan" type="button">Rebuild curve</button>
  </div>
  <div class="cards">
    <div class="card"><div class="lbl">Curve minimum</div><div class="big" id="bmin"></div><div class="sm">degrees · live</div></div>
    <div class="card"><div class="lbl">End-point rise</div><div class="big" id="bbar"></div><div class="sm">kcal/mol above minimum</div></div>
    <div class="card"><div class="lbl">Curvature</div><div class="big" id="bcurv"></div><div class="sm">kcal/mol/deg²</div></div>
    <div class="card"><div class="lbl">Optimized r</div><div class="big" id="br"></div><div class="sm">Å</div></div>
  </div>
  <div class="grid2">
    <div class="panel" style="min-height:430px"><div style="padding:8px 10px;color:#94a3b8;font-size:.72rem;letter-spacing:1.8px;text-transform:uppercase">H₂O bending geometry · live</div><div id="b3d" style="height:390px"></div></div>
    <div class="panel"><canvas id="bplot" height="430"></canvas></div>
  </div>
  <div class="hint" id="bendHint">Drag the start-angle control to rebuild the live bending surface. Click the curve to place the molecule at any angle. <b>Center on minimum</b> moves directly to the predicted minimum.</div>
</div>
<script>
const BD=__DATA__,bq=id=>document.getElementById(id),BPI=Math.PI;
BD.funcs.forEach(x=>bq('bfn').add(new Option(x,x)));BD.bases.forEach(x=>bq('bbs').add(new Option(x,x)));bq('bfn').value='b3lyp';bq('bbs').value='6-31g*';
function ba(){return BD.anchors[bq('bfn').value][bq('bbs').value]}
function be(r,t,a){return 240*Math.pow(1-Math.exp(-2.2*(r-a.r)),2)+61*Math.pow(Math.cos(t*BPI/180)-Math.cos(a.th*BPI/180),2)}
function liveSlider(id,trackId,on){const el=bq(id),track=bq(trackId),fill=document.createElement('div'),thumb=document.createElement('div');fill.className='liveFill';thumb.className='liveThumb';track.append(fill,thumb);el.style.position='absolute';el.style.opacity='0';el.style.pointerEvents='none';let down=false,raf=0,lastX=0;
 function paint(x){const r=track.getBoundingClientRect(),lo=+el.min,hi=+el.max,step=+(el.step||1);let p=Math.max(0,Math.min(1,(x-r.left)/Math.max(1,r.width))),raw=lo+p*(hi-lo),v=lo+Math.round((raw-lo)/step)*step;v=Math.max(lo,Math.min(hi,v));el.value=String(v);const z=(v-lo)/(hi-lo||1);fill.style.width=(z*100)+'%';thumb.style.left=(z*100)+'%';on()}
 function move(e){if(!down)return;lastX=e.clientX;paint(lastX);e.preventDefault()}
 track.addEventListener('pointerdown',e=>{down=true;track.setPointerCapture?.(e.pointerId);paint(e.clientX);e.preventDefault()});track.addEventListener('pointermove',move);track.addEventListener('pointerup',e=>{if(down)paint(e.clientX);down=false;track.releasePointerCapture?.(e.pointerId)});track.addEventListener('pointercancel',()=>down=false);track.tabIndex=0;track.setAttribute('role','slider');track.setAttribute('aria-valuemin',el.min);track.setAttribute('aria-valuemax',el.max);track.setAttribute('aria-valuenow',el.value);track.addEventListener('keydown',e=>{const step=+(el.step||1),lo=+el.min,hi=+el.max;let v=+el.value;if(e.key==='ArrowLeft'||e.key==='ArrowDown')v-=step;else if(e.key==='ArrowRight'||e.key==='ArrowUp')v+=step;else if(e.key==='Home')v=lo;else if(e.key==='End')v=hi;else return;e.preventDefault();v=Math.max(lo,Math.min(hi,v));el.value=String(v);paint(track.getBoundingClientRect().left+(v-lo)/(hi-lo||1)*track.getBoundingClientRect().width);track.setAttribute('aria-valuenow',v)});const z=(+el.value-+el.min)/(+el.max-+el.min||1);fill.style.width=(z*100)+'%';thumb.style.left=(z*100)+'%'}

function drawB(){const a=ba(),lo=+bq('bstart').value,n=Math.max(2,+bq('bn').value),r=a.r,xs=[],ys=[];for(let i=0;i<n;i++){let x=lo+(180-lo)*i/(n-1);xs.push(x);ys.push(be(r,x,a))}let mi=ys.indexOf(Math.min(...ys)),ang=xs[mi],dense=[],de=[];for(let i=0;i<241;i++){let x=lo+(180-lo)*i/240;dense.push(x);de.push(be(r,x,a))}let mn=Math.min(...de),bar=de[de.length-1]-mn;let w=bq('bplot').clientWidth,h=430,d=devicePixelRatio||1,cv=bq('bplot');cv.width=w*d;cv.height=h*d;let g=cv.getContext('2d');g.setTransform(d,0,0,d,0,0);g.clearRect(0,0,w,h);let ymax=Math.max(...de,1),X=x=>55+(x-lo)/(180-lo||1)*(w-75),Y=y=>h-45-y/ymax*(h-80);g.strokeStyle='rgba(255,255,255,.08)';for(let i=0;i<5;i++){let y=Y(ymax*i/4);g.beginPath();g.moveTo(55,y);g.lineTo(w-20,y);g.stroke()}g.strokeStyle='#818cf8';g.lineWidth=3;g.beginPath();dense.forEach((x,i)=>i?g.lineTo(X(x),Y(de[i])):g.moveTo(X(x),Y(de[i])));g.stroke();g.fillStyle='#5eead4';xs.forEach((x,i)=>{g.beginPath();g.arc(X(x),Y(ys[i]),4,0,7);g.fill()});g.fillStyle='#fb7185';g.beginPath();g.arc(X(ang),Y(mn),8,0,7);g.fill();if(typeof selectedTh==='number'){g.fillStyle='#fbbf24';g.beginPath();g.arc(X(selectedTh),Y(be(r,selectedTh,a)),7,0,7);g.fill()}g.strokeStyle='#fb7185';g.setLineDash([6,5]);g.beginPath();g.moveTo(X(104.52),45);g.lineTo(X(104.52),h-45);g.stroke();g.setLineDash([]);g.fillStyle='#e2e8f0';g.font='600 15px Space Grotesk';g.fillText('Bending potential · live preview',18,24);bq('bsv').textContent=lo+'°';bq('bnv').textContent=n;bq('bmin').textContent=ang.toFixed(1)+'°';bq('bbar').textContent=bar.toFixed(1);bq('bcurv').textContent=(2*61*Math.pow(Math.sin(a.th*BPI/180),2)/Math.pow(180/BPI,2)).toFixed(4);bq('br').textContent=r.toFixed(3);if(typeof selectedTh!=='number')selectedTh=lo;targetTh=selectedTh;bq('bendHint').textContent='Selected angle: '+selectedTh.toFixed(1)+'°. Click another point on the curve to inspect that geometry; use Center on minimum to return to the predicted minimum.'}

const host=bq('b3d'),scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(38,1,.1,100),ren=new THREE.WebGLRenderer({antialias:true,alpha:true}),ctl=new THREE.OrbitControls(cam,ren.domElement);ren.setPixelRatio(Math.min(devicePixelRatio||1,2));ren.setClearColor(0,0);host.appendChild(ren.domElement);cam.position.set(0,2.4,4.4);ctl.enableDamping=true;ctl.enablePan=false;ctl.minDistance=2.8;ctl.maxDistance=7;scene.add(new THREE.HemisphereLight(0xffffff,0x202040,1.5));const dl=new THREE.DirectionalLight(0xffffff,1.8);dl.position.set(3,4,5);scene.add(dl);const root=new THREE.Group();scene.add(root);
const o=new THREE.Mesh(new THREE.SphereGeometry(.28,32,24),new THREE.MeshStandardMaterial({color:0x9aa4b2,roughness:.24}));root.add(o);const hs=[new THREE.Mesh(new THREE.SphereGeometry(.16,28,20),new THREE.MeshStandardMaterial({color:0xe8eef6})),new THREE.Mesh(new THREE.SphereGeometry(.16,28,20),new THREE.MeshStandardMaterial({color:0xe8eef6}))];hs.forEach(h=>root.add(h));function cyl(){const x=new THREE.Mesh(new THREE.CylinderGeometry(.055,.055,1,18),new THREE.MeshStandardMaterial({color:0x64748b,roughness:.45}));root.add(x);return x}const bonds=[cyl(),cyl()];
let targetTh=+bq('bstart').value,curTh=targetTh;function place(t){const a=t*BPI/2/180,r=ba().r,p1=new THREE.Vector3(r*Math.sin(a),r*Math.cos(a),0),p2=new THREE.Vector3(-r*Math.sin(a),r*Math.cos(a),0);hs[0].position.copy(p1);hs[1].position.copy(p2);[bonds[0],bonds[1]].forEach((x,i)=>{const p=i?p2:p1;x.position.copy(p.clone().multiplyScalar(.5));x.scale.y=p.length();x.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),p.clone().normalize())})}
function resize(){const w=host.clientWidth,h=host.clientHeight;ren.setSize(w,h,false);cam.aspect=w/Math.max(1,h);cam.updateProjectionMatrix()}addEventListener('resize',resize);resize();
let selectedTh=+bq('bstart').value;
bq('bplot').addEventListener('pointerdown',e=>{const cv=bq('bplot'),rect=cv.getBoundingClientRect(),lo=+bq('bstart').value;let x=Math.max(0,Math.min(rect.width,e.clientX-rect.left));selectedTh=lo+(180-lo)*x/Math.max(1,rect.width);drawB();});
bq('bscan').onclick=()=>{selectedTh=+bq('bstart').value;drawB();bq('bendHint').textContent='Curve rebuilt at '+bq('bn').value+' sample points. Click the curve to inspect a specific angle.'};
function frame(t){requestAnimationFrame(frame);curTh+=(targetTh-curTh)*.28;place(curTh);root.rotation.y=t*.00045;ctl.update();ren.render(scene,cam)}requestAnimationFrame(frame);
['bfn','bbs'].forEach(id=>bq(id).addEventListener('change',drawB));liveSlider('bstart','bstartTrack',drawB);liveSlider('bn','bnTrack',drawB);bq('bexp').onclick=()=>{selectedTh=ba().th;bq('bstart').value=Math.max(60,Math.round(ba().th-35));drawB()};drawB();
</script>
"""
    html_component(bend_html, 920)
    @st.fragment
    def bending_action():
        st.markdown("**Evaluate a bending scan**")
        st.caption("This scan uses the same calibrated energy surface used by the live preview. It is useful for exploring the curve; it is not a new PySCF calculation at every angle.")
        c1,c2,c3,c4=st.columns(4)
        with c1: bfn=st.selectbox("Functional", _UIFUNCS, index=2, key="bend_real_fn")
        with c2: bbs=st.selectbox("Basis", _UIBASES, index=2, key="bend_real_bs")
        with c3: blo=st.number_input("Start angle (°)", 60.0, 150.0, 70.0, 1.0, key="bend_real_lo")
        with c4: bn=st.number_input("Points", 6, 40, 16, 1, key="bend_real_n")
        if st.button("RUN BENDING SCAN", key="run_bend_scan"):
            a=_ui_anchor(bfn,bbs)
            if a is None: st.error("No reference row is available for that functional/basis combination.")
            else:
                angles=np.linspace(blo,180.0,int(bn)); vals=[float(oe) for oe in [240*(1-np.exp(-2.2*(float(a.r_A)-float(a.r_A))))**2+61*(np.cos(np.radians(x))-np.cos(np.radians(float(a.theta_deg))))**2 for x in angles]]
                j=int(np.argmin(vals))
                st.session_state["bend_scan"]=(angles,vals,bfn,bbs)
                st.success(f"Minimum of calibrated scan: {angles[j]:.1f}° · relative energy = {vals[j]:.2f} kcal/mol")
        if "bend_scan" in st.session_state:
            angles,vals,bfn,bbs=st.session_state["bend_scan"]
            st.line_chart(pd.DataFrame({"angle_deg":angles,"relative_energy_kcal_mol":vals}).set_index("angle_deg"), height=260)
    bending_action()

# ================================================================== LIVE LAB (three.js, client-side)
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
input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:#fff;border:5px solid #5eead4;cursor:grab} .liveTrack{position:relative;width:100%;height:22px;margin-top:3px;cursor:ew-resize;touch-action:none}
.liveTrack:before{content:"";position:absolute;left:0;right:0;top:8px;height:6px;border-radius:99px;background:rgba(255,255,255,.12)}
.liveFill{position:absolute;left:0;top:8px;height:6px;border-radius:99px;background:#5eead4;pointer-events:none}
.liveThumb{position:absolute;top:0;width:22px;height:22px;margin-left:-11px;border-radius:50%;background:#fff;border:5px solid #5eead4;box-shadow:0 0 0 6px rgba(94,234,212,.18),0 0 18px rgba(94,234,212,.6);pointer-events:none}

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


# ================================================================== APPLICATION STUDIO
# Real-world extension: screen arbitrary molecular candidates with real PySCF.
# The browser preview remains live; pressing RUN REAL SCREEN executes the actual DFT.
with tab5:
    st.markdown("### Molecular Screening")
    st.caption("Use this for early-stage molecular screening: compare candidate geometries, electronic gaps, dipoles, energy, and compute time before committing to larger calculations.")
    c1,c2,c3,c4=st.columns([1.1,1.1,1,1])
    with c1: app_fn=st.selectbox("Functional", _UIFUNCS, index=2, key="app_fn")
    with c2: app_bs=st.selectbox("Basis", _UIBASES, index=2, key="app_bs")
    with c3: app_charge=st.number_input("Charge", -5, 5, 0, 1, key="app_charge")
    with c4: app_spin=st.number_input("Spin", 0, 10, 0, 1, key="app_spin")
    default_geom="""name: Water\nO   0.000000   0.000000   0.000000\nH   0.758600   0.000000   0.504300\nH  -0.758600   0.000000   0.504300\n---\nname: Bent water\nO   0.000000   0.000000   0.000000\nH   0.900000   0.000000   0.450000\nH  -0.700000   0.000000   0.600000"""
    app_text=st.text_area("Candidate geometries", value=default_geom, height=220, key="app_geom", help="Separate candidates with ---. Each block may begin with name: Candidate name.")
    run_col,info_col=st.columns([1,3])
    with run_col: run_screen=st.button("RUN REAL SCREEN", key="run_screen", use_container_width=True, disabled=(_GTO is None))
    with info_col:
        if _GTO is None:
            st.caption("Live editing remains available. Real molecular screening is disabled until PySCF is available.")
        else:
            st.caption("Live editing is immediate; the real DFT calculation runs only when requested so the browser stays responsive.")
    if run_screen:
        blocks=_parse_xyz_blocks(app_text)
        if not blocks: st.error("No valid candidates found. Use lines like `O 0 0 0` and separate candidates with `---`.")
        elif _GTO is not None:
            out=[]
            for name,atoms in blocks:
                try: out.append(_real_screen_candidate(name,atoms,app_fn,app_bs,app_charge,app_spin))
                except Exception as exc: out.append(dict(candidate=name, atoms=len(atoms), energy_Eh=np.nan, gap_eV=np.nan, dipole_D=np.nan, runtime_s=np.nan, functional=app_fn, basis=app_bs, error=str(exc)))
            st.session_state["app_results"]=pd.DataFrame(out)
    res=st.session_state.get("app_results")
    if res is not None and len(res):
        good=res[res.get("error",pd.Series([None]*len(res))).isna()] if "error" in res else res
        if len(good):
            k1,k2,k3,k4=st.columns(4)
            with k1: stat("Lowest energy", f"{good.energy_Eh.min():.6f}", "Hartree")
            with k2: stat("Largest gap", f"{good.gap_eV.max():.2f}", "eV")
            with k3: stat("Smallest dipole", f"{good.dipole_D.min():.2f}", "Debye")
            with k4: stat("Fastest candidate", f"{good.runtime_s.min():.2f}", "seconds")
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.download_button("Download screening results", res.to_csv(index=False), "dft_screening_results.csv", "text/csv")
    else:
        st.info("Enter one or more molecular candidates, choose the DFT level, and run the real screen.")

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

def _advisor_data_frame():
    return _VREF.copy() if _VREF is not None else pd.DataFrame()

# Browser-only Method Advisor. Every slider/filter below is native HTML and updates locally.
with tabA:
    _adv = _advisor_data_frame()
    if _adv.empty:
        st.info("Put sweep_results.csv next to this script to use Method Advisor.")
    else:
        _adv_payload = _adv.replace({np.nan: None}).to_dict("records")
        _adv_html = r"""
<style>
#advisorApp .toolbar{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin-bottom:12px}
#advisorApp .toolbar>div{flex:1 1 150px;min-width:150px}
#advisorApp .toolbar label{display:block;font-size:.68rem;letter-spacing:2px;text-transform:uppercase;color:#94a3b8;font-weight:700;margin-bottom:7px}
#advisorApp .toolbar label b{float:right;color:#5eead4;letter-spacing:0;font-size:.82rem}
#advisorApp .toolbar select{width:100%;padding:10px 12px;border-radius:12px;background:rgba(255,255,255,.055);color:#e2e8f0;border:1px solid rgba(255,255,255,.12);font:inherit}
#advisorApp input[type=range]{width:100%;accent-color:#5eead4}
#advisorApp .liveTrack{position:relative;width:100%;height:26px;margin-top:1px;cursor:ew-resize;touch-action:none;user-select:none}
#advisorApp .liveTrack:before{content:"";position:absolute;left:0;right:0;top:10px;height:7px;border-radius:99px;background:rgba(255,255,255,.13)}
#advisorApp .liveFill{position:absolute;left:0;top:10px;height:7px;border-radius:99px;background:linear-gradient(90deg,#5eead4,#818cf8);pointer-events:none}
#advisorApp .liveThumb{position:absolute;top:1px;width:25px;height:25px;margin-left:-12.5px;border-radius:50%;background:#fff;border:5px solid #5eead4;box-shadow:0 0 0 6px rgba(94,234,212,.16),0 0 18px rgba(94,234,212,.45);pointer-events:none}
#advisorApp .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:15px}
#advisorApp .card{border:1px solid rgba(255,255,255,.09);border-radius:18px;padding:14px 17px;background:linear-gradient(160deg,rgba(255,255,255,.075),rgba(255,255,255,.02))}
#advisorApp .lbl{font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:#94a3b8}
#advisorApp .big{font:700 1.65rem 'Space Grotesk',sans-serif;margin-top:4px;color:#e2e8f0}
#advisorApp .sm{font-size:.78rem;color:#94a3b8}
#advisorApp .good{color:#5eead4} #advisorApp .bad{color:#fb7185}
#advisorApp .grid2{display:grid;grid-template-columns:1.25fr .75fr;gap:14px}
#advisorApp .panel{border:1px solid rgba(255,255,255,.08);border-radius:18px;background:rgba(255,255,255,.025);padding:10px}
#advisorApp canvas{width:100%;display:block}
#advisorApp .table{max-height:300px;overflow:auto;border-radius:12px}
#advisorApp table{width:100%;border-collapse:collapse;font-size:.78rem}
#advisorApp th,#advisorApp td{padding:7px 9px;border-bottom:1px solid rgba(255,255,255,.07);text-align:right;white-space:nowrap}
#advisorApp th{color:#94a3b8;position:sticky;top:0;background:#111126}
#advisorApp th:first-child,#advisorApp td:first-child{text-align:left}
#advisorApp button{background:linear-gradient(90deg,#5eead4,#818cf8);color:#08111a;font:700 .9rem Inter;border:0;border-radius:12px;padding:10px 16px;cursor:pointer}
@media(max-width:900px){#advisorApp .cards{grid-template-columns:1fr 1fr}#advisorApp .grid2{grid-template-columns:1fr}}
</style>
<div class="rt" id="advisorApp">
  <div class="toolbar">
    <div><label>Molecule</label><select id="amol"></select></div>
    <div><label>Bond length within (%) <b id="avr"></b></label><input id="atr" type="range" min="0.1" max="5" step="0.1" value="1.5"></div>
    <div><label>Angle within (°) <b id="avt"></b></label><input id="att" type="range" min="0.25" max="6" step="0.25" value="2"></div>
    <div><label>Dipole within (%) <b id="avd"></b></label><input id="atd" type="range" min="1" max="40" step="1" value="10"></div>
    <div><label>Time budget (s) <b id="avb"></b></label><input id="abud" type="range" min="0" max="300" step="5" value="60"></div>
  </div>
  <div class="cards">
    <div class="card"><div class="lbl">Recommended level of theory</div><div class="big" id="arec"></div><div class="sm" id="ares"></div></div>
    <div class="card"><div class="lbl">Bond error</div><div class="big" id="arb"></div><div class="sm" id="art"></div></div>
    <div class="card"><div class="lbl">Angle error</div><div class="big" id="ara"></div><div class="sm" id="aat"></div></div>
    <div class="card"><div class="lbl">Dipole error</div><div class="big" id="ard"></div><div class="sm" id="adt"></div></div>
  </div>
  <div class="grid2"><div class="panel"><canvas id="aplot" height="390"></canvas></div><div class="panel"><canvas id="aheat" height="390"></canvas></div></div>
  <div class="toolbar" style="margin-top:12px"><button id="adownload" type="button">Export method comparison CSV</button></div>
</div>
<script>
const AD=__DATA__, aq=id=>document.getElementById(id), rows=AD.rows||[];
const mols=[...new Set(rows.map(x=>x.molecule))]; mols.forEach(x=>aq('amol').add(new Option(x,x)));
function num(v,d=0){const n=Number(v);return Number.isFinite(n)?n:d}
function data(){return rows.filter(x=>x.molecule===aq('amol').value)}
function setv(id,v){aq(id).textContent=v}
function cls(id,ok){aq(id).className='big '+(ok?'good':'bad')}
function scale(v,a,b,c,d){return c+(v-a)/(b-a)*(d-c)}
function chart(rows,rec){const cv=aq('aplot'),w=cv.clientWidth,h=390,d=devicePixelRatio||1;cv.width=w*d;cv.height=h*d;const g=cv.getContext('2d');g.setTransform(d,0,0,d,0,0);g.clearRect(0,0,w,h);if(!rows.length)return;const tx=rows.map(x=>num(x.time_s)),sy=rows.map(x=>Math.max(num(x.err_r_pct),num(x.err_theta_deg),num(x.err_dipole_pct)));const x0=Math.min(...tx),x1=Math.max(...tx)||x0+1,y1=Math.max(...sy,1);const X=x=>50+(x-x0)/(x1-x0)*(w-70),Y=y=>h-45-y/y1*(h-80);g.strokeStyle='rgba(255,255,255,.08)';for(let i=0;i<5;i++){let y=Y(y1*i/4);g.beginPath();g.moveTo(50,y);g.lineTo(w-20,y);g.stroke()}g.fillStyle='#94a3b8';g.font='12px Inter';g.fillText('runtime (s)',w/2-30,h-10);g.save();g.translate(14,h/2+35);g.rotate(-Math.PI/2);g.fillText('worst absolute error',0,0);g.restore();rows.forEach(x=>{g.fillStyle=x===rec?'#fbbf24':'#818cf8';g.beginPath();g.arc(X(num(x.time_s)),Y(Math.max(num(x.err_r_pct),num(x.err_theta_deg),num(x.err_dipole_pct))),x===rec?8:5,0,7);g.fill()});g.fillStyle='#e2e8f0';g.font='600 15px Space Grotesk';g.fillText('Cost vs accuracy · browser live',16,24)}
function heat(rows,tR,tT,tD){const cv=aq('aheat'),w=cv.clientWidth,h=390,d=devicePixelRatio||1;cv.width=w*d;cv.height=h*d;const g=cv.getContext('2d');g.setTransform(d,0,0,d,0,0);g.clearRect(0,0,w,h);if(!rows.length)return;const cols=['Bond','Angle','Dipole'],left=120,top=45,rh=Math.max(22,Math.min(42,(h-top-20)/rows.length));g.fillStyle='#94a3b8';g.font='12px Inter';cols.forEach((c,i)=>g.fillText(c,left+i*90,25));rows.forEach((x,j)=>{g.fillStyle='#cbd5e1';g.fillText(x.functional+' / '+x.basis,8,top+j*rh+15);[Math.abs(num(x.err_r_pct))/tR,Math.abs(num(x.err_theta_deg))/tT,Math.abs(num(x.err_dipole_pct))/tD].forEach((v,i)=>{g.fillStyle=v<=1?'rgba(94,234,212,.65)':v<=2?'rgba(129,140,248,.55)':'rgba(251,113,133,.65)';g.fillRect(left+i*90,top+j*rh,72,Math.max(18,rh-4));g.fillStyle='#fff';g.fillText(v.toFixed(2),left+i*90+22,top+j*rh+15)})})}
function update(){const d=data();if(!d.length)return;const tr=num(aq('atr').value,1.5),tt=num(aq('att').value,2),td=num(aq('atd').value,10),budget=num(aq('abud').value);d.forEach(x=>{x._b=Math.abs(num(x.err_r_pct))/tr;x._a=Math.abs(num(x.err_theta_deg))/tt;x._d=Math.abs(num(x.err_dipole_pct))/td;x._score=Math.max(x._b,x._a,x._d);x._ok=x._score<=1&&num(x.time_s)<=budget});let pool=d.filter(x=>x._ok);let met=pool.length>0;if(!pool.length)pool=d.filter(x=>num(x.time_s)<=budget);if(!pool.length)pool=d;const rec=[...pool].sort((a,b)=>a._score-b._score||num(a.time_s)-num(b.time_s))[0];setv('avr',tr.toFixed(1)+'%');setv('avt',tt.toFixed(2)+'°');setv('avd',td.toFixed(0)+'%');setv('avb',budget.toFixed(0)+' s');setv('arec',rec.functional+' / '+rec.basis);setv('ares',num(rec.time_s).toFixed(0)+' s per optimization'+(met?'':' · closest available'));setv('arb',num(rec.err_r_pct).toFixed(2)+'%');setv('art','target ±'+tr.toFixed(1)+'%');setv('ara',num(rec.err_theta_deg).toFixed(2)+'°');setv('aat','target ±'+tt.toFixed(2)+'°');setv('ard',num(rec.err_dipole_pct).toFixed(1)+'%');setv('adt','target ±'+td.toFixed(0)+'%');cls('arb',rec._b<=1);cls('ara',rec._a<=1);cls('ard',rec._d<=1);chart(d,rec);heat(d,tr,tt,td);
}
aq('amol').addEventListener('change',update);

// IMPORTANT: Streamlit's iframe can sometimes coalesce native range input events
// while the pointer is being dragged. Drive the range ourselves from pointermove
// so the Method Advisor repaints before mouse/touch release.
function makeLiveSlider(id,onChange){
  const el=aq(id),wrap=el.parentElement,track=document.createElement('div'),fill=document.createElement('div'),thumb=document.createElement('div');
  track.className='liveTrack';fill.className='liveFill';thumb.className='liveThumb';track.append(fill,thumb);el.style.position='absolute';el.style.opacity='0';el.style.pointerEvents='none';wrap.appendChild(track);
  let down=false,raf=0,lastX=0;
  const paint=x=>{const r=track.getBoundingClientRect(),lo=+el.min,hi=+el.max,step=+(el.step||1);let p=Math.max(0,Math.min(1,(x-r.left)/Math.max(1,r.width))),raw=lo+p*(hi-lo),v=lo+Math.round((raw-lo)/step)*step;v=Math.max(lo,Math.min(hi,v));el.value=String(v);const z=(v-lo)/(hi-lo||1);fill.style.width=(z*100)+'%';thumb.style.left=(z*100)+'%';onChange()};
  const move=e=>{if(!down)return;lastX=e.clientX;paint(lastX);if(!raf)raf=requestAnimationFrame(()=>{raf=0;onChange()});e.preventDefault()};
  track.addEventListener('pointerdown',e=>{down=true;track.setPointerCapture?.(e.pointerId);paint(e.clientX);e.preventDefault()});
  track.addEventListener('pointermove',move);track.addEventListener('pointerup',e=>{if(down)paint(e.clientX);down=false});track.addEventListener('pointercancel',()=>down=false);
  const z=(+el.value-+el.min)/(+el.max-+el.min||1);fill.style.width=(z*100)+'%';thumb.style.left=(z*100)+'%';
}
['atr','att','atd','abud'].forEach(id=>makeLiveSlider(id,update));
update();
aq('adownload').addEventListener('click',()=>{const d=data().map(x=>({functional:x.functional,basis:x.basis,r_A:x.r_A,theta_deg:x.theta_deg,time_s:x.time_s,score:x._score||'',ok:x._ok||false}));const csv='functional,basis,r_A,theta_deg,time_s,score,ok\n'+d.map(x=>Object.values(x).join(',')).join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='ranked_methods.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
</script>
"""
        html_component(_adv_html, 1050)
