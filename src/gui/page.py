"""The single self-contained HTML page — inline CSS + JS, **no CDN, no fonts, no network**.

Built to be filmed: black background, oversized type, one thing moving at a time, and
state changes a viewer can follow without narration. The page is dumb on purpose — it
polls ``GET /state`` four times a second and draws exactly what
:meth:`src.gui.app.DashboardApp.snapshot` gives it (including overlay geometry already
projected into the served crop), so all the logic stays in Python where it is testable.
"""

from __future__ import annotations

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Picking Cell — live pipeline</title>
<style>
  :root{
    --bg:#070a0e; --panel:#111823; --panel2:#0d131c; --line:#1f2c3d;
    --ink:#e9f1fb; --dim:#8ea3bd; --cyan:#38bdf8; --green:#22c55e;
    --amber:#f59e0b; --red:#ef4444; --violet:#a78bfa;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:"DejaVu Sans",Verdana,Geneva,sans-serif;
       font-size:16px;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1900px;margin:0 auto;padding:14px 18px 22px}

  /* ---------- header ---------- */
  header{display:flex;align-items:center;gap:18px;flex-wrap:wrap;
         border-bottom:2px solid var(--line);padding-bottom:12px;margin-bottom:14px}
  h1{font-size:26px;margin:0;letter-spacing:.16em;text-transform:uppercase;font-weight:800}
  h1 small{display:block;font-size:12px;letter-spacing:.22em;color:var(--dim);font-weight:600;margin-top:4px}
  .badges{display:flex;gap:10px;flex-wrap:wrap;margin-left:auto;align-items:center}
  .badge{border:2px solid var(--line);border-radius:8px;padding:7px 12px;font-size:13px;
         letter-spacing:.1em;font-weight:700;text-transform:uppercase;white-space:nowrap}
  .badge b{color:var(--ink)}
  .badge span{color:var(--dim);font-weight:600;margin-right:6px}
  .badge.mock{border-color:var(--violet);color:var(--violet)}
  .badge.sim{border-color:var(--cyan);color:var(--cyan)}
  .badge.live{border-color:var(--red);color:var(--red)}
  .badge.on{border-color:var(--green);color:var(--green)}
  .badge.off{border-color:#4b5c72;color:var(--dim)}
  .badge.id{font-family:"DejaVu Sans Mono",monospace;letter-spacing:0;text-transform:none;font-size:12px}
  a.badge{text-decoration:none;color:var(--cyan);border-color:var(--cyan)}

  /* ---------- pipeline ---------- */
  .pipe{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin-bottom:16px}
  .stage{position:relative;background:var(--panel2);border:2px solid var(--line);border-radius:12px;
         padding:12px 12px 10px;min-height:104px;transition:border-color .18s,background .18s,transform .18s}
  .stage .n{font-size:21px;font-weight:800;letter-spacing:.14em;color:#54687f}
  .stage .b{font-size:11.5px;color:#54687f;margin-top:3px;min-height:15px}
  .stage .d{font-size:12.5px;color:var(--dim);margin-top:7px;line-height:1.35;min-height:34px}
  .stage .ms{position:absolute;top:10px;right:12px;font-size:12px;font-weight:700;color:#54687f;
             font-family:"DejaVu Sans Mono",monospace}
  .stage.active{border-color:var(--cyan);background:#0e2130;transform:translateY(-3px);
                box-shadow:0 0 0 3px rgba(56,189,248,.18),0 0 34px rgba(56,189,248,.30)}
  .stage.active .n{color:var(--cyan)}
  .stage.active .ms{color:var(--cyan)}
  .stage.done{border-color:#1d6b3f;background:#0c1a14}
  .stage.done .n{color:var(--green)}
  .stage.done .ms{color:var(--green)}
  .stage.failed{border-color:var(--red);background:#220f12}
  .stage.failed .n{color:var(--red)}
  .stage.failed .ms{color:var(--red)}
  .stage.active::after{content:"";position:absolute;inset:-2px;border-radius:12px;
                       border:2px solid var(--cyan);animation:pulse 1.15s infinite}
  @keyframes pulse{0%{opacity:.9}50%{opacity:.15}100%{opacity:.9}}

  /* ---------- motion honesty: four states, never two ---------- */
  .stage .mo{margin-top:7px;font-size:12px;font-weight:800;letter-spacing:.08em;
             border-radius:6px;padding:4px 7px;line-height:1.3;border:2px solid currentColor}
  /* These override the plain done/failed colouring: a move that was only COMMANDED must
     never look like a move the encoders confirmed. That equivalence was the bug. */
  .stage.m-commanded{border-color:var(--amber)!important;border-style:dashed!important;
                     background:#1b1408!important}
  .stage.m-commanded .n,.stage.m-commanded .ms{color:var(--amber)!important}
  .stage.m-commanded .mo{color:var(--amber);animation:pulse 1.1s infinite}
  .stage.m-verified{border-color:var(--green)!important;background:#0c1a14!important}
  .stage.m-verified .n,.stage.m-verified .ms{color:var(--green)!important}
  .stage.m-verified .mo{color:var(--green)}
  .stage.m-mismatch{border-color:var(--red)!important;background:#210e11!important}
  .stage.m-mismatch .n,.stage.m-mismatch .ms{color:var(--red)!important}
  .stage.m-mismatch .mo{color:var(--red);animation:blink .8s infinite}
  .stage.m-unverified{border-color:var(--amber)!important;background:
      repeating-linear-gradient(45deg,#1b1408,#1b1408 9px,#241a0a 9px,#241a0a 18px)!important}
  .stage.m-unverified .n,.stage.m-unverified .ms{color:var(--amber)!important}
  .stage.m-unverified .mo{color:var(--amber)}

  .mrow{display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px solid var(--line)}
  .mrow:last-child{border-bottom:none}
  .mrow .ms{font-size:17px;font-weight:800;letter-spacing:.12em;width:78px;color:#54687f}
  .mrow .mstate{font-size:17px;font-weight:800;letter-spacing:.08em;padding:4px 11px;
                border-radius:8px;border:2px solid currentColor;white-space:nowrap}
  .mrow .mtxt{font-size:13.5px;color:var(--dim);line-height:1.35}
  .mrow .mtxt b{color:var(--ink);font-family:"DejaVu Sans Mono",monospace;font-size:14px}
  .t-good{color:var(--green)} .t-warn{color:var(--amber)} .t-bad{color:var(--red)}
  .t-idle{color:#54687f}
  .mlegend{display:flex;gap:7px;flex-wrap:wrap;margin-top:11px;padding-top:10px;
           border-top:1px solid var(--line)}
  .mlegend div{font-size:10.5px;font-weight:800;letter-spacing:.1em;border:1.5px solid currentColor;
               border-radius:5px;padding:3px 7px}
  .mnote{font-size:12px;color:#54687f;margin-top:8px;line-height:1.45}

  /* ---------- layout ---------- */
  .grid{display:grid;grid-template-columns:minmax(360px,0.85fr) minmax(520px,1.25fr);gap:16px;align-items:start}
  .card{background:var(--panel);border:2px solid var(--line);border-radius:14px;padding:14px 16px;margin-bottom:14px}
  .card h2{margin:0 0 10px;font-size:12px;letter-spacing:.2em;text-transform:uppercase;color:var(--dim);font-weight:700}

  .order{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  .oid{font-family:"DejaVu Sans Mono",monospace;font-size:15px;color:var(--cyan);letter-spacing:.05em}
  .item{font-size:38px;font-weight:800;line-height:1.05}
  .arrow{font-size:26px;color:var(--dim)}
  .binbig{font-size:38px;font-weight:800;padding:0 16px;border-radius:10px;line-height:1.2}
  .kv{display:flex;gap:10px;font-size:14px;color:var(--dim);margin-top:8px;flex-wrap:wrap}
  .kv b{color:var(--ink);font-family:"DejaVu Sans Mono",monospace}

  .reason{font-size:19px;line-height:1.4;font-weight:600}
  .qcrow{display:flex;gap:12px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
  .grade{font-size:26px;font-weight:800;padding:4px 14px;border-radius:9px;letter-spacing:.06em}
  .defect{font-size:17px;color:var(--amber)}
  .divert{margin-top:10px;font-size:16px;font-weight:700;color:var(--amber);
          border-left:5px solid var(--amber);padding-left:12px}

  .outcome{border-radius:14px;padding:18px 20px;border:3px solid var(--line);background:var(--panel)}
  .outcome .big{font-size:44px;font-weight:800;letter-spacing:.06em;line-height:1}
  .outcome .sub{font-size:16px;color:var(--dim);margin-top:8px;line-height:1.4}
  .outcome.fulfilled{border-color:var(--green);background:#0b1a12}
  .outcome.fulfilled .big{color:var(--green)}
  .outcome.retried{border-color:var(--amber);background:#1c1408}
  .outcome.retried .big{color:var(--amber)}
  .outcome.failed{border-color:var(--red);background:#210e11}
  .outcome.failed .big{color:var(--red)}
  .outcome.running .big{color:var(--cyan)}

  .alert{border-left:6px solid var(--dim);background:var(--panel2);border-radius:8px;
         padding:10px 12px;margin-bottom:9px}
  .alert .an{font-size:16px;font-weight:700}
  .alert .ad{font-size:13px;color:var(--dim);margin-top:4px;line-height:1.4}
  .alert.info{border-color:var(--green)} .alert.warning{border-color:var(--amber)}
  .alert.error,.alert.critical{border-color:var(--red)}
  .tag{display:inline-block;font-size:10.5px;letter-spacing:.12em;font-weight:800;text-transform:uppercase;
       border:1.5px solid currentColor;border-radius:5px;padding:2px 7px;margin-left:8px;vertical-align:2px}
  .tag.sent{color:var(--green)} .tag.mockt{color:var(--violet)} .tag.offl{color:var(--amber)}

  /* ---------- camera ---------- */
  .cam{position:relative;border:2px solid var(--line);border-radius:14px;overflow:hidden;background:#000}
  .cam img{display:block;width:100%;height:auto}
  .zone{position:absolute;border:3px solid #fff;border-radius:4px;transition:box-shadow .2s,border-color .2s}
  .zone .zl{position:absolute;top:-13px;left:-3px;font-size:12px;font-weight:800;letter-spacing:.1em;
            padding:1px 7px;border-radius:5px;color:#06121b;white-space:nowrap}
  .zone.active{box-shadow:0 0 0 3px rgba(255,255,255,.28),0 0 26px rgba(255,255,255,.25)}
  .zone.ok{border-color:var(--green)!important;box-shadow:0 0 0 4px rgba(34,197,94,.35),0 0 40px rgba(34,197,94,.5)}
  .zone.fail{border-color:var(--red)!important;box-shadow:0 0 0 4px rgba(239,68,68,.4),0 0 40px rgba(239,68,68,.55);
             animation:blink .7s 4}
  @keyframes blink{50%{opacity:.35}}
  .pt{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%;
      border:3px solid var(--cyan);box-shadow:0 0 16px rgba(56,189,248,.9)}
  .pt::before,.pt::after{content:"";position:absolute;background:var(--cyan)}
  .pt::before{left:10px;top:-14px;width:2px;height:12px}
  .pt::after{left:-14px;top:10px;width:12px;height:2px}
  .pt .lab{position:absolute;left:32px;top:-6px;white-space:nowrap;font-size:13px;font-weight:700;
           color:#06121b;background:var(--cyan);padding:3px 9px;border-radius:6px}
  .ring{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%;
        border:3px solid var(--cyan);animation:ring 1.4s infinite}
  @keyframes ring{0%{transform:scale(1);opacity:.85}100%{transform:scale(3.2);opacity:0}}
  .vmark{position:absolute;font-size:30px;font-weight:900;margin:-22px 0 0 -15px;
         text-shadow:0 0 12px rgba(0,0,0,.95)}
  .camfoot{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:10px;
           font-size:13px;color:var(--dim)}
  .camfoot b{color:var(--ink);font-family:"DejaVu Sans Mono",monospace}

  /* ---------- controls ---------- */
  .ctl{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
  button{font:inherit;font-size:14px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;
         background:var(--panel);color:var(--dim);border:2px solid var(--line);border-radius:10px;
         padding:11px 17px;cursor:pointer;transition:all .15s}
  button:hover{color:var(--ink);border-color:#3a5170}
  button.on{background:#0e2130;border-color:var(--cyan);color:var(--cyan)}
  .ctl .sep{width:1px;height:26px;background:var(--line);margin:0 5px}
  .ctl .lbl{font-size:11px;letter-spacing:.18em;color:#54687f;font-weight:700;text-transform:uppercase}
  .blurb{font-size:14px;color:var(--dim);margin:-4px 0 14px}

  .log{font-family:"DejaVu Sans Mono",monospace;font-size:12.5px;line-height:1.65;color:var(--dim);
       max-height:190px;overflow:hidden}
  .log .t{color:#46586e}
</style>

<div class="wrap">
  <header>
    <h1>Order-Driven Zero-Shot Picking Cell<small id="modeblurb">&nbsp;</small></h1>
    <div class="badges" id="badges"></div>
  </header>

  <div class="pipe" id="pipe"></div>

  <div class="ctl" id="ctl"></div>
  <div class="blurb" id="blurb"></div>

  <div class="grid">
    <div>
      <div class="card"><h2>Incoming order</h2><div id="order"></div></div>
      <div class="card" id="routecard"><h2>Routing decision</h2><div id="route"></div></div>
      <div class="card" id="motioncard">
        <h2>Motion confirmation — did the arm actually move?</h2>
        <div id="motion"></div>
        <div class="mlegend" id="mlegend"></div>
        <div class="mnote">MotionExecutor publishes to MQTT and prints “plan complete” with no
          servo acknowledgement. Only an encoder read-back (verify_pose, ±<span id="tol">5</span>°)
          can turn COMMANDED into VERIFIED.</div>
      </div>
      <div class="outcome running" id="outcome"><div class="big">WAITING</div><div class="sub"></div></div>
      <div class="card" style="margin-top:14px"><h2>Platform alerts</h2><div id="alerts"></div></div>
    </div>
    <div>
      <div class="card" style="padding:12px">
        <h2>Overhead camera · overlays</h2>
        <div class="cam" id="cam"><img id="frame" alt="overhead frame"><div id="ov"></div></div>
        <div class="camfoot" id="camfoot"></div>
      </div>
      <div class="card"><h2>Event log</h2><div class="log" id="log"></div></div>
    </div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
let LAST_TOKEN = "", LAST_RUN = -1, STAGES = [];

function el(tag, cls, txt){ const e=document.createElement(tag); if(cls)e.className=cls;
  if(txt!==undefined&&txt!==null)e.textContent=txt; return e; }

function buildPipe(stages){
  const pipe = $("pipe"); pipe.innerHTML = "";
  STAGES = stages.map(s => {
    const d = el("div","stage");
    d.appendChild(el("div","n",s.name));
    d.appendChild(el("div","b",s.blurb||""));
    d.appendChild(el("div","d",""));
    d.appendChild(el("div","ms",""));
    d.appendChild(el("div","mo",""));
    pipe.appendChild(d);
    return d;
  });
}

/* {"elbow": 7.4} -> "elbow +7.4°" — the per-joint encoder error, worst first. */
function errText(errors){
  const keys = Object.keys(errors||{});
  if(!keys.length) return "";
  keys.sort((a,b) => Math.abs(errors[b]) - Math.abs(errors[a]));
  return keys.map(k => k+" "+(errors[k]>=0?"+":"")+errors[k].toFixed(1)+"°").join(" · ");
}

const MOTION_ICON = {commanded:"◷", verified:"✓", mismatch:"✗", unverified:"⚠"};

function drawPipe(stages){
  if(STAGES.length !== stages.length) buildPipe(stages);
  stages.forEach((s,i) => {
    const d = STAGES[i];
    const m = s.motion;
    d.className = "stage " + (s.status==="idle" ? "" : s.status) + (m ? " m-"+m.state : "");
    d.children[2].textContent = s.detail || "";
    d.children[3].textContent = s.ms===null||s.ms===undefined ? "" :
      (s.ms >= 1000 ? (s.ms/1000).toFixed(1)+"s" : Math.round(s.ms)+"ms");
    const mo = d.children[4];
    if(m){
      const e = errText(m.errors);
      mo.textContent = (MOTION_ICON[m.state]||"") + " " + m.label + (e ? " · " + e : "");
      mo.style.display = "block";
    } else {
      mo.textContent = ""; mo.style.display = "none";
    }
  });
}

function drawMotion(d){
  const box = $("motion"); box.innerHTML = "";
  (d.motion||[]).forEach(m => {
    const row = el("div","mrow");
    row.appendChild(el("div","ms", m.stage));
    const st = el("div","mstate t-"+(m.tone||"idle"),
                  (MOTION_ICON[m.state]||"") + " " + (m.label||"—"));
    row.appendChild(st);
    const txt = el("div","mtxt");
    if(m.state==="mismatch" && m.summary){
      txt.appendChild(el("b", m.summary));
      txt.appendChild(document.createTextNode("  (tolerance ±"+m.tolerance_deg.toFixed(0)+"°)"));
    } else {
      txt.textContent = m.meaning || "";
    }
    row.appendChild(txt);
    box.appendChild(row);
  });
  const lg = $("mlegend"); lg.innerHTML = "";
  (d.motion_legend||[]).forEach(l => {
    const c = el("div","t-"+l.tone, (MOTION_ICON[l.state]||"")+" "+l.label);
    c.title = l.meaning; lg.appendChild(c);
  });
  const tol = (d.motion||[]).find(m => m.tolerance_deg);
  if(tol) $("tol").textContent = tol.tolerance_deg.toFixed(0);
}

function badges(d){
  const p = d.platform || {}, id = p.identity || {};
  const rt = (p.runtime||"MOCK").toUpperCase();
  const rtc = rt==="MOCK" ? "mock" : (rt==="LIVE" ? "live" : "sim");
  const st = p.status||"mock";
  const stc = st==="connected" ? "on" : (st==="mock" ? "mock" : "off");
  const b = $("badges"); b.innerHTML = "";
  const add = (cls, label, val) => {
    const e = el("div","badge "+cls);
    if(label){ const s=el("span",null,label); e.appendChild(s); }
    e.appendChild(el("b",null,val)); b.appendChild(e);
  };
  add(rtc, "runtime", rt);
  add(d.source==="live" ? "on" : "mock", "data",
      d.source==="live" ? "LIVE RUN EVENTS" : "MOCK SCRIPT");
  add(stc, "cyberwave", st==="connected" ? (p.dry_run ? "CONNECTED · ALERTS DRY-RUN" : "CONNECTED") :
                        st==="mock" ? "MOCK — NOT SENT" :
                        st==="connecting" ? "CONNECTING…" : "OFFLINE");
  add("id", "env", id.environment_id || "—");
  add("id", "twin", (id.twin_id || "—"));
  const a = el("a","badge"); a.href = id.dashboard_url || "#"; a.target="_blank";
  a.textContent = "CYBERWAVE DASHBOARD ↗"; b.appendChild(a);
}

function controls(d){
  const c = $("ctl"); c.innerHTML = "";
  c.appendChild(el("div","lbl","mode"));
  ["order","qc","fusion"].forEach(m => {
    const btn = el("button", d.mode===m ? "on" : "", m);
    btn.onclick = () => post("/mode", {mode:m});
    c.appendChild(btn);
  });
  if(d.controls && d.controls.mock){
    c.appendChild(el("div","sep"));
    c.appendChild(el("div","lbl","canned outcome"));
    [["auto","auto · all three"],["ok","verified"],["retry","mismatch + retry"],
     ["fail","ghost arm + alert"]].forEach(([k,label]) => {
      const btn = el("button", d.controls.outcome===k ? "on" : "", label);
      btn.onclick = () => post("/replay", {outcome:k});
      c.appendChild(btn);
    });
    if(d.controls.outcome==="auto" && d.controls.playing){
      c.appendChild(el("div","lbl","playing: "+d.controls.playing));
    }
    c.appendChild(el("div","sep"));
    const r = el("button","","▶ replay"); r.onclick = () => post("/replay", {});
    c.appendChild(r);
  }
  $("blurb").textContent = d.mode_blurb || "";
  $("modeblurb").textContent = (d.mode||"").toUpperCase() + " MODE";
}

function zoneCss(d, label){
  const z = (d.overlay.zones||[]).find(z => z.label===label);
  return z ? z.css : "#94a3b8";
}

function drawOrder(d){
  const o = $("order"); o.innerHTML = "";
  if(!d.order){ o.appendChild(el("div","kv","waiting for POST /orders …")); return; }
  const row = el("div","order");
  row.appendChild(el("div","oid", d.order.order_id || "(no id)"));
  row.appendChild(el("div","item", d.order.item || "—"));
  row.appendChild(el("div","arrow","→"));
  const dest = (d.routing && d.routing.bin) || d.order.bin;
  const bb = el("div","binbig", dest ? dest : "?");
  if(dest){ bb.style.background = zoneCss(d, dest); bb.style.color = "#06121b"; }
  else { bb.style.color = "#8ea3bd"; bb.title = "no bin in the order — QC decides"; }
  row.appendChild(bb);
  o.appendChild(row);
  const kv = el("div","kv");
  if(!d.order.bin) kv.appendChild(el("span",null,"order carries NO bin — QC decides the zone"));
  if(d.attempts>1) kv.appendChild(el("span",null,"attempts: "+d.attempts));
  o.appendChild(kv);
}

function drawRoute(d){
  const r = $("route"); r.innerHTML = "";
  if(d.qc){
    const row = el("div","qcrow");
    const g = el("div","grade", d.qc.grade.toUpperCase());
    const gc = d.qc.grade==="pass" ? "#22c55e" : (d.qc.grade==="review" ? "#f59e0b" : "#ef4444");
    g.style.background = gc; g.style.color = "#06121b";
    row.appendChild(el("div","kv","VLM QC"));
    row.appendChild(g);
    if(d.qc.confidence) row.appendChild(el("div","kv", (d.qc.confidence*100).toFixed(0)+"% conf"));
    r.appendChild(row);
    if(d.qc.defect) r.appendChild(el("div","defect","defect: "+d.qc.defect));
    if(d.qc.detail) r.appendChild(el("div","kv", d.qc.detail));
  }
  if(d.routing){
    r.appendChild(el("div","reason", d.routing.reason || ""));
    if(d.routing.diverted){
      r.appendChild(el("div","divert",
        "DIVERTED — ordered bin " + (d.routing.requested_bin||"?") +
        " → zone " + d.routing.bin + " (" + ((d.routing.zone&&d.routing.zone.color)||"") + ")"));
    }
  } else if(!d.qc){
    r.appendChild(el("div","kv","not decided yet"));
  }
}

function drawOutcome(d){
  const o = $("outcome");
  o.className = "outcome " + d.outcome;
  const words = {running:"RUNNING", fulfilled:"FULFILLED", retried:"FULFILLED (RETRIED)", failed:"FAILED"};
  o.children[0].textContent = words[d.outcome] || d.outcome.toUpperCase();
  let sub = d.outcome_detail || "";
  if(d.verification && !sub) sub = d.verification.reason;
  o.children[1].textContent = sub;
}

function drawAlerts(d){
  const a = $("alerts"); a.innerHTML = "";
  if(!d.alerts.length){ a.appendChild(el("div","kv","none yet")); return; }
  d.alerts.forEach(al => {
    const box = el("div","alert "+(al.severity||"info"));
    const n = el("div","an", al.name);
    const p = al.platform || {};
    /* Four honest answers, never "green by default": a mock alert, a dry-run alert, a
       really-dispatched alert, and one that failed / was raised elsewhere. */
    const cls = p.mock ? "mockt" : (p.dispatched ? "sent" : "offl");
    const txt = p.mock ? "MOCK — NOT SENT"
              : p.dry_run ? "DRY RUN — NOT SENT"
              : p.dispatched ? "SENT TO CYBERWAVE"
              : (p.detail || "offline");
    const tag = el("span","tag " + cls, txt);
    n.appendChild(tag);
    box.appendChild(n);
    box.appendChild(el("div","ad", al.description||""));
    a.appendChild(box);
  });
}

function drawCam(d){
  if(d.frame_token !== LAST_TOKEN){
    LAST_TOKEN = d.frame_token;
    $("frame").src = "/frame.jpg?t=" + encodeURIComponent(LAST_TOKEN);
  }
  const ov = $("ov"); ov.innerHTML = "";
  const o = d.overlay || {};
  (o.zones||[]).forEach(z => {
    const box = el("div","zone" + (z.active ? " active" : "") + (z.verdict ? " "+z.verdict : ""));
    box.style.left = (z.x*100)+"%"; box.style.top = (z.y*100)+"%";
    box.style.width = (z.w*100)+"%"; box.style.height = (z.h*100)+"%";
    box.style.borderColor = z.css;
    const lab = el("div","zl", "ZONE "+z.label + (z.active ? " ◀ TARGET" : ""));
    lab.style.background = z.css;
    box.appendChild(lab);
    ov.appendChild(box);
  });
  if(o.point){
    const ring = el("div","ring");
    ring.style.left = (o.point.x*100)+"%"; ring.style.top = (o.point.y*100)+"%";
    ov.appendChild(ring);
    const p = el("div","pt");
    p.style.left = (o.point.x*100)+"%"; p.style.top = (o.point.y*100)+"%";
    const mm = o.point_mm ? "  ·  " + o.point_mm[0].toFixed(0) + ", " + o.point_mm[1].toFixed(0) + " mm" : "";
    p.appendChild(el("div","lab", (o.point_label||"detected") + mm));
    ov.appendChild(p);
  }
  if(o.verify){
    const v = el("div","vmark", o.verify_ok ? "✓" : "✗");
    v.style.left = (o.verify.x*100)+"%"; v.style.top = (o.verify.y*100)+"%";
    v.style.color = o.verify_ok ? "#22c55e" : "#ef4444";
    ov.appendChild(v);
  }
  const f = $("camfoot"); f.innerHTML = "";
  const add = (t) => f.appendChild(el("div",null,t));
  add("frame " + d.frame_size[0] + "×" + d.frame_size[1] +
      "  ·  view " + d.view.w + "×" + d.view.h + " @ " + d.view.x + "," + d.view.y);
  add(d.calibrated ? "zones: hardware/config/bin-regions.json (surveyed)" : "zones: FALLBACK (not surveyed)");
  if(o.point && o.point_mm) add("homography px → table mm: OK");
  add("t+" + (d.elapsed||0).toFixed(1) + "s");
}

function drawLog(d){
  const l = $("log"); l.innerHTML = "";
  (d.log||[]).slice().reverse().forEach(e => {
    const line = el("div");
    line.appendChild(el("span","t", "+" + e.t.toFixed(1).padStart(5) + "s  "));
    line.appendChild(document.createTextNode(e.text));
    l.appendChild(line);
  });
}

function render(d){
  if(d.run_id !== LAST_RUN){ LAST_RUN = d.run_id; }
  badges(d); controls(d); drawPipe(d.stages); drawOrder(d); drawRoute(d);
  drawMotion(d); drawOutcome(d); drawAlerts(d); drawCam(d); drawLog(d);
}

async function post(path, body){
  try{
    await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                       body: JSON.stringify(body||{})});
    await tick();
  }catch(e){ /* offline UI must keep working */ }
}

async function tick(){
  try{
    const r = await fetch("/state", {cache:"no-store"});
    render(await r.json());
  }catch(e){ /* server restarting — keep the last paint */ }
}

tick();
setInterval(tick, 250);
</script>
"""


def render_page() -> bytes:
    return PAGE.encode("utf-8")
