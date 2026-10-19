// Render the dashboard's machine grid outside a browser.
//
// The page's whole script block is evaluated by dashboard_page.mjs (the shared
// harness); this file drives what that gives back. The grid is rendered from
// hand-built overview rows -- online, offline, killswitched, reserve, hub,
// self -- and the resulting HTML is asserted on. Catches exactly what a syntax
// check cannot: a template that throws on a null status, a pill on the wrong
// card.
import vm from "node:vm";

import {ctx, sink, html, src} from "./dashboard_page.mjs";

const spec = (over = {}) => ({cpu: "CPU", gpu: "GPU", ram_gb: 128, vram_gb: 96,
                             mem_bw_gbs: 256, gpu_tflops: 40, ...over});
const liveStatus = {
  host: {cpu_count: 32, cpu_percent: 12, mem: {total: 1e11, used: 4e10},
         gpu: [{vram_total: 1e11, vram_used: 2e10, busy_percent: 30}],
         net: {interfaces: [{name: "eth0", wireless: false, rx_rate: 1e6, tx_rate: 2e5}]}},
  models_running: [{id: "qwen3.8-27b", state: "ready"}],
};
const HOSTS = [
  {name: "hub-box", kind: "self", online: true, routed: true,
   specs: spec({role: "hub"}), status: liveStatus},
  {name: "box-a", kind: "peer", online: true, routed: true,
   specs: spec(), status: liveStatus},
  {name: "box-b", kind: "peer", online: true, routed: false,
   specs: spec({mem_bw_gbs: 100}), status: liveStatus},
  // The case the old renderer never had to survive: no status at all.
  {name: "box-c", kind: "peer", online: false, routed: true,
   specs: spec({mem_bw_gbs: 200}), status: null, last_seen: "2026-08-28T00:00:00+00:00"},
  {name: "box-d", kind: "peer", online: false, routed: false,
   specs: spec({mem_bw_gbs: 50}), status: null},
  // Somebody's personal machine: routed, but flagged reserve in its specs.
  {name: "box-e", kind: "peer", online: true, routed: true,
   specs: spec({mem_bw_gbs: 150, reserve: true, rank: 0}), status: liveStatus},
];

ctx.renderMachines(HOSTS);
const out = sink["#machines"];
const cards = out.split('<div class="card machine').slice(1);
const nameOf = (c) => (c.match(/class="mname">([^<]+)/) || [])[1];

const fail = [];
const ok = (cond, msg) => { if (!cond) fail.push(msg); };

ok(cards.length === 6, `6 cards rendered, got ${cards.length}`);
ok(JSON.stringify(cards.map(nameOf))
   === JSON.stringify(["box-a", "box-e", "box-b", "box-c", "box-d", "hub-box"]),
   "order: online by bandwidth, then offline, then the hub — got "
   + JSON.stringify(cards.map(nameOf)));

const byName = Object.fromEntries(cards.map((c) => [nameOf(c), c]));
ok(byName["box-c"].includes('<span class="pill no">offline</span>'),
   "an offline box gets the red offline pill");
ok(byName["box-a"].includes('<span class="pill ok">serving</span>'),
   "a serving box keeps its green pill");

// The routing policy is edited on the Configurations tab now; the cards only
// MIRROR it. No card carries the old killswitch toggle any more, and each
// non-hub card says which of the three states it is in: a place in the
// routing order, reserve, or not in use.
ok(!out.includes("toggleRouting") && !out.includes('class="krow"'),
   "no machine card carries the old routing toggle");
ok(byName["box-b"].includes(">not in use</span>"),
   "an online-but-killed box says 'not in use'");
ok(byName["box-d"].includes('<span class="pill no">offline</span>')
   && byName["box-d"].includes(">not in use</span>"),
   "offline AND killed shows both");
ok(byName["box-e"].includes(">reserve</span>"),
   "a reserve box carries the reserve pill");
ok(!byName["box-e"].includes("routing #"),
   "a reserve box gets no routing number — its position is policy");
// Routed, non-reserve, no rank in specs: ordered by name, so box-a then
// box-c (offline still holds its place: positions are policy, not liveness).
ok(byName["box-a"].includes("routing #1"),
   "the first routed box shows routing #1");
ok(byName["box-c"].includes("routing #2"),
   "an offline routed box keeps its place in the order");
ok(!byName["hub-box"].includes("routing #")
   && byName["hub-box"].includes(">hub</span>"),
   "the hub shows its own pill, never a routing number");

// The argument crosses two layers now (JSON.stringify then esc(), see jsArg
// in index.html) so a plain host name renders as a double-quoted JS string
// literal with its quotes HTML-escaped: switchHost(&quot;box-a&quot;).
ok(byName["box-c"].includes(`showHostDetail(&quot;box-c&quot;)`),
   "an offline card opens its last-known detail, not a dead switchHost");
ok(byName["box-a"].includes(`switchHost(&quot;box-a&quot;)`),
   "an online card still switches to the machine");
ok(byName["box-c"].includes("CPU · 0 threads"),
   "a status-less card renders instead of throwing");
// The split above ate the `<div class="card machine` prefix, so what is left
// at the head of an offline card is the modifier itself.
ok(byName["box-c"].startsWith(' off"'), "an offline card is marked stale");
ok(byName["box-a"].startsWith('"'), "an online card carries no stale marker");

// And the empty fleet, which used to say "nothing online".
ctx.renderMachines([]);
ok(sink["#machines"].includes("no machines registered"), "empty fleet copy");

// ---------------------------------------------------------------------------
// The killswitch endpoint's client. The switch moved from the machine cards
// into the Configurations tab's box detail, but the function is the same one
// and so is its historical bug: the first version shipped writing `routed:
// !killed` -- which for a live box is `!false`, i.e. the state it already
// had. Every click was a no-op that then announced "killswitch is ON". So
// what is asserted here is the DIRECTION of the write, not just that a write
// happens: on -> {routed:false}, off -> {routed:true}.
// ---------------------------------------------------------------------------
const sw = (checked) => {
  const cls = new Set(["ksw"]);
  const state = {textContent: checked ? "on" : "off",
                 classList: {toggle(c, v) { state[c] = !!v; }}};
  return {
    attrs: {"aria-checked": String(checked)},
    getAttribute(k) { return this.attrs[k]; },
    setAttribute(k, v) { this.attrs[k] = v; },
    classList: {contains: (c) => cls.has(c), toggle(c, v) { v ? cls.add(c) : cls.delete(c); }},
    parentElement: {querySelector: () => state},
    state, busy: () => cls.has("busy"),
  };
};

// Every rawApi call, of which the killswitch's PUT is only one: a successful
// flip also refetches the fleet overview AND the routing config (both views
// mirror the same flag), so select rather than index.
const calls = [];
const puts = () => calls.filter((c) => c.method === "PUT");
ctx.rawApi = async (path, opts) => { calls.push({path, ...opts}); return {}; };
const toasts = [];
ctx.toast = (m, bad) => toasts.push({m, bad});

const onSwitch = sw(true);
await ctx.toggleRouting(onSwitch, "box-a");
ok(puts().length === 1, `one PUT per click, got ${puts().length}`);
ok(puts()[0].path === "/peers/box-a/routed", `PUT path, got ${puts()[0].path}`);
ok(calls.some((c) => c.path === "/fleet/overview"),
   "a successful flip redraws from the hub rather than trusting itself");
ok(calls.some((c) => c.path === "/fleet-config"),
   "a successful flip also refreshes the Configurations lists");
ok(puts()[0].body.routed === false,
   `flipping a ROUTED box must send routed:false, sent ${JSON.stringify(puts()[0].body)}`);
ok(onSwitch.getAttribute("aria-checked") === "false", "the knob ends off");
ok(onSwitch.state.textContent === "off", "the word next to it ends off");
ok(!onSwitch.busy(), "the busy marker is cleared");

const offSwitch = sw(false);
await ctx.toggleRouting(offSwitch, "box-b");
ok(puts()[1].body.routed === true,
   `flipping a KILLED box must send routed:true, sent ${JSON.stringify(puts()[1].body)}`);
ok(offSwitch.getAttribute("aria-checked") === "true", "the knob ends on");
ok(toasts.length === 0, "a switch that worked says nothing — it just moved");

// A hub that refuses puts the knob back rather than lying about the state.
ctx.rawApi = async () => { throw new Error("boom"); };
const failing = sw(true);
await ctx.toggleRouting(failing, "box-c");
ok(failing.getAttribute("aria-checked") === "true", "a refused flip slides back");
ok(failing.state.textContent === "on", "and so does its label");
ok(!failing.busy(), "and it is usable again");
ok(toasts.length === 1 && toasts[0].bad, "a refused flip does say why");

// ---------------------------------------------------------------------------
// The Configurations tab. Its lists render from a fleet-config payload; the
// order list numbers rows, marks this box (the hub) pinned and undraggable,
// and carries the reserve pill; the not-in-use list holds the killed boxes.
// ---------------------------------------------------------------------------
const cfgFixture = {
  hosts: {
    "hub-box": {name: "hub-box", self: true, klass: "hub", reserve: false,
                rank: null, routed: true},
    "box-a": {name: "box-a", self: false, klass: "gpu", reserve: false,
              rank: 0, routed: true},
    "box-e": {name: "box-e", self: false, klass: "gpu", reserve: true,
              rank: 1, routed: true},
    "box-b": {name: "box-b", self: false, klass: "small", reserve: false,
              rank: null, routed: false},
  },
  order: ["box-a", "box-e", "hub-box"],
  not_in_use: ["box-b"],
  saved: {},
};
vm.runInContext(
  "CFG = " + JSON.stringify(cfgFixture) + "; OVERVIEW = null; renderConfig();",
  ctx);
const orderHtml = sink["#cfg-order"] || "";
const outHtml = sink["#cfg-out"] || "";
const rowsOf = (h) => h.split('<div class="cfgrow').slice(1);
ok(rowsOf(orderHtml).length === 3, "three rows in the routing order");
ok(rowsOf(outHtml).length === 1, "one row not in use");
ok(orderHtml.indexOf("box-a") < orderHtml.indexOf("box-e"),
   "the order list renders in order");
const hubRow = rowsOf(orderHtml).find((r) => r.includes("hub-box")) || "";
ok(hubRow.startsWith(' pinned"'), "this box's own row is pinned");
ok(!hubRow.includes('draggable="true"'), "and cannot be dragged");
ok(!hubRow.includes("#"), "and carries no routing number");
const aRow = rowsOf(orderHtml).find((r) => r.includes("box-a")) || "";
ok(aRow.includes('draggable="true"'), "a peer row drags");
ok(aRow.includes("#1"), "the first normal box is #1");
const eRow = rowsOf(orderHtml).find((r) => r.includes("box-e")) || "";
ok(eRow.includes(">reserve</span>"), "a reserve box says so in the list");
// Numbering counts only the boxes competing in the normal order -- reserve
// and self rows carry pills instead -- so this list and the Home page's
// "routing #N" pills (routePositions) always agree on a box's number.
ok(!eRow.includes("#"), "a reserve row carries a pill, not a number");
ok(outHtml.includes("box-b"), "the killed box sits in Not in use");
vm.runInContext(
  'CFG = {hosts: {"only-box": {name: "only-box", self: true}}, '
  + 'order: ["only-box"], not_in_use: []}; renderConfig();', ctx);
ok((sink["#cfg-order"] || "").includes("no peers"),
   "a box with no peers says the order lives on the hub");

// The static side of the tab: the nav button, the section, the tab set.
ok(html.includes('<button data-t="config">'),
   "the fleet nav has a Configurations button");
ok(html.includes('<section id="config">'), "and the section exists");
ok(/FLEET_TABS = new Set\(\[[^\]]*"config"/.test(src),
   "config is a fleet-level tab");
// Routing classes: an editor on the hub, so a fleet tab with a loader and no
// poll.
ok(html.includes('<button data-t="routing">Routing</button>'),
   "the fleet nav has a Routing button");
ok(html.includes('<section id="routing">'), "and the section exists");
ok(/FLEET_TABS = new Set\(\[[^\]]*"routing"/.test(src),
   "routing is a fleet-level tab");
ok(/LOADERS = \{[^}]*routing: loadRouting/.test(src),
   "and the tab loads its classes when it opens");
// Each setInterval(...) call, parenthesis to parenthesis. A pattern that stops
// at the first `)` only ever saw the bare `setInterval(loadX, 5000)`: the
// page's own pollers are arrow functions, `() => { ... loadX(); }`, which carry
// a `)` before the name.
const pollers = (code) => {
  const out = [];
  for (let i = code.indexOf("setInterval("); i >= 0;
       i = code.indexOf("setInterval(", i + 1)) {
    let depth = 0, j = i + "setInterval".length;
    do {
      const c = code[j++];
      depth += c === "(" ? 1 : c === ")" ? -1 : 0;
    } while (depth > 0 && j < code.length);
    out.push(code.slice(i, j));
  }
  return out;
};
const polls = (code, fn) => pollers(code).some((p) => new RegExp("\\b" + fn + "\\b").test(p));
ok(polls('setInterval(() => { if ($("#routing").classList.contains("on")) loadRouting(); }, 5000);',
         "loadRouting"), "the poll check catches an arrow-function poller");
ok(polls("setInterval(loadRouting, 30000);", "loadRouting"), "and the bare form");
ok(polls(src, "loadHome"), "and it finds the page's real pollers (loadHome is one)");
ok(!polls(src, "loadRouting"), "an editor is not polled");

// ---------------------------------------------------------------------------
// The Overview card's helpers. fillPick must not rewrite a <select> the
// operator is mid-choice in (the 5s poll used to reset it), and
// watchManualLoad announces a dashboard-started load exactly once.
// ---------------------------------------------------------------------------
const selStub = () => {
  const s = {dataset: {}, value: "", _html: "", options: []};
  Object.defineProperty(s, "innerHTML", {
    set(v) {
      s._html = v;
      s.options = [...v.matchAll(/value="([^"]*)"/g)].map((m) => ({value: m[1]}));
    },
    get() { return s._html; },
  });
  return s;
};
const pick = selStub();
ctx.fillPick(pick, [{v: "", t: "—"}, {v: "a", t: "a"}], true);
pick.value = "a";
const before = pick._html;
ctx.fillPick(pick, [{v: "", t: "—"}, {v: "a", t: "a"}], true);
ok(pick._html === before && pick.value === "a",
   "an unchanged option set leaves the select (and the choice) alone");
ctx.fillPick(pick, [{v: "", t: "—"}, {v: "a", t: "a"}, {v: "b", t: "b"}], true);
ok(pick.value === "a", "a grown option set keeps the operator's choice");
ok(pick.options.length === 3, "and carries the new option");

toasts.length = 0;
ctx.watchManualLoad({model: "m1", status: "loading", at: "t1"});
ok(toasts.length === 0, "a load merely starting says nothing");
ctx.watchManualLoad({model: "m1", status: "loading", at: "t1"});
ok(toasts.length === 0, "and repeating the same state stays quiet");
ctx.watchManualLoad({model: "m1", status: "ok", at: "t2"});
ok(toasts.length === 1 && !toasts[0].bad && toasts[0].m.includes("m1"),
   "a finished load is announced once");
ctx.watchManualLoad({model: "m1", status: "ok", at: "t2"});
ok(toasts.length === 1, "and only once");
ctx.watchManualLoad({model: "m2", status: "loading", at: "t3"});
ctx.watchManualLoad({model: "m2", status: "failed", why: "boom", at: "t4"});
ok(toasts.length === 2 && toasts[1].bad && toasts[1].m.includes("boom"),
   "a failed load says why, loudly");

// ---------------------------------------------------------------------------
// The Public tab is gone (2026-09-17): its whole surface -- Fleet Pass keys,
// employers, auto-issue, the public settings -- is now the CareerOps admin's
// own Public page, driven through the same /admin/api/public/* routes this
// gateway still serves. What the dashboard kept is the one thing the Teams
// and API-keys tables borrow, so check it is still here rather than checking
// the sub-nav that left with the tab.
// ---------------------------------------------------------------------------
ok(!html.includes('<section id="public">'), "the Public section is gone");
ok(!html.includes('data-t="public"'), "...and so is its nav button");
ok(!/\bFLEET_TABS\s*=\s*new Set\([^)]*"public"/.test(src),
   "...and the tab registry no longer names it");
ok(/async function publicKeyIds\(/.test(src),
   "publicKeyIds() stayed -- the Teams and API-keys tables badge a profile "
   + "'fleet-pass' from it, and it is the only piece of the tab they used");
ok((src.match(/await publicKeyIds\(\)/g) || []).length === 2,
   "both of its callers are still calling it");

// ---------------------------------------------------------------------------
// A node whose model runs on a device attached to its box (`hw: "spec"`):
// the gateway's live telemetry is the PC under it -- 11 GB, an iGPU, its NIC
// -- and none of that may be drawn as the node. The sheet is the node.
// ---------------------------------------------------------------------------
const pcStatus = {
  host: {cpu_count: 12, cpu_percent: 40, mem: {total: 11.4e9, used: 9e9},
         gpu: [{vram_total: 2e9, vram_used: 1e9, busy_percent: 80}],
         net: {interfaces: [{name: "Wi-Fi", wireless: true, rx_rate: 1e6, tx_rate: 1e6}]}},
  models_running: [{id: "glm-4.7-flash", state: "ready"}],
};
const npuBox = {name: "npu-box", kind: "peer", online: true, routed: false,
  specs: {cpu: "12-core ARMv9.2", gpu: "dNPU (48 GB) + G720", ram_gb: 32, vram_gb: 48,
          npu_tops: 190, cpu_cores: 12, hw: "spec", reserve: true},
  status: pcStatus};
const GiB = vm.runInContext("GB", ctx);
const npuRes = ctx.hostRes(npuBox);
ok(npuRes.ramT === 32 * GiB && npuRes.vramT === 48 * GiB,
   "a spec-hw node's memory is its sheet's, not the PC's");
ok(npuRes.ramU === 0 && npuRes.vramU === 0 && npuRes.coresU === 0 && npuRes.tf === 0,
   "...and none of the PC's load is counted as the node's, nor any GPU TFLOPS");
ok(npuRes.cores === 12, "its cores come from the sheet");
ctx.renderMachines([npuBox]);
const npuCard = sink["#machines"];
ok(npuCard.includes("NPU compute") && npuCard.includes("190 TOPS"), "the card shows NPU compute");
ok(npuCard.includes("NPU memory"), "...and calls the 48 GB NPU memory, not VRAM");
ok(!npuCard.includes("Wi-Fi") && !npuCard.includes("threads"),
   "...and shows neither the PC's network nor its threads");
ok(npuCard.includes(">not in use</span>"), "a node registered Not in use says so");

// ---------------------------------------------------------------------------
// Per-key model access (API keys tab). The control is a checklist dropdown;
// newKey/saveKeyEdit send its ids and compare the node's echo with them, so a
// node on an older gateway (which ignores the field and mints an unrestricted
// key) is caught instead of trusted.
// ---------------------------------------------------------------------------
const mopts = {roles: ["fast", "small"], models: ["m-b", "m-a"]};
const picker = ctx.mselHtml("kx", ["fast", "gone-model"], mopts);
ok(picker.includes("<summary>2 selected</summary>"),
   "a restricted key's picker says how many are selected");
ok(/data-msel-v="fast"\s+checked/.test(picker) && /data-msel-v="gone-model"\s+checked/.test(picker),
   "ticked ids are checked, including one the node no longer lists");
ok(picker.indexOf('data-msel-v="m-a"') < picker.indexOf('data-msel-v="m-b"'),
   "the models are sorted");
ok(ctx.mselHtml("kx", [], mopts).includes("<summary>all models</summary>"),
   "no list reads as all models");
ok(ctx.mselHtml("kx", [], mopts, true).includes("<summary>0 selected</summary>"),
   "an unreadable stored list opens on 'Only these' with nothing ticked");
ok(ctx.mselHtml("kx", ['a"b'], null).includes("a&quot;b"), "ids are HTML-escaped");
ok(ctx.modelsEchoed(["a", "b"], {allowed_models: ["b", "a"]}), "the same set echoed is accepted");
ok(!ctx.modelsEchoed(["a"], {allowed_models: []}), "an emptied echo is a mismatch");
ok(!ctx.modelsEchoed(["a"], {id: 1, key: "k"}), "no echo at all (an older gateway) is a mismatch");
ok(ctx.modelsEchoed([], {id: 1, key: "k"}), "...except when nothing was asked for");
ok(/allowed_models: models, \.\.\.workers\}\}\);\s*const lost = \[!modelsEchoed\(models, r\) && "model list",\s*!workersEchoed\(workers, r\)/.test(src),
   "newKey sends allowed_models and the worker setting, and checks both echoes before showing the key");
ok(/hostApi\(host, "\/keys\/" \+ r\.id, \{method: "DELETE"\}\)/.test(src),
   "a mismatched mint archives the key it just made, on the node it was minted on");
ok(/body\.allowed_models = mselValue\("ke_models"\)/.test(src),
   "saveKeyEdit sends the list too");
ok(vm.runInContext('modelsStr({allowed_models_invalid: true})', ctx)
     .includes("blocked: stored model list unreadable"),
   "an unreadable stored list shows as blocked, not as all models");
ok(vm.runInContext('modelsStr({allowed_models: ["a", "b", "c", "d"]})', ctx)
     .includes("models: a, b +2"),
   "the key list shows the first two models and a count");

// The picker's panel, its labels and its keyboard/mouse dismissal.
ok(/table \.mselp\{position:static/.test(html),
   "inside a table (the edit row) the panel stays in the row's flow: the scrolling card would clip a floating one");
ok(picker.includes('aria-label="models"') && picker.includes('aria-label="filter models"'),
   "the picker and its filter box have accessible names");
ok(/key !== "Escape"\) return;\s*\$\$\("details\.msel\[open\]"\)/.test(src),
   "Escape closes an open model checklist");
ok(/closest\("\.msel"\)[\s\S]{0,120}d\.open = false/.test(src),
   "a click outside an open model checklist closes it");
ok(/keys: \(\) => loadKeys\(true\)/.test(src),
   "entering the Keys tab re-reads the model choices");

// ---- behaviour: a mint belongs to the node it was started on ----
// The harness has no real page, so `$` is pointed at hand-made fields and
// rawApi at a recorder (hostApi/api reach the network only through it).
// SELF/CUR are the page's own lexical bindings.
{
  const set = (code) => vm.runInContext(code, ctx);
  const realQuery = ctx.document.querySelector, realApi = ctx.rawApi;
  const fields = {"#keyname": {value: "mykey"}, "#keyexp": {value: ""},
                  "#keyrpd": {value: ""}, "#keytpd": {value: ""}, "#keytt": {value: ""}};
  const calls = [];
  try {
    set('SELF = "hub"; CUR = "peer1"');

    // No picker on the page (its choices still loading, or the node just
    // switched): reading it as "all models" would mint an unrestricted key.
    ctx.document.querySelector = (s) => s === "#keymodels_sel" ? null : (fields[s] || realQuery(s));
    ctx.rawApi = async (path, opts = {}) => { calls.push([opts.method || "GET", path]); return {}; };
    let msg = "";
    try { ctx.mselValue("keymodels_sel"); } catch (e) { msg = e.message; }
    ok(/still loading/.test(msg), "an absent picker is an error, never 'all models'");
    await ctx.newKey();
    ok(calls.length === 0, "Mint key with no picker on the page sends nothing");

    // The picker has m-a ticked; the operator moves to the hub while the POST
    // is in flight, and the node (an older gateway) answers with no echo.
    fields["#keymodels_sel"] = {querySelector: () => ({checked: true}),
                                querySelectorAll: () => [{dataset: {mselV: "m-a"}}]};
    ctx.document.querySelector = (s) => fields[s] || realQuery(s);
    ctx.rawApi = async (path, opts = {}) => {
      calls.push([opts.method || "GET", path]);
      if (opts.method === "POST") {
        set('CUR = "hub"');
        return {id: 9, key: "sk-secret-value", name: "mykey"};
      }
      return {};
    };
    await ctx.newKey();
    ok(calls[0] && calls[0][0] === "POST" && calls[0][1] === "/fleet/peer1/keys",
       "the mint is sent to the node the operator was on");
    const del = calls.find((c) => c[0] === "DELETE");
    ok(del && del[1] === "/fleet/peer1/keys/9",
       "the rollback archives the key on the node that minted it, not the one picked meanwhile");
    ok(!(sink["#newkey"] || "").includes("sk-secret-value"),
       "the secret of a rolled-back mint is never shown");

    // One node's choices are filed under that node's name even when the pick
    // moves on before the reply.
    set('KEYMODELS = {}; CUR = "peer1"');
    ctx.rawApi = async (path) => {
      const roles = path.endsWith("/roles");
      if (roles) set('CUR = "hub"');
      return roles ? {roles: {fast: {}}} : {models: [{model: "peer-only"}]};
    };
    await ctx.keyModelOptions("peer1", true);
    const filed = JSON.parse(set("JSON.stringify(KEYMODELS)"));
    ok(filed.peer1 && filed.peer1.models.includes("peer-only") && !filed.hub,
       "model choices are cached under the node they were fetched for");

    // Another node's picker is gone the moment this node's mount starts.
    set('KEYMODELS = {}; KEYMODELS_FOR = "peer1"; CUR = "hub"');
    sink["#keymodels"] = "<details>peer1's picker</details>";
    ctx.rawApi = async () => { throw new Error("offline"); };
    const mounting = ctx.mountKeyModels();
    ok((sink["#keymodels"] || "").includes("loading"),
       "a picker built for another node is replaced by 'loading' at once");
    await mounting;
  } finally {
    ctx.document.querySelector = realQuery;
    ctx.rawApi = realApi;
  }
}

// ---------------------------------------------------------------------------
// Max parallel workers (API keys tab): off / this node only / up to N other
// nodes, with a routing class for the workers. Like the model list, newKey and
// saveKeyEdit compare the node's echo with what they sent, so a node on an older
// gateway (which would mint a plain key) is caught instead of trusted.
// ---------------------------------------------------------------------------
{
  const values = (h) => [...h.matchAll(/value="([^"]*)"/g)].map((m) => m[1]);
  ok(html.includes('id="keyworkers"') && html.includes('id="keyclass"'),
     "the Create-a-key card has the two worker selects");
  ok(/<label class="f">max parallel workers<\/label>/.test(html) && /<label class="f">worker class<\/label>/.test(html),
     "...labelled max parallel workers and worker class");
  ok(src.includes('"keys:max parallel workers"') && src.includes('"keys:worker class"'),
     "both have a keys-section HELP entry, which beats the Teams tab's un-prefixed one");
  ok(JSON.stringify(values(ctx.workerNodeOptions(0, null))) === '["","0"]',
     "a node that knows no peers offers only off and this node only");
  ok(JSON.stringify(values(ctx.workerNodeOptions(3, null))) === '["","0","1","2","3"]',
     "a node with 3 peers offers 1..3 other nodes");
  ok(values(ctx.workerNodeOptions(2, 5)).includes("5"),
     "a stored cap the node can no longer offer stays on the list, or saving would change it");
  ok(/value="0" selected/.test(ctx.workerNodeOptions(2, 0)) && /value="" selected/.test(ctx.workerNodeOptions(2, null)),
     "0 and off are told apart when picking what is selected");
  ok(/value="fast" selected/.test(ctx.workerClassOptions(["small", "fast"], "")),
     "the worker class defaults to fast");
  ok(/value="mine" selected/.test(ctx.workerClassOptions(["small"], "mine")),
     "a stored class the node no longer lists stays on the list");

  const set = (code) => vm.runInContext(code, ctx);
  const realQuery = ctx.document.querySelector, realApi = ctx.rawApi;
  const fields = {"#keyname": {value: "wkey"}, "#keyexp": {value: ""}, "#keyrpd": {value: ""},
                  "#keytpd": {value: ""}, "#keytt": {value: ""},
                  "#keyworkers": {value: "2"}, "#keyclass": {value: "small"},
                  "#keymodels_sel": {querySelector: () => ({checked: true}),
                                     querySelectorAll: () => [{dataset: {mselV: "m-a"}}]}};
  let sent = [];
  try {
    ctx.document.querySelector = (s) => fields[s] || realQuery(s);
    // The node is switched away mid-call so loadKeys (which this harness has no
    // network for) is not started; it is the request and the verdict on the
    // reply that are under test.
    const answer = (reply) => async (path, opts = {}) => {
      sent.push([opts.method || "GET", path, opts.body]);
      if (opts.method === "POST") set('CUR = "hub"');
      return opts.method === "POST" ? reply : {};
    };
    set('SELF = "hub"; CUR = "peer1"');

    ctx.rawApi = answer({id: 11, key: "sk-kept", allowed_models: ["m-a"], max_worker_nodes: 2, worker_class: "small"});
    await ctx.newKey();
    const post = sent.find((c) => c[0] === "POST");
    ok(post && post[2].max_worker_nodes === 2 && post[2].worker_class === "small",
       "newKey sends max_worker_nodes as a number and the chosen worker_class");
    ok(!sent.some((c) => c[0] === "DELETE") && (sink["#newkey"] || "").includes("sk-kept"),
       "a node that echoes both is believed: the key is shown, nothing is rolled back");

    sent = []; set('CUR = "peer1"'); sink["#newkey"] = ""; fields["#keyname"].value = "wkey";
    ctx.rawApi = answer({id: 12, key: "sk-lost", allowed_models: ["m-a"]});
    await ctx.newKey();
    const del = sent.find((c) => c[0] === "DELETE");
    ok(del && del[1] === "/fleet/peer1/keys/12" && !(sink["#newkey"] || "").includes("sk-lost"),
       "no worker echo with workers asked for (an older gateway): the key is archived on that node and never shown");

    sent = []; set('CUR = "peer1"'); sink["#newkey"] = ""; fields["#keyname"].value = "wkey";
    ctx.rawApi = answer({id: 13, key: "sk-wrong", allowed_models: ["m-a"], max_worker_nodes: 1, worker_class: "small"});
    await ctx.newKey();
    ok(sent.some((c) => c[0] === "DELETE"), "an echo of a different cap is a mismatch too");

    fields["#keyworkers"].value = "";   // off
    sent = []; set('CUR = "peer1"'); sink["#newkey"] = ""; fields["#keyname"].value = "wkey";
    ctx.rawApi = answer({id: 14, key: "sk-plain", allowed_models: ["m-a"]});
    await ctx.newKey();
    const off = sent.find((c) => c[0] === "POST");
    ok(off && off[2].max_worker_nodes === null && off[2].worker_class === "",
       "off is sent as null with no class");
    ok(!sent.some((c) => c[0] === "DELETE") && (sink["#newkey"] || "").includes("sk-plain"),
       "...and an older gateway that echoes nothing is fine when off was asked for");

    // The edit row: workers are sent only when the selects moved.
    Object.assign(fields, {"#ke_name": {value: "wkey"}, "#ke_exp": {value: ""}, "#ke_rpd": {value: ""},
      "#ke_tpd": {value: ""}, "#ke_tt": {value: ""}, "#ke_workers": {value: "3"}, "#ke_class": {value: "fast"}});
    ctx.document.querySelector = (s) => s === "#ke_models" ? null : (fields[s] || realQuery(s));
    const patched = (reply) => async (path, opts = {}) => {
      sent.push([opts.method || "GET", path, opts.body]);
      if (opts.method === "PATCH") set('CUR = "hub"');
      return opts.method === "PATCH" ? reply : {};
    };
    set('KEYS = [{id: 5, name: "wkey", max_worker_nodes: 3, worker_class: "fast"}]; CUR = "peer1"');
    sent = [];
    ctx.rawApi = patched({id: 5, max_worker_nodes: 3, worker_class: "fast"});
    await ctx.saveKeyEdit(5);
    const same = sent.find((c) => c[0] === "PATCH");
    ok(same && !("max_worker_nodes" in same[2]) && !("worker_class" in same[2]),
       "an edit that leaves the workers alone does not send them");

    fields["#ke_workers"].value = "0"; fields["#ke_class"].value = "small";
    sent = []; set('CUR = "peer1"');
    ctx.rawApi = patched({id: 5, max_worker_nodes: 0, worker_class: "small"});
    await ctx.saveKeyEdit(5);
    const moved = sent.find((c) => c[0] === "PATCH");
    ok(moved && moved[2].max_worker_nodes === 0 && moved[2].worker_class === "small",
       "saveKeyEdit sends 0 (this node only) as a number, not as off");
  } finally {
    ctx.document.querySelector = realQuery;
    ctx.rawApi = realApi;
  }

  ok(ctx.workersEchoed({max_worker_nodes: 2, worker_class: "fast"}, {max_worker_nodes: 2, worker_class: "fast"}),
     "the same cap and class echoed is accepted");
  ok(!ctx.workersEchoed({max_worker_nodes: 0, worker_class: "fast"}, {max_worker_nodes: null, worker_class: ""}),
     "0 echoed back as off is a mismatch");
  ok(!ctx.workersEchoed({max_worker_nodes: 2, worker_class: "fast"}, {id: 1}),
     "no echo at all is a mismatch when workers were asked for");
  ok(ctx.workersEchoed({max_worker_nodes: null, worker_class: ""}, {id: 1}),
     "...but not when off was");

  const str = (k) => vm.runInContext(`workersStr(${JSON.stringify(k)})`, ctx);
  ok(str({}).includes("workers: off") && str({max_worker_nodes: null, worker_class: ""}).includes("workers: off"),
     "a key with no setting (or from an older gateway) reads workers: off");
  ok(str({max_worker_nodes: 0, worker_class: "fast"}).includes("workers: this node only · fast"),
     "0 reads workers: this node only · class");
  ok(str({max_worker_nodes: 3, worker_class: "small"}).includes("workers: up to 3 other nodes · small"),
     "N reads workers: up to N other nodes · class");
  ok(str({max_worker_nodes: 1, worker_class: "fast"}).includes("up to 1 other node ·"),
     "...singular for one");
  ok(str({team: true, max_worker_nodes: null, worker_class: ""}).includes(">team<"),
     "a key fronting a Teams-tab team reads team");
  ok(str({max_worker_nodes: 1, worker_class: '<b>"x"'}).includes("&lt;b&gt;"),
     "the class name is HTML-escaped");

  ok(vm.runInContext('workersEditHtml({name: "hub-aero"})', ctx) === "",
     "a hub routing key's edit row has no worker selects");
  const teamEdit = vm.runInContext('workersEditHtml({name: "t", team: true})', ctx);
  ok(!teamEdit.includes("<select") && teamEdit.includes("Teams tab"),
     "a key fronting a team has none either, and says where to edit it");
  const plainEdit = vm.runInContext('workersEditHtml({name: "k", max_worker_nodes: null, worker_class: ""})', ctx);
  ok(plainEdit.includes('id="ke_workers"') && /id="ke_class"[^>]*disabled/.test(plainEdit),
     "an ordinary key gets both selects, the class disabled while workers are off");
  ok(!/id="ke_class"[^>]*disabled/.test(
       vm.runInContext('workersEditHtml({name: "k", max_worker_nodes: 0, worker_class: "fast"})', ctx)),
     "...and enabled once workers are on");

  // A key with an agent profile is not offered workers: the team loop is taken
  // before the profile is read, so the profile would silently stop applying.
  const agentEdit = vm.runInContext(
    'workersEditHtml({name: "k", agent: "pinned", max_worker_nodes: null, worker_class: ""})', ctx);
  ok(!agentEdit.includes("<select") && agentEdit.includes("Agents tab"),
     "a key with an agent profile gets no worker selects, and is told where the profile is");

  // An edit row opened before the node's choices arrived says so rather than
  // offering a stunted list (no classes, no other nodes) for the rest of that
  // edit; the real controls are swapped in, in place, the moment they arrive.
  {
    const run = (code) => vm.runInContext(code, ctx);
    const realQ = ctx.document.querySelector, realApi = ctx.rawApi;
    const keyRow = '{id: 5, name: "k", allowed_models: ["m-a"], max_worker_nodes: null, worker_class: ""}';
    const box = {h: "", loading: true,
                 set innerHTML(v) { this.h = v; }, get innerHTML() { return this.h; },
                 querySelector(s) { return this.loading && s === ".kloading" ? {} : null; }};
    try {
      run('KEYMODELS = {}; KEYMODELS_FOR = null; CUR = "peer1"; KEDIT = 5; KEYS = [' + keyRow + ']');
      const waiting = run('keyChoicesHtml(' + keyRow + ')');
      ok(waiting.includes("kloading") && !waiting.includes('id="ke_models"') && !waiting.includes('id="ke_workers"'),
         "before the node's choices arrive the edit row offers no pickers, only 'loading'");
      const hub = run('keyChoicesHtml({id: 6, name: "hub-aero"})');
      ok(hub.includes("hub routing key") && !hub.includes("kloading"),
         "a hub routing key never waits: it has neither control");

      ctx.document.querySelector = (s) => s === "#ke_choices" ? box : realQ(s);
      ctx.rawApi = async (path) => path.endsWith("/roles") ? {roles: {fast: {}, small: {}}}
        : path.endsWith("/peers") ? [{name: "a"}, {name: "b"}] : {models: [{model: "m-a"}]};
      await ctx.mountKeyModels(true);
      ok(box.h.includes('id="ke_models"') && box.h.includes('id="ke_workers"') && box.h.includes('value="2"')
           && box.h.includes('value="small"'),
         "when the choices arrive the waiting edit row is filled with the model list, the 2 peers and the classes");

      box.h = "typed into"; box.loading = false;
      ctx.fillKeyChoices();
      ok(box.h === "typed into", "an edit row that already has its pickers is never rewritten");
    } finally {
      run('KEYMODELS = {}; KEYMODELS_FOR = null; KEDIT = null; KEYS = []');
      ctx.document.querySelector = realQ;
      ctx.rawApi = realApi;
    }
  }

  // The Teams tab shows a blank primary as the client's model.
  ok(/client's model<\/span>/.test(src), "the Teams tab names a blank primary");

  // And offers it in the editor only for a team the keys form made: the
  // gateway refuses a blank primary on any other team (Fleet Pass included),
  // where it would let the key's holder pick any fleet model.
  {
    const run = (code) => vm.runInContext(code, ctx);
    const realQ = ctx.document.querySelector, oldModels = run("FLEETMODELS");
    ctx.document.querySelector = (s) => Object.assign(realQ(s), {scrollIntoView() {}});
    try {
      run('FLEETMODELS = ["m-a"]');
      const primaryOptions = (team) => { ctx.fillTeamEditor(team); return sink["#tm_primary"]; };
      const form = primaryOptions({name: "k", primary_model: "", worker_class: "fast",
                                   worker_models: '["fast"]'});
      ok(form.includes('<option value="" selected>client\'s model</option>'),
         "a keys-form team's editor offers the client's model, selected while the primary is blank");
      const pinned = primaryOptions({name: "k", primary_model: "m-a", worker_class: "fast"});
      ok(pinned.includes('<option value="m-a" selected>') && !pinned.includes('value="" selected'),
         "and keeps the model a keys-form team was pinned to");
      const fleetPass = primaryOptions({name: "fp", primary_model: "m-a", worker_class: ""});
      ok(fleetPass.includes('<option value="m-a" selected>') && !fleetPass.includes("client's model"),
         "any other team's editor does not offer the client's model");
      ok(!primaryOptions(null).includes("client's model"),
         "nor does the editor for a new team");

      // With nothing served the select must say so, or saving it is a bare 400
      // ("a team needs a primary model") with nothing on screen to explain it.
      run("FLEETMODELS = []");
      const none = '<option value="">no models served anywhere</option>';
      ok(primaryOptions(null) === none, "an empty fleet says so for a new team");
      ok(primaryOptions({name: "fp", primary_model: "m-a", worker_class: ""}) === none,
         "...and for any other team, a Fleet Pass one included");
      const bare = primaryOptions({name: "k", primary_model: "", worker_class: "fast"});
      ok(bare.includes("client's model") && !bare.includes("no models served anywhere"),
         "a keys-form team has the client's model to choose, so it needs no such message");
    } finally {
      ctx.document.querySelector = realQ;
      run("FLEETMODELS = " + JSON.stringify(oldModels));
    }
  }
}

if (fail.length) { console.error("FAIL:\n - " + fail.join("\n - ")); process.exit(1); }
console.log(`ok — ${cards.length} cards, all assertions passed`);
