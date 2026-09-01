// Render the dashboard's machine grid outside a browser.
//
// The page's whole script block is evaluated in a vm with a DOM stub thin
// enough to be obvious and thick enough for renderMachines(): everything it
// touches is a querySelector, an innerHTML assignment or an event listener
// that never fires. Then the grid is rendered from hand-built overview rows
// -- online, offline, killswitched, reserve, hub, self -- and the resulting
// HTML is asserted on. Catches exactly what a syntax check cannot: a template
// that throws on a null status, a pill on the wrong card.
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import {fileURLToPath} from "node:url";

// Resolved from this file, never written out: an absolute path here would
// carry a developer's home directory into the public export, which denylists
// it. gateway/tests/ -> gateway/static/index.html.
const HERE = path.dirname(fileURLToPath(import.meta.url));
const PAGE = path.join(HERE, "..", "static", "index.html");
const html = fs.readFileSync(PAGE, "utf8");
const src = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));

const sink = {};                       // selector -> last innerHTML written
const el = (sel) => ({
  set innerHTML(v) { sink[sel] = v; },
  get innerHTML() { return sink[sel] || ""; },
  set textContent(v) { sink[sel + ":text"] = v; },
  get textContent() { return sink[sel + ":text"] || ""; },
  classList: {add() {}, remove() {}, contains: () => false, toggle() {}},
  setAttribute() {}, getAttribute: () => null, removeAttribute() {},
  addEventListener() {}, appendChild() {}, insertBefore() {}, remove() {},
  querySelector: () => null, querySelectorAll: () => [],
  style: {}, dataset: {}, closest: () => null, focus() {}, scrollTo() {},
});
const doc = {
  querySelector: (s) => el(s),
  querySelectorAll: () => [],
  getElementById: (s) => el("#" + s),
  createElement: () => el("<new>"),
  addEventListener() {}, body: el("body"), documentElement: el("html"),
  cookie: "", hidden: false,
};
const ctx = {
  document: doc, window: {addEventListener() {}, location: {search: "", hash: ""}},
  location: {search: "", hash: "", pathname: "/"},
  localStorage: {getItem: () => null, setItem() {}, removeItem() {}},
  fetch: async () => { throw new Error("no network in this harness"); },
  console, setTimeout, clearTimeout, setInterval: () => 0, clearInterval() {},
  requestAnimationFrame: () => 0, navigator: {clipboard: {writeText: async () => {}}},
  EventSource: class { addEventListener() {} close() {} },
  Intl, Date, Math, JSON, URLSearchParams, URL, TextEncoder, TextDecoder,
  Promise, Error, Array, Object, Map, Set, RegExp, String, Number, Boolean,
  AbortController, Headers: class {}, FormData: class {}, Blob: class {},
  addEventListener() {}, removeEventListener() {}, matchMedia: () => ({matches: false,
    addEventListener() {}}), getComputedStyle: () => ({}), alert() {}, confirm: () => true,
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(src, ctx, {filename: "index.html<script>"});

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

ok(byName["box-c"].includes(`showHostDetail('box-c')`),
   "an offline card opens its last-known detail, not a dead switchHost");
ok(byName["box-a"].includes(`switchHost('box-a')`),
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
// The Public tab's vertical sub-nav. A nav entry whose pane does not exist
// (or a pane no entry points at) is a section that cannot be reached at all
// -- which is how the auto-issue card spent its first day invisible, buried
// mid-scroll behind seven other subjects. This is a static read of the markup
// and the script: a selector stub cannot tell you the two lists agree.
// ---------------------------------------------------------------------------
const pub = html.slice(html.indexOf('<section id="public">'),
                       html.indexOf("<!-- ================= CONFIGURATIONS"));
const entries = [...pub.matchAll(/data-p="([a-z]+)"/g)].map((m) => m[1]);
const panes = [...pub.matchAll(/id="pub-([a-z]+)"/g)].map((m) => m[1]);
const declared = JSON.parse(
  (src.match(/const PUB_PANES = (\[[\s\S]*?\])/) || [])[1].replace(/\s+/g, " "));

ok(entries.length === 8, `8 sub-nav entries, got ${entries.length}`);
ok(JSON.stringify(entries) === JSON.stringify(panes),
   `every entry has its pane, in order -- nav ${JSON.stringify(entries)} `
   + `vs panes ${JSON.stringify(panes)}`);
ok(JSON.stringify(declared) === JSON.stringify(panes),
   `PUB_PANES matches the markup -- declared ${JSON.stringify(declared)}`);
ok(entries.includes("auto"), "auto-issue is reachable from the sub-nav");
ok((pub.match(/class="pane on"/g) || []).length === 1, "exactly one pane starts open");
ok(pub.includes('<div class="subnav" id="pubnav"'),
   "the sub-nav is a div -- showTab() drives every `nav button` on the page as a "
   + "top-level tab, so a <nav> here would fight it for the .on class");
ok(pub.includes('role="tablist"') && (pub.match(/role="tabpanel"/g) || []).length === 8,
   "tablist, and one tabpanel per pane");

// And it runs: an unknown pane falls back rather than leaving the tab blank.
ok(typeof ctx.showPubPane === "function", "showPubPane is defined");
try {
  ctx.showPubPane("auto");
  ctx.showPubPane("nonsense-pane");
} catch (e) {
  fail.push("showPubPane threw: " + e.message);
}

if (fail.length) { console.error("FAIL:\n - " + fail.join("\n - ")); process.exit(1); }
console.log(`ok — ${cards.length} cards, all assertions passed`);
