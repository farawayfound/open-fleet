// Render the dashboard's machine grid outside a browser.
//
// The page's whole script block is evaluated in a vm with a DOM stub thin
// enough to be obvious and thick enough for renderMachines(): everything it
// touches is a querySelector, an innerHTML assignment or an event listener
// that never fires. Then the grid is rendered from hand-built overview rows
// -- online, offline, killswitched, hub, self -- and the resulting HTML is
// asserted on. Catches exactly what a syntax check cannot: a template that
// throws on a null status, a toggle attached to the wrong card.
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
];

ctx.renderMachines(HOSTS);
const out = sink["#machines"];
const cards = out.split('<div class="card machine').slice(1);
const nameOf = (c) => (c.match(/class="mname">([^<]+)/) || [])[1];

const fail = [];
const ok = (cond, msg) => { if (!cond) fail.push(msg); };

ok(cards.length === 5, `5 cards rendered, got ${cards.length}`);
ok(JSON.stringify(cards.map(nameOf))
   === JSON.stringify(["box-a", "box-b", "box-c", "box-d", "hub-box"]),
   "order: online by bandwidth, then offline, then the hub — got "
   + JSON.stringify(cards.map(nameOf)));

const byName = Object.fromEntries(cards.map((c) => [nameOf(c), c]));
ok(byName["box-c"].includes('<span class="pill no">offline</span>'),
   "an offline box gets the red offline pill");
ok(byName["box-a"].includes('<span class="pill ok">serving</span>'),
   "a serving box keeps its green pill");
ok(byName["box-b"].includes('<span class="pill no">killed</span>'),
   "an online-but-killed box says so");
ok(byName["box-d"].includes('<span class="pill no">offline</span>')
   && byName["box-d"].includes('<span class="pill no">killed</span>'),
   "offline AND killed shows both");
ok(!byName["hub-box"].includes("toggleRouting"),
   "the hub has no killswitch");
ok(!byName["hub-box"].includes('class="krow"'), "the hub has no routing row");
for (const n of ["box-a", "box-b", "box-c", "box-d"]) {
  ok(byName[n].includes(`toggleRouting(this, '${n}')`), `${n} has a routing switch`);
  ok(byName[n].includes('role="switch"'), `${n}'s control is a switch, not a button`);
  ok(byName[n].includes("event.stopPropagation()"),
     `${n}'s switch does not also open the machine`);
}
// The switch reports the state it is in, both ways round -- the half that a
// button labelled "killswitch" could never do.
ok(byName["box-a"].includes('aria-checked="true"')
   && byName["box-a"].includes('>on</span>'), "a routed box shows routing on");
ok(byName["box-b"].includes('aria-checked="false"')
   && byName["box-b"].includes('>off</span>'), "a killed box shows routing off");
ok(byName["box-d"].includes('aria-checked="false"'),
   "a box that is offline AND killed still shows its routing state");
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
// The switch itself. The first version of this shipped writing `routed:
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
// flip also triggers loadHome()'s refetch, so select rather than index.
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

if (fail.length) { console.error("FAIL:\n - " + fail.join("\n - ")); process.exit(1); }
console.log(`ok — ${cards.length} cards, all assertions passed`);
