// The Models table's boot column, driven outside a browser.
//
// `preload` is the one model a box loads when its engine starts. models.json
// allows exactly one (check_preload_count refuses the save otherwise), so the
// column is a radio group rather than a per-row switch — and a radio group
// has a state HTML cannot express by itself: none selected. Clicking the
// checked circle has to clear it, which means the handler reads the flag it
// is about to overwrite. That is the logic this file exercises: the page's
// script block is evaluated by the shared harness, MODELS is set by hand, and
// renderModels() + bootPick() are called the way a click calls them.
import {ctx, sink, html, run} from "./dashboard_page.mjs";

const fail = [];
const ok = (cond, msg) => { if (!cond) fail.push(msg); };

const model = (id, over = {}) => ({
  id, path: "/models/" + id + ".gguf", enabled: true, ctx: 32768, ngl: 99,
  n_cpu_moe: 0, ttl: 1800, reasoning: "auto", extra_flags: "",
  preload: false, persistent: false, ...over,
});

// A box shaped like apu-box-1: one big model warm at boot, a pinned little one
// beside the swap group, and others that load on demand.
const fresh = () => [
  model("qwen3.8-flash-next", {preload: true, ttl: 0}),
  model("deepseek-v4-flash", {ctx: 131072}),
  model("nomic-embed", {persistent: true}),
];
const load = (models) => run(`LOCAL = []; UPSTREAM = {enabled: false}; `
  + `MODELS = ${JSON.stringify(models)}; renderModels();`);
const preloads = () => JSON.parse(run("JSON.stringify(MODELS.map(m => !!m.preload))"));
const pinned = (i) => JSON.parse(run(`JSON.stringify(!!MODELS[${i}].persistent)`));
load(fresh());

const rows = () => (sink["#mtab tbody"] || "").split("<tr>").slice(1);
const radioOf = (row) => (row.match(/<input type="radio"[^>]*>/) || [""])[0];
const checked = () => rows().map(radioOf).map((r) => r.includes(" checked"));

ok(rows().length === 3, `3 rows rendered, got ${rows().length}`);
ok(rows().every((r) => radioOf(r).includes('name="bootmodel"')),
   "every row's boot control is part of one radio group");
ok(JSON.stringify(checked()) === JSON.stringify([true, false, false]),
   "the preloaded model is the one checked circle — got " + JSON.stringify(checked()));
ok(radioOf(rows()[2]).includes("disabled"),
   "a pinned model cannot take the boot slot — it is resident from start-up anyway");
ok(!radioOf(rows()[0]).includes("disabled"),
   "an ordinary model's circle is live");

// Picking another model moves the slot rather than adding a second one: two
// preloads is not two warm models, it is a cold load that evicts the first.
run("bootPick(1)");
ok(JSON.stringify(preloads()) === JSON.stringify([false, true, false]),
   "picking a row clears every other row's preload — got " + JSON.stringify(preloads()));
ok(JSON.stringify(checked()) === JSON.stringify([false, true, false]),
   "and the re-render follows the data");

// The whole point of the column: clicking the selected circle turns the boot
// model off, which a radio group has no native way to say.
run("bootPick(1)");
ok(preloads().every((p) => !p),
   "clicking the selected circle again leaves the box with no boot model");
ok(checked().every((c) => !c), "and nothing is checked afterwards");

// Clicking a cleared row picks it again (the toggle is per-row state, not a
// latch that stays off).
run("bootPick(1)");
ok(preloads()[1] === true, "clicking a cleared row selects it again");

// A pinned row is warm at boot by definition (render_swap_config preloads
// every persistent model), so pinning gives up the boot slot instead of
// holding one the radio would have to show as unavailable.
load(fresh());
run('updPinned(0, "persistent")');
ok(pinned(0) === true && preloads()[0] === false, "pinning a model clears its boot flag");
ok(radioOf(rows()[0]).includes("disabled"), "and disables its circle");
run('updPinned(0, "")');
ok(pinned(0) === false && preloads()[0] === false,
   "unpinning does not silently hand the boot slot back");

// An empty table still renders (the colspan has to cover every column, or the
// "no models configured yet" line sits under a short row).
load([]);
const empty = sink["#mtab tbody"] || "";
ok(empty.includes("no models configured yet"), "an empty table says so");
ok(/colspan="(\d+)"/.test(empty), "and does it in one spanning cell");

// The span must match the header count, or the copy sits under a short row.
const head = (html.match(/<table id="mtab">\s*<thead><tr>([\s\S]*?)<\/tr>/) || ["", ""])[1];
const cols = (head.match(/<th/g) || []).length;
const span = +(empty.match(/colspan="(\d+)"/) || [0, 0])[1];
ok(cols === span, `the empty row spans every column: ${cols} headers vs colspan ${span}`);
ok(/<th>boot<\/th>/.test(head) && /<th>pinned<\/th>/.test(head),
   "the table has both a boot and a pinned column");

if (fail.length) { console.error("FAIL:\n - " + fail.join("\n - ")); process.exit(1); }
console.log(`ok — ${cols} columns, all assertions passed`);
