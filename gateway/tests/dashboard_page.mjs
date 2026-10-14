// The dashboard's script block, evaluated outside a browser.
//
// Everything in gateway/static/index.html's one <script> runs here in a node
// vm against a DOM stub thin enough to be obvious and thick enough for the
// render functions: everything they touch is a querySelector, an innerHTML
// assignment or an event listener that never fires. Importing this gives you
// the live context (`ctx`, every top-level function and variable of the page)
// and `sink`, the last innerHTML written per selector.
//
// Shared by the checks that drive the page - the machine grid and its routing
// switch (dashboard_switch_check.mjs), the Models table's boot radio
// (models_boot_radio_check.mjs) - so a stub fix lands in one place.
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

// The page's top-level `let`/`const` bindings (MODELS, CFG, and the arrow
// functions beside them) are lexical: they never become properties of the vm's
// global object, so `ctx.MODELS = ...` would quietly create a second variable
// the page cannot see. Run a line of code inside the context instead. Function
// DECLARATIONS (renderMachines, renderModels) are global properties and can be
// called off `ctx` directly.
const run = (code) => vm.runInContext(code, ctx);

export {ctx, sink, el, html, src, run};
