// The Routing tab's class editor, driven outside a browser.
//
// The page's script block is evaluated by dashboard_page.mjs (the shared
// harness). RTDATA is set by hand to what GET /admin/api/routing-classes
// returns, rawApi is a recorder, and rtNew() / rtEdit() / rtSave() are called
// the way a click calls them. What this pins is what a Python test of the
// endpoint cannot see: what the editor puts in the PUT, and whether it can be
// reached and used from the keyboard.
import {ctx, sink, el, html, run} from "./dashboard_page.mjs";

const fail = [];
const ok = (cond, msg) => { if (!cond) fail.push(msg); };
let asserted = 0;
const check = (cond, msg) => { asserted++; ok(cond, msg); };

// One stub element per selector, so what the editor writes to #rt-name is
// still there when rtSave() reads it back.
const nodes = {};
const stub = (sel) => (nodes[sel] ??= Object.assign(
  el(sel), {value: "", readOnly: false, disabled: false, scrollIntoView() {}}));
ctx.document.querySelector = stub;
const lists = {};
ctx.document.querySelectorAll = (sel) => lists[sel] || [];

const toasts = [];
ctx.toast = (msg, bad) => toasts.push({msg, bad: !!bad});
const calls = [];
const DATA = {
  classes: [
    {name: "fast", models: ["qwen3.5-4b", "qwen3.8-27b"], nodes: [], builtin: true,
     overridden: false, default_models: ["qwen3.5-4b", "qwen3.8-27b"]},
    {name: "reviewers", models: ["qwen3.8-27b"], nodes: ["peer1"], builtin: false,
     overridden: false, default_models: []},
  ],
  holders: {
    "qwen3.5-4b": [
      {host: "peer1", name: "qwen3.5-4b", fit: "vram", resident: false, online: true, reserve: false},
      {host: "peer2", name: "qwen3.5-4b", fit: "", resident: false, online: false, reserve: false}],
    "qwen3.8-27b": [
      {host: "peer1", name: "qwen3.8-27b", fit: "vram", resident: true, online: true, reserve: false}],
  },
  models: ["qwen3.5-4b", "qwen3.6-35b", "qwen3.8-27b"],
  nodes: [{name: "peer1", online: true, self: false}, {name: "peer2", online: false, self: false}],
};
ctx.rawApi = async (path, opts = {}) => {
  calls.push({path, method: opts.method || "GET", body: opts.body});
  return JSON.parse(JSON.stringify(DATA));
};
const reset = () => { calls.length = 0; toasts.length = 0; };
const saved = () => calls.filter((c) => c.method === "PUT");

run(`RTDATA = ${JSON.stringify(DATA)}`);
ctx.renderRouting();

// ---------------------------------------------------------------------------
// The class table can be opened without a mouse: each name is a real button,
// which is focusable and answers Enter and Space by clicking itself -- a click
// that bubbles to the row's own handler.
// ---------------------------------------------------------------------------
const table = sink["#rt-tab tbody"] || "";
const buttons = table.match(/<button [^>]*>[^<]*<\/button>/g) || [];
check(buttons.length === 2, `one button per class row, got ${buttons.length}`);
check(table.includes('aria-label="edit class fast"')
      && table.includes('aria-label="edit class reviewers"'),
      "each button says which class it opens");
check((table.match(/<tr data-name="[^"]+" onclick="rtEdit\(this\.dataset\.name\)">/g) || []).length === 2,
      "and the row still opens the editor for a click anywhere on it");
check(/<input id="rt-filter"[^>]*aria-label="[^"]+"/.test(html),
      "and the filter box has a name a screen reader can announce");

// ---------------------------------------------------------------------------
// "+ New class" cannot silently overwrite a class that exists: the PUT is an
// upsert, so the draft would replace its members and its nodes.
// ---------------------------------------------------------------------------
for (const typed of ["fast", "FAST", " Reviewers "]) {
  reset();
  ctx.rtNew();
  nodes["#rt-name"].value = typed;
  ctx.rtAdd("qwen3.6-35b");
  await ctx.rtSave();
  check(saved().length === 0, `a new class named ${JSON.stringify(typed)} sends no PUT`);
  check(toasts.length === 1 && toasts[0].bad && /already exists/.test(toasts[0].msg),
        `and says why: ${JSON.stringify(toasts)}`);
}

reset();
ctx.rtNew();
nodes["#rt-name"].value = "writers";
ctx.rtAdd("qwen3.6-35b");
await ctx.rtSave();
check(saved().length === 1, "a genuinely new class is saved");
check(saved()[0].path === "/routing-classes"
      && JSON.stringify(saved()[0].body)
         === '{"classes":{"writers":{"models":["qwen3.6-35b"],"nodes":[]}}}',
      "as that one class: " + JSON.stringify(saved()[0]));
check(toasts.length === 1 && !toasts[0].bad, "with a confirmation toast, not an error");

// An existing class still saves under its own name -- the guard is for drafts.
reset();
ctx.rtEdit("reviewers");
await ctx.rtSave();
check(saved().length === 1 && "reviewers" in saved()[0].body.classes,
      "an open class saves under its own name");
check(JSON.stringify(saved()[0].body.classes.reviewers)
      === '{"models":["qwen3.8-27b"],"nodes":["peer1"]}',
      "with the nodes it was opened with: " + JSON.stringify(saved()[0].body));

// A save that strands some key's workers goes through, and says so: one error
// toast per warning the gateway returned, beside the usual confirmation.
{
  const realApi = ctx.rawApi;
  ctx.rawApi = async (path, opts = {}) => {
    const out = await realApi(path, opts);
    return opts.method === "PUT"
      ? {...out, warnings: ["key 'a' (#3): <b>cap 0</b>", "key 'b' (#4): cap 0"]} : out;
  };
  reset();
  ctx.rtEdit("reviewers");
  await ctx.rtSave();
  ctx.rawApi = realApi;
  const bad = toasts.filter((t) => t.bad).map((t) => t.msg);
  check(JSON.stringify(bad) === JSON.stringify(["key 'a' (#3): <b>cap 0</b>", "key 'b' (#4): cap 0"]),
        "each warning from the save is toasted as given: " + JSON.stringify(toasts));
  check(toasts.some((t) => !t.bad && /saved/.test(t.msg)), "and the save is still confirmed");
}

// ---------------------------------------------------------------------------
// Every redraw replaces the control the keyboard was on, so the editor hands
// the focus to the control's twin in the new markup.
// ---------------------------------------------------------------------------
const realFocus = ctx.rtFocus;
const asked = [];
let refuse = new Set();
ctx.rtFocus = (box, key) => { asked.push([box, key]); return !refuse.has(key); };

ctx.rtEdit("fast");
const members = sink["#rt-members"] || "";
for (const key of ["up:0", "down:0", "rm:0", "up:1", "down:1", "rm:1"])
  check(members.includes(`data-f="${key}"`), `member row control ${key} is addressable`);
check(members.includes('aria-label="move qwen3.8-27b up"')
      && members.includes('aria-label="remove qwen3.5-4b"'),
      "and the arrows and remove button are named for their model");
const nodeList = sink["#rt-nodes"] || "";
check(nodeList.includes('data-f="n:peer1"') && nodeList.includes('data-f="n:peer2"'),
      "node checkboxes are addressable");
check(nodeList.includes('role="img" aria-label="online"')
      && nodeList.includes('role="img" aria-label="offline"'),
      "and the online dot has a text alternative, not only a tooltip");

asked.length = 0;
ctx.rtMove(1, -1);
check(JSON.stringify(asked) === '[["#rt-members","up:0"]]',
      "moving a row up keeps the focus on its up arrow: " + JSON.stringify(asked));
asked.length = 0;
refuse = new Set(["down:1"]);   // the last row's down arrow is disabled
ctx.rtMove(0, 1);
check(JSON.stringify(asked) === '[["#rt-members","down:1"],["#rt-members","up:1"]]',
      "at the end of the list the focus falls to the other arrow: " + JSON.stringify(asked));
refuse = new Set();
asked.length = 0;
ctx.rtDrop(0);
check(JSON.stringify(asked) === '[["#rt-members","rm:0"]]',
      "removing a row hands the focus to the remove button that took its place");
asked.length = 0;
ctx.rtNode("peer1", true);
check(JSON.stringify(asked) === '[["#rt-nodes","n:peer1"]]',
      "ticking a node keeps the focus on its checkbox");
let filterFocused = 0;
stub("#rt-filter").focus = () => { filterFocused++; };
ctx.rtAdd("qwen3.6-35b");
check(filterFocused === 1, "adding a model from the checklist moves the focus to the filter box");

// The real rtFocus: picks the twin by its key and never focuses a disabled one.
ctx.rtFocus = realFocus;
const hits = [];
const fake = (f, disabled = false) => ({dataset: {f}, disabled, focus() { hits.push(f); }});
lists["#rt-members [data-f]"] = [fake("up:0"), fake("down:0", true)];
check(ctx.rtFocus("#rt-members", "down:0") === false && hits.length === 0,
      "a disabled control does not take the focus");
check(ctx.rtFocus("#rt-members", "up:0") === true && hits[0] === "up:0",
      "an enabled one does");
check(ctx.rtFocus("#rt-members", "rm:9") === false, "and a key with no twin is a no-op");

if (fail.length) { console.error("FAIL:\n - " + fail.join("\n - ")); process.exit(1); }
console.log(`ok — ${asserted} checks, all assertions passed`);
