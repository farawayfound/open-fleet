// Library per-quant package downloads: the pure render functions, driven
// outside a browser.
//
// hf/quants used to render one row per FILE (LM Studio users kept asking why
// an 8-shard quant was eight separate download buttons). renderQuantRows
// groups that into one row per quant level instead, and renderJobRows mirrors
// the same grouping on the Downloads panel so a package's shards share one
// progress bar. Both are declared with `function` (not a closure over page
// state) so this harness can hand them a fixture directly, the same idiom as
// dashboard_switch_check.mjs and models_boot_radio_check.mjs. renderModels
// gained a filter (only a `kind === "main"` LOCAL entry is a loadable model)
// and switchSection was split out of showTab so addQuantModel can land on the
// Models tab without re-fetching /models over the record it just staged --
// both regressions this file exists to catch.
import {ctx, sink, html, run} from "./dashboard_page.mjs";

const fail = [];
const ok = (cond, msg) => { if (!cond) fail.push(msg); };

// ---- renderQuantRows -------------------------------------------------------

// Base shape matches what /hf/quants adds on top of _group_hf_quants' output
// (spec sections 3 and 7): label/files/bytes/shards/expected_shards/complete/
// mmproj_choices from the grouping, fit/disk_ok/status/main_dest/
// suggested_id/ollama_tag from the endpoint.
const quant = (over = {}) => ({
  label: "Q4_K_M",
  files: [
    {path: "org/repo-GGUF/repo-Q4_K_M-00001-of-00002.gguf", size: 3 * 1024 ** 3},
    {path: "org/repo-GGUF/repo-Q4_K_M-00002-of-00002.gguf", size: 2 * 1024 ** 3},
  ],
  bytes: 5 * 1024 ** 3, shards: 2, expected_shards: 2, complete: true,
  mmproj_choices: [],
  fit: "vram", disk_ok: true, status: "none",
  main_dest: "D:\\models\\org__repo-GGUF\\repo-Q4_K_M-00001-of-00002.gguf",
  suggested_id: "repo-Q4_K_M", ollama_tag: "hf.co/org/repo-GGUF:Q4_K_M",
  ...over,
});
const REPO = "org/repo-GGUF";
const quants = [
  quant(),
  // Missing shard 4 of 4 upstream: never downloadable, no matter how much of
  // it already sits on disk (spec's "never hide an incomplete set" rule).
  quant({label: "Q5_K_M", complete: false, shards: 3, expected_shards: 4,
         status: "partial", done: 3, total: 4}),
  // On disk already, and ships a projector -- the row that offers both the
  // mmproj <select> and the Add to Models button.
  quant({label: "Q8_0", status: "done",
         mmproj_choices: [{path: "org/repo-GGUF/mmproj-model-f16.gguf", size: 900 * 1024 * 1024}],
         // The gateway names the projector that is actually on disk; the row
         // never derives one from mmproj_choices (a record pointing at a
         // projector nobody downloaded would fail to launch).
         mmproj_dest: "D:\\models\\org__repo-GGUF\\mmproj-model-f16.gguf"}),
];
const modelInfo = {params: 27e9, architecture: "llama", context_length: 131072,
                    pipeline_tag: "text-generation", downloads: 12345};
const excluded = [{path: "org/repo-GGUF/imatrix_unsloth.gguf", reason: "importance matrix, not a model"}];

const renderQuants = (over = {}) => run(`renderQuantRows(${JSON.stringify(REPO)}, `
  + `${JSON.stringify({quants, excluded, model_info: modelInfo, upstream_only: false, ...over})})`);

// A row's own <tr> ends at the first </tr> after its label -- the details
// <tr> right after it is a second, separate <tr> this must not swallow.
const rowFor = (out, label) => {
  const i = out.indexOf(`style="font-weight:bold">${label}<`);
  ok(i !== -1, `expected a ${label} row in the rendered output`);
  return out.slice(i, out.indexOf("</tr>", i));
};

const out = renderQuants();
ok((out.match(/style="font-weight:bold"/g) || []).length === quants.length,
   `one row per quant level (${quants.length}), not per file (5) -- got `
   + `${(out.match(/style="font-weight:bold"/g) || []).length} label rows`);
ok((out.match(/<details>/g) || []).length === quants.length,
   "one file-list <details> per quant, not per file");
ok(out.includes("27B") && out.includes("llama"),
   "header shows the model's param count and architecture");
ok(out.includes("6 files"), "header's file count sums every quant's files");

const q4 = rowFor(out, "Q4_K_M");
ok(q4.includes(">Download<") && q4.includes("onclick=\"pullQuant(0)\""),
   "a complete, not-yet-downloaded quant offers Download");
ok(!q4.includes("<select id=\"hfq-mm-"),
   "no mmproj <select> when the quant ships no projector choices");

const q5 = rowFor(out, "Q5_K_M");
ok(!q5.includes("Download"),
   "an incomplete-on-HF quant (missing a shard upstream) has no Download button, however much is on disk");
ok(q5.includes("incomplete on HF (3 of 4 shards)"),
   "the incomplete pill names how many of the upstream shards exist");

const q8 = rowFor(out, "Q8_0");
ok(q8.includes("<select id=\"hfq-mm-2\">"),
   "the mmproj <select> appears (indexed to this quant's position) when choices exist");
ok(q8.includes("Add to Models") && q8.includes("addQuantModel("),
   "a quant already fully on disk offers Add to Models instead of Download");
ok(q8.includes(run('jsArg("D:\\\\models\\\\org__repo-GGUF\\\\mmproj-model-f16.gguf")')),
   "Add to Models stages the projector path the gateway reported on disk (mmproj_dest), through jsArg");
const q8NoMm = rowFor(run(`renderQuantRows(${JSON.stringify(REPO)}, ${JSON.stringify(
  {quants: [quant({label: "Q8_0", status: "done", mmproj_choices: quants[2].mmproj_choices, mmproj_dest: ""})],
   excluded: [], model_info: {}, upstream_only: false})})`), "Q8_0");
ok(q8NoMm.includes("," + run('jsArg("")') + ')">Add to Models'),
   "with no projector on disk the staged mmproj is empty -- never guessed from the choices");

// upstream_only wins over every other status: an Ollama-managed box never
// gets a Download (or Add to Models) button, even for the quant that is
// already fully on disk.
const upstreamOut = renderQuants({upstream_only: true});
ok(!upstreamOut.includes(">Download<"), "Ollama-managed box: no Download button anywhere");
ok(!upstreamOut.includes("Add to Models"), "Ollama-managed box: no Add to Models either -- upstream_only wins over status===done");
ok(rowFor(upstreamOut, "Q4_K_M").includes("copyOllamaPull(") && upstreamOut.includes("copy ollama pull"),
   "every row gets the copy-to-clipboard ollama pull button instead");

// complete === false blocks every affirmative action regardless of status --
// the backend counts only the files HF actually offers, so a set missing a
// shard upstream reads "done" the moment every offered file is on disk, yet
// it can never be loaded. "downloading" is the one status that still needs
// its stop button while the (never going to finish) transfer is in flight.
const incompleteDoneOut = renderQuants({quants: [quant({label: "Q4_K_M", complete: false,
  shards: 3, expected_shards: 4, status: "done"})]});
const incompleteDoneRow = rowFor(incompleteDoneOut, "Q4_K_M");
ok(!incompleteDoneRow.includes("Add to Models"),
   "complete:false quant reading status done: no Add to Models button");
ok(!incompleteDoneRow.includes(">Download<"),
   "complete:false quant reading status done: no Download button either");
ok(incompleteDoneRow.includes("incomplete on HF (3 of 4 shards)"),
   "complete:false quant reading status done still shows the incomplete-on-HF pill");

const incompleteDownloadingOut = renderQuants({quants: [quant({label: "Q4_K_M", complete: false,
  shards: 3, expected_shards: 4, status: "downloading", bytes_done: 100, bytes_total: 400})]});
const incompleteDownloadingRow = rowFor(incompleteDownloadingOut, "Q4_K_M");
ok(incompleteDownloadingRow.includes("cancelPackage(") && incompleteDownloadingRow.includes(">stop<"),
   "complete:false quant that is actively downloading still offers the stop button");

// A root file and a subfolder re-cut sharing a basename land on the same
// local file (dest flattens to the basename); clash_with names the other
// quant it collides with, escaped through esc same as everything else, and
// blocks Download the same way an incomplete-on-HF set does.
const clashLabel = "Q4_K_M (Q4_K_M/)";
const clashEsc = run(`esc(${JSON.stringify(clashLabel)})`);
const clashOut = renderQuants({quants: [quant({label: "Q4_K_M", clash_with: clashLabel})]});
const clashRow = rowFor(clashOut, "Q4_K_M");
ok(clashRow.includes(`class="pill no"`) && clashRow.includes(`name clash with ${clashEsc}`),
   "a clash_with quant shows a name-clash pill, HTML-escaped through esc");
ok(!clashRow.includes(">Download<"), "a clash_with quant has no Download button");

// ---- renderJobRows ----------------------------------------------------------

// A pkg === "" row (every download queued before this feature existed, and
// the per-file "pull" escape hatch after it) must render through the
// UNCHANGED original per-row template -- this is a literal snapshot of that
// template's actual output for a fixed row, not a re-derivation of it, so an
// accidental edit to the template's markup trips this test rather than
// quietly reformatting both sides of the comparison.
const plainRow = {id: 701, repo: "acme/bar-GGUF", pkg: "", filename: "acme/bar-GGUF/bar-Q4.gguf",
                  status: "downloading", bytes_done: 104857600, bytes_total: 209715200};
const EXPECT_PLAIN_ROW =
  '<tr><td style="width:100%">\n' +
  '        <div class="mono" style="font-size:12px">bar-Q4.gguf</div>\n' +
  '        <div class="bar"><i style="width:50%"></i></div>\n' +
  '        <div class="sub">100 MB / 200 MB\n' +
  '          </div>\n' +
  '      </td><td style="text-align:right;white-space:nowrap">\n' +
  '        <span class="pill mid">downloading</span>\n' +
  '        <button class="act" onclick="cancelJob(701)">stop</button>\n' +
  '      </td></tr>';
const plainOut = run(`renderJobRows(${JSON.stringify([plainRow])})`);
ok(plainOut === EXPECT_PLAIN_ROW,
   "a pkg==='' row still renders byte-identical to the pre-grouping template -- got:\n" + plainOut);

// Two shards of one package, one still queued: they fold into a single bar
// rather than two rows, and the "n of m files" count is files done, not bytes.
run("LOCAL = [];");
const partialGroup = [
  {id: 601, repo: "unsloth/Qwen3.8-27B-GGUF", pkg: "Q4_K_M",
   filename: "unsloth/Qwen3.8-27B-GGUF/file-Q4_K_M-00001-of-00002.gguf",
   status: "downloading", bytes_done: 100 * 1024 * 1024, bytes_total: 200 * 1024 * 1024,
   role: "main", dest: "D:\\models\\unsloth__Qwen3.8-27B-GGUF\\file-Q4_K_M-00001-of-00002.gguf"},
  {id: 602, repo: "unsloth/Qwen3.8-27B-GGUF", pkg: "Q4_K_M",
   filename: "unsloth/Qwen3.8-27B-GGUF/file-Q4_K_M-00002-of-00002.gguf",
   status: "queued", bytes_done: 0, bytes_total: 200 * 1024 * 1024,
   role: "shard", dest: "D:\\models\\unsloth__Qwen3.8-27B-GGUF\\file-Q4_K_M-00002-of-00002.gguf"},
];
const partialOut = run(`renderJobRows(${JSON.stringify(partialGroup)})`);
ok((partialOut.match(/<tr>/g) || []).length === 1,
   "two shards of one package render as one row, not two");
ok(partialOut.includes("0 of 2 files"), "the group counts finished files, not bytes");
ok(partialOut.includes('<span class="pill mid">downloading</span>'),
   "a group with anything still queued or downloading is pill mid, not done");
ok(partialOut.includes("cancelPackage(") && partialOut.includes("stop"),
   "an active group offers a single stop button for the whole package");
ok(!partialOut.includes("Add to Models"), "an incomplete group never offers Add to Models");

// All files done: the fallback slug is used when no LOCAL entry has picked up
// suggested_id yet (a group whose files finished before /models was refreshed).
const doneGroupNoLocal = [
  {id: 611, repo: "acme/foo-GGUF", pkg: "Q8_0", filename: "acme/foo-GGUF/foo-Q8_0.gguf",
   status: "done", bytes_done: 500, bytes_total: 500, role: "main",
   dest: "D:\\models\\acme__foo-GGUF\\foo-Q8_0.gguf"},
  {id: 612, repo: "acme/foo-GGUF", pkg: "Q8_0", filename: "acme/foo-GGUF/mmproj-model-f16.gguf",
   status: "done", bytes_done: 200, bytes_total: 200, role: "mmproj",
   dest: "D:\\models\\acme__foo-GGUF\\mmproj-model-f16.gguf"},
];
const doneOutNoLocal = run(`renderJobRows(${JSON.stringify(doneGroupNoLocal)})`);
const fallbackId = run(`fallbackSlug(${JSON.stringify("acme/foo-GGUF")}, ${JSON.stringify("Q8_0")})`);
ok(fallbackId === "foo-q8_0", `fallbackSlug("acme/foo-GGUF","Q8_0") unexpected: ${fallbackId}`);
ok(doneOutNoLocal.includes('<span class="pill ok">done</span>'), "an all-done group is pill ok");
ok(doneOutNoLocal.includes("Add to Models") && doneOutNoLocal.includes("addQuantModel(")
   && doneOutNoLocal.includes(run(`jsArg(${JSON.stringify(fallbackId)})`)),
   "an all-done group with no LOCAL match yet uses the client-side fallback slug as the staged id");

// Same package, but LOCAL now has an entry for the main file's dest (the
// gateway's own suggested_id) -- that must win over the fallback slug.
run(`LOCAL = ${JSON.stringify([{path: "D:\\models\\acme__foo-GGUF\\foo-Q8_0.gguf", suggested_id: "foo-Q8_0-fleet"}])};`);
const doneOutLocal = run(`renderJobRows(${JSON.stringify(doneGroupNoLocal)})`);
ok(doneOutLocal.includes(run('jsArg("foo-Q8_0-fleet")')),
   "an all-done group prefers LOCAL's suggested_id over the fallback slug once /models has caught up");
run("LOCAL = [];");

// All-done package group whose rows carry NO role === "main" row: the main
// file was already on disk when the package was queued, so download-package
// reported it instead of queuing it, and nothing here knows its path --
// staging a record with an empty one would be worse than pointing at the
// rows that do know it.
const doneGroupNoMainRow = [
  {id: 621, repo: "acme/baz-GGUF", pkg: "Q4_K_M", filename: "acme/baz-GGUF/baz-Q4_K_M-00002-of-00002.gguf",
   status: "done", bytes_done: 500, bytes_total: 500, role: "shard",
   dest: "D:\\models\\acme__baz-GGUF\\baz-Q4_K_M-00002-of-00002.gguf"},
  {id: 622, repo: "acme/baz-GGUF", pkg: "Q4_K_M", filename: "acme/baz-GGUF/mmproj-model-f16.gguf",
   status: "done", bytes_done: 200, bytes_total: 200, role: "mmproj",
   dest: "D:\\models\\acme__baz-GGUF\\mmproj-model-f16.gguf"},
];
const doneOutNoMainRow = run(`renderJobRows(${JSON.stringify(doneGroupNoMainRow)})`);
ok(!doneOutNoMainRow.includes("addQuantModel("),
   "an all-done group with no main-role row renders no Add to Models button");
ok(doneOutNoMainRow.includes("ready — add it from the quant row or On disk below"),
   "...and shows the muted ready-elsewhere text instead");

// ---- renderModels: only a loadable GGUF is offered -------------------------

const local = [
  {path: "D:\\models\\a\\main1.gguf", name: "main1.gguf", kind: "main", bytes: 5 * 1024 ** 3},
  {path: "D:\\models\\a\\shard2.gguf", name: "shard2.gguf", kind: "shard", size: 5 * 1024 ** 3},
  {path: "D:\\models\\a\\mmproj.gguf", name: "mmproj.gguf", kind: "mmproj", size: 900 * 1024 * 1024},
];
const modelRec = (id, path) => ({id, path, enabled: true, ctx: 4096, ngl: 99, n_cpu_moe: 0,
  ttl: 1800, reasoning: "auto", extra_flags: "", preload: false, persistent: false, mmproj: ""});
const models = [modelRec("m1", "D:\\models\\a\\main1.gguf"), modelRec("m2", "D:\\models\\a\\shard2.gguf")];
run(`LOCAL = ${JSON.stringify(local)}; MODELS = ${JSON.stringify(models)}; `
  + `UPSTREAM = {enabled: false}; DEFAULTS = {}; SIZING = {}; renderModels();`);
const mrows = (sink["#mtab tbody"] || "").split("<tr>").slice(1);
ok(mrows.length === 2, `2 model rows rendered, got ${mrows.length}`);
ok(mrows[0].includes('value="D:\\models\\a\\main1.gguf" selected') && !mrows[0].includes("shard2.gguf"),
   "a record pointed at a loadable main GGUF offers it, selected, and nothing about the shard");
ok(mrows[1].includes("shard/projector — not a loadable model") && mrows[1].includes('value="D:\\models\\a\\shard2.gguf" selected'),
   "a record already pointed at a shard keeps a flagged synthetic selected option rather than going blank");
// A row's GGUF picker is the FIRST <select>...</select> in the row -- the
// projector <select data-help="projector"> that mmSelect renders right after
// it, in the same <td>, is a second, separate <select> this must not swallow.
const ggufSelectOf = (row) => {
  const i = row.indexOf('<select onchange="updPath(');
  return i === -1 ? "" : row.slice(i, row.indexOf("</select>", i) + "</select>".length);
};
ok(mrows.every(r => !ggufSelectOf(r).includes("mmproj.gguf")),
   "a projector file is never offered in the GGUF picker, under any record");

// ---- Models tab: the projector <select> next to the GGUF picker (mmSelect) -

// A projector beside the record's own GGUF ("near") sorts right after
// "— no projector —" and renders selected when the record already names it;
// one from another folder ("far") sits under an "other folders" optgroup
// labelled "<folder> / <name>".
const mmSelectOf = (row) => {
  const i = row.indexOf('<select data-help="projector"');
  return i === -1 ? "" : row.slice(i, row.indexOf("</select>", i) + "</select>".length);
};
const mmLocal = [
  {path: "D:\\models\\a\\main1.gguf", name: "main1.gguf", kind: "main", bytes: 5 * 1024 ** 3},
  {path: "D:\\models\\a\\proj-a.gguf", name: "proj-a.gguf", kind: "mmproj", size: 900 * 1024 * 1024},
  {path: "D:\\models\\b\\proj-b.gguf", name: "proj-b.gguf", kind: "mmproj", size: 800 * 1024 * 1024},
];
const mmModelRec = (path, mmproj) => Object.assign(modelRec("m1", path), {mmproj});
run(`LOCAL = ${JSON.stringify(mmLocal)}; MODELS = ${JSON.stringify(
  [mmModelRec("D:\\models\\a\\main1.gguf", "D:\\models\\a\\proj-a.gguf")])}; `
  + `UPSTREAM = {enabled: false}; DEFAULTS = {}; SIZING = {}; renderModels();`);
const mmRow0 = (sink["#mtab tbody"] || "").split("<tr>").slice(1)[0];
const mmSel0 = mmSelectOf(mmRow0);
ok(mmSel0 !== "", "a box with a projector on disk gets the projector <select>");
ok(mmSel0.indexOf("— no projector —") < mmSel0.indexOf("proj-a.gguf"),
   '"— no projector —" is always the first option');
ok(new RegExp('value="[^"]*proj-a\\.gguf" selected').test(mmSel0),
   "the record's own sibling projector renders selected");
ok(mmSel0.includes('<optgroup label="other folders">') && mmSel0.includes("b / proj-b.gguf"),
   'a projector from another folder sits under an "other folders" optgroup, labelled "<folder> / <name>"');
ok(mmSel0.indexOf('<optgroup label="other folders">') > mmSel0.indexOf("proj-a.gguf"),
   "the near projector (same folder) is listed ahead of the far optgroup");

// Folder matching for near/far is normDir()'s job, so it has to ignore slash
// direction and case exactly the way normDir does -- a record path typed by
// hand (forward slashes, lower-case drive letter) still matches a backslash,
// upper-case LOCAL entry in the "same" folder.
run(`LOCAL = ${JSON.stringify([
  {path: "D:\\models\\a\\proj-a.gguf", name: "proj-a.gguf", kind: "mmproj", size: 900 * 1024 * 1024},
])}; MODELS = ${JSON.stringify([mmModelRec("d:/models/a/main1.gguf", "")])}; `
  + `UPSTREAM = {enabled: false}; DEFAULTS = {}; SIZING = {}; renderModels();`);
const slashRow = (sink["#mtab tbody"] || "").split("<tr>").slice(1)[0];
const slashSel = mmSelectOf(slashRow);
ok(slashSel.includes("proj-a.gguf") && !slashSel.includes("other folders"),
   "folder matching ignores slash direction and case -- a lower-case, forward-slash record path "
   + "still counts a backslash, upper-case LOCAL projector in the same folder as near, not far");

// A record's mmproj that is not among LOCAL's projectors (an LM Studio import
// pointing at LM Studio's own copy, say) keeps a synthetic selected option
// instead of going blank on the next save.
run(`LOCAL = ${JSON.stringify(mmLocal)}; MODELS = ${JSON.stringify(
  [mmModelRec("D:\\models\\a\\main1.gguf", "C:\\lmstudio\\models\\x\\vision.gguf")])}; `
  + `UPSTREAM = {enabled: false}; DEFAULTS = {}; SIZING = {}; renderModels();`);
const outsideRow = (sink["#mtab tbody"] || "").split("<tr>").slice(1)[0];
const outsideSel = mmSelectOf(outsideRow);
ok(/selected>[^<]*\(outside the models folder\)<\/option>/.test(outsideSel),
   'a record\'s mmproj that is not among LOCAL\'s projectors keeps a synthetic selected option ending "(outside the models folder)"');

// No projector on the box and none named on the record: no control at all --
// this is what keeps models_boot_radio_check.mjs's radio-column checks green
// (a projector <select> in the row would shift where its own radio sits).
run(`LOCAL = ${JSON.stringify([
  {path: "D:\\models\\a\\main1.gguf", name: "main1.gguf", kind: "main", bytes: 5 * 1024 ** 3},
])}; MODELS = ${JSON.stringify([mmModelRec("D:\\models\\a\\main1.gguf", "")])}; `
  + `UPSTREAM = {enabled: false}; DEFAULTS = {}; SIZING = {}; renderModels();`);
const noneRow = (sink["#mtab tbody"] || "").split("<tr>").slice(1)[0];
ok(mmSelectOf(noneRow) === "",
   "no projector select at all when LOCAL has no mmproj entries and the record names none");

// ---- updPath: switching the GGUF moves the projector with it --------------

const updLocal = [
  {path: "D:\\models\\a\\main1.gguf", name: "main1.gguf", kind: "main", bytes: 5e9, mmproj: "D:\\models\\a\\proj-a.gguf"},
  {path: "D:\\models\\a\\main1b.gguf", name: "main1b.gguf", kind: "main", bytes: 5e9, mmproj: "D:\\models\\a\\proj-a.gguf"},
  {path: "D:\\models\\a\\proj-a.gguf", name: "proj-a.gguf", kind: "mmproj", size: 9e8},
  {path: "D:\\models\\a\\proj-a2.gguf", name: "proj-a2.gguf", kind: "mmproj", size: 8e8},
  {path: "D:\\models\\b\\main2.gguf", name: "main2.gguf", kind: "main", bytes: 5e9, mmproj: "D:\\models\\b\\proj-b.gguf"},
  {path: "D:\\models\\b\\proj-b.gguf", name: "proj-b.gguf", kind: "mmproj", size: 8e8},
  {path: "D:\\models\\c\\main3.gguf", name: "main3.gguf", kind: "main", bytes: 5e9, mmproj: ""},
];
// (a) moving to a GGUF in another folder picks up THAT folder's own sibling projector.
run(`LOCAL = ${JSON.stringify(updLocal)}; MODELS = ${JSON.stringify(
  [mmModelRec("D:\\models\\a\\main1.gguf", "D:\\models\\a\\proj-a.gguf")])}; `
  + `UPSTREAM = {enabled: false}; DEFAULTS = {}; SIZING = {};`);
run(`updPath(0, ${JSON.stringify("D:\\models\\b\\main2.gguf")})`);
ok(run("MODELS[0].path") === "D:\\models\\b\\main2.gguf", "updPath sets the new path");
ok(run("MODELS[0].mmproj") === "D:\\models\\b\\proj-b.gguf",
   "moving to a GGUF in another folder replaces the projector with that folder's own sibling projector");

// (b) moving on to a GGUF with no sibling projector clears it.
run(`updPath(0, ${JSON.stringify("D:\\models\\c\\main3.gguf")})`);
ok(run("MODELS[0].mmproj") === "", "moving to a GGUF with no sibling projector clears the projector");

// (c) moving between two GGUFs in the SAME folder keeps whatever projector is
// already picked there, even when it is not that GGUF's own auto-suggested one.
run(`LOCAL = ${JSON.stringify(updLocal)}; MODELS = ${JSON.stringify(
  [mmModelRec("D:\\models\\a\\main1b.gguf", "D:\\models\\a\\proj-a2.gguf")])}; `
  + `UPSTREAM = {enabled: false}; DEFAULTS = {}; SIZING = {};`);
run(`updPath(0, ${JSON.stringify("D:\\models\\a\\main1.gguf")})`);
ok(run("MODELS[0].mmproj") === "D:\\models\\a\\proj-a2.gguf",
   "switching to another GGUF already in the same folder keeps the projector that already sits there");

// (d) What decides it is whether the WEIGHTS changed folder, never where the
// projector is. The first cut compared the projector's folder with the new
// GGUF's, so a record kept text-only on purpose (no projector, to save its
// VRAM) had one silently attached the moment it moved to a sibling quant.
run(`LOCAL = ${JSON.stringify(updLocal)}; MODELS = ${JSON.stringify(
  [mmModelRec("D:\\models\\a\\main1b.gguf", "")])}; UPSTREAM = {enabled: false}; DEFAULTS = {}; SIZING = {};`);
run(`updPath(0, ${JSON.stringify("D:\\models\\a\\main1.gguf")})`);
ok(run("MODELS[0].mmproj") === "",
   "a text-only record stays text-only when it moves to a sibling quant in the same folder");

// (e) ...and a projector paired by hand from somewhere else entirely (an LM
// Studio copy) survives that same-folder move too.
run(`MODELS = ${JSON.stringify([mmModelRec("D:\\models\\a\\main1b.gguf", "C:\\lmstudio\\x\\proj.gguf")])};`);
run(`updPath(0, ${JSON.stringify("D:\\models\\a\\main1.gguf")})`);
ok(run("MODELS[0].mmproj") === "C:\\lmstudio\\x\\proj.gguf",
   "a hand-picked out-of-folder projector is kept across a same-folder GGUF change");

// (f) A brand-new row ("+ Add model", no path yet) picking its first GGUF is
// a folder change from nowhere, so it gets that GGUF's own projector.
run(`MODELS = ${JSON.stringify([mmModelRec("", "")])};`);
run(`updPath(0, ${JSON.stringify("D:\\models\\b\\main2.gguf")})`);
ok(run("MODELS[0].mmproj") === "D:\\models\\b\\proj-b.gguf",
   "a new row's first GGUF pick pairs that GGUF's own sibling projector");

// The pickers match a record's path to the listing by samePath, not ===: a
// record typed by hand says d:/models/a/main1.gguf where the gateway's own
// listing says D:\models\a\main1.gguf, and it is the same loadable file.
run(`MODELS = ${JSON.stringify([mmModelRec("d:/models/a/main1.gguf", "d:/models/a/proj-a.gguf")])}; renderModels();`);
const spRow = (sink["#mtab tbody"] || "");
ok(!spRow.includes("not a loadable model") && !spRow.includes("outside the models folder"),
   "a differently-spelled path still selects the on-disk GGUF and projector instead of a synthetic option");
ok(/value="D:\\models\\a\\main1\.gguf" selected/.test(spRow) && /value="D:\\models\\a\\proj-a\.gguf" selected/.test(spRow),
   "...the listing's own options are the ones marked selected");

// ---- unregisteredLocal: what "Add all" and the On-disk banner both count --

// Excludes: a shard or projector kind, an incomplete split, one the gateway
// already resolved to a record (in_models), and one already staged (even if
// not yet saved). Includes a kind-less entry -- an older gateway that sends
// no `kind` field at all still counts as "main".
const ulLocal = [
  {path: "D:\\models\\p\\main.gguf", name: "main.gguf", kind: "main", complete: true},
  {path: "D:\\models\\p\\shard.gguf", name: "shard.gguf", kind: "shard", complete: true},
  {path: "D:\\models\\p\\mmproj.gguf", name: "mmproj.gguf", kind: "mmproj", complete: true},
  {path: "D:\\models\\p\\broken.gguf", name: "broken.gguf", kind: "main", complete: false},
  {path: "D:\\models\\p\\known.gguf", name: "known.gguf", kind: "main", complete: true, in_models: "known-id"},
  {path: "D:\\models\\p\\staged.gguf", name: "staged.gguf", kind: "main", complete: true},
  {path: "D:\\models\\p\\old.gguf", name: "old.gguf", complete: true},
];
const ulModels = [{id: "staged-already", path: "D:\\models\\p\\staged.gguf"}];
const ulResult = JSON.parse(run(`JSON.stringify(unregisteredLocal(${JSON.stringify(ulLocal)}, ${JSON.stringify(ulModels)}))`));
ok(ulResult.length === 2 && ulResult.some(l => l.name === "main.gguf") && ulResult.some(l => l.name === "old.gguf"),
   `unregisteredLocal keeps only main.gguf and the kind-less old.gguf -- got ${ulResult.map(l => l.name).join(", ")}`);

// ---- renderLocalRows: the On-disk table's backlog banner -------------------

const lrLocalFixture = () => [
  {path: "D:\\models\\p\\a.gguf", name: "a.gguf", kind: "main", bytes: 1e9,
   complete: true, suggested_id: "a-id", mmproj: "", in_models: ""},
  {path: "D:\\models\\p\\b.gguf", name: "b.gguf", kind: "main", bytes: 1e9,
   complete: true, suggested_id: "b-id", mmproj: "", in_models: "already-b"},
  {path: "D:\\models\\p\\c.gguf", name: "c.gguf", kind: "main", bytes: 1e9,
   complete: false, suggested_id: "c-id", mmproj: "", in_models: ""},
  {path: "D:\\models\\p\\d.gguf", name: "d.gguf", kind: "main", bytes: 1e9,
   complete: true, suggested_id: "d-id", mmproj: "", in_models: ""},
];
const rowOf = (out, needle) => {
  const i = out.indexOf(needle);
  return out.slice(out.lastIndexOf("<tr>", i), out.indexOf("</tr>", i) + "</tr>".length);
};
const lrOut = run(`renderLocalRows(${JSON.stringify(lrLocalFixture())}, [], false)`);
ok(lrOut.includes("Add all 2 to Models") && lrOut.includes('onclick="addAllLocalModels()"'),
   "two unregistered entries (a.gguf and d.gguf, out of 4) get the backlog banner");
ok(rowOf(lrOut, "d.gguf").includes("Add to Models"),
   "...and each of those two still offers its own row-level Add to Models button");
ok(rowOf(lrOut, "b.gguf").includes("in Models as") && rowOf(lrOut, "b.gguf").includes("already-b"),
   '"in Models as <id>" is read from in_models even though `configured` (the saved records) is empty');
ok(rowOf(lrOut, "c.gguf").includes("incomplete") && !rowOf(lrOut, "c.gguf").includes("Add to Models"),
   "an incomplete entry (complete === false) is listed with the incomplete pill but no Add to Models button");

const lrOutOne = run(`renderLocalRows(${JSON.stringify([lrLocalFixture()[0]])}, [], false)`);
ok(!lrOutOne.includes("not in Models yet") && !lrOutOne.includes("Add all"),
   "exactly one unregistered entry: no banner (the row's own button is already the same click)");

const lrOutUpstream = run(`renderLocalRows(${JSON.stringify(lrLocalFixture())}, [], true)`);
ok(!lrOutUpstream.includes("Add all") && !lrOutUpstream.includes("Add to Models"),
   "an Ollama-managed box: no backlog banner and no per-row Add to Models button either");

// A row staged on the Models tab but not saved yet is neither "in Models" nor
// backlog: the Library used to keep offering "Add all 2" for models already
// sitting staged one tab over.
const lrStaged = [{id: "staged-a", path: "d:/models/p/a.gguf"}];
const lrOutStaged = run(`renderLocalRows(${JSON.stringify(lrLocalFixture())}, [], false, ${JSON.stringify(lrStaged)})`);
ok(rowOf(lrOutStaged, "a.gguf").includes("staged as") && rowOf(lrOutStaged, "a.gguf").includes("staged-a")
   && !rowOf(lrOutStaged, "a.gguf").includes("Add to Models"),
   "a staged-but-unsaved row reads 'staged as <id>' (path matched across slash/case spelling), with no second Add button");
ok(!lrOutStaged.includes("Add all"),
   "...and no longer counts toward the backlog banner (one left: below the banner's threshold of two)");

// An OLDER gateway answers without `in_models`. A path compare alone cannot
// see a GGUF already registered under its LM Studio hard link, so the
// add-everything pass is withheld there; the per-row buttons remain.
const lrOld = lrLocalFixture().map(({in_models, ...rest}) => rest);
ok(run(`bulkAddSupported(${JSON.stringify(lrLocalFixture())})`) === true
   && run(`bulkAddSupported(${JSON.stringify(lrOld)})`) === false
   && run("bulkAddSupported([])") === false,
   "bulkAddSupported: only when every loadable entry carries the in_models key");
const lrOutOld = run(`renderLocalRows(${JSON.stringify(lrOld)}, [], false)`);
ok(!lrOutOld.includes("Add all") && rowOf(lrOutOld, "a.gguf").includes("Add to Models"),
   "an older gateway: no backlog banner, per-row Add to Models still offered");

// ---- adoptServerModels: a reload keeps what is staged ----------------------

// Every /models reload used to replace MODELS wholesale, so the SECOND Add to
// Models click (which reloads first) silently threw away what the first had
// staged -- as did a visit to another tab.
run(`CUR = "boxA"; MODELS = []; adoptServerModels([{id: "s1", path: "/m/s1.gguf"}]);
     MODELS.push({id: "new", path: "/m/new.gguf"}); MODELS.push({id: "blank", path: ""});`);
ok(JSON.stringify(JSON.parse(run("JSON.stringify(stagedRows().map(m => m.id))"))) === JSON.stringify(["new", "blank"]),
   "stagedRows: the rows the gateway has never seen, not the ones it sent");
run(`adoptServerModels([{id: "s1", path: "/m/s1.gguf"}]);`);
ok(JSON.stringify(JSON.parse(run("JSON.stringify(MODELS.map(m => m.id))"))) === JSON.stringify(["s1", "new", "blank"]),
   "a reload adopts the gateway's records and puts the still-unsaved rows back");
run(`adoptServerModels([{id: "s1", path: "/m/s1.gguf"}, {id: "new-saved", path: "/M/NEW.gguf"}]);`);
ok(JSON.stringify(JSON.parse(run("JSON.stringify(MODELS.map(m => m.id))"))) === JSON.stringify(["s1", "new-saved", "blank"]),
   "a staged row is dropped once the gateway serves that file (a save landed) -- matched across path spelling");
run(`CUR = "boxB"; adoptServerModels([{id: "b1", path: "/m/b1.gguf"}]);`);
ok(JSON.stringify(JSON.parse(run("JSON.stringify(MODELS.map(m => m.id))"))) === JSON.stringify(["b1"]),
   "staged rows belong to the host they were staged for and never follow a host switch");
run(`CUR = null; MODELS = []; adoptServerModels([]);`);

// ---- switchSection vs showTab ----------------------------------------------

// addQuantModel's whole reason to call switchSection instead of showTab: it
// must NOT trigger tabLoad("models"), which would re-fetch /models and
// overwrite the record addQuantModel just pushed onto MODELS.
run("window.__tabLoadCalls = 0; tabLoad = () => { window.__tabLoadCalls++; };");
run("switchSection('models')");
ok(run("window.__tabLoadCalls") === 0, "switchSection does not call tabLoad");
run("showTab('models')");
ok(run("window.__tabLoadCalls") === 1, "showTab still calls tabLoad, exactly like before the split");

// ---- hfQuants: a 404-shaped error falls back to hfFiles() exactly once ----

// A peer whose gateway predates /hf/quants answers 404 through the fleet
// proxy; hfQuants degrades to the flat file list on that one failure shape,
// but a *different* failure of the same repo (a real outage, a typo) must
// not also trigger the flat-file fallback -- only the specific 404 case does.
run(`api = async () => { throw new Error("Not Found"); };
     hfFiles = () => { globalThis.__hfFilesCalls = (globalThis.__hfFilesCalls || 0) + 1; };
     toast = () => {};`);
await run("hfQuants('org/repo')");
ok(run("globalThis.__hfFilesCalls") === 1, "a 404 from /hf/quants falls back to hfFiles() exactly once");
ok(run("HFQ") === null, "HFQ is cleared to null on the 404 fallback");

run(`api = async () => { throw new Error("boom"); };`);
await run("hfQuants('org/repo')");
ok(run("globalThis.__hfFilesCalls") === 1, "a non-404 failure does not also fall back to hfFiles()");

// ---- toolbar: Files stayed a plain hfFiles() button; help text trimmed ----

ok(html.includes('onclick="hfFiles()"'),
   "the toolbar keeps a plain hfFiles() button (the flat per-file fallback view)");
ok(!html.includes("List files"),
   'the hf-query help text no longer mentions the old "List files" wording');

// ---- addQuantModel: double-click guard -------------------------------------

// Two overlapping calls each await their own loadModels(); whichever
// resolves last used to replace MODELS wholesale, including the record the
// other call had just pushed. ADDQ_BUSY is what makes the second call return
// before ever touching MODELS -- the regression this section exists to catch.
run(`loadModels = () => new Promise(r => { globalThis.__release = r; });
     MODELS = []; DEFAULTS = {}; renderModels = () => {}; switchSection = () => {}; toast = () => {};`);
const p1 = run("addQuantModel('a','/m/a.gguf','')");
const p2 = run("addQuantModel('b','/m/b.gguf','')");
run("globalThis.__release()");
await Promise.all([p1, p2]);
ok(run("MODELS.length") === 1,
   "addQuantModel's double-click guard: exactly one record staged from two overlapping calls");
ok(run("MODELS[0] && MODELS[0].id") === "a",
   "...and it is the first call's record, not the second's");

// ---- addAllLocalModels: stage the whole backlog in one pass ---------------

// Stubbed the same way as addQuantModel's own double-click guard above, since
// this is the tail end of the file and nothing after it needs the real
// switchSection/renderModels/loadModels/toast.
const aaLocal = [
  {path: "D:\\models\\p\\one.gguf", name: "one.gguf", kind: "main", complete: true,
   suggested_id: "keep-me", mmproj: "D:\\models\\p\\proj.gguf", in_models: ""},
  {path: "D:\\models\\q\\two.gguf", name: "two.gguf", kind: "main", complete: true,
   suggested_id: "alias-taken", mmproj: "", in_models: ""},
  {path: "D:\\models\\r\\three.gguf", name: "three.gguf", kind: "main", complete: true,
   suggested_id: "fresh-id", mmproj: "", in_models: ""},
];
const aaExisting = [{id: "keep-me", path: "D:\\models\\existing.gguf", aliases: ["alias-taken"]}];
run(`loadModels = async () => {}; toast = () => {}; switchSection = () => {}; renderModels = () => {}; `
  + `UPSTREAM = {enabled: false}; DEFAULTS = {}; `
  + `LOCAL = ${JSON.stringify(aaLocal)}; MODELS = ${JSON.stringify(aaExisting)};`);
await run("addAllLocalModels()");
const aaStaged = JSON.parse(run("JSON.stringify(MODELS)"));
ok(aaStaged.length === 4, `the existing record plus all 3 on-disk entries are staged -- got ${aaStaged.length}`);
const aaByPath = Object.fromEntries(aaStaged.map(m => [m.path, m]));
ok(aaByPath["D:\\models\\p\\one.gguf"] && aaByPath["D:\\models\\p\\one.gguf"].id === "keep-me-2",
   "an id colliding with an existing record's own id gets a -2 suffix");
ok(aaByPath["D:\\models\\p\\one.gguf"].mmproj === "D:\\models\\p\\proj.gguf",
   "the on-disk entry's mmproj is staged onto the new record");
ok(aaByPath["D:\\models\\q\\two.gguf"] && aaByPath["D:\\models\\q\\two.gguf"].id === "alias-taken-2",
   "an id colliding with an existing record's ALIAS also gets a -2 suffix");
ok(aaByPath["D:\\models\\r\\three.gguf"] && aaByPath["D:\\models\\r\\three.gguf"].id === "fresh-id"
   && aaByPath["D:\\models\\r\\three.gguf"].mmproj === "",
   "an id with no collision stages unchanged, with an empty mmproj when the entry names none");

// A second call finds nothing new -- every path from the first pass is now
// staged on MODELS (even though nothing has been Saved yet).
await run("addAllLocalModels()");
ok(run("MODELS.length") === 4, "a second call stages nothing more");

// An Ollama-managed box: nothing here can serve a GGUF record.
run(`MODELS = ${JSON.stringify(aaExisting)}; UPSTREAM = {enabled: true};`);
await run("addAllLocalModels()");
ok(run("MODELS.length") === 1, "an Ollama-managed box: addAllLocalModels stages nothing");

// An older gateway (no in_models on its entries): the pass is withheld, since
// it could re-register weights already served under their LM Studio link.
run(`MODELS = ${JSON.stringify(aaExisting)}; UPSTREAM = {enabled: false}; `
  + `LOCAL = ${JSON.stringify(aaLocal.map(({in_models, ...rest}) => rest))};`);
await run("addAllLocalModels()");
ok(run("MODELS.length") === 1, "an older gateway without in_models: addAllLocalModels stages nothing");

// An id is at most 64 characters and one over-long id fails the WHOLE save,
// so a collision suffix is cut in under the limit rather than appended past it.
const longId = "x".repeat(64);
run(`MODELS = ${JSON.stringify([{id: longId, path: "D:\\models\\existing.gguf"}])}; `
  + `LOCAL = ${JSON.stringify([{path: "D:\\models\\z\\long.gguf", name: "long.gguf", kind: "main",
       complete: true, suggested_id: longId, mmproj: "", in_models: ""}])};`);
await run("addAllLocalModels()");
const longStaged = run("MODELS[1] && MODELS[1].id") || "";
ok(longStaged.length <= 64 && longStaged.endsWith("-2") && longStaged !== longId,
   `a colliding 64-character id still fits the limit after its suffix -- got ${longStaged.length} chars`);

if (fail.length) { console.error("FAIL:\n - " + fail.join("\n - ")); process.exit(1); }
console.log("all assertions passed");
