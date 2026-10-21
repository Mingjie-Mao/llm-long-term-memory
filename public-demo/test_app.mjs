// Exercise interrupted user journeys with scripted HTTP transport, never model calls.
import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";

const source = readFileSync(new URL("./site/app.js", import.meta.url), "utf8")
  .replace(/^import .*;\n/, "")
  .split('$("resumeScenario").addEventListener')[0];
const presets = readFileSync(new URL("./site/content.js", import.meta.url), "utf8")
  .replace("export const", "const");
function page(fail) {
  const nodes = new Map(), requests = [], delays = [];
  const node = () => ({
    dataset: {}, value: "", textContent: "", children: [],
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    setAttribute() {},
  });
  const document = {
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); },
    createElement: node, querySelectorAll: () => [],
  };
  const context = vm.createContext({
    document, location: { hostname: "localhost" }, Date, AbortController,
    crypto: { randomUUID: () => `request-${requests.length}` },
    sessionStorage: { setItem() {}, removeItem() {} },
    setInterval: () => 1, clearInterval() {},
    setTimeout(callback, ms) { delays.push(ms); if (ms === 5000) callback(); return 1; },
    clearTimeout() {},
    async fetch(url, options = {}) {
      const path = new URL(url).pathname;
      const body = options.body && JSON.parse(options.body);
      const request = { path, method: options.method, body };
      requests.push(request);
      const error = fail?.(request, requests);
      const data = error || (path === "/demo/session" ? { token: "test-namespace" }
        : path === "/demo/messages" ? { mode: "llm", memories: [], extracted: [] }
        : path === "/demo/answer" ? { mode: "llm", answer_profile: "t4", answer: "scripted test" }
        : { memories: [] });
      return { ok: !error, status: error ? (error.detail.error === "demo_capacity" ? 429 : 503) : 200,
        headers: { get: (name) => name === "content-type" ? "application/json" : null },
        json: async () => data };
    },
  });
  vm.runInContext(presets + "\n" + source + '\nready = true; renderMemories = () => {}; renderAnswer = () => {}; renderUsage = () => {};', context);
  return { context, nodes, requests, delays, run: (code) => vm.runInContext(code, context) };
}
const unavailable = (code) => ({ detail: { error: "llm_unavailable", kind: code ? "ServerError" : "ReadTimeout", provider_code: code } });
const capacity = (message) => ({ detail: { error: "demo_capacity", message } });
const calls = (p, path) => p.requests.filter((r) => r.path === path);

test("resume a failed answer without repeating successful extraction", async () => {
  let failed = false;
  const p = page((r) => r.path === "/demo/answer" && !failed && (failed = true) && unavailable());
  await p.run('runLiveScenario("update")');
  assert.equal(calls(p, "/demo/messages").length, 2);
  assert.equal(p.run("scenarioRun.nextTurn"), 2);
  await p.run("action(advanceScenario)");
  assert.equal(calls(p, "/demo/messages").length, 2);
  assert.equal(calls(p, "/demo/answer").length, 2);
  assert.equal(calls(p, "/demo/session").length, 1);
  assert.equal(p.run("scenarioRun"), null);
});

test("resume a partial write with the same request identity and original language", async () => {
  let writes = 0;
  const p = page((r) => r.path === "/demo/messages" && ++writes === 2 && unavailable());
  await p.run('runLiveScenario("update")');
  assert.equal(p.run("scenarioRun.nextTurn"), 1);
  await p.run('lang = "zh"; action(advanceScenario)');
  const sent = calls(p, "/demo/messages");
  assert.equal(sent.length, 3);
  assert.deepEqual(sent[1].body, sent[2].body);
  assert.match(sent[2].body.content, /switched/);
  assert.equal(sent.filter((r) => /I use TensorFlow/.test(r.body.content)).length, 1);
  assert.equal(p.nodes.get("chat").children.filter((el) => el.className === "bubble user").length, 3);
});

test("retry an as_of read without calling the successful answer again", async () => {
  let failed = false;
  const p = page((r) => r.path === "/demo/as_of" && !failed && (failed = true) && unavailable());
  await p.run('runLiveScenario("temporal")');
  assert.equal(p.run("scenarioRun.answerDone"), true);
  await p.run("action(advanceScenario)");
  assert.equal(calls(p, "/demo/answer").length, 1);
  assert.equal(calls(p, "/demo/messages").length, 2);
  assert.equal(p.nodes.get("historyDetails").open, true);
});

test("rate limit pauses the preset and blocks attempts until the cooldown ends", async () => {
  let failed = false;
  const p = page((r) => r.path === "/demo/answer" && !failed && (failed = true) && capacity("Model rate limit reached; retry later."));
  await p.run('runLiveScenario("recall")');
  assert.equal(p.nodes.get("resumeScenario").disabled, true);
  const before = p.requests.length;
  await p.run("action(advanceScenario)");
  assert.equal(p.requests.length, before);
  await p.run("cooldownUntil = Date.now() - 1; renderCooldown(); action(advanceScenario)");
  assert.equal(calls(p, "/demo/messages").length, 1);
  assert.equal(p.run("scenarioRun"), null);
});

test("provider 429 also enters cooldown without an automatic retry", async () => {
  const p = page((r) => r.path === "/demo/answer" && unavailable(429));
  await p.run('runLiveScenario("recall")');
  assert.equal(calls(p, "/demo/answer").length, 1);
  assert.equal(p.nodes.get("resumeScenario").disabled, true);
});

test("exhausted capacity ends the run instead of retrying or clearing the namespace", async () => {
  const p = page((r) => r.path === "/demo/answer" && capacity("Demo capacity exhausted; try again after the UTC day resets."));
  await p.run('runLiveScenario("recall")');
  assert.equal(p.run("ready"), false);
  assert.equal(p.run("scenarioRun"), null);
  assert.equal(calls(p, "/demo/answer").length, 1);
  assert.equal(calls(p, "/demo/session").length, 1);
});

test("provider 503 still gets only one page retry, preserving the write request", async () => {
  const p = page((r) => r.path === "/demo/messages" && unavailable(503));
  await p.run('runLiveScenario("recall")');
  const writes = calls(p, "/demo/messages");
  assert.equal(writes.length, 2);
  assert.deepEqual(writes[0].body, writes[1].body);
  assert.equal(p.delays.filter((ms) => ms === 5000).length, 1);
  assert.equal(p.run("scenarioRun.nextTurn"), 0);
});
