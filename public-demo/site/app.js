import { SCENARIOS } from "./content.js";

const LOCAL = location.hostname === "localhost" || location.hostname === "127.0.0.1";
const API = LOCAL ? "http://127.0.0.1:8100" : "https://lltm-playground.onrender.com";
const COLD_START_MS = 90_000;

let lang = /^zh\b/i.test(navigator.language || "") ? "zh" : "en";
let currentTour = "changed";
let tourRan = false;
let token = null;
let liveReady = false;
let liveBusy = false;
let wakingTimer = null;
let lastLive = null;
let releaseData = null;
let liveHealth = null;
let currentEngineMessage = {
  zh: "正在连接真实引擎…",
  en: "Connecting to the live engine…",
};

const $ = (id) => document.getElementById(id);
const T = (zh, en) => (lang === "zh" ? zh : en);
const esc = (value) => String(value ?? "").replace(
  /[&<>"]/g,
  (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char],
);

function setText(id, text) {
  $(id).textContent = text;
}

function setStep(journeyId, step) {
  $(journeyId).dataset.step = String(step);
  document.querySelectorAll(`[data-step-nav="${journeyId}"]`).forEach((button) => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.step) === step));
  });
}

function renderTourStep(step) {
  let body = `<p>${esc(step.lead)}</p>`;
  if (step.quote) body += `<blockquote class="quote">${esc(step.quote)}</blockquote>`;
  if (step.memories) {
    body += `<div class="memory-mini">${step.memories.map((memory) => `
      <div class="${memory.old ? "old" : ""}">
        <strong>${esc(memory.value)}</strong><small>${esc(memory.state)}</small>
      </div>`).join("")}</div>`;
  }
  if (step.reason) body += `<p class="reason">${esc(step.reason)}</p>`;
  return body;
}

function renderTour() {
  const copy = SCENARIOS[currentTour].tour[lang];
  copy.steps.forEach((step, index) => {
    setText(`tourStep${index + 1}Title`, step.title);
    const body = $(`tourStep${index + 1}Body`);
    body.innerHTML = index === 0 || tourRan
      ? renderTourStep(step)
      : `<p class="empty">${esc(T("运行演示后显示。", "Run the tour to reveal this step."))}</p>`;
  });
  [$("tourCard2"), $("tourCard3")].forEach((card) => {
    card.classList.toggle("revealed", tourRan);
  });
  setText("runTour", tourRan ? T("重新运行演示", "Run the tour again") : T("运行 30 秒演示", "Run the 30-second tour"));
}

function runTour() {
  tourRan = true;
  $("tourJourney").setAttribute("aria-busy", "true");
  renderTour();
  setStep("tourJourney", 1);
  $("tourJourney").setAttribute("aria-busy", "false");
  setText("tourAnnounce", SCENARIOS[currentTour].tour[lang].announce);
  $("tour").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}

function selectTour(name) {
  currentTour = name;
  tourRan = false;
  document.querySelectorAll("[data-tour]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.tour === name));
  });
  setText("tourAnnounce", "");
  setStep("tourJourney", 1);
  renderTour();
}

function renderMetrics() {
  if (!releaseData) return;
  const release = releaseData;
  const heldout = release.heldout;
  const engineering = release.engineering;
  setText("metricAccuracy", `${heldout.accuracy.toFixed(1)}%`);
  setText(
    "metricAccuracySource",
    T(`${heldout.dataset} · 冻结后单次运行`, `${heldout.dataset} · one frozen run`),
  );
  setText("metricContext", heldout.median_context_tokens.toLocaleString("en-US"));
  setText(
    "metricContextSource",
    T(
      `整段对话中位数 ${heldout.whole_transcript_median_tokens.toLocaleString("en-US")}`,
      `whole-transcript median ${heldout.whole_transcript_median_tokens.toLocaleString("en-US")}`,
    ),
  );
  setText("metricArchive", `+${heldout.raw_archive_recovery_pp}pp`);
  setText(
    "metricArchiveSource",
    T(
      `结构化记忆 ${heldout.structured_memory_accuracy.toFixed(1)}% → 最终 ${heldout.accuracy.toFixed(1)}%`,
      `structured memory ${heldout.structured_memory_accuracy.toFixed(1)}% → final ${heldout.accuracy.toFixed(1)}%`,
    ),
  );
  setText("metricTests", engineering.tests.toLocaleString("en-US"));
  setText(
    "metricTestsSource",
    T(`行覆盖率 ${engineering.line_coverage}%`, `${engineering.line_coverage}% line coverage`),
  );
  setText(
    "releaseMeta",
    T(
      `${release.release} 指标 · 工程验证于 ${engineering.verified_at}`,
      `${release.release} metrics · engineering verified ${engineering.verified_at}`,
    ),
  );
  setText("releaseCommit", engineering.source_commit);
  $("releaseCommit").href = `https://github.com/Mingjie-Mao/llm-long-term-memory/commit/${engineering.source_commit}`;
}

async function loadMetrics() {
  try {
    const response = await fetch("./release.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`release manifest returned ${response.status}`);
    releaseData = await response.json();
    renderMetrics();
  } catch (error) {
    console.warn("release metrics unavailable", error);
    setText("releaseMeta", T("指标清单暂时无法读取。", "Metric manifest unavailable."));
  }
}

function showTrace(method, path, body, response, status, elapsed) {
  setText(
    "api",
    `${method} ${path}  →  ${status}   ${elapsed}ms\n`
      + (body ? `\nrequest\n${JSON.stringify(body, null, 2)}\n` : "")
      + `\nresponse\n${JSON.stringify(response, null, 2)}`,
  );
}

function errorMessage(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail.message === "string") return detail.message;
  // FastAPI reports request-validation failures as a list of per-field errors.
  if (Array.isArray(detail) && detail.length) {
    const field = detail[0].loc?.filter((part) => part !== "body").join(".");
    const reason = detail[0].msg || fallback;
    return field ? `${field}: ${reason}` : reason;
  }
  return fallback;
}

async function api(method, path, body, { cold = false } = {}) {
  const started = performance.now();
  const headers = { "content-type": "application/json" };
  if (token) headers["x-demo-token"] = token;
  let response;
  let payload;
  try {
    response = await fetch(API + path, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(cold ? COLD_START_MS : 20_000),
    });
    payload = await response.json().catch(() => ({}));
  } catch (error) {
    showTrace(method, path, body, { error: error.message }, "network error", Math.round(performance.now() - started));
    throw error;
  }
  showTrace(method, path, body, payload, `${response.status} ${response.statusText}`, Math.round(performance.now() - started));
  if (!response.ok) throw new Error(errorMessage(payload, response.statusText));
  return payload;
}

function setLiveControls() {
  document.querySelectorAll("[data-requires-live]").forEach((element) => {
    element.disabled = !liveReady || liveBusy;
  });
  $("liveJourney").setAttribute("aria-busy", String(liveBusy));
}

function setEngineState(state, zh, en) {
  currentEngineMessage = { zh, en };
  $("engineStatus").setAttribute("aria-busy", String(state === "loading"));
  $("dot").className = `dot${state === "ready" ? " on" : state === "error" ? " off" : ""}`;
  setText("statusText", T(zh, en));
  $("retry").classList.toggle("hidden", state !== "error");
}

function showReadyState() {
  if (!liveHealth || !token) return;
  setEngineState(
    "ready",
    `真实引擎在线 · 临时会话 ${token.split(".")[0]} · ${liveHealth.session_ttl_minutes} 分钟后失效`,
    `Live engine ready · temporary session ${token.split(".")[0]} · expires after ${liveHealth.session_ttl_minutes} min`,
  );
}

function beginWakeClock() {
  let seconds = 0;
  clearInterval(wakingTimer);
  wakingTimer = setInterval(() => {
    seconds += 1;
    if (seconds >= 3) {
      setEngineState(
        "loading",
        `真实引擎正在唤醒，已等待 ${seconds} 秒。引导体验仍可立即使用。`,
        `The live engine is waking up (${seconds}s). The guided tour remains available now.`,
      );
    }
  }, 1000);
}

async function connectEngine() {
  liveReady = false;
  liveBusy = false;
  token = null;
  setLiveControls();
  setEngineState("loading", "正在连接真实引擎…", "Connecting to the live engine…");
  beginWakeClock();
  try {
    const health = await api("GET", "/demo/health", null, { cold: true });
    const session = await api("POST", "/demo/session");
    clearInterval(wakingTimer);
    token = session.token;
    liveHealth = health;
    liveReady = true;
    setText("enc", health.encoder);
    showReadyState();
    setLiveControls();
  } catch (error) {
    clearInterval(wakingTimer);
    setText("enc", "—");
    setEngineState(
      "error",
      LOCAL ? `本地后端未连接：${error.message}` : `真实引擎暂时不可用：${error.message}`,
      LOCAL ? `Local backend unavailable: ${error.message}` : `Live engine unavailable: ${error.message}`,
    );
  }
}

async function freshSession() {
  if (token) {
    try { await api("DELETE", "/demo/session"); } catch { /* expired or already removed */ }
  }
  token = null;
  const session = await api("POST", "/demo/session");
  token = session.token;
}

function renderConversation(name) {
  const rows = SCENARIOS[name].view[lang].conversation;
  $("liveConversation").innerHTML = rows.map((row) => `
    <div class="bubble ${row.role === "assistant" ? "assistant" : ""}">
      <small>${esc(row.role)}</small>${esc(row.text)}
      ${row.gloss ? `<p class="gloss">${esc(row.gloss)}</p>` : ""}
    </div>`).join("");
}

function memoryCard(memory) {
  const superseded = memory.status === "superseded";
  const start = (memory.valid_from || memory.event_time || "").slice(0, 10) || "—";
  const end = (memory.valid_to || "").slice(0, 10);
  return `<div class="memory-card ${superseded ? "superseded" : ""}">
    <span class="key">${esc(memory.predicate || "memory")}</span>
    <strong>${esc(memory.object || memory.content)}</strong>
    <small>${esc(superseded
      ? T(`已取代 · ${start} 至 ${end || "—"}`, `superseded · ${start} to ${end || "—"}`)
      : T(`当前有效 · ${start} 起`, `current · since ${start}`))}</small>
  </div>`;
}

function renderMemories(data) {
  const rows = [...(data.active || []), ...(data.superseded || [])];
  $("liveMemory").innerHTML = rows.length
    ? rows.map(memoryCard).join("")
    : `<p class="empty">${esc(T("没有结构化记忆。", "No structured memories."))}</p>`;
}

function renderResult(name, result, mode) {
  const summary = name ? SCENARIOS[name].view[lang].answer : T("查询已完成。", "Query complete.");
  let html = `<div class="result-card"><div class="answer">${esc(summary)}</div>`;
  if (mode === "raw") {
    html += `<small>${esc(T(`原文命中 ${result.turns.length} 条`, `${result.turns.length} raw turn(s) matched`))}</small></div>`;
    html += result.turns.map((turn) => `<div class="memory-card"><span class="key">raw turn ${turn.turn_index}</span><strong>${esc(turn.content)}</strong><small>${esc(turn.role)}</small></div>`).join("");
  } else {
    html += `<small>${esc(T(`选中 ${result.memories.length} 条，排除 ${result.rejected.length} 条`, `${result.memories.length} selected, ${result.rejected.length} rejected`))}</small></div>`;
    html += result.memories.map((memory) => `<div class="memory-card"><span class="key">selected · ${Number(memory.score).toFixed(2)}</span><strong>${esc(memory.content)}</strong><small>${esc(memory.predicate || "memory")}</small></div>`).join("");
    html += result.rejected.map((memory) => `<div class="reject-card"><strong>${esc(memory.content)}</strong><small>${esc(memory.reason)}${memory.superseded_by ? ` → ${esc(memory.superseded_by)}` : ""}</small></div>`).join("");
  }
  $("liveResult").innerHTML = html;
}

async function runLiveScenario(name) {
  if (!liveReady || liveBusy) return;
  liveBusy = true;
  setLiveControls();
  document.querySelectorAll("[data-live]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.live === name));
  });
  setEngineState(
    "loading",
    "正在清理临时会话并运行场景…",
    "Clearing the temporary session and running the scenario…",
  );
  setText("liveAnnounce", T("场景运行中。", "Scenario running."));
  setStep("liveJourney", 1);
  renderConversation(name);
  setText("liveMemory", "…");
  setText("liveResult", "…");
  const scenario = SCENARIOS[name];
  try {
    // Every preset starts from a newly minted namespace. A partially failed run can
    // therefore never contaminate the next scenario or a repeated click.
    await freshSession();
    for (const fact of scenario.engine.facts || []) await api("POST", "/demo/facts", fact);
    for (const turn of scenario.engine.turns || []) await api("POST", "/demo/turns", turn);
    const memories = await api("GET", "/demo/memories");
    const result = scenario.engine.searchMode === "raw"
      ? await api("POST", "/demo/raw/search", { query: scenario.engine.query, limit: 3 })
      : await api("POST", "/demo/search", { query: scenario.engine.query, limit: 5 });
    renderMemories(memories);
    renderResult(name, result, scenario.engine.searchMode);
    lastLive = { name, memories, result, mode: scenario.engine.searchMode };
    setStep("liveJourney", 3);
    setText("liveAnnounce", T("真实引擎场景运行完成，已跳到第三步结果。", "Live engine scenario complete; moved to step three, the result."));
  } catch (error) {
    $("liveResult").innerHTML = `<p class="message error" role="alert">${esc(error.message)}</p>`;
    setText("liveAnnounce", T(`场景失败：${error.message}`, `Scenario failed: ${error.message}`));
    if (!token) {
      liveReady = false;
      setEngineState(
        "error",
        `无法创建干净的临时会话：${error.message}`,
        `Could not create a clean temporary session: ${error.message}`,
      );
    }
  } finally {
    if (liveReady && token) showReadyState();
    liveBusy = false;
    setLiveControls();
  }
}

function advancedMessage(text, error = false) {
  setText("advancedMessage", text);
  $("advancedMessage").classList.toggle("error", error);
}

async function refreshAdvancedMemory() {
  const memories = await api("GET", "/demo/memories");
  renderMemories(memories);
  return memories;
}

async function writeFact() {
  const body = {
    predicate: $("predicate").value.trim(),
    object: $("object").value.trim(),
    content: $("content").value.trim(),
    event_time: $("when").value.trim() || null,
    replaces_previous: $("replaces").checked,
  };
  if (!body.predicate || !body.object || !body.content) {
    advancedMessage(T("原始表述、时间键和值都必须填写。", "Source wording, temporal key, and value are required."), true);
    return;
  }
  liveBusy = true;
  setLiveControls();
  try {
    const result = await api("POST", "/demo/facts", body);
    await refreshAdvancedMemory();
    advancedMessage(T(`写入成功；关闭 ${result.changed_by_this_write.length} 条旧值。`, `Written; ${result.changed_by_this_write.length} old value(s) closed.`));
  } catch (error) {
    advancedMessage(error.message, true);
  } finally {
    liveBusy = false;
    setLiveControls();
  }
}

async function writeTurn() {
  const content = $("turn").value.trim();
  if (!content) return advancedMessage(T("请先填写原始对话。", "Enter a raw turn first."), true);
  liveBusy = true;
  setLiveControls();
  try {
    await api("POST", "/demo/turns", { role: "assistant", content });
    advancedMessage(T("已存入原文档案。", "Stored in the raw archive."));
  } catch (error) {
    advancedMessage(error.message, true);
  } finally {
    liveBusy = false;
    setLiveControls();
  }
}

async function customSearch(mode) {
  const query = $("query").value.trim();
  if (!query) return advancedMessage(T("请先填写查询。", "Enter a query first."), true);
  liveBusy = true;
  setLiveControls();
  try {
    const path = mode === "raw" ? "/demo/raw/search" : "/demo/search";
    const result = await api("POST", path, { query, limit: mode === "raw" ? 3 : 5 });
    renderResult(null, result, mode);
    setStep("liveJourney", 3);
    advancedMessage(T("查询完成，结果显示在上方第三步。", "Query complete; the result is shown in step three above."));
  } catch (error) {
    advancedMessage(error.message, true);
  } finally {
    liveBusy = false;
    setLiveControls();
  }
}

function applyLanguage(next) {
  lang = next;
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.body.dataset.lang = lang;
  document.querySelectorAll("[data-zh]").forEach((element) => {
    const value = element.getAttribute(lang === "zh" ? "data-zh" : "data-en");
    if (value !== null) element.textContent = value;
  });
  document.querySelectorAll("[data-placeholder-zh]").forEach((element) => {
    element.placeholder = element.getAttribute(lang === "zh" ? "data-placeholder-zh" : "data-placeholder-en");
  });
  document.querySelectorAll("[data-aria-zh]").forEach((element) => {
    element.setAttribute("aria-label", element.getAttribute(lang === "zh" ? "data-aria-zh" : "data-aria-en"));
  });
  $("bzh").setAttribute("aria-pressed", String(lang === "zh"));
  $("ben").setAttribute("aria-pressed", String(lang === "en"));
  setText("statusText", T(currentEngineMessage.zh, currentEngineMessage.en));
  renderMetrics();
  try { localStorage.setItem("lltm.demo.lang", lang); } catch { /* private mode */ }
  renderTour();
  if (lastLive) {
    renderConversation(lastLive.name);
    renderMemories(lastLive.memories);
    renderResult(lastLive.name, lastLive.result, lastLive.mode);
  }
}

$("runTour").addEventListener("click", runTour);
document.querySelectorAll("[data-tour]").forEach((button) => button.addEventListener("click", () => selectTour(button.dataset.tour)));
document.querySelectorAll("[data-live]").forEach((button) => button.addEventListener("click", () => runLiveScenario(button.dataset.live)));
document.querySelectorAll("[data-step-nav]").forEach((button) => {
  button.addEventListener("click", () => setStep(button.dataset.stepNav, Number(button.dataset.step)));
});
$("retry").addEventListener("click", connectEngine);
$("save").addEventListener("click", writeFact);
$("saveTurn").addEventListener("click", writeTurn);
$("ask").addEventListener("click", () => customSearch("memory"));
$("askRaw").addEventListener("click", () => customSearch("raw"));
$("query").addEventListener("keydown", (event) => { if (event.key === "Enter") customSearch("memory"); });
$("bzh").addEventListener("click", () => applyLanguage("zh"));
$("ben").addEventListener("click", () => applyLanguage("en"));

try {
  const saved = localStorage.getItem("lltm.demo.lang");
  if (saved === "zh" || saved === "en") lang = saved;
} catch { /* private mode */ }

applyLanguage(lang);
loadMetrics();
connectEngine();
