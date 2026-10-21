import { SCENARIOS } from "./content.js";

const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const API = LOCAL ? "http://127.0.0.1:8100" : "https://lltm-playground.onrender.com";
// A sleeping free instance measured 81.5 s to boot in a constrained container; the
// first request also waits for Render to route to it, so 90 s left almost no margin.
const TIMEOUT_MS = 150000;
const SAVED = "lltm-demo-session";
const $ = (id) => document.getElementById(id);
let lang = "en", ready = false, busy = false, token = null, conversation = 1;
let manifest = null, pendingWrite = null, statusRecord = null, wakingTimer = null;
let scenarioRun = null, cooldownUntil = 0, cooldownTimer = null, recording = null;
const t = (zh, en) => lang === "zh" ? zh : en;
const conversationId = () => `conversation-${conversation}`;

// Status and error copy, as [zh, en] pairs so a language switch re-renders them.
const COPY = {
  disabled: ["真实 LLM 服务尚未启用。不会显示规则生成的替代答案。", "Live LLM service is not enabled. Rule-generated substitutes are disabled."],
  modelFailed: ["模型调用失败，本次没有生成替代答案。可以直接再发一次；同一条消息不会重复写入。", "The model call failed and no substitute was generated. You can send it again; the same message will not be written twice."],
  ready: ["真实 LLM 已就绪 · t4", "Live LLM ready · t4"],
  deleted: ["数据已删除，点击重试可新建演示。", "Data deleted. Retry to start a new demo."],
  timeout: ["后端在 150 秒内没有响应，请稍后点击重试。", "The backend did not respond within 150 s. Retry in a moment."],
  network: ["无法连接后端，请检查网络后重试。", "Cannot reach the backend. Check your connection and retry."],
  badResponse: ["后端返回了非预期的响应（可能仍在启动），请稍后重试。", "The backend returned an unexpected response (it may still be starting). Retry in a moment."],
  expired: ["演示空间已过期或失效，点击重试可新建演示。", "This demo namespace expired. Retry to start a new demo."],
  capacity: ["演示额度已用完，请在 UTC 日切换后再试。", "Demo capacity is exhausted; try again after the UTC day resets."],
  rateLimited: ["模型请求触发限流，请等倒计时结束再试；若仍失败，可先看录制演示。", "Model requests are rate limited. Wait for the countdown, then retry; if it persists, watch the recording."],
  operationTimeout: ["这一步超过了 60 秒的处理上限，请重试。", "This step exceeded its 60-second limit. Please retry."],
};
const waking = (seconds) => [
  `正在唤醒后端 · ${seconds}s（免费实例休眠后首次唤醒可能需要 1–2 分钟）`,
  `Waking the backend · ${seconds}s (a sleeping free instance can take 1–2 minutes)`,
];
function failure(key, status) {
  const error = new Error(COPY[key][1]);
  error.key = key;
  error.status = status;
  return error;
}
// A diagnostic suffix (the failure's class and provider status) is shown verbatim in
// both languages, so a visitor's report names the cause.
const pairOf = (error) => {
  if (!error.key) return [error.message, error.message];
  const extra = error.diagnostic ? ` (${error.diagnostic})` : "";
  return COPY[error.key].map((text) => text + extra);
};

// The token lives in this tab's sessionStorage so a refresh returns to the same memories.
// Storage can be unavailable (private mode, blocked site data); the demo then works for
// the life of the page, as it did before.
function loadSaved() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(SAVED) || "null");
    if (saved && typeof saved.token === "string" && Number.isInteger(saved.conversation)) return saved;
  } catch {}
  return null;
}
function save() {
  try {
    if (token) sessionStorage.setItem(SAVED, JSON.stringify({ token, conversation }));
    else sessionStorage.removeItem(SAVED);
  } catch {}
}

function applyLanguage() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  for (const el of document.querySelectorAll("[data-zh][data-en]")) el.textContent = el.dataset[lang];
  for (const el of document.querySelectorAll("[data-placeholder-zh]")) {
    el.placeholder = el.dataset[lang === "zh" ? "placeholderZh" : "placeholderEn"];
  }
  $("bzh").setAttribute("aria-pressed", String(lang === "zh"));
  $("ben").setAttribute("aria-pressed", String(lang === "en"));
  for (const el of document.querySelectorAll("[data-aria-zh][data-aria-en]")) {
    el.setAttribute("aria-label", el.dataset[lang === "zh" ? "ariaZh" : "ariaEn"]);
  }
  title();
  renderMetrics();
  renderRecording();
  renderCooldown();
  if (statusRecord) status(statusRecord.state, statusRecord.pair);
}
// Only the release manifest supplies measured values. Each table keeps its own
// experiment scope; recorded provider usage is never mixed with context estimates.
function metricRows(id, rows) {
  $(id).replaceChildren(...rows.map((values) => {
    const row = document.createElement("tr");
    for (const [index, value] of values.entries()) {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) cell.scope = "row";
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  }));
}
function renderMetrics() {
  if (!manifest) return;
  const { result, release, regression, engineering } = manifest;
  const percent = (n) => `${Number(n.toFixed(2))}%`;
  const number = (n) => n.toLocaleString("en-US");
  const link = $("releaseCommit");
  $("regressionComparison").hidden = !regression;
  if (regression) {
    $("headline").textContent = t(
      `原 test100 回归：${percent(regression.candidate_accuracy)} 正确率；同期旧版 ${percent(regression.control_accuracy)}，提升 ${Number(regression.mean_difference_pp.toFixed(2))} 个百分点。`,
      `Original test100 regression: ${percent(regression.candidate_accuracy)} accuracy; ${percent(regression.control_accuracy)} for the concurrent baseline, a ${Number(regression.mean_difference_pp.toFixed(2))}-point gain.`,
    );
    metricRows("regressionRows", [
      [t("同期旧 v2", "Concurrent v2 control"), `${regression.control_correct.join(" / ")} (${t("每轮", "each out of")} ${regression.questions})`, percent(regression.control_accuracy), number(regression.control_recorded_tokens)],
      [t("限定范围 t4 候选", "Scoped t4 candidate"), `${regression.candidate_correct.join(" / ")} (${t("每轮", "each out of")} ${regression.questions})`, percent(regression.candidate_accuracy), number(regression.candidate_recorded_tokens)],
    ]);
    $("footnote").textContent = t(
      `已暴露题集，${regression.questions} 道独立题，每组重复 ${regression.repeats_per_arm} 次；模型未换，${regression.reader_rows} 份回答与 ${regression.grade_rows} 份评分通过完整性审计。上表为回答＋评分已记录 token，不含原有记忆抽取和未报告的失败用量。`,
      `An exposed set of ${regression.questions} unique questions, ${regression.repeats_per_arm} repeats per arm; models unchanged, ${regression.reader_rows} answers and ${regression.grade_rows} grades audited. Tokens above cover recorded answer + grading usage; they exclude prior extraction and unreported failed-call usage.`,
    );
    $("releaseMeta").textContent = t("LongMemEval-S · 已暴露题集回归", "LongMemEval-S · exposed-set regression");
    link.href = `https://github.com/Mingjie-Mao/llm-long-term-memory/blob/main/${regression.gate_path}`;
    link.textContent = t("完整结果与审计口径", "full results and audit scope");
  } else {
    $("releaseMeta").textContent = t(`${release} · 历史终测`, `${release} · archived final test`);
    link.href = "https://github.com/Mingjie-Mao/llm-long-term-memory/blob/main/results/final/test100-aggregate.md";
    link.textContent = t("实验报告", "full report");
  }
  $("historicalCaption").textContent = t(
    `历史冻结单次终测 · ${result.dataset} · ${result.questions} 题`,
    `Historical frozen one-shot test · ${result.dataset} · ${result.questions} questions`,
  );
  metricRows("historicalRows", [
    ["Naive RAG", percent(result.naive_rag_accuracy)],
    [t("LLTM v2 · 记忆与条件回退", "LLTM v2 · memory + conditional recovery"), percent(result.accuracy)],
    [t("Full Context · 整段历史", "Full Context · full history"), percent(result.full_context_accuracy)],
  ]);
  const significance = result.significant_vs_naive_rag
    ? t("显著", "significant") : t("不显著", "not significant");
  $("historicalResult").textContent = t(
    `每臂只运行一次；v2 相对 Naive RAG 的差异${significance}（p=${result.vs_naive_rag_p_value}）。历史基准与上面的回归分别报告，不能排成版本进步曲线。`,
    `One run per arm; v2’s difference from Naive RAG was ${significance} (p=${result.vs_naive_rag_p_value}). Historical benchmarks and the regression above are separate experiments, not a version-progress curve.`,
  );
  $("engineeringTests").textContent = number(engineering.tests);
  $("engineeringCoverage").textContent = `${engineering.line_coverage}%`;
  const platformNames = { ubuntu: "Linux", windows: "Windows", macos: "macOS" };
  $("engineeringPlatforms").textContent = engineering.ci_platforms.map((p) => platformNames[p] || p).join(" / ");
  $("engineeringNote").textContent = t(
    `工程数据核验于 ${engineering.verified_at}；另有 ${number(engineering.document_checks)} 项文档检查，不计入工程测试数。指标对应发布清单记录的源码树，不代表生产可用性保证。`,
    `Engineering figures verified ${engineering.verified_at}; ${number(engineering.document_checks)} document checks are counted separately. Figures refer to the source tree recorded in the manifest, not a production availability guarantee.`,
  );
}

async function loadManifest() {
  try {
    const response = await fetch("./release.json", { cache: "no-store" });
    manifest = await response.json();
    renderMetrics();
  } catch {
    $("releaseMeta").textContent = "Release manifest unavailable.";
  }
}


async function api(method, path, body) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    let response;
    try {
      response = await fetch(API + path, {
        method, signal: controller.signal,
        headers: { "content-type": "application/json", ...(token ? { "x-demo-token": token } : {}) },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
    } catch (error) {
      throw failure(error.name === "AbortError" ? "timeout" : "network");
    }
    // A waking or failing Render instance answers with an HTML page, not JSON.
    let data = null;
    if ((response.headers.get("content-type") || "").includes("application/json")) {
      try { data = await response.json(); } catch {}
    }
    if (data === null) throw failure("badResponse", response.status);
    if (!response.ok) {
      const detail = data.detail;
      if (detail?.error === "llm_unavailable") {
        const error = failure(detail.provider_code === 429 ? "rateLimited" : "modelFailed", response.status);
        error.diagnostic = [detail.kind, detail.provider_code].filter(Boolean).join(" ");
        throw error;
      }
      // demo_capacity also covers transient rate limits and deadlines; only exhaustion is final.
      if (detail?.error === "demo_capacity") {
        const kind = /exhausted/.test(detail.message) ? "capacity" : /rate limit/.test(detail.message) ? "rateLimited" : /timed out/.test(detail.message) ? "operationTimeout" : null;
        if (kind) {
          const error = failure(kind, response.status);
          const after = Number(response.headers.get("retry-after"));
          error.retryAfter = Number.isFinite(after) && after > 0 ? Math.min(after, 300) : 60;
          throw error;
        }
      }
      if (response.status === 401 || response.status === 410) throw failure("expired", response.status);
      const error = new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join("; ") : detail?.message || detail || response.statusText);
      error.status = response.status;
      throw error;
    }
    return data;
  } finally { clearTimeout(timer); }
}

function controls() {
  const cooling = cooldownUntil > Date.now();
  for (const element of document.querySelectorAll("[data-requires-live]")) element.disabled = !ready || busy || cooling || !!scenarioRun;
  $("deleteSession").disabled = !token || busy;
  $("retry").disabled = busy || cooling;
  $("scenarioActions").hidden = !scenarioRun;
  $("resumeScenario").disabled = !ready || busy || cooling;
  $("leaveScenario").disabled = busy;
}
function renderCooldown() {
  const seconds = Math.max(0, Math.ceil((cooldownUntil - Date.now()) / 1000));
  $("cooldown").textContent = seconds ? t(
    `模型调用已限流，约 ${seconds}s 后可重试。已完成的预设步骤会保留。`,
    `Model calls are rate limited. Retry in about ${seconds}s; completed preset steps are kept.`,
  ) : "";
  if (!seconds) { clearInterval(cooldownTimer); cooldownTimer = null; }
  controls();
}
function startCooldown(seconds) {
  cooldownUntil = Date.now() + seconds * 1000;
  clearInterval(cooldownTimer);
  cooldownTimer = setInterval(renderCooldown, 1000);
  renderCooldown();
}
function renderRecording() {
  if (!recording) return;
  $("recordingTranscript").replaceChildren();
  const meta = document.createElement("p");
  meta.textContent = t(`录制于 ${recording.recorded_at_utc}；场景：${recording.scenario}；语言：${recording.language}。`, `Recorded ${recording.recorded_at_utc}; scenario: ${recording.scenario}; language: ${recording.language}.`);
  $("recordingTranscript").append(meta);
  for (const item of recording.transcript) {
    const p = document.createElement("p");
    p.textContent = `${item.role}: ${item.text}`;
    $("recordingTranscript").append(p);
  }
}
async function loadRecording() {
  try {
    const response = await fetch("./recording.json");
    if (!response.ok) throw new Error("Recording metadata unavailable");
    recording = await response.json();
    renderRecording();
  } catch {
    $("recordingTranscript").textContent = t("录制信息暂不可用，请稍后重试。", "Recording details unavailable; retry later.");
  }
}

function title() { $("conversationTitle").textContent = t(`对话 ${conversation}`, `Conversation ${conversation}`); }
// state: "waking" while connecting, "up" when live, "down" otherwise. Retry is offered
// only when there is something to retry.
function status(state, pair) {
  statusRecord = { state, pair };
  $("engineStatus").dataset.state = state;
  $("engineStatus").setAttribute("aria-busy", String(state === "waking"));
  $("statusText").textContent = pair[lang === "zh" ? 0 : 1];
  $("retry").hidden = state !== "down";
}
function say(role, text) {
  const li = document.createElement("li");
  li.className = `bubble ${role}`;
  // A [zh, en] pair follows language switches; visitor and model text stays verbatim.
  if (Array.isArray(text)) [li.dataset.zh, li.dataset.en] = text;
  li.textContent = Array.isArray(text) ? t(...text) : text;
  $("chat").append(li);
}
function emptyState(id, zh, en, tag = "p") {
  const element = document.createElement(tag);
  element.className = "empty";
  element.dataset.zh = zh;
  element.dataset.en = en;
  element.textContent = t(zh, en);
  $(id).append(element);
}
function clearInspector() {
  for (const id of ["chat", "memoryState", "sources", "modelInputs", "asOfState", "recovery", "notSelected"]) $(id).replaceChildren();
  $("usage").textContent = "";
  $("progress").textContent = "";
  for (const id of ["historyDetails", "selectionDetails", "promptDetails"]) $(id).open = false;
  delete $("progress").dataset.zh;
  delete $("progress").dataset.en;
  emptyState("memoryState", "保存一段话后查看抽取事实。", "Save a message to inspect extracted facts.", "li");
  emptyState("asOfState", "选一个日期，查看那天结束时成立的事实。只读取已存的时间区间，不调用模型。", "Pick a date to see the facts in force at the end of that day. Reads stored intervals only; no model call.", "li");
  emptyState("sources", "提问后查看选中的来源。", "Ask a question to inspect selected sources.");
  emptyState("notSelected", "提问后展示。", "Shown after a question.", "li");
}
async function freshSession() {
  scenarioRun = null;
  // Delete before replacing the credential; a failed deletion keeps the old token.
  // An already expired namespace has nothing left to delete.
  if (token) {
    try { await api("DELETE", "/demo/session"); }
    catch (error) { if (error.status !== 401 && error.status !== 410) throw error; }
  }
  token = null;
  save();
  const issued = await api("POST", "/demo/session");
  token = issued.token;
  conversation = 1;
  pendingWrite = null;
  save();
  clearInspector();
  title();
}
async function connect() {
  busy = true; ready = false; controls();
  const started = Date.now();
  const tick = () => status("waking", waking(Math.round((Date.now() - started) / 1000)));
  tick();
  clearInterval(wakingTimer);
  wakingTimer = setInterval(tick, 1000);
  try {
    const health = await api("GET", "/demo/health");
    if (!health.llm_enabled || !(health.capabilities || []).includes("llm_answer")) throw failure("disabled");
    let restored = null;
    if (token) {
      try { restored = await api("GET", "/demo/memories"); }
      catch (error) {
        if (error.status === 401 || error.status === 410) { token = null; save(); }
        else throw error;
      }
    }
    if (!token) await freshSession();
    else if (restored) {
      clearInspector();
      title();
      if (restored.total) {
        renderMemories([...restored.active, ...restored.superseded]);
        say("assistant", [
          `已恢复本标签页的演示空间：${restored.total} 条记忆。`,
          `Restored this tab's demo namespace: ${restored.total} ${restored.total === 1 ? "memory" : "memories"}.`,
        ]);
      }
    }
    ready = true;
    clearInterval(wakingTimer);
    status("up", COPY.ready);
  } catch (error) {
    clearInterval(wakingTimer);
    status("down", pairOf(error));
  } finally { busy = false; controls(); }
}
// Server timestamps carry microseconds; seconds are kept because demo writes land
// seconds apart and the order is the point. The raw value stays in the tooltip.
const when = (value) => value ? String(value).replace("T", " ").replace(/\.\d+$/, "") : "?";
const day = (value) => value ? String(value).slice(0, 10) : "?";
// The visitor's calendar date, which is what a date input shows and compares against.
const today = () => { const d = new Date(); return new Date(d - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10); };
// `tag` is plain text, or a [zh, en] pair that follows language switches.
function factRow(memory, tag) {
  const li = document.createElement("li");
  li.className = `row ${memory.status}`;
  const key = document.createElement("code");
  key.textContent = `${memory.predicate || "fact"} = ${memory.object || memory.content}`;
  const label = document.createElement("span");
  label.className = "tag";
  if (Array.isArray(tag)) [label.dataset.zh, label.dataset.en] = tag;
  label.textContent = Array.isArray(tag) ? t(...tag) : tag;
  li.append(key, label);
  return li;
}
function renderMemories(memories) {
  $("memoryState").replaceChildren();
  if (!memories.length) emptyState("memoryState", "本次未抽取到事实。", "No facts extracted yet.", "li");
  const start = (m) => m.occurred_at || m.valid_from || "";
  for (const memory of [...memories].sort((a, b) => String(start(a)).localeCompare(String(start(b))))) {
    const li = factRow(memory, `${memory.status}${memory.scope ? " · " + memory.scope : ""}`);
    const fact = document.createElement("small");
    fact.className = "origin-note";
    fact.textContent = memory.content;
    // The interval the timeline uses runs from the fact's own time; the conversation it
    // was said in is shown separately when the two differ.
    const time = document.createElement("small");
    time.className = "origin-note";
    const end = memory.valid_to ? when(memory.valid_to) : (memory.status === "active" ? t("至今", "present") : "—");
    const said = memory.observed_at && day(memory.observed_at) !== day(start(memory))
      ? t(` · 说于 ${day(memory.observed_at)}`, ` · said ${day(memory.observed_at)}`) : "";
    time.textContent = `${memory.session_id || ""} · ${when(start(memory))} → ${end}${said}`;
    time.title = `occurred_at=${memory.occurred_at || ""} observed_at=${memory.observed_at || ""} valid_from=${memory.valid_from || ""} valid_to=${memory.valid_to || ""}`;
    li.append(fact, time);
    $("memoryState").append(li);
  }
}
function renderAsOf(results) {
  $("asOfState").replaceChildren();
  for (const { at, memories } of results) {
    const head = document.createElement("li");
    head.className = "as-of-day";
    head.textContent = at;
    $("asOfState").append(head);
    if (!memories.length) emptyState("asOfState", "那天还没有已知事实。", "Nothing known yet on that day.", "li");
    for (const memory of memories) $("asOfState").append(factRow(memory, [`自 ${day(memory.occurred_at || memory.valid_from)}`, `since ${day(memory.occurred_at || memory.valid_from)}`]));
  }
}
async function showAsOf(dates) {
  const results = [];
  for (const at of dates) results.push(await api("GET", `/demo/as_of?at=${encodeURIComponent(at)}`));
  renderAsOf(results);
}
function renderRecovery(data) {
  $("recovery").replaceChildren();
  const summary = document.createElement("p");
  summary.className = "recovery";
  const selected = data.memories_selected ?? (data.selected || []).length;
  const level = data.fallback_level || "none";
  const recovered = data.recovered_turns || [];
  const primary = data.raw_primary_turns || [];
  let zh = `结构化记忆：选中 ${selected} 条（候选 ${data.candidates_considered ?? "?"} 条）。`;
  let en = `Structured memory: ${selected} selected of ${data.candidates_considered ?? "?"} candidates. `;
  if (primary.length) {
    zh += `首轮请求同时附带了 ${primary.length} 段相关原话。`;
    en += `The first request also carried ${primary.length} relevant archived ${primary.length === 1 ? "turn" : "turns"}. `;
  }
  if (level === "none") {
    zh += "原文回退：未触发，第一次请求的证据已足够回答。";
    en += "Raw fallback: not triggered; the first request had enough evidence.";
  } else if (recovered.length) {
    summary.classList.add("used");
    zh += `原文回退：已触发（${level}），从对话存档恢复了 ${recovered.length} 段原话，再次请求模型。`;
    en += `Raw fallback: triggered (${level}); ${recovered.length} archived ${recovered.length === 1 ? "turn was" : "turns were"} recovered and the model was asked again.`;
  } else {
    summary.classList.add("used");
    zh += `原文回退：已触发（${level}），但没有找到新的原话。`;
    en += `Raw fallback: triggered (${level}), but no new archived turn was found.`;
  }
  [summary.dataset.zh, summary.dataset.en] = [zh, en];
  summary.textContent = t(zh, en);
  $("recovery").append(summary);
  const card = (turn, zh, en) => {
    const p = document.createElement("p");
    p.className = "evidence-card";
    const body = `${turn.session_id} · turn ${turn.turn_index}: ${turn.text}`;
    [p.dataset.zh, p.dataset.en] = [`${zh} · ${body}`, `${en} · ${body}`];
    p.textContent = t(p.dataset.zh, p.dataset.en);
    $("recovery").append(p);
  };
  for (const turn of primary) card(turn, "首轮附带的原话", "Archived turn in the first request");
  for (const turn of recovered) card(turn, "回退恢复的原话", "Recovered by the fallback");
}
function renderNotSelected(rows) {
  $("notSelected").replaceChildren();
  if (!rows.length) emptyState("notSelected", "所有记忆都进入了本次回答。", "Every memory reached this answer.", "li");
  const reasons = {
    not_current: ["已被替代或结束，不作为当前事实", "replaced or ended; not current"],
    not_ranked: ["当前有效，但没有排进本次上下文", "current, but not ranked into this context"],
  };
  for (const row of rows) $("notSelected").append(factRow(row, reasons[row.reason] || row.reason));
}
function renderAnswer(data) {
  renderRecovery(data);
  renderNotSelected(data.not_selected || []);
  $("sources").replaceChildren();
  if (!(data.selected_sources || []).length) emptyState("sources", "没有选中结构化记忆；原文恢复见下方实际请求。", "No structured memory selected; see the request below for raw recovery.");
  for (const item of data.selected_sources || []) {
    const p = document.createElement("p");
    p.className = "evidence-card";
    const source = item.source;
    p.textContent = source
      ? `${source.session_id} · turn ${source.turn_index}: ${source.text}`
      : `${item.memory.content} (${t("未定位到原句", "source span unavailable")})`;
    $("sources").append(p);
  }
  $("modelInputs").replaceChildren();
  for (const [i, input] of (data.model_inputs || []).entries()) {
    const heading = document.createElement("h4");
    heading.textContent = `${t("模型请求", "Model request")} ${i + 1} · ${input.model}`;
    const pre = document.createElement("pre");
    pre.textContent = input.prompt;
    $("modelInputs").append(heading, pre);
  }
}
function renderUsage(data) {
  const used = data.usage || {};
  const seconds = data.latency_ms ? ` · ${(data.latency_ms / 1000).toFixed(1)}s` : "";
  $("usage").textContent = t(
    `本步模型调用 ${used.total_requests || 0} 次，已记录 ${used.total_tokens || 0} tokens${seconds}。本空间返回时剩余预留调用额度：${Math.max(0, (data.capacity_limits?.calls || 0) - (data.capacity_reserved?.calls || 0))}；全站额度另行共享，重试与失败也占额度。`,
    `This step: ${used.total_requests || 0} model calls, ${used.total_tokens || 0} recorded tokens${seconds}. Namespace attempt capacity remaining at this response: ${Math.max(0, (data.capacity_limits?.calls || 0) - (data.capacity_reserved?.calls || 0))}. Global capacity is shared separately; retries and failures consume it too.`,
  );
}
// A provider 503 is the model being overloaded, seen several times in a day even after
// the backend's own two attempts. One more try after a pause costs a call only when it
// is needed; writes are idempotent by request_id, and questions write nothing.
async function retryOverloaded(run) {
  try { return await run(); }
  catch (error) {
    if (error.key !== "modelFailed" || !/\b503\b/.test(error.diagnostic || "")) throw error;
    say("assistant", ["模型服务繁忙（503），5 秒后自动重试一次…", "The model service is busy (503); retrying once in 5 s…"]);
    await new Promise((resolve) => setTimeout(resolve, 5000));
    return await run();
  }
}
// Today is sent as no date, so the server's own clock decides it; a visitor ahead of
// the server's timezone would otherwise be told their today is in the future.
function conversationDate() {
  const value = $("conversationDate").value;
  return value && value !== today() ? value : null;
}
async function remember(content) {
  const observed = conversationDate();
  if (!pendingWrite || pendingWrite.content !== content || pendingWrite.session_id !== conversationId()
      || (pendingWrite.observed_on || null) !== observed) {
    pendingWrite = { content, session_id: conversationId(), request_id: crypto.randomUUID(), ...(observed ? { observed_on: observed } : {}) };
  }
  const result = await retryOverloaded(() => api("POST", "/demo/messages", pendingWrite));
  pendingWrite = null;
  if (result.mode !== "llm") throw new Error("Unexpected non-LLM extraction response.");
  renderMemories(result.memories);
  renderUsage(result);
  const n = result.extracted.length;
  say("assistant", [`抽取完成：${n} 条事实。`, `Extraction complete: ${n} ${n === 1 ? "fact" : "facts"}.`]);
}
async function ask(query) {
  const result = await retryOverloaded(() => api("POST", "/demo/answer", { query }));
  if (result.mode !== "llm" || result.answer_profile !== "t4") throw new Error("Unexpected non-LLM answer response.");
  say("answer", `${t("LLM 回答", "LLM answer")}: ${result.answer}`);
  renderAnswer(result);
  renderUsage(result);
}
async function action(work) {
  if (!ready || busy || cooldownUntil > Date.now()) return;
  busy = true; controls();
  try { await work(); }
  catch (error) {
    say("error", pairOf(error));
    if (error.key === "rateLimited") startCooldown(error.retryAfter || 60);
    // A failed model call leaves the session usable: the backend already retried it, and
    // a resend reuses the write's request_id. Only a lost namespace or spent capacity
    // ends the session.
    if (error.key === "expired") { token = null; save(); }
    if (error.key === "expired" || error.key === "capacity" || !token) { scenarioRun = null; ready = false; status("down", pairOf(error)); }
  } finally { busy = false; controls(); }
}
// Progress the page can actually know: which request is running and for how long. The
// backend does one synchronous request per step, so finer stages would be invented.
let progressTimer = null;
function progress(step, total, label) {
  clearInterval(progressTimer);
  delete $("progress").dataset.zh;
  delete $("progress").dataset.en;
  const started = Date.now();
  const render = () => {
    const seconds = Math.round((Date.now() - started) / 1000);
    $("progress").textContent = `${t("步骤", "Step")} ${step}/${total} · ${t(...label)} · ${seconds}s`;
  };
  render();
  progressTimer = setInterval(render, 1000);
}
function progressDone(started) {
  clearInterval(progressTimer);
  const seconds = Math.round((Date.now() - started) / 1000);
  const element = $("progress");
  [element.dataset.zh, element.dataset.en] = [`完成 · 共 ${seconds}s`, `Done · ${seconds}s total`];
  element.textContent = t(element.dataset.zh, element.dataset.en);
}
async function runLiveScenario(name) {
  if (scenarioRun) return;
  await action(async () => {
    await freshSession();
    scenarioRun = { name, language: lang, nextTurn: 0, answerDone: false, announced: null, started: Date.now() };
    await advanceScenario();
  });
}
// Only advance after a successful response. A paused write retains pendingWrite's
// identity; a completed answer is never called again just to retry as_of reads.
async function advanceScenario() {
  const run = scenarioRun;
  const scenario = SCENARIOS[run.name];
  const total = scenario.turns.length + 1;
  try {
    while (run.nextTurn < scenario.turns.length) {
      const turn = scenario.turns[run.nextTurn];
      const text = turn.text[run.language];
      const on = turn.on;
      $("conversationDate").value = on || "";
      progress(run.nextTurn + 1, total, [`写入对话 ${conversation}${on ? `（${on}）` : ""}：抽取与时间线`, `Writing conversation ${conversation}${on ? ` (${on})` : ""}: extraction and timeline`]);
      if (run.announced !== run.nextTurn) {
        say("user", on ? `[${on}] ${text}` : text);
        run.announced = run.nextTurn;
      }
      await remember(text);
      run.nextTurn += 1;
      conversation += 1;
      save(); title();
    }
    $("conversationDate").value = "";
    if (!run.answerDone) {
      progress(total, total, ["检索与回答", "Retrieval and answer"]);
      const question = scenario.question[run.language];
      if (run.announced !== "answer") { say("user", question); run.announced = "answer"; }
      await ask(question);
      run.answerDone = true;
    }
    if (scenario.asOf) {
      await showAsOf(scenario.asOf);
      $("historyDetails").open = true;
    }
    progressDone(run.started);
    scenarioRun = null;
  } catch (error) {
    clearInterval(progressTimer);
    const element = $("progress");
    [element.dataset.zh, element.dataset.en] = ["预设已暂停；可继续未完成步骤。离开或刷新页面后无法恢复进度。", "Preset paused; resume the unfinished step. Progress cannot resume after leaving or refreshing the page."];
    element.textContent = t(element.dataset.zh, element.dataset.en);
    throw error;
  }
}
$("resumeScenario").addEventListener("click", () => { if (scenarioRun) action(advanceScenario); });
$("leaveScenario").addEventListener("click", () => {
  if (busy) return;
  scenarioRun = null;
  $("progress").textContent = "";
  delete $("progress").dataset.zh;
  delete $("progress").dataset.en;
  controls();
});

$("messageForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("message").value.trim();
  if (!text || scenarioRun) return;
  action(async () => {
    say("user", text);
    if ($("operation").value === "remember") await remember(text);
    else await ask(text);
    $("message").value = "";
  });
});
$("newConversation").addEventListener("click", () => {
  conversation += 1; pendingWrite = null; save(); title(); $("chat").replaceChildren();
  $("conversationDate").value = "";
});
$("deleteSession").addEventListener("click", async () => {
  if (!token || busy) return;
  busy = true; controls();
  try {
    await api("DELETE", "/demo/session");
    token = null; ready = false; pendingWrite = null; scenarioRun = null; save(); clearInspector();
    status("down", COPY.deleted);
  } catch (error) { status("down", pairOf(error)); }
  finally { busy = false; controls(); }
});
for (const button of document.querySelectorAll("[data-live]")) button.addEventListener("click", () => runLiveScenario(button.dataset.live));
$("asOfForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const at = $("asOfDate").value;
  if (at) action(() => showAsOf([at]));
});
$("retry").addEventListener("click", connect);
$("bzh").addEventListener("click", () => { lang = "zh"; applyLanguage(); });
$("ben").addEventListener("click", () => { lang = "en"; applyLanguage(); });
for (const id of ["conversationDate", "asOfDate"]) $(id).max = today();
const saved = loadSaved();
if (saved) ({ token, conversation } = saved);
applyLanguage();
loadManifest();
loadRecording();
connect();
