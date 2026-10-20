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
const pairOf = (error) => error.key ? COPY[error.key] : [error.message, error.message];

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
  title();
  renderMetrics();
  if (statusRecord) status(statusRecord.state, statusRecord.pair);
}
function renderMetrics() {
  if (!manifest) return;
  const { result, release, regression } = manifest;
  const acc = result.accuracy;
  const rag = result.naive_rag_accuracy;
  const full = result.full_context_accuracy;
  const significance = result.significant_vs_naive_rag
    ? { zh: "显著", en: "significant" }
    : { zh: "不显著", en: "not significant" };
  const link = $("releaseCommit");
  const history = $("historicalResult");
  history.hidden = !regression;
  if (regression) {
    const percent = (n) => Number(n.toFixed(2));
    const tokens = (n) => n.toLocaleString("en-US");
    $("headline").textContent = lang === "zh"
      ? `原 test100 回归：${percent(regression.candidate_accuracy)}% 正确率；同期旧版 ${percent(regression.control_accuracy)}%，提升 ${percent(regression.mean_difference_pp)} 个百分点。`
      : `Original test100 regression: ${percent(regression.candidate_accuracy)}% accuracy; ${percent(regression.control_accuracy)}% for the concurrent baseline, a ${percent(regression.mean_difference_pp)}-point gain.`;
    $("footnote").textContent = lang === "zh"
      ? `已暴露题集，${regression.questions} 道独立题，每组重复 ${regression.repeats_per_arm} 次；模型未换，${regression.reader_rows} 份回答与 ${regression.grade_rows} 份评分通过完整性审计。回答＋评分已记录 token：新版 ${tokens(regression.candidate_recorded_tokens)}，旧版 ${tokens(regression.control_recorded_tokens)}；不含原有记忆抽取和未报告的失败用量。`
      : `An exposed set of ${regression.questions} unique questions, ${regression.repeats_per_arm} repeats per arm; models unchanged, ${regression.reader_rows} answers and ${regression.grade_rows} grades audited. Recorded answer + grading tokens: ${tokens(regression.candidate_recorded_tokens)} candidate, ${tokens(regression.control_recorded_tokens)} baseline; excludes prior extraction and unreported failed-call usage.`;
    history.textContent = lang === "zh"
      ? `历史冻结单次终测：v2 ${acc}%，naive RAG ${rag}%，整段历史 ${full}%；v2 相对 RAG 的差异${significance.zh}（p=${result.vs_naive_rag_p_value}）。历史终测与这次回归分别报告。`
      : `Historical frozen one-shot test: v2 ${acc}%, naive RAG ${rag}%, full history ${full}%; v2's difference from RAG was ${significance.en} (p=${result.vs_naive_rag_p_value}). Historical and regression results are reported separately.`;
    $("releaseMeta").textContent = lang === "zh"
      ? "LongMemEval-S · 已暴露题集回归"
      : "LongMemEval-S · exposed-set regression";
    link.href = `https://github.com/Mingjie-Mao/llm-long-term-memory/blob/main/${regression.gate_path}`;
    link.textContent = lang === "zh" ? "完整结果与审计口径" : "full results and audit scope";
    return;
  }

  $("headline").textContent = lang === "zh"
    ? `冻结 v2：${acc}% 正确率；naive RAG ${rag}%，整段历史 ${full}%。`
    : `Frozen v2: ${acc}% accuracy; naive RAG ${rag}%, full history ${full}%.`;

  $("footnote").textContent = result.significant_vs_naive_rag
    ? (lang === "zh"
        ? `LongMemEval-S ${result.questions} 题，每臂一次；相对 naive RAG 的差异显著（p=${result.vs_naive_rag_p_value}）。`
        : `LongMemEval-S, ${result.questions} questions, one run per arm; the difference from naive RAG was significant (p=${result.vs_naive_rag_p_value}).`)
    : (lang === "zh"
        ? `LongMemEval-S ${result.questions} 题，每臂一次；相对 naive RAG 的差异不显著（p=${result.vs_naive_rag_p_value}）。`
        : `LongMemEval-S, ${result.questions} questions, one run per arm; the difference from naive RAG was not significant (p=${result.vs_naive_rag_p_value}).`);

  $("releaseMeta").textContent = lang === "zh"
    ? `${release} · 历史终测`
    : `${release} · archived final test`;
  link.href = "https://github.com/Mingjie-Mao/llm-long-term-memory/blob/main/results/final/test100-aggregate.md";
  link.textContent = lang === "zh" ? "实验报告" : "full report";
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
      if (detail?.error === "llm_unavailable") throw failure("modelFailed", response.status);
      // demo_capacity also covers transient rate limits and deadlines; only exhaustion is final.
      if (detail?.error === "demo_capacity" && /exhausted/.test(detail.message)) throw failure("capacity", response.status);
      if (response.status === 401 || response.status === 410) throw failure("expired", response.status);
      const error = new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join("; ") : detail?.message || detail || response.statusText);
      error.status = response.status;
      throw error;
    }
    return data;
  } finally { clearTimeout(timer); }
}

function controls() {
  for (const element of document.querySelectorAll("[data-requires-live]")) element.disabled = !ready || busy;
  $("deleteSession").disabled = !token || busy;
  $("retry").disabled = busy;
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
  for (const id of ["chat", "memoryState", "sources", "modelInputs"]) $(id).replaceChildren();
  $("usage").textContent = "";
  emptyState("memoryState", "保存一段话后查看抽取事实。", "Save a message to inspect extracted facts.", "li");
  emptyState("sources", "提问后查看选中的来源。", "Ask a question to inspect selected sources.");
}
async function freshSession() {
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
function renderMemories(memories) {
  $("memoryState").replaceChildren();
  if (!memories.length) emptyState("memoryState", "本次未抽取到事实。", "No facts extracted yet.", "li");
  for (const memory of [...memories].sort((a, b) => String(a.valid_from || "").localeCompare(String(b.valid_from || "")))) {
    const li = document.createElement("li");
    li.className = `row ${memory.status}`;
    const key = document.createElement("code");
    key.textContent = `${memory.predicate || "fact"} = ${memory.object || memory.content}`;
    const label = document.createElement("span");
    label.className = "tag";
    label.textContent = `${memory.status}${memory.scope ? " · " + memory.scope : ""}`;
    const time = document.createElement("small");
    time.className = "origin-note";
    time.textContent = `${memory.session_id || ""} · ${when(memory.valid_from)} → ${memory.valid_to ? when(memory.valid_to) : (memory.status === "active" ? t("至今", "present") : "—")}`;
    time.title = `valid_from=${memory.valid_from || ""} valid_to=${memory.valid_to || ""}`;
    const fact = document.createElement("small");
    fact.className = "origin-note";
    fact.textContent = memory.content;
    li.append(key, label, fact, time);
    $("memoryState").append(li);
  }
}
function renderAnswer(data) {
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
  $("usage").textContent = t(
    `本步模型调用 ${used.total_requests || 0} 次，已记录 ${used.total_tokens || 0} tokens。额度按保守预留计量，重试与失败也占额度。`,
    `This step: ${used.total_requests || 0} model calls, ${used.total_tokens || 0} recorded tokens. Capacity is conservatively reserved; retries and failures consume it too.`,
  );
}
async function remember(content) {
  if (!pendingWrite || pendingWrite.content !== content || pendingWrite.session_id !== conversationId()) {
    pendingWrite = { content, session_id: conversationId(), request_id: crypto.randomUUID() };
  }
  const result = await api("POST", "/demo/messages", pendingWrite);
  pendingWrite = null;
  if (result.mode !== "llm") throw new Error("Unexpected non-LLM extraction response.");
  renderMemories(result.memories);
  renderUsage(result);
  const n = result.extracted.length;
  say("assistant", [`抽取完成：${n} 条事实。`, `Extraction complete: ${n} ${n === 1 ? "fact" : "facts"}.`]);
}
async function ask(query) {
  const result = await api("POST", "/demo/answer", { query });
  if (result.mode !== "llm" || result.answer_profile !== "t4") throw new Error("Unexpected non-LLM answer response.");
  say("answer", `${t("LLM 回答", "LLM answer")}: ${result.answer}`);
  renderAnswer(result);
  renderUsage(result);
}
async function action(work) {
  if (!ready || busy) return;
  busy = true; controls();
  try { await work(); }
  catch (error) {
    say("error", pairOf(error));
    // A failed model call leaves the session usable: the backend already retried it, and
    // a resend reuses the write's request_id. Only a lost namespace or spent capacity
    // ends the session.
    if (error.key === "expired") { token = null; save(); }
    if (error.key === "expired" || error.key === "capacity") { ready = false; status("down", pairOf(error)); }
  } finally { busy = false; controls(); }
}
async function runLiveScenario(name) {
  await action(async () => {
    await freshSession();
    for (const text of SCENARIOS[name].turns) {
      say("user", text);
      await remember(text);
      conversation += 1;
      save();
      title();
    }
    const query = SCENARIOS[name].question;
    say("user", query);
    await ask(query);
  });
}
$("messageForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("message").value.trim();
  if (!text) return;
  action(async () => {
    say("user", text);
    if ($("operation").value === "remember") await remember(text);
    else await ask(text);
    $("message").value = "";
  });
});
$("newConversation").addEventListener("click", () => {
  conversation += 1; pendingWrite = null; save(); title(); $("chat").replaceChildren();
});
$("deleteSession").addEventListener("click", async () => {
  if (!token || busy) return;
  busy = true; controls();
  try {
    await api("DELETE", "/demo/session");
    token = null; ready = false; pendingWrite = null; save(); clearInspector();
    status("down", COPY.deleted);
  } catch (error) { status("down", pairOf(error)); }
  finally { busy = false; controls(); }
});
for (const button of document.querySelectorAll("[data-live]")) button.addEventListener("click", () => runLiveScenario(button.dataset.live));
$("retry").addEventListener("click", connect);
$("bzh").addEventListener("click", () => { lang = "zh"; applyLanguage(); });
$("ben").addEventListener("click", () => { lang = "en"; applyLanguage(); });
const saved = loadSaved();
if (saved) ({ token, conversation } = saved);
applyLanguage();
loadManifest();
connect();
