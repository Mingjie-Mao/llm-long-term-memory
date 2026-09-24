import { WALKTHROUGH, PATTERNS, SUGGESTIONS, COPY } from "./content.js";

const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const API = LOCAL ? "http://127.0.0.1:8100" : "https://lltm-playground.onrender.com";

let lang = "en";
let ready = false;
// The demo session is carried by a header, not a cookie: the backend reads
// `x-demo-token` and namespaces every write under it. Nothing is stored across reloads,
// so closing the tab is enough to abandon a session.
//
// Two sessions, because the page has two independent demos. Sharing one namespace meant
// the walkthrough's facts appeared in the visitor's own state panel, and re-running the
// walkthrough wiped whatever the visitor had just typed.
const session = { walk: null, try: null };

// One visitor, several conversations. The box's namespace is the user, and a conversation
// is an engine session inside it: every fact is written with the conversation it was
// stated in, and search spans all of them. An answer that names an earlier conversation
// therefore came from memory — that conversation's transcript is gone from the page, and
// a question sends nothing but its own text.
let conversation = 1;
let spokeInConversation = false;
const conversationId = (n) => `conversation-${n}`;
const conversationNumber = (id) => Number(/^conversation-(\d+)$/.exec(id ?? "")?.[1]) || null;

const $ = (id) => document.getElementById(id);
const t = (pair) => pair[lang] ?? pair.en;

// ----------------------------------------------------------------- language

function applyLanguage() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  for (const el of document.querySelectorAll("[data-zh][data-en]")) {
    el.textContent = el.dataset[lang];
  }
  for (const el of document.querySelectorAll("[data-placeholder-zh][data-placeholder-en]")) {
    el.placeholder = el.dataset[`placeholder${lang === "zh" ? "Zh" : "En"}`];
  }
  for (const el of document.querySelectorAll("[data-aria-zh][data-aria-en]")) {
    el.setAttribute("aria-label", el.dataset[`aria${lang === "zh" ? "Zh" : "En"}`]);
  }
  $("bzh").setAttribute("aria-pressed", String(lang === "zh"));
  $("ben").setAttribute("aria-pressed", String(lang === "en"));
  // Manifest copy is built from data, so it has to be rebuilt rather than swapped.
  renderMetrics();
  renderSuggestions();
  renderConversationTitle();
  // So are the state rows, which carry translated provenance.
  for (const [target, memories] of Object.entries(lastState)) {
    if (memories.length) renderState(target, memories);
  }
  const status = statusText();
  if (status) $("statusText").textContent = status;
}

// ----------------------------------------------------------------- manifest

let manifest = null;

function renderMetrics() {
  if (!manifest) return;
  const { result, release } = manifest;
  const acc = result.accuracy;
  const rag = result.naive_rag_accuracy;
  const full = result.full_context_accuracy;

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
  const link = $("releaseCommit");
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

// ----------------------------------------------------------------- transport

async function api(method, path, body, { cold = false, as = null } = {}) {
  // No `credentials`: the session travels in a header, and asking for credentials mode
  // makes a wildcard CORS origin fail the preflight for no benefit.
  const options = { method, headers: {} };
  const held = as ? session[as] : null;
  if (held) options.headers["x-demo-token"] = held;
  if (body !== undefined && body !== null) {
    options.headers["content-type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  // The free tier sleeps. A cold call can take most of a minute, and a short timeout
  // would report the engine as broken when it is only waking up.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), cold ? 90000 : 25000);
  options.signal = controller.signal;
  try {
    const response = await fetch(API + path, options);
    const text = await response.text();
    const data = text ? JSON.parse(text) : null;
    if (!response.ok) {
      // FastAPI reports validation failures as a list of per-field errors.
      const detail = data?.detail;
      throw new Error(
        Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || response.statusText,
      );
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

// Remembered so a language switch can re-render it. The status is written once when the
// engine answers, and without this it stayed in whichever language was active then.
let statusKey = null;

function statusText() {
  if (!statusKey) return null;
  const { kind, value } = statusKey;
  if (kind === "waking") return lang === "zh" ? "正在唤醒引擎…" : "Waking the live engine…";
  if (kind === "up") {
    return lang === "zh" ? `引擎就绪 · 编码器 ${value}` : `Engine ready · encoder ${value}`;
  }
  return lang === "zh" ? "引擎暂不可用，请重试。" : "Engine unavailable. Please retry.";
}

function setStatus(state, message) {
  $("engineStatus").dataset.state = state;
  $("engineStatus").setAttribute("aria-busy", String(state === "waking"));
  $("statusText").textContent = message;
  $("retry").classList.toggle("hidden", state !== "down");
}

function setReady(value) {
  ready = value;
  for (const el of document.querySelectorAll("[data-requires-live]")) el.disabled = !value;
  // Starting another conversation means nothing until this one has said something.
  $("tryNewConversation").disabled = !value || !spokeInConversation;
}

async function freshSession(which = "walk") {
  // Every run starts in its own namespace, so one visitor's facts can never appear in
  // another's state panel, and a re-run never reads what the previous run wrote.
  if (session[which]) {
    try {
      await api("DELETE", "/demo/session", null, { as: which });
    } catch { /* expired or never issued */ }
  }
  session[which] = null;
  const issued = await api("POST", "/demo/session");
  session[which] = issued.token;
  return issued;
}

async function sessionFor(which) {
  if (!session[which]) await freshSession(which);
  return session[which];
}

async function connect() {
  statusKey = { kind: "waking", value: "" };
  setStatus("waking", statusText());
  setReady(false);
  try {
    const health = await api("GET", "/demo/health", null, { cold: true });
    statusKey = { kind: "up", value: health.encoder };
    // An engine that cannot record conversations would turn the button into a transcript
    // wipe, which proves nothing about memory. Hide it rather than offer the imitation.
    $("tryNewConversation").hidden = !(health.capabilities || []).includes("conversations");
    setStatus("up", statusText());
    setReady(true);
  } catch {
    statusKey = { kind: "down", value: "" };
    setStatus("down", statusText());
  }
}

// ----------------------------------------------------------------- rendering

// The last state each panel drew, so a language switch can rebuild it.
const lastState = { walkState: [], tryState: [] };

// The engine's own comparison (`_value` in temporal/resolve.py), so the page calls a row a
// restatement exactly when the resolver folded it as one.
const sameValue = (a, b) => {
  const value = (m) => String(m.object || m.content || "").trim().toLowerCase();
  return value(a) === value(b);
};

function stateRow(memory, all) {
  const li = document.createElement("li");
  const superseded = memory.status === "superseded";
  // Saying the same thing twice is folded into the first mention and stored as
  // superseded, but nothing was replaced. Tagged as a restatement, so repeating a fact in
  // a later conversation does not read as a change.
  const owner = superseded ? all.find((m) => m.id === memory.superseded_by) : null;
  const restated = Boolean(owner) && sameValue(owner, memory);
  const plan = memory.scope === "plan";
  li.className = `row${superseded ? " superseded" : ""}${restated ? " restated" : ""}${plan ? " plan" : ""}`;
  const key = document.createElement("code");
  key.textContent = `${memory.predicate} = ${memory.object ?? memory.content}`;
  li.append(key);
  const origin = conversationNumber(memory.session_id);
  if (origin) {
    const from = document.createElement("span");
    from.className = "origin";
    from.textContent = lang === "zh" ? `对话 ${origin}` : `conversation ${origin}`;
    li.append(from);
  }
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = restated ? "restated" : superseded ? "superseded" : plan ? "plan" : "active";
  li.append(tag);
  return li;
}

function renderState(target, memories) {
  lastState[target] = memories;
  const list = $(target);
  list.replaceChildren();
  if (!memories.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = lang === "zh" ? "还没有事实。" : "No facts yet.";
    list.append(li);
    return;
  }
  // Superseded first, so the change reads top to bottom the way it happened.
  const order = (m) => (m.status === "superseded" ? 0 : m.scope === "plan" ? 2 : 1);
  for (const memory of [...memories].sort((a, b) => order(a) - order(b))) {
    list.append(stateRow(memory, memories));
  }
}

function bubble(role, text, gloss, note, noteKind = "note") {
  const li = document.createElement("li");
  li.className = `bubble ${role}`;
  const line = document.createElement("span");
  line.textContent = text;
  li.append(line);
  if (gloss && lang === "zh") {
    const g = document.createElement("small");
    g.className = "gloss";
    g.textContent = gloss;
    li.append(g);
  }
  if (note) {
    const n = document.createElement("small");
    n.className = noteKind;
    n.textContent = t(note);
    li.append(n);
  }
  return li;
}

// ----------------------------------------------------------------- walkthrough

async function runLiveScenario(name) {
  const button = $("walkRun");
  button.disabled = true;
  button.setAttribute("aria-pressed", "true");
  $("walkChat").replaceChildren();
  $("walkAsk").hidden = true;
  $("walkAnnounce").textContent = lang === "zh" ? "正在运行…" : "Running…";

  try {
    await freshSession("walk");
    for (const turn of WALKTHROUGH.turns) {
      $("walkChat").append(bubble("user", turn.text, turn.gloss, turn.note));
      const result = await api("POST", "/demo/facts", turn.fact, { as: "walk" });
      renderState("walkState", result.memories);
      await new Promise((r) => setTimeout(r, 650));
    }

    const question = WALKTHROUGH.question;
    $("walkQuestion").textContent = question.text
      + (lang === "zh" ? `  （${question.gloss}）` : "");
    const found = await api("POST", "/demo/search", { query: question.text, limit: 5 },
                           { as: "walk" });
    const hit = (found.memories || []).find((m) => m.predicate === "lives_in");
    $("walkAnswer").textContent = hit ? (hit.object ?? hit.content) : t(COPY.noMatch);
    $("walkAsk").hidden = false;
    $("walkAnnounce").textContent = lang === "zh" ? "完成。" : "Done.";
    $("walkReset").hidden = false;
  } catch (error) {
    $("walkAnnounce").textContent = error.message;
    setStatus("down", lang === "zh" ? `运行失败：${error.message}` : `Run failed: ${error.message}`);
  } finally {
    button.disabled = !ready;
    button.setAttribute("aria-pressed", "false");
  }
}

function walkPlaceholder() {
  const li = document.createElement("li");
  li.className = "empty";
  li.dataset.zh = "点下面的按钮，看三轮对话如何改变记忆。";
  li.dataset.en = "Press the button below to watch three turns change the memory.";
  li.textContent = li.dataset[lang];
  return li;
}

function resetWalkthrough() {
  $("walkChat").replaceChildren(walkPlaceholder());
  $("walkAsk").hidden = true;
  $("walkReset").hidden = true;
  renderState("walkState", []);
  $("walkAnnounce").textContent = "";
}

// ----------------------------------------------------------------- try it

const looksLikeQuestion = (text) =>
  /\?\s*$/.test(text) || /^(where|what|which|who|when|do i|am i)\b/i.test(text.trim());

function parse(text) {
  for (const pattern of PATTERNS) {
    const match = pattern.re.exec(text);
    if (!match) continue;
    const object = match[1].trim().replace(/[.!,;]+$/, "");
    if (!object) continue;
    return {
      subject: "user",
      predicate: pattern.predicate,
      object,
      content: pattern.say(object),
      replaces_previous: pattern.replaces,
      ...(pattern.scope ? { scope: pattern.scope } : {}),
    };
  }
  return null;
}

function say(role, text, note, noteKind) {
  const list = $("tryChat");
  list.querySelector(".empty")?.remove();
  list.append(bubble(role, text, null, note, noteKind));
  list.scrollTop = list.scrollHeight;
  if (!spokeInConversation) {
    spokeInConversation = true;
    $("tryNewConversation").disabled = !ready;
  }
}

// Where an answer's memory was written. The box exists to show this naming a
// conversation other than the one on screen.
function provenance(memories) {
  const numbers = [...new Set(memories.map((m) => conversationNumber(m.session_id)))]
    .filter(Boolean)
    .sort((a, b) => a - b);
  const parts = [];
  if (numbers.length === 1 && numbers[0] === conversation) {
    parts.push({ zh: "记忆来自本次对话", en: "Memory from this conversation" });
  } else if (numbers.length) {
    parts.push({
      zh: `记忆来自对话 ${numbers.join("、")}`,
      en: `Memory from conversation ${numbers.join(", ")}`,
    });
  }
  if (memories.length > 1) {
    parts.push({
      zh: `${memories.length} 个值同时有效，没有一句说替换`,
      en: `${memories.length} values are active at once; nothing said one replaced another`,
    });
  }
  if (!parts.length) return null;
  return { zh: parts.map((p) => p.zh).join("；"), en: parts.map((p) => p.en).join(". ") };
}

async function handleTurn(text) {
  say("user", text);
  try {
    await sessionFor("try");
    if (looksLikeQuestion(text)) {
      const found = await api("POST", "/demo/search", { query: text, limit: 10 },
                             { as: "try" });
      const memories = found.memories || [];
      if (!memories.length) {
        say("answer", t(COPY.askedNothing));
        return;
      }
      // The answer is read out of the state, not the ranking. No model runs here, and the
      // page says so rather than letting the shape imply otherwise. Every value still
      // active on the top memory's key is part of the answer: reading out only the best
      // ranked one once answered "TensorFlow" to a visitor whose last word was PyTorch,
      // because that sentence happened to sit closer to the question.
      const top = memories[0];
      const current = memories.filter(
        (m) => m.status === "active" && m.subject === top.subject && m.predicate === top.predicate,
      );
      say("answer", current.map((m) => m.object ?? m.content).join(lang === "zh" ? "、" : ", "),
          provenance(current), "origin-note");
      return;
    }

    const fact = parse(text);
    if (!fact) {
      say("answer", t(COPY.parseFailed));
      return;
    }
    const result = await api("POST", "/demo/facts", { ...fact, session_id: conversationId(conversation) },
                             { as: "try" });
    renderState("tryState", result.memories);
    const retired = (result.changed_by_this_write || []).filter((m) => m.status === "superseded");
    const named = (m) => {
      const origin = conversationNumber(m.session_id);
      const where = origin && origin !== conversation
        ? (lang === "zh" ? `（对话 ${origin}）` : ` (conversation ${origin})`)
        : "";
      return `${m.predicate} = ${m.object ?? m.content}${where}`;
    };
    say("answer", retired.length
      ? (lang === "zh"
          ? `已记住，并取代了：${retired.map(named).join("、")}`
          : `Remembered, superseding ${retired.map(named).join(", ")}`)
      : (lang === "zh" ? "已记住。" : "Remembered."));
    $("tryAnnounce").textContent = lang === "zh" ? "记忆已更新。" : "Memory updated.";
  } catch (error) {
    say("answer", error.message);
  }
}

function renderConversationTitle() {
  $("tryConversationTitle").textContent =
    lang === "zh" ? `对话 ${conversation}` : `Conversation ${conversation}`;
}

function startConversation() {
  conversation += 1;
  spokeInConversation = false;
  $("tryNewConversation").disabled = true;
  renderConversationTitle();
  // A new conversation starts with an empty transcript, as it would for an agent. The
  // state panel is left alone: memory is the only thing that carries over.
  const li = document.createElement("li");
  li.className = "empty";
  li.dataset.zh = COPY.newConversation.zh;
  li.dataset.en = COPY.newConversation.en;
  li.textContent = t(COPY.newConversation);
  $("tryChat").replaceChildren(li);
  $("tryAnnounce").textContent = lang === "zh"
    ? `已开始对话 ${conversation}。记忆保留，之前的聊天记录不保留。`
    : `Conversation ${conversation} started. Memory is kept; the earlier chat is not.`;
  $("tryInput").focus();
}

function renderSuggestions() {
  const box = $("trySuggestions");
  box.replaceChildren();
  for (const text of SUGGESTIONS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = text;
    chip.disabled = !ready;
    chip.addEventListener("click", () => {
      $("tryInput").value = text;
      $("tryInput").focus();
    });
    box.append(chip);
  }
}

// ----------------------------------------------------------------- wiring

$("bzh").addEventListener("click", () => { lang = "zh"; applyLanguage(); });
$("ben").addEventListener("click", () => { lang = "en"; applyLanguage(); });
$("retry").addEventListener("click", connect);
$("walkRun").addEventListener("click", () => runLiveScenario("walkthrough"));
$("walkReset").addEventListener("click", resetWalkthrough);
$("tryNewConversation").addEventListener("click", startConversation);
$("tryForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("tryInput");
  const text = input.value.trim();
  if (!text || !ready) return;
  input.value = "";
  $("trySend").disabled = true;
  await handleTurn(text);
  $("trySend").disabled = !ready;
});

applyLanguage();
loadManifest();
connect().then(renderSuggestions);
