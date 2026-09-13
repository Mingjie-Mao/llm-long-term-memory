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
  const status = statusText();
  if (status) $("statusText").textContent = status;
}

// ----------------------------------------------------------------- manifest

let manifest = null;

function renderMetrics() {
  if (!manifest) return;
  const { result, engineering, release } = manifest;
  const acc = result.accuracy;
  const ctx = result.median_context_tokens;
  const rag = result.naive_rag_accuracy;
  const ragCtx = result.naive_rag_median_context_tokens;

  // One sentence. Everything that qualifies it goes in the footnote below, and
  // everything else lives in the repository.
  $("headline").textContent = lang === "zh"
    ? `在 LongMemEval-S 的 100 题终测上准确率 ${acc}%，上下文中位数 ${ctx.toLocaleString()} token；naive RAG 为 ${rag}%，用 ${ragCtx.toLocaleString()} token。`
    : `${acc}% accuracy on LongMemEval-S test100 with ${ctx.toLocaleString()} median context tokens, versus ${rag}% for naive RAG using ${ragCtx.toLocaleString()} tokens.`;

  $("footnote").textContent = result.significant_vs_naive_rag
    ? (lang === "zh"
        ? `在这 ${result.questions} 题上，${acc}% 与 ${rag}% 的差距达到统计显著。`
        : `The ${acc}% vs ${rag}% difference was statistically significant on this ${result.questions}-question test.`)
    : (lang === "zh"
        ? `在这 ${result.questions} 题上，${acc}% 与 ${rag}% 的差距未达统计显著。`
        : `The ${acc}% vs ${rag}% difference was not statistically significant on this ${result.questions}-question test.`);

  $("releaseMeta").textContent = lang === "zh"
    ? `发布 ${release} · ${engineering.tests} 项测试 · ${engineering.verified_at} 核验`
    : `Release ${release} · ${engineering.tests} tests · verified ${engineering.verified_at}`;
  // The reference commit is a convenience and may not survive a history rewrite, so the
  // link degrades to the repository rather than to a 404 when it is absent.
  const commit = engineering.source_commit_for_reference;
  const link = $("releaseCommit");
  link.textContent = commit ? commit.slice(0, 7) : "source";
  link.href = commit
    ? `https://github.com/Mingjie-Mao/llm-long-term-memory/commit/${commit}`
    : "https://github.com/Mingjie-Mao/llm-long-term-memory";
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
  return lang === "zh" ? `引擎不可用：${value}` : `Engine unavailable: ${value}`;
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
    setStatus("up", statusText());
    setReady(true);
  } catch (error) {
    statusKey = { kind: "down", value: error.message };
    setStatus("down", statusText());
  }
}

// ----------------------------------------------------------------- rendering

function stateRow(memory) {
  const li = document.createElement("li");
  const superseded = memory.status === "superseded";
  const plan = memory.scope === "plan";
  li.className = `row${superseded ? " superseded" : ""}${plan ? " plan" : ""}`;
  const key = document.createElement("code");
  key.textContent = `${memory.predicate} = ${memory.object ?? memory.content}`;
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = superseded ? "superseded" : plan ? "plan" : "active";
  li.append(key, tag);
  return li;
}

function renderState(target, memories) {
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
    list.append(stateRow(memory));
  }
}

function bubble(role, text, gloss, note) {
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
    n.className = "note";
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

function say(role, text, note) {
  const list = $("tryChat");
  list.querySelector(".empty")?.remove();
  list.append(bubble(role, text, null, note));
  list.scrollTop = list.scrollHeight;
}

async function handleTurn(text) {
  say("user", text);
  try {
    await sessionFor("try");
    if (looksLikeQuestion(text)) {
      const found = await api("POST", "/demo/search", { query: text, limit: 5 },
                             { as: "try" });
      const memories = found.memories || [];
      if (!memories.length) {
        say("answer", t(COPY.askedNothing));
        return;
      }
      // The answer is the top retrieved memory's value, read out. No model runs here,
      // and the page says so rather than letting the shape imply otherwise.
      const top = memories[0];
      say("answer", top.object ?? top.content);
      return;
    }

    const fact = parse(text);
    if (!fact) {
      say("answer", t(COPY.parseFailed));
      return;
    }
    const result = await api("POST", "/demo/facts", fact, { as: "try" });
    renderState("tryState", result.memories);
    const retired = (result.changed_by_this_write || []).filter((m) => m.status === "superseded");
    say("answer", retired.length
      ? (lang === "zh"
          ? `已记住，并取代了：${retired.map((m) => `${m.predicate} = ${m.object ?? m.content}`).join("、")}`
          : `Remembered, superseding ${retired.map((m) => `${m.predicate} = ${m.object ?? m.content}`).join(", ")}`)
      : (lang === "zh" ? "已记住。" : "Remembered."));
    $("tryAnnounce").textContent = lang === "zh" ? "记忆已更新。" : "Memory updated.";
  } catch (error) {
    say("answer", error.message);
  }
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
