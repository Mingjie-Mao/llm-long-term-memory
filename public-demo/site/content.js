// Scenario data for the public page.
//
// `text` is what the engine is actually sent, so it is English and never translated —
// a Chinese bubble would misrepresent the request. `gloss` carries the Chinese reading.
//
// The walkthrough writes real facts to the real service. Nothing here is a recording.

export const WALKTHROUGH = {
  // Three turns, and the middle one is the point. "I'm moving to Sydney" is a plan: it
  // must not retire the fact that the user still lives in Canberra. Only the third turn,
  // which asserts the move happened, supersedes it.
  turns: [
    {
      text: "I live in Canberra.",
      gloss: "我住在堪培拉。",
      fact: { subject: "user", predicate: "lives_in", object: "Canberra",
              content: "The user lives in Canberra.", replaces_previous: false },
    },
    {
      text: "I'm moving to Sydney next month.",
      gloss: "我下个月要搬去悉尼。",
      note: { zh: "这是计划，不是已发生的事实", en: "a plan, not something that happened" },
      fact: { subject: "user", predicate: "plans_move_to", object: "Sydney", scope: "plan",
              content: "The user plans to move to Sydney next month.", replaces_previous: false },
    },
    {
      text: "I've moved to Sydney.",
      gloss: "我已经搬到悉尼了。",
      note: { zh: "现在旧事实才被取代", en: "now the old fact is superseded" },
      fact: { subject: "user", predicate: "lives_in", object: "Sydney",
              content: "The user lives in Sydney.", replaces_previous: true },
    },
  ],
  question: { text: "Where do I live now?", gloss: "我现在住在哪里？" },
};

// What the interactive box understands. Pattern matching, not extraction: the product
// uses a model to read a turn, and this page has no model behind it. Stated on the page
// rather than implied, because a visitor would otherwise read a parser failure as an
// extraction failure.
//
// `replaces` stands in for the model's verdict on what a statement does to the value
// before it. The product's keying prompt says `replaces` when the user signals a change
// or when the attribute plainly holds one value at a time, and every attribute here but a
// plan holds one. "I live in X" and "I use X" were once `false`, which left two values
// active on one key; the answer was then whichever sentence ranked closer to the
// question, not what the visitor said last.
export const PATTERNS = [
  { re: /\b(?:i(?:'ve| have)? (?:just )?moved to|i now live in|i relocated to)\s+([A-Za-z][\w' -]{0,40})/i,
    predicate: "lives_in", replaces: true,
    say: (o) => `The user lives in ${o}.` },
  { re: /\bi(?:'m| am) moving to\s+([A-Za-z][\w' -]{0,40})/i,
    predicate: "plans_move_to", replaces: false, scope: "plan",
    say: (o) => `The user plans to move to ${o}.` },
  { re: /\bi live in\s+([A-Za-z][\w' -]{0,40})/i,
    predicate: "lives_in", replaces: true,
    say: (o) => `The user lives in ${o}.` },
  { re: /\bi(?:'ve| have)? switched to\s+([A-Za-z][\w'. -]{0,40})/i,
    predicate: "uses_framework", replaces: true,
    say: (o) => `The user uses ${o}.` },
  { re: /\bi use\s+([A-Za-z][\w'. -]{0,40})/i,
    predicate: "uses_framework", replaces: true,
    say: (o) => `The user uses ${o}.` },
  { re: /\bi(?:'m| am) now (?:a|an)\s+([A-Za-z][\w' -]{0,40})/i,
    predicate: "works_as", replaces: true,
    say: (o) => `The user works as ${o}.` },
  { re: /\bi work as (?:a|an)?\s*([A-Za-z][\w' -]{0,40})/i,
    predicate: "works_as", replaces: true,
    say: (o) => `The user works as ${o}.` },
  { re: /\bi work at\s+([A-Za-z][\w'. -]{0,40})/i,
    predicate: "works_at", replaces: true,
    say: (o) => `The user works at ${o}.` },
];

export const SUGGESTIONS = [
  "I live in Canberra.",
  "I use TensorFlow.",
  "I've switched to PyTorch.",
  "Where do I live?",
];

export const COPY = {
  parseFailed: {
    zh: "这句没有匹配到句式。这个演示用模式匹配识别句子，产品用模型抽取。试试 “I live in X”、“I've moved to X”、“I use X”、“I've switched to X”。",
    en: "No pattern matched. This demo recognises sentences by pattern; the product uses a model to extract them. Try “I live in X”, “I've moved to X”, “I use X”, “I've switched to X”.",
  },
  askedNothing: {
    zh: "还没有写入任何事实。先说一句关于你自己的话，再提问。",
    en: "Nothing has been written yet. State a fact about yourself first, then ask.",
  },
  noMatch: {
    zh: "检索没有命中任何记忆。",
    en: "Retrieval matched no memory.",
  },
  newConversation: {
    zh: "新对话。之前的聊天记录不在这里，提问时也不会发送；问一件之前说过的事，答案只能来自记忆。",
    en: "New conversation. The earlier chat is not here and is never sent with a question — ask about something you said before, and the answer can only come from memory.",
  },
};
