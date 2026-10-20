// Presets contain only the utterances sent to the real extractor, never facts or answers.
// Each utterance has a Chinese and an English wording; the page sends the one in the
// visitor's language. Every wording here was run against the real extractor and answerer
// before it was published (public-demo/deployments/ records the runs).
export const SCENARIOS = {
  recall: {
    title: { zh: "跨会话记忆", en: "Cross-session recall" },
    turns: [
      { text: { zh: "我喜欢有户外座位的安静咖啡馆，而且我不吃乳制品。", en: "I prefer quiet cafes with outdoor seating, and I avoid dairy." } },
    ],
    question: {
      zh: "我喜欢什么样的咖啡馆氛围和座位？你应该记住我的什么饮食限制？",
      en: "What cafe atmosphere and seating do I prefer, and what dietary restriction should you remember?",
    },
  },
  // Dated conversations: the second is written in a later conversation, and as_of shows
  // which framework held on a day between them without asking the model.
  update: {
    title: { zh: "事实变化", en: "Fact updates" },
    turns: [
      { text: { zh: "我的个人项目用的是 TensorFlow。", en: "I use TensorFlow for my personal projects." }, on: "2024-03-10" },
      { text: { zh: "我的个人项目已经从 TensorFlow 换成了 PyTorch。", en: "I have switched from TensorFlow to PyTorch for my personal projects." }, on: "2026-09-01" },
    ],
    question: { zh: "我现在用什么框架？之前用的是什么？", en: "Which framework do I use now, and what did I use before?" },
    asOf: ["2025-01-01", "2026-09-30"],
  },
  // Conversation time versus event time: the move is reported in October but happened
  // on 15 August, so on 1 September the user already lived in Sydney.
  temporal: {
    title: { zh: "对话时间与事实时间", en: "Said later, happened earlier" },
    turns: [
      { text: { zh: "我住在堪培拉。", en: "I live in Canberra." }, on: "2026-03-01" },
      { text: { zh: "我在 2026 年 8 月 15 日搬到了悉尼。", en: "I moved to Sydney on 15 August 2026." }, on: "2026-10-01" },
    ],
    question: {
      zh: "2026 年 9 月 1 日我住在哪里？我是什么时候搬过去的？",
      en: "Where did I live on 1 September 2026, and when did I move there?",
    },
    asOf: ["2026-07-01", "2026-09-01"],
  },
};
