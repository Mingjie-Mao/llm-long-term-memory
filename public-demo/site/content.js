// Presets contain only the utterances sent to the real extractor, never facts or answers.
const ago = (days) => new Date(Date.now() - days * 86400000).toISOString().slice(0, 10);
export const SCENARIOS = {
  recall: {
    title: { zh: "跨会话记忆", en: "Cross-session recall" },
    turns: ["I prefer quiet cafes with outdoor seating, and I avoid dairy."],
    question: "What cafe atmosphere and seating do I prefer, and what dietary restriction should you remember?",
  },
  update: {
    title: { zh: "事实变化", en: "Fact updates" },
    turns: ["I use TensorFlow for my personal projects.", "I have switched from TensorFlow to PyTorch for my personal projects."],
    question: "Which framework do I use now, and what did I use before?",
  },
  temporal: {
    title: { zh: "时间推理", en: "Temporal reasoning" },
    turns: [`I started a sculpture course on ${ago(28)}.`, `I finished that sculpture course on ${ago(7)}.`],
    question: "How many weeks elapsed between the start and finish of my sculpture course?",
  },
};
