export const SCENARIOS = {
  changed: {
    tour: {
      zh: {
        announce: "演示完成：Sydney 被标记为历史，Melbourne 成为当前事实，回答排除了旧值。",
        steps: [
          {
            title: "对话报告变化",
            lead: "同一个人先后给出两个居住地，并明确说自己搬家了。",
            quote: "我以前住在悉尼。上个月我搬到了墨尔本。",
          },
          {
            title: "旧值关闭，但不删除",
            lead: "明确且无歧义的 replacement 会结束 Sydney 的有效期。",
            memories: [
              { value: "Melbourne", state: "当前有效 · 2026-06 起", old: false },
              { value: "Sydney", state: "已取代 · 2025-01 至 2026-06", old: true },
            ],
          },
          {
            title: "回答只使用当前事实",
            lead: "问：我现在住在哪里？",
            quote: "你现在住在墨尔本。",
            reason: "Sydney 被排除：superseded → Melbourne",
          },
        ],
      },
      en: {
        announce: "Tour complete: Sydney became history, Melbourne became current, and retrieval excluded the old value.",
        steps: [
          {
            title: "The conversation reports a change",
            lead: "The same person gives two locations and explicitly says they moved.",
            quote: "I used to live in Sydney. Last month I moved to Melbourne.",
          },
          {
            title: "The old value closes, but stays",
            lead: "An explicit, unambiguous replacement ends Sydney's validity window.",
            memories: [
              { value: "Melbourne", state: "current · since 2026-06", old: false },
              { value: "Sydney", state: "superseded · 2025-01 to 2026-06", old: true },
            ],
          },
          {
            title: "The answer uses only current truth",
            lead: "Question: where do I live now?",
            quote: "You live in Melbourne now.",
            reason: "Sydney rejected: superseded → Melbourne",
          },
        ],
      },
    },
    view: {
      zh: {
        conversation: [
          { role: "user", text: "The user lives in Sydney.", gloss: "用户住在悉尼。" },
          { role: "user", text: "The user moved to Melbourne.", gloss: "用户搬到了墨尔本。" },
        ],
        answer: "当前记忆是 Melbourne；Sydney 被作为 superseded 历史排除。",
      },
      en: {
        conversation: [
          { role: "user", text: "The user lives in Sydney." },
          { role: "user", text: "The user moved to Melbourne." },
        ],
        answer: "Melbourne is current; Sydney is returned as superseded history.",
      },
    },
    engine: {
      facts: [
        {
          predicate: "lives_in",
          object: "Sydney",
          content: "The user lives in Sydney.",
          event_time: "2025-01-10T00:00:00",
          replaces_previous: false,
        },
        {
          predicate: "lives_in",
          object: "Melbourne",
          content: "The user moved to Melbourne.",
          event_time: "2026-06-04T00:00:00",
          replaces_previous: true,
        },
      ],
      query: "where do I live now?",
      searchMode: "memory",
    },
  },

  detail: {
    tour: {
      zh: {
        announce: "演示完成：结构化记忆没有预订编号，系统按需从原始对话找回了 VX-928173。",
        steps: [
          {
            title: "对话里有精确编号",
            lead: "助手给出建议，同时留下一个之后可能被追问的逐字细节。",
            quote: "我推荐这盏台灯。你的预订编号是 VX-928173。",
          },
          {
            title: "结构化记忆保留语义",
            lead: "抽取结果记得“推荐过台灯”，但精确编号不在事实摘要里。",
            memories: [{ value: "推荐过一盏台灯", state: "当前有效 · 编号未包含", old: false }],
          },
          {
            title: "需要时回到原文",
            lead: "问：预订编号是什么？",
            quote: "VX-928173",
            reason: "structured memory insufficient → raw archive BM25",
          },
        ],
      },
      en: {
        announce: "Tour complete: structured memory lacked the booking code, so the raw archive recovered VX-928173 on demand.",
        steps: [
          {
            title: "The conversation contains an exact code",
            lead: "The assistant gives advice and an exact artifact that may be requested later.",
            quote: "I recommend this desk lamp. Your booking reference is VX-928173.",
          },
          {
            title: "Structured memory keeps the meaning",
            lead: "Extraction remembers the lamp recommendation, but the exact booking code is absent from the summary.",
            memories: [{ value: "recommended a desk lamp", state: "current · code omitted", old: false }],
          },
          {
            title: "Raw turns recover it on demand",
            lead: "Question: what was the booking reference?",
            quote: "VX-928173",
            reason: "structured memory insufficient → raw archive BM25",
          },
        ],
      },
    },
    view: {
      zh: {
        conversation: [
          { role: "assistant", text: "The assistant recommended a desk lamp.", gloss: "助手推荐了一盏台灯。" },
          {
            role: "assistant",
            text: "Your booking reference is VX-928173 and the lamp ships to level 3.",
            gloss: "你的预订编号是 VX-928173，台灯送到三楼。",
          },
        ],
        answer: "原文档案找回了预订编号 VX-928173。",
      },
      en: {
        conversation: [
          { role: "assistant", text: "The assistant recommended a desk lamp." },
          { role: "assistant", text: "Your booking reference is VX-928173 and the lamp ships to level 3." },
        ],
        answer: "The raw archive recovered booking reference VX-928173.",
      },
    },
    engine: {
      facts: [
        {
          predicate: "assistant_recommendation",
          object: "a desk lamp",
          content: "The assistant recommended a desk lamp.",
          event_time: "2026-05-20T00:00:00",
          replaces_previous: false,
          source_role: "assistant",
        },
      ],
      turns: [
        {
          role: "assistant",
          content: "Your booking reference is VX-928173 and the lamp ships to level 3.",
        },
      ],
      query: "booking reference",
      searchMode: "raw",
    },
  },

  multi: {
    tour: {
      zh: {
        announce: "演示完成：cat 和 dog 都保持有效；多值键没有生成虚假的替换关系。",
        steps: [
          {
            title: "两个值可以同时为真",
            lead: "“拥有宠物”不是单值属性，新增一只狗不代表猫已经不存在。",
            quote: "我有一只猫。最近又领养了一只狗。",
          },
          {
            title: "解析器拒绝编造取代",
            lead: "两个事实都保持 current，也没有 superseded_by 箭头。",
            memories: [
              { value: "cat", state: "当前有效", old: false },
              { value: "dog", state: "当前有效", old: false },
            ],
          },
          {
            title: "查询返回两个值",
            lead: "问：我有哪些宠物？",
            quote: "你有一只猫和一只狗。",
            reason: "multi-valued key → both facts remain eligible",
          },
        ],
      },
      en: {
        announce: "Tour complete: cat and dog both stayed current; the multi-valued key invented no replacement edge.",
        steps: [
          {
            title: "Two values can be true together",
            lead: "Owning a pet is not single-valued. Adopting a dog does not imply the cat disappeared.",
            quote: "I have a cat. I recently adopted a dog too.",
          },
          {
            title: "The resolver refuses to invent a replacement",
            lead: "Both facts remain current and neither receives a superseded_by edge.",
            memories: [
              { value: "cat", state: "current", old: false },
              { value: "dog", state: "current", old: false },
            ],
          },
          {
            title: "Retrieval returns both values",
            lead: "Question: what pets do I have?",
            quote: "You have a cat and a dog.",
            reason: "multi-valued key → both facts remain eligible",
          },
        ],
      },
    },
    view: {
      zh: {
        conversation: [
          { role: "user", text: "The user owns a cat.", gloss: "用户养了一只猫。" },
          { role: "user", text: "The user owns a dog.", gloss: "用户又养了一只狗。" },
        ],
        answer: "cat 和 dog 都保持 current，没有生成替换关系。",
      },
      en: {
        conversation: [
          { role: "user", text: "The user owns a cat." },
          { role: "user", text: "The user owns a dog." },
        ],
        answer: "Cat and dog both remain current; no replacement edge was created.",
      },
    },
    engine: {
      facts: [
        {
          predicate: "owns_pet",
          object: "a cat",
          content: "The user owns a cat.",
          event_time: "2025-05-02T00:00:00",
          replaces_previous: false,
        },
        {
          predicate: "owns_pet",
          object: "a dog",
          content: "The user owns a dog.",
          event_time: "2026-02-01T00:00:00",
          replaces_previous: false,
        },
      ],
      query: "what pets do I have?",
      searchMode: "memory",
    },
  },
};
