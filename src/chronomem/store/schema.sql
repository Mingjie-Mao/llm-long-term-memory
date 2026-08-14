-- ChronoMem storage schema.
--
-- Design notes (the "why" lives in docs/DECISIONS.md):
--   * Bi-temporal from day one. `event_time`/`valid_from`/`valid_to` describe when a
--     fact was true in the world; `ingested_at` describes when we learned it. P4's
--     supersede logic needs both axes, and retrofitting them later would mean
--     re-ingesting the whole corpus.
--   * Superseded memories are never deleted, only marked. Ablations need to compare
--     "with temporal resolution" against "without", and that requires the old rows.
--   * `token_count` is computed at write time because the P6 packer treats memory
--     selection as a knapsack problem and needs the weight of every item cheaply.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    started_at  TEXT NOT NULL,          -- ISO-8601 UTC
    source      TEXT                    -- e.g. 'longmemeval:<question_id>'
);

CREATE TABLE IF NOT EXISTS turns (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_index  INTEGER NOT NULL,
    role        TEXT NOT NULL,          -- 'user' | 'assistant'
    content     TEXT NOT NULL,
    ts          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_index);

CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    type            TEXT NOT NULL,      -- semantic|episodic|preference|procedural|profile
    content         TEXT NOT NULL,

    -- Triple form. Supersede detection in P4 matches on (user_id, subject, predicate),
    -- so these are indexed even though `content` is the human-readable payload.
    subject         TEXT,
    predicate       TEXT,
    object          TEXT,

    importance      REAL NOT NULL DEFAULT 0.5,   -- [0,1], assigned at write time
    confidence      REAL NOT NULL DEFAULT 1.0,   -- [0,1], lowered by consolidation

    -- Temporal axis 1: when the fact holds in the world.
    event_time      TEXT,
    valid_from      TEXT,
    valid_to        TEXT,               -- NULL => still true

    -- Temporal axis 2: when we learned it.
    ingested_at     TEXT NOT NULL,

    -- Set when the user's own wording signalled a replacement ("I switched to X",
    -- "I no longer do Y"). A static predicate-arity list cannot catch this: "uses
    -- PyTorch" and "uses TensorFlow" can both be true, and what makes one supersede
    -- the other is the language, not the key. Captured during extraction, so it
    -- costs no extra request. The arity list is the default; this is the override.
    replaces_previous INTEGER NOT NULL DEFAULT 0,

    -- What Stage B said this fact does to earlier facts on the same key:
    -- coexists | replaces | removes | none. Stored raw rather than collapsed into
    -- `replaces_previous` because `removes` ends an attribute with no successor,
    -- which is a different timeline shape from a replacement and will need its own
    -- handling. Keeping the distinction now avoids re-ingesting to recover it.
    update_op         TEXT NOT NULL DEFAULT 'coexists',

    superseded_by   TEXT REFERENCES memories(id),
    status          TEXT NOT NULL DEFAULT 'active',  -- active|superseded|evicted

    -- Decay / reinforcement (P5). `strength` decays with time since last access and
    -- is bumped on retrieval; eviction sorts on strength * importance.
    strength         REAL NOT NULL DEFAULT 1.0,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT,
    -- The point in time `strength` itself was last brought forward to. This is
    -- separate from access: otherwise applying decay twice at the same moment
    -- would compound the same interval twice.
    strength_updated_at TEXT,

    token_count       INTEGER NOT NULL,
    source_session_id TEXT REFERENCES sessions(id),
    source_turn_index INTEGER,
    source_char_start INTEGER,
    source_char_end   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_mem_user_status ON memories(user_id, status);
CREATE INDEX IF NOT EXISTS idx_mem_sp          ON memories(user_id, subject, predicate)
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_mem_type        ON memories(user_id, type, status);
CREATE INDEX IF NOT EXISTS idx_mem_valid       ON memories(user_id, valid_from, valid_to);

-- Lexical half of hybrid retrieval. FTS5 gives BM25 for free; the alternative was
-- standing up Elasticsearch, which is not worth a service dependency at this scale.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- Entity overlap is the fifth retrieval signal. Entities are stored normalized so
-- "PyTorch" and "pytorch" collapse to one node.
CREATE TABLE IF NOT EXISTS entities (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    surface_form   TEXT NOT NULL,
    type           TEXT,
    UNIQUE(user_id, canonical_name)
);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_me_entity ON memory_entities(entity_id);

-- Consolidation provenance: which raw memories back a synthesized semantic memory.
-- This is what lets the demo answer "why do you believe that?".
CREATE TABLE IF NOT EXISTS evidence (
    memory_id        TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    source_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, source_memory_id)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
