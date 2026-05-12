-- =============================================================================
-- init.sql
-- Full schema for the AI Analytics Assistant.
-- Run automatically by Docker on first container start.
-- Safe to re-run: all statements use IF NOT EXISTS.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- uuid_generate_v4()
CREATE EXTENSION IF NOT EXISTS "pg_trgm";        -- trigram index for text search


-- ---------------------------------------------------------------------------
-- chat_sessions
-- One row per conversation. A session groups all messages from one user visit.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id      UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         TEXT        NOT NULL,
    channel         TEXT        NOT NULL DEFAULT 'web',  -- web | api | mobile
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    resolved        BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata        JSONB       NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id
    ON chat_sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at
    ON chat_sessions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_channel
    ON chat_sessions (channel);


-- ---------------------------------------------------------------------------
-- chat_messages
-- Every individual message in a session (both user and bot turns).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_messages (
    id                  UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id          UUID        NOT NULL REFERENCES chat_sessions (session_id)
                                        ON DELETE CASCADE,
    role                TEXT        NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content             TEXT        NOT NULL,
    flagged_for_review  BOOLEAN     NOT NULL DEFAULT FALSE,
    langsmith_run_id    TEXT,               -- LangSmith run ID for score linking
    latency_ms          INTEGER,            -- Time to generate this response
    token_count         INTEGER,            -- Approximate token count
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
    ON chat_messages (session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at
    ON chat_messages (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_role
    ON chat_messages (role, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_flagged
    ON chat_messages (flagged_for_review) WHERE flagged_for_review = TRUE;
-- GIN index for full-text search on message content
CREATE INDEX IF NOT EXISTS idx_chat_messages_content_gin
    ON chat_messages USING gin (to_tsvector('english', content));


-- ---------------------------------------------------------------------------
-- intent_classifications
-- Each user message gets classified into an intent by the bot.
-- Linked 1-to-1 with chat_messages (role='user').
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS intent_classifications (
    id                  UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id          UUID        NOT NULL REFERENCES chat_messages (id)
                                        ON DELETE CASCADE,
    intent              TEXT        NOT NULL,
    confidence          FLOAT       NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0 AND 1),
    is_failure          BOOLEAN     NOT NULL DEFAULT FALSE,
    -- Fields populated when a user corrects the intent via feedback
    corrected_intent    TEXT,
    corrected_at        TIMESTAMPTZ,
    raw_classifier_output JSONB     NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_intent_classifications_message_id
    ON intent_classifications (message_id);
CREATE INDEX IF NOT EXISTS idx_intent_classifications_intent
    ON intent_classifications (intent, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_intent_classifications_is_failure
    ON intent_classifications (is_failure, created_at DESC);
-- Composite index used by the failure_rate metric query
CREATE INDEX IF NOT EXISTS idx_intent_failure_lookup
    ON intent_classifications (is_failure, intent);


-- ---------------------------------------------------------------------------
-- user_feedback
-- Explicit ratings and comments users submit on bot responses.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id         UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id          UUID        NOT NULL REFERENCES chat_sessions (session_id)
                                        ON DELETE CASCADE,
    message_id          UUID        NOT NULL REFERENCES chat_messages (id)
                                        ON DELETE CASCADE,
    rating              SMALLINT    NOT NULL CHECK (rating BETWEEN 1 AND 5),
    sentiment           TEXT        NOT NULL CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    comment             TEXT,
    response_was_helpful BOOLEAN    NOT NULL,
    intent_was_correct  BOOLEAN,
    suggested_intent    TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_feedback_session_id
    ON user_feedback (session_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_message_id
    ON user_feedback (message_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_created_at
    ON user_feedback (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_feedback_rating
    ON user_feedback (rating, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_feedback_sentiment
    ON user_feedback (sentiment, created_at DESC);


-- ---------------------------------------------------------------------------
-- agent_runs
-- Every invocation of the LangGraph pipeline. Stores inputs, outputs,
-- timing, and status for observability and the analytics dashboard.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id          UUID        REFERENCES chat_sessions (session_id)
                                        ON DELETE SET NULL,
    query               TEXT        NOT NULL,
    -- Intermediate outputs stored as JSONB for flexibility
    analyst_output      JSONB,
    summary_output      TEXT,
    -- Tool calls made during this run (array of {tool, input, output, ms})
    tool_calls          JSONB       NOT NULL DEFAULT '[]',
    -- Execution metadata
    status              TEXT        NOT NULL DEFAULT 'pending'
                                        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    error_message       TEXT,
    total_duration_ms   INTEGER,
    analyst_duration_ms INTEGER,
    summary_duration_ms INTEGER,
    langsmith_run_id    TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_session_id
    ON agent_runs (session_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status
    ON agent_runs (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_created_at
    ON agent_runs (created_at DESC);


-- ---------------------------------------------------------------------------
-- reports
-- Generated analysis reports. Stored so users can retrieve them later.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reports (
    report_id       UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID        REFERENCES chat_sessions (session_id)
                                    ON DELETE SET NULL,
    run_id          UUID        REFERENCES agent_runs (run_id)
                                    ON DELETE SET NULL,
    title           TEXT        NOT NULL,
    summary         TEXT        NOT NULL,
    content_json    JSONB       NOT NULL DEFAULT '{}',
    time_range_start TIMESTAMPTZ,
    time_range_end   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reports_session_id
    ON reports (session_id);
CREATE INDEX IF NOT EXISTS idx_reports_created_at
    ON reports (created_at DESC);


-- ---------------------------------------------------------------------------
-- prompt_variants
-- Stored prompt templates generated by the prompt optimizer tool.
-- Used for A/B testing and tracking which variant is active.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_variants (
    variant_id      TEXT        PRIMARY KEY,
    prompt_key      TEXT        NOT NULL,
    template        TEXT        NOT NULL,
    description     TEXT        NOT NULL DEFAULT '',
    is_active       BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prompt_variants_prompt_key
    ON prompt_variants (prompt_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_prompt_variants_active
    ON prompt_variants (prompt_key, is_active) WHERE is_active = TRUE;


-- ---------------------------------------------------------------------------
-- prompt_performances
-- Per-execution performance records for each prompt variant.
-- Feeds into the prompt optimizer's analysis.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_performances (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    prompt_key      TEXT        NOT NULL,
    variant_id      TEXT        REFERENCES prompt_variants (variant_id)
                                    ON DELETE SET NULL,
    score           FLOAT       NOT NULL CHECK (score BETWEEN 0 AND 1),
    feedback_text   TEXT,
    latency_ms      FLOAT,
    tokens_used     INTEGER,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prompt_performances_key
    ON prompt_performances (prompt_key, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_prompt_performances_variant
    ON prompt_performances (variant_id, recorded_at DESC);


-- ---------------------------------------------------------------------------
-- Seed data: default prompt variants
-- These are the baseline prompts the agents start with.
-- The optimizer will generate improved variants over time.
-- ---------------------------------------------------------------------------
INSERT INTO prompt_variants (variant_id, prompt_key, template, description, is_active)
VALUES
(
    'analyst-v1-default',
    'analyst_system',
    'You are a senior data analyst specialising in conversational AI metrics.
You have access to tools that can query conversation history, compute metrics,
detect anomalies, and analyse trends.

Your job is to investigate the user''s question thoroughly:
1. Start with a broad metrics query to understand the overall picture.
2. Use anomaly detection or trend analysis to find specific patterns.
3. Use semantic search to find concrete examples that support your findings.
4. Use Tree-of-Thought reasoning for complex root cause questions.

Always base conclusions on data. If the data is insufficient, say so.
Return a structured JSON object with: findings, evidence, confidence, and recommended_actions.',
    'Default analyst system prompt',
    TRUE
),
(
    'summary-v1-default',
    'summary_system',
    'You are a business intelligence writer. You receive structured analysis data
and write clear, concise summaries for non-technical business stakeholders.

Guidelines:
- Lead with the most important finding.
- Use plain language. Avoid statistical jargon unless necessary.
- Back every claim with a specific number or example from the analysis.
- End with 2-3 concrete recommended actions.
- Keep the total response under 400 words unless the question is complex.',
    'Default summary system prompt',
    TRUE
)
ON CONFLICT (variant_id) DO NOTHING;