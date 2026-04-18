/**
 * Cache trace logging integration tests.
 *
 * Verifies that MERIDIAN_TRACE_CACHE gates per-request JSONL traces,
 * records the key cache/resume/query/usage events when enabled, and stays
 * silent when disabled.
 */

import { describe, it, expect, mock, beforeEach, afterEach } from "bun:test"
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs"
import { join } from "node:path"
import { tmpdir } from "node:os"

let capturedQueryCalls: any[] = []
let passthroughServerCreateCount = 0
const MOCK_SDK_SESSION = "trace-sdk-session"

mock.module("@anthropic-ai/claude-agent-sdk", () => ({
  query: (params: any) => {
    capturedQueryCalls.push(params)
    return (async function* () {
      yield {
        type: "assistant",
        message: {
          id: `msg-${capturedQueryCalls.length}`,
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "ok" }],
          model: "claude-sonnet-4-5",
          stop_reason: "end_turn",
          usage: {
            input_tokens: 42,
            output_tokens: 9,
            cache_read_input_tokens: 30,
            cache_creation_input_tokens: 7,
          },
        },
        session_id: MOCK_SDK_SESSION,
        uuid: `assistant-uuid-${capturedQueryCalls.length}`,
      }
    })()
  },
  createSdkMcpServer: ({ name }: { name: string }) => ({
    type: "sdk",
    name,
    _id: ++passthroughServerCreateCount,
    instance: {
      registerTool: () => ({}),
    },
  }),
  tool: () => ({}),
}))

mock.module("../logger", () => ({
  claudeLog: () => {},
  withClaudeLogContext: (_ctx: any, fn: any) => fn(),
}))

mock.module("../mcpTools", () => ({
  createOpencodeMcpServer: () => ({ type: "sdk", name: "opencode", instance: {} }),
}))

const { createProxyServer, clearSessionCache } = await import("../proxy/server")
const { setSessionStoreDir } = await import("../proxy/sessionStore")

const TOOL_READ = {
  name: "read",
  description: "Read a file",
  input_schema: {
    type: "object",
    properties: { path: { type: "string" } },
    required: ["path"],
  },
}

const TOOL_WRITE = {
  name: "write",
  description: "Write a file",
  input_schema: {
    type: "object",
    properties: {
      path: { type: "string" },
      content: { type: "string" },
    },
    required: ["path", "content"],
  },
}

function createTestApp() {
  const { app } = createProxyServer({ port: 0, host: "127.0.0.1" })
  return app
}

function createBody(overrides: Record<string, unknown> = {}) {
  return {
    model: "claude-sonnet-4-5",
    max_tokens: 256,
    stream: false,
    messages: [{ role: "user", content: "hello" }],
    ...overrides,
  }
}

async function post(app: any, requestId: string, body: any, headers: Record<string, string> = {}) {
  return app.fetch(new Request("http://localhost/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-request-id": requestId,
      ...headers,
    },
    body: JSON.stringify(body),
  }))
}

function readTraceEvents(traceDir: string, requestId: string): Array<Record<string, unknown>> {
  const path = join(traceDir, `meridian-cache-trace-${requestId}.jsonl`)
  const text = readFileSync(path, "utf8").trim()
  if (!text) return []
  return text.split("\n").filter(Boolean).map(line => JSON.parse(line) as Record<string, unknown>)
}

function getTracePath(traceDir: string, requestId: string): string {
  return join(traceDir, `meridian-cache-trace-${requestId}.jsonl`)
}

function findEvent(events: Array<Record<string, unknown>>, name: string): Record<string, unknown> {
  const event = events.find(entry => entry.event === name)
  expect(event).toBeDefined()
  return event!
}

describe("cache trace logging", () => {
  const originalTrace = process.env.MERIDIAN_TRACE_CACHE
  const originalLegacyTrace = process.env.CLAUDE_PROXY_TRACE_CACHE
  const originalTraceDir = process.env.MERIDIAN_TRACE_CACHE_DIR
  const originalLegacyTraceDir = process.env.CLAUDE_PROXY_TRACE_CACHE_DIR
  const originalPassthrough = process.env.MERIDIAN_PASSTHROUGH
  const originalLegacyPassthrough = process.env.CLAUDE_PROXY_PASSTHROUGH

  let tmpDir: string

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "meridian-cache-trace-"))
    capturedQueryCalls = []
    passthroughServerCreateCount = 0
    process.env.MERIDIAN_TRACE_CACHE_DIR = tmpDir
    delete process.env.MERIDIAN_TRACE_CACHE
    delete process.env.CLAUDE_PROXY_TRACE_CACHE
    delete process.env.CLAUDE_PROXY_TRACE_CACHE_DIR
    delete process.env.MERIDIAN_PASSTHROUGH
    delete process.env.CLAUDE_PROXY_PASSTHROUGH
    setSessionStoreDir(join(tmpDir, "sessions"))
    clearSessionCache()
  })

  afterEach(() => {
    clearSessionCache()
    setSessionStoreDir(null)
    if (originalTrace === undefined) delete process.env.MERIDIAN_TRACE_CACHE
    else process.env.MERIDIAN_TRACE_CACHE = originalTrace
    if (originalLegacyTrace === undefined) delete process.env.CLAUDE_PROXY_TRACE_CACHE
    else process.env.CLAUDE_PROXY_TRACE_CACHE = originalLegacyTrace
    if (originalTraceDir === undefined) delete process.env.MERIDIAN_TRACE_CACHE_DIR
    else process.env.MERIDIAN_TRACE_CACHE_DIR = originalTraceDir
    if (originalLegacyTraceDir === undefined) delete process.env.CLAUDE_PROXY_TRACE_CACHE_DIR
    else process.env.CLAUDE_PROXY_TRACE_CACHE_DIR = originalLegacyTraceDir
    if (originalPassthrough === undefined) delete process.env.MERIDIAN_PASSTHROUGH
    else process.env.MERIDIAN_PASSTHROUGH = originalPassthrough
    if (originalLegacyPassthrough === undefined) delete process.env.CLAUDE_PROXY_PASSTHROUGH
    else process.env.CLAUDE_PROXY_PASSTHROUGH = originalLegacyPassthrough
    try { rmSync(tmpDir, { recursive: true, force: true }) } catch {}
  })

  it("stays quiet when MERIDIAN_TRACE_CACHE is disabled", async () => {
    const app = createTestApp()
    const requestId = "trace-disabled"

    const res = await post(app, requestId, createBody(), {
      "x-opencode-session": "trace-disabled-session",
    })

    expect(res.status).toBe(200)
    expect(existsSync(getTracePath(tmpDir, requestId))).toBe(false)
  })

  it("writes structured request/query/store/usage events when enabled", async () => {
    process.env.MERIDIAN_TRACE_CACHE = "1"
    const app = createTestApp()
    const requestId = "trace-enabled-basic"

    const res = await post(app, requestId, createBody(), {
      "x-opencode-session": "trace-basic-session",
    })

    expect(res.status).toBe(200)

    const events = readTraceEvents(tmpDir, requestId)
    const eventNames = events.map(event => event.event)
    for (const eventName of ["request.identity", "resume.delta", "sdk.query", "session.store", "response.usage"]) {
      expect(eventNames).toContain(eventName)
    }

    const identity = findEvent(events, "request.identity")
    expect(identity.adapter).toBe("opencode")
    expect(identity.lineageType).toBe("new")
    expect(identity.scopedSessionId).toBe("trace-basic-session")
    expect(identity.passthroughCacheKey).toBe("trace-basic-session")

    const delta = findEvent(events, "resume.delta")
    expect(delta.resumePath).toBe("fresh_all")
    expect(delta.promptMode).toBe("text")
    expect(delta.assistantReplayFiltered).toBe(false)
    expect(delta.resumeBoundaryKind).toBe("other")
    expect(typeof (delta.allMessages as Record<string, unknown>).digest).toBe("string")

    const sdkQuery = findEvent(events, "sdk.query")
    expect(sdkQuery.passthrough).toBe(true)
    expect((sdkQuery.options as Record<string, unknown>).resume).toBeUndefined()
    expect((sdkQuery.options as Record<string, unknown>).maxTurns).toBe(2)
    expect((sdkQuery.prompt as Record<string, unknown>).promptMode).toBe("text")
    expect(typeof ((sdkQuery.prompt as Record<string, unknown>).promptDigest)).toBe("string")
    expect(typeof (((sdkQuery.options as Record<string, unknown>).allowedToolDigest))).toBe("string")

    const sessionStore = findEvent(events, "session.store")
    expect(sessionStore.decision).toBe("stored")
    expect(sessionStore.keyType).toBe("session")
    expect(sessionStore.key).toBe("trace-basic-session")
    expect(sessionStore.claudeSessionId).toBe(MOCK_SDK_SESSION)
    expect(sessionStore.messageCount).toBe(1)

    const usage = findEvent(events, "response.usage")
    expect((usage.usage as Record<string, unknown>).inputTokens).toBe(42)
    expect((usage.usage as Record<string, unknown>).cacheReadInputTokens).toBe(30)
    expect((usage.usage as Record<string, unknown>).cacheCreationInputTokens).toBe(7)
  })

  it("records passthrough tool restore and MCP reuse decisions on resumed turns", async () => {
    process.env.MERIDIAN_TRACE_CACHE = "1"
    process.env.MERIDIAN_PASSTHROUGH = "1"
    const app = createTestApp()
    const sessionId = "trace-passthrough-session"

    const first = await post(app, "trace-passthrough-1", createBody({
      tools: [TOOL_READ, TOOL_WRITE],
    }), {
      "x-opencode-session": sessionId,
    })
    expect(first.status).toBe(200)

    const second = await post(app, "trace-passthrough-2", createBody({
      tools: [],
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: [{ type: "text", text: "ok" }] },
        { role: "user", content: "continue" },
      ],
    }), {
      "x-opencode-session": sessionId,
    })
    expect(second.status).toBe(200)

    const events = readTraceEvents(tmpDir, "trace-passthrough-2")

    const identity = findEvent(events, "request.identity")
    expect(identity.lineageType).toBe("continuation")
    expect(identity.resumeSessionId).toBe(MOCK_SDK_SESSION)
    expect(identity.passthroughCacheKey).toBe(sessionId)

    const toolCache = findEvent(events, "cache.tools")
    expect(toolCache.decision).toBe("restored")
    expect(toolCache.reason).toBe("client_omitted_tools")
    expect(toolCache.effectiveToolCount).toBe(2)

    const mcpCache = findEvent(events, "cache.mcp")
    expect(mcpCache.decision).toBe("reused")
    expect(mcpCache.reason).toBe("tool_set_match")
    expect(mcpCache.toolCount).toBe(2)
    expect(mcpCache.toolNames).toEqual(["read", "write"])
    expect(typeof mcpCache.toolSetKeySha256).toBe("string")

    const delta = findEvent(events, "resume.delta")
    expect(delta.resumePath).toBe("resume_delta")
    expect(delta.assistantReplayFiltered).toBe(true)
    expect((delta.messagesToConvert as Record<string, unknown>).count).toBe(2)
    expect(delta.resumeBoundaryKind).toBe("assistant_text_to_user_text")
    expect(typeof (delta.messagesToConvert as Record<string, unknown>).digest).toBe("string")

    const sdkQuery = findEvent(events, "sdk.query")
    expect((sdkQuery.options as Record<string, unknown>).resume).toBe(MOCK_SDK_SESSION)
    expect((sdkQuery.options as Record<string, unknown>).allowedToolCount).toBe(2)
    expect((sdkQuery.options as Record<string, unknown>).strictMcpConfig).toBe(true)
    expect(typeof ((sdkQuery.prompt as Record<string, unknown>).promptDigest)).toBe("string")

    expect(passthroughServerCreateCount).toBe(2)
  })

  it("records structured prompt digests and boundary classification for Pi tool_result resumes", async () => {
    process.env.MERIDIAN_TRACE_CACHE = "1"
    process.env.MERIDIAN_PASSTHROUGH = "1"
    const app = createTestApp()
    const piHeaders = {
      "x-meridian-agent": "pi",
      "x-meridian-source": "main",
    }
    const piSystem = "Current working directory: /Users/cartwmic/git/meridian"

    const first = await post(app, "trace-pi-structured-1", createBody({
      system: piSystem,
      tools: [TOOL_WRITE],
      messages: [{ role: "user", content: "Create foo.txt with hello" }],
    }), piHeaders)
    expect(first.status).toBe(200)

    const second = await post(app, "trace-pi-structured-2", createBody({
      system: piSystem,
      tools: [TOOL_WRITE],
      messages: [
        { role: "user", content: "Create foo.txt with hello" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "I'll create the file." },
            { type: "tool_use", id: "toolu_123", name: "write", input: { path: "foo.txt", content: "hello" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_123", content: "File written." },
          ],
        },
      ],
    }), piHeaders)
    expect(second.status).toBe(200)

    const events = readTraceEvents(tmpDir, "trace-pi-structured-2")

    const identity = findEvent(events, "request.identity")
    expect(identity.adapter).toBe("pi")
    expect(identity.source).toBe("main")
    expect(identity.lineageType).toBe("continuation")
    expect(identity.resumeSessionId).toBe(MOCK_SDK_SESSION)

    const delta = findEvent(events, "resume.delta")
    expect(delta.promptMode).toBe("structured")
    expect(delta.useStructuredPrompt).toBe(true)
    expect(delta.hasStructuredResumeUserBlocks).toBe(true)
    expect(delta.resumeBoundaryKind).toBe("assistant_tool_use_to_user_tool_result")
    expect(typeof (delta.messagesToConvert as Record<string, unknown>).digest).toBe("string")

    const sdkQuery = findEvent(events, "sdk.query")
    const prompt = sdkQuery.prompt as Record<string, unknown>
    const options = sdkQuery.options as Record<string, unknown>
    expect(prompt.promptMode).toBe("structured")
    expect(typeof prompt.promptDigest).toBe("string")
    expect(Array.isArray(prompt.promptMessageDigests)).toBe(true)
    expect((prompt.promptMessageDigests as unknown[]).length).toBe(1)
    expect(typeof options.allowedToolDigest).toBe("string")
    expect(typeof ((options.systemPrompt as Record<string, unknown>).digest)).toBe("string")
  })
})
