/**
 * Tests for session tool caching — when a client drops tools on a
 * continuation request, Meridian reuses the last-seen tool set to
 * preserve prompt cache stability.
 */

import { describe, it, expect, mock, beforeEach, afterEach } from "bun:test"
import { assistantMessage, makeRequest } from "./helpers"

let capturedQueryParams: any = null
let mockMessages: any[] = []

mock.module("@anthropic-ai/claude-agent-sdk", () => ({
  query: (params: any) => {
    capturedQueryParams = params
    return (async function* () {
      for (const msg of mockMessages) yield msg
    })()
  },
  createSdkMcpServer: () => ({
    type: "sdk",
    name: "test",
    instance: { tool: () => {}, registerTool: () => ({}) },
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

const SESSION_ID = "tool-cache-test-session"

const TOOL_A = {
  name: "read_file",
  description: "Read a file",
  input_schema: { type: "object", properties: { path: { type: "string" } } },
}

const TOOL_B = {
  name: "write_file",
  description: "Write a file",
  input_schema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } } },
}

function createTestApp() {
  const { app } = createProxyServer({ port: 0, host: "127.0.0.1" })
  return app
}

async function post(app: any, body: any, sessionId = SESSION_ID) {
  return app.fetch(new Request("http://localhost/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-opencode-session": sessionId,
    },
    body: JSON.stringify(body),
  }))
}

async function postPi(app: any, body: any, headers: Record<string, string> = {}) {
  return app.fetch(new Request("http://localhost/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-meridian-agent": "pi",
      ...headers,
    },
    body: JSON.stringify(body),
  }))
}

function getPassthroughMcp(options: any) {
  const mcpServers = options?.mcpServers
  return mcpServers?.oc
}

describe("Session tool cache", () => {
  const originalPassthrough = process.env.MERIDIAN_PASSTHROUGH

  beforeEach(() => {
    clearSessionCache()
    capturedQueryParams = null
    mockMessages = [
      assistantMessage([{ type: "text", text: "Done." }]),
    ]
    process.env.MERIDIAN_PASSTHROUGH = "1"
  })

  afterEach(() => {
    if (originalPassthrough === undefined) delete process.env.MERIDIAN_PASSTHROUGH
    else process.env.MERIDIAN_PASSTHROUGH = originalPassthrough
  })

  it("caches tools from first request and reuses when client sends none", async () => {
    const app = createTestApp()

    // Request 1: client sends tools
    await post(app, makeRequest({
      stream: false,
      tools: [TOOL_A, TOOL_B],
      messages: [{ role: "user", content: "hello" }],
    }))

    // Verify tools were registered
    const opts1 = capturedQueryParams?.options
    expect(opts1?.mcpServers).toBeDefined()

    capturedQueryParams = null

    // Request 2: same session, no tools — should reuse cached
    await post(app, makeRequest({
      stream: false,
      tools: [],
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: "Done." },
        { role: "user", content: "continue" },
      ],
    }))

    const opts2 = capturedQueryParams?.options
    expect(opts2?.mcpServers).toBeDefined()
  })

  it("does not reuse tools for a different session", async () => {
    const app = createTestApp()

    // Request 1: session A sends tools
    await post(app, makeRequest({
      stream: false,
      tools: [TOOL_A],
      messages: [{ role: "user", content: "hello" }],
    }), "session-a")

    capturedQueryParams = null

    // Request 2: session B sends no tools — should NOT get session A's tools
    await post(app, makeRequest({
      stream: false,
      tools: [],
      messages: [{ role: "user", content: "hello" }],
    }), "session-b")

    const opts2 = capturedQueryParams?.options
    expect(getPassthroughMcp(opts2)).toBeUndefined()
  })

  it("restores the cached tool set for resumed Pi/headerless passthrough turns when the client omits tools", async () => {
    const app = createTestApp()
    const system = "Current working directory: /Users/cartwmic/git/meridian"

    await postPi(app, makeRequest({
      stream: false,
      system,
      tools: [TOOL_A, TOOL_B],
      messages: [{ role: "user", content: "hello" }],
    }))

    capturedQueryParams = null

    await postPi(app, makeRequest({
      stream: false,
      system,
      tools: [],
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: "Done." },
        { role: "user", content: "continue" },
      ],
    }))

    const opts2 = capturedQueryParams?.options
    expect(opts2?.resume).toBe("test-session")
    expect(getPassthroughMcp(opts2)).toBeDefined()
  })

  it("does not restore cached tools for a headered request that no longer resumes the prior session", async () => {
    const app = createTestApp()

    await post(app, makeRequest({
      stream: false,
      tools: [TOOL_A],
      messages: [{ role: "user", content: "hello" }],
    }), "session-replay")

    capturedQueryParams = null

    // Same session header, but this is a replay/fresh turn rather than a continuation.
    await post(app, makeRequest({
      stream: false,
      tools: [],
      messages: [{ role: "user", content: "hello" }],
    }), "session-replay")

    const opts2 = capturedQueryParams?.options
    expect(opts2?.resume).toBeUndefined()
    expect(getPassthroughMcp(opts2)).toBeUndefined()
  })

  it("does not restore cached tools for a Pi/headerless request that no longer resumes the prior session", async () => {
    const app = createTestApp()
    const system = "Current working directory: /Users/cartwmic/git/meridian"

    await postPi(app, makeRequest({
      stream: false,
      system,
      tools: [TOOL_A],
      messages: [{ role: "user", content: "hello" }],
    }))

    capturedQueryParams = null

    // Same fingerprint, but verifyLineage classifies this as a replay/diverged turn.
    await postPi(app, makeRequest({
      stream: false,
      system,
      tools: [],
      messages: [{ role: "user", content: "hello" }],
    }))

    const opts2 = capturedQueryParams?.options
    expect(opts2?.resume).toBeUndefined()
    expect(getPassthroughMcp(opts2)).toBeUndefined()
  })

  it("updates cached tools when client sends a new set", async () => {
    const app = createTestApp()

    // Request 1: send TOOL_A
    await post(app, makeRequest({
      stream: false,
      tools: [TOOL_A],
      messages: [{ role: "user", content: "hello" }],
    }))

    // Request 2: send TOOL_A + TOOL_B (updated set)
    await post(app, makeRequest({
      stream: false,
      tools: [TOOL_A, TOOL_B],
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: "Done." },
        { role: "user", content: "continue" },
      ],
    }))

    capturedQueryParams = null

    // Request 3: no tools — should get the updated set (TOOL_A + TOOL_B)
    await post(app, makeRequest({
      stream: false,
      tools: [],
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: "Done." },
        { role: "user", content: "continue" },
        { role: "assistant", content: "Done." },
        { role: "user", content: "more" },
      ],
    }))

    const opts3 = capturedQueryParams?.options
    expect(opts3?.mcpServers).toBeDefined()
  })

  it("does not cache tools when not in passthrough mode", async () => {
    const app = createTestApp()

    // Set passthrough off
    const originalPassthrough = process.env.MERIDIAN_PASSTHROUGH
    process.env.MERIDIAN_PASSTHROUGH = "0"

    try {
      // Request with tools in non-passthrough mode
      await post(app, makeRequest({
        stream: false,
        tools: [TOOL_A],
        messages: [{ role: "user", content: "hello" }],
      }))

      capturedQueryParams = null

      // Request without tools — no cache should apply
      await post(app, makeRequest({
        stream: false,
        tools: [],
        messages: [
          { role: "user", content: "hello" },
          { role: "assistant", content: "Done." },
          { role: "user", content: "continue" },
        ],
      }))

      const opts2 = capturedQueryParams?.options
      expect(getPassthroughMcp(opts2)).toBeUndefined()
    } finally {
      if (originalPassthrough === undefined) delete process.env.MERIDIAN_PASSTHROUGH
      else process.env.MERIDIAN_PASSTHROUGH = originalPassthrough
    }
  })
})
