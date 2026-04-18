/**
 * Regression tests for passthrough MCP session caching.
 *
 * Desired contract:
 * - Headered passthrough sessions detect same-tool resumed turns via the
 *   session cache, but still create a fresh synthetic passthrough MCP server
 *   object for each request.
 * - Pi/headerless passthrough sessions that resume via fingerprint cache also
 *   detect same-tool resumed turns while creating a fresh MCP server object
 *   per request.
 * - Changing the tool set still reports recreation semantics.
 * - Requests with no passthrough tools do not create or attach the synthetic
 *   passthrough MCP server.
 */

import { describe, it, expect, mock, beforeEach, afterEach } from "bun:test"

let capturedQueryCalls: any[] = []
let passthroughServerCreateCount = 0
const MOCK_SDK_SESSION = "sdk-session-passthrough-cache"

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
          usage: { input_tokens: 10, output_tokens: 5 },
        },
        session_id: MOCK_SDK_SESSION,
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
    max_tokens: 1024,
    stream: false,
    messages: [{ role: "user", content: "hello" }],
    ...overrides,
  }
}

async function post(app: any, body: any, headers: Record<string, string> = {}) {
  return app.fetch(new Request("http://localhost/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...headers,
    },
    body: JSON.stringify(body),
  }))
}

function getPassthroughServer(callIndex: number) {
  return capturedQueryCalls[callIndex]?.options?.mcpServers?.oc
}

describe("Passthrough MCP session cache", () => {
  const originalPassthrough = process.env.MERIDIAN_PASSTHROUGH

  beforeEach(() => {
    clearSessionCache()
    capturedQueryCalls = []
    passthroughServerCreateCount = 0
    process.env.MERIDIAN_PASSTHROUGH = "1"
  })

  afterEach(() => {
    if (originalPassthrough === undefined) delete process.env.MERIDIAN_PASSTHROUGH
    else process.env.MERIDIAN_PASSTHROUGH = originalPassthrough
  })

  it("creates a fresh passthrough MCP server per headered resumed turn even when the tool set matches", async () => {
    const app = createTestApp()
    const headers = { "x-opencode-session": "headered-cache-reuse" }

    await (await post(app, createBody({
      tools: [TOOL_READ, TOOL_WRITE],
    }), headers)).json()

    await (await post(app, createBody({
      tools: [TOOL_READ, TOOL_WRITE],
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: [{ type: "text", text: "ok" }] },
        { role: "user", content: "continue" },
      ],
    }), headers)).json()

    expect(capturedQueryCalls).toHaveLength(2)
    expect(capturedQueryCalls[1]?.options?.resume).toBe(MOCK_SDK_SESSION)
    expect(getPassthroughServer(0)).toBeDefined()
    expect(getPassthroughServer(1)).toBeDefined()
    expect(getPassthroughServer(1)).not.toBe(getPassthroughServer(0))
    expect(passthroughServerCreateCount).toBe(2)
  })

  it("creates a fresh passthrough MCP server per Pi/headerless resumed turn even when the tool set matches", async () => {
    const app = createTestApp()
    const headers = { "x-meridian-agent": "pi" }
    const system = "Current working directory: /Users/cartwmic/git/meridian"

    await (await post(app, createBody({
      system,
      tools: [TOOL_READ, TOOL_WRITE],
    }), headers)).json()

    await (await post(app, createBody({
      system,
      tools: [TOOL_READ, TOOL_WRITE],
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: [{ type: "text", text: "ok" }] },
        { role: "user", content: "continue" },
      ],
    }), headers)).json()

    expect(capturedQueryCalls).toHaveLength(2)
    expect(capturedQueryCalls[1]?.options?.resume).toBe(MOCK_SDK_SESSION)
    expect(getPassthroughServer(0)).toBeDefined()
    expect(getPassthroughServer(1)).toBeDefined()
    expect(getPassthroughServer(1)).not.toBe(getPassthroughServer(0))
    expect(passthroughServerCreateCount).toBe(2)
  })

  it("restores tools but still creates a fresh passthrough MCP server for resumed Pi/headerless turns when the client omits tools", async () => {
    const app = createTestApp()
    const headers = { "x-meridian-agent": "pi" }
    const system = "Current working directory: /Users/cartwmic/git/meridian"

    await (await post(app, createBody({
      system,
      tools: [TOOL_READ, TOOL_WRITE],
    }), headers)).json()

    await (await post(app, createBody({
      system,
      tools: [],
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: [{ type: "text", text: "ok" }] },
        { role: "user", content: "continue" },
      ],
    }), headers)).json()

    expect(capturedQueryCalls).toHaveLength(2)
    expect(capturedQueryCalls[1]?.options?.resume).toBe(MOCK_SDK_SESSION)
    expect(getPassthroughServer(0)).toBeDefined()
    expect(getPassthroughServer(1)).toBeDefined()
    expect(getPassthroughServer(1)).not.toBe(getPassthroughServer(0))
    expect(passthroughServerCreateCount).toBe(2)
  })

  it("recreates the passthrough MCP server for headered sessions when the tool set changes", async () => {
    const app = createTestApp()
    const headers = { "x-opencode-session": "headered-tools-changed" }

    await (await post(app, createBody({
      tools: [TOOL_READ],
    }), headers)).json()

    await (await post(app, createBody({
      tools: [TOOL_READ, TOOL_WRITE],
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: [{ type: "text", text: "ok" }] },
        { role: "user", content: "continue" },
      ],
    }), headers)).json()

    expect(capturedQueryCalls).toHaveLength(2)
    expect(capturedQueryCalls[1]?.options?.resume).toBe(MOCK_SDK_SESSION)
    expect(getPassthroughServer(0)).toBeDefined()
    expect(getPassthroughServer(1)).toBeDefined()
    expect(getPassthroughServer(1)).not.toBe(getPassthroughServer(0))
    expect(passthroughServerCreateCount).toBe(2)
  })

  it("does not create or attach a passthrough MCP server when no passthrough tools are present", async () => {
    const app = createTestApp()
    const headers = { "x-opencode-session": "headered-no-tools" }

    await (await post(app, createBody({
      tools: [],
    }), headers)).json()

    await (await post(app, createBody({
      tools: [],
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: [{ type: "text", text: "ok" }] },
        { role: "user", content: "continue" },
      ],
    }), headers)).json()

    expect(capturedQueryCalls).toHaveLength(2)
    expect(getPassthroughServer(0)).toBeUndefined()
    expect(getPassthroughServer(1)).toBeUndefined()
    expect(passthroughServerCreateCount).toBe(0)
  })
})
