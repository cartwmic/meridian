import { describe, expect, it } from "bun:test"
import type { Options, Query } from "@anthropic-ai/claude-agent-sdk"

import {
  buildHooksBinding,
  makePersistentCreateRuntime,
  type PersistentWiringDeps,
  type RuntimeRef,
} from "../proxy/session/persistentWiring"
import { getDispatchState } from "../proxy/session/persistentDispatch"
import { createMockQuery } from "./helpers/mockQuery"
import type { ReopenCriticalOptions } from "../proxy/session/runtime"
import type { InPlaceOptions } from "../proxy/session/optionsClassifier"

const baseReopen: ReopenCriticalOptions = { cwd: "/p", systemPrompt: "sp" }
const baseInPlace: InPlaceOptions = { model: "claude-sonnet-4-5" }

describe("buildHooksBinding", () => {
  it("returns {} for ToolSearch calls so the SDK proceeds", async () => {
    const runtimeRef: RuntimeRef = { current: null }
    const binding = buildHooksBinding(runtimeRef)
    const hooks = (binding.hooks as any).PreToolUse[0].hooks
    const result = await hooks[0]({ tool_name: "ToolSearch", tool_use_id: "toolu_x" })
    expect(result).toEqual({})
    // No runtime yet, so no enqueue; nothing crashes.
  })

  it("enqueues tool_use_id into the runtime's FIFO and returns {} (no block)", async () => {
    const { query } = createMockQuery({ turns: [{ events: [] as never[] }] })
    // Minimal runtime stub that just exposes enqueueToolUseId.
    const enqueued: Array<{ toolName: string; toolUseId: string }> = []
    const runtime = {
      enqueueToolUseId: (toolName: string, toolUseId: string) => enqueued.push({ toolName, toolUseId }),
    } as any
    query.close() // keep the mock clean
    const runtimeRef: RuntimeRef = { current: runtime }
    const binding = buildHooksBinding(runtimeRef)
    const hooks = (binding.hooks as any).PreToolUse[0].hooks

    const result = await hooks[0]({ tool_name: "mcp__oc__read", tool_use_id: "toolu_a" })

    expect(result).toEqual({})
    expect(enqueued).toEqual([{ toolName: "read", toolUseId: "toolu_a" }])
  })

  it("strips the mcp__oc__ prefix before enqueueing", async () => {
    const enqueued: Array<{ toolName: string }> = []
    const runtime = { enqueueToolUseId: (toolName: string) => enqueued.push({ toolName }) } as any
    const runtimeRef: RuntimeRef = { current: runtime }
    const binding = buildHooksBinding(runtimeRef)
    const hooks = (binding.hooks as any).PreToolUse[0].hooks

    await hooks[0]({ tool_name: "mcp__oc__write", tool_use_id: "toolu_1" })
    await hooks[0]({ tool_name: "plain_tool_name", tool_use_id: "toolu_2" })

    expect(enqueued.map((x) => x.toolName)).toEqual(["write", "plain_tool_name"])
  })

  it("does nothing when runtimeRef.current is null (before late binding resolves)", async () => {
    const runtimeRef: RuntimeRef = { current: null }
    const binding = buildHooksBinding(runtimeRef)
    const hooks = (binding.hooks as any).PreToolUse[0].hooks
    const result = await hooks[0]({ tool_name: "mcp__oc__read", tool_use_id: "toolu_x" })
    expect(result).toEqual({})
  })
})

describe("makePersistentCreateRuntime", () => {
  it("constructs a runtime with options snapshot and returns it", async () => {
    const { query } = createMockQuery({ turns: [{ events: [] as never[] }] })
    const startQueryCalls: Array<Options> = []
    const deps: PersistentWiringDeps = {
      startQuery: ({ options }) => {
        startQueryCalls.push(options)
        return query as Query
      },
      buildOptions: (args) => ({
        executable: "node",
        model: args.inPlace.model,
        ...(args.resumeSessionId ? { resume: args.resumeSessionId } : {}),
        ...(args.forkSession ? { forkSession: true, resumeSessionAt: args.resumeSessionAt } : {}),
      } as unknown as Options),
      getPassthroughSpec: () => null, // non-passthrough for this test
    }
    const createRuntime = makePersistentCreateRuntime(deps)

    const runtime = await createRuntime({
      profileSessionId: "session-A",
      reopenCritical: baseReopen,
      inPlace: baseInPlace,
    })

    expect(runtime.profileSessionId).toBe("session-A")
    expect(runtime.closed).toBe(false)
    expect(startQueryCalls).toHaveLength(1)
    expect((startQueryCalls[0] as any).model).toBe("claude-sonnet-4-5")

    // The dispatch state should be attached so drift detection works on the next turn.
    const state = getDispatchState(runtime)
    expect(state).toBeDefined()
    expect(state!.snapshot.reopenCriticalHash).toMatch(/^[0-9a-f]{16}$/)
  })

  it("forwards resumeSessionId / forkSession / resumeSessionAt into the options builder", async () => {
    const { query } = createMockQuery({ turns: [{ events: [] as never[] }] })
    let capturedArgs: any = null
    const deps: PersistentWiringDeps = {
      startQuery: () => query as Query,
      buildOptions: (args) => {
        capturedArgs = args
        return {} as Options
      },
      getPassthroughSpec: () => null,
    }
    const createRuntime = makePersistentCreateRuntime(deps)
    await createRuntime({
      profileSessionId: "session-A",
      reopenCritical: baseReopen,
      inPlace: baseInPlace,
      resumeSessionId: "sdk-sess-123",
      forkSession: true,
      resumeSessionAt: "uuid-rollback",
    })

    expect(capturedArgs.resumeSessionId).toBe("sdk-sess-123")
    expect(capturedArgs.forkSession).toBe(true)
    expect(capturedArgs.resumeSessionAt).toBe("uuid-rollback")
  })

  it("binds the PreToolUse hook so it can reach the runtime after construction (late-binding)", async () => {
    // This test verifies that when passthrough is on, the hook binding is
    // produced by buildHooksBinding (which closes over runtimeRef). We
    // can't trigger buildPassthroughBinding in the unit test because that
    // path requires real MCP construction (server.ts overrides it). So we
    // focus on the hook half.
    const runtimeRef: RuntimeRef = { current: null }
    const binding = buildHooksBinding(runtimeRef)

    // Simulate the factory populating runtimeRef after runtime construction.
    const enqueued: string[] = []
    runtimeRef.current = { enqueueToolUseId: (_n: string, id: string) => enqueued.push(id) } as any

    // Now invoke the hook — it should reach the runtime via the late reference.
    const hooks = (binding.hooks as any).PreToolUse[0].hooks
    await hooks[0]({ tool_name: "mcp__oc__read", tool_use_id: "toolu_late" })
    expect(enqueued).toEqual(["toolu_late"])
  })
})
