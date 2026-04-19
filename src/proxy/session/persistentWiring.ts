/**
 * Construction glue for persistent-mode runtimes.
 *
 * The `dispatchPersistentTurn` dispatcher is SDK-agnostic — it takes a
 * `CreateRuntimeFn` factory and calls it when it needs a new runtime.
 * This module owns that factory: it knows how to build the passthrough
 * MCP with deferred handlers, wire the PreToolUse hook to enqueue
 * tool_use_ids, start `query()` with a streaming-input queue, and wrap
 * everything into a `SessionRuntime`.
 *
 * Kept as its own module so `server.ts` only needs to construct a
 * `PersistentWiring` instance once and hand its `createRuntime`
 * function to the dispatcher. Unit-testable in isolation because the
 * `startQuery` and `createPassthroughMcp` callbacks are injected.
 */

import type { Query, SDKUserMessage, Options } from "@anthropic-ai/claude-agent-sdk"
import {
  createAsyncQueue,
  createSessionRuntime,
  type SessionRuntime,
  type ReopenCriticalOptions,
} from "./runtime"
import {
  attachDispatchState,
  type CreateRuntimeArgs,
  type CreateRuntimeFn,
} from "./persistentDispatch"
import {
  snapshotOptions,
  type InPlaceOptions,
} from "./optionsClassifier"

// --- Types -----------------------------------------------------------------

/**
 * Contract the caller (server.ts) fills in. These callbacks encapsulate the
 * SDK-aware construction steps so this module has no direct SDK dependency
 * beyond types — swap them in tests for the mock Query.
 */
export interface PersistentWiringDeps {
  /**
   * Start a live SDK `query()` with the given streaming-input queue and
   * options. Returns the resulting Query.
   */
  startQuery: (params: { inputQueue: AsyncIterable<SDKUserMessage>; options: Options }) => Query

  /**
   * Build SDK `Options` for a new query(). The caller already knows the
   * adapter-specific pieces (cwd, system prompt, MCP mappings, tool filters);
   * here we forward the persistent-mode-specific pieces:
   *
   *   resumeSessionId → passed through as `options.resume`
   *   forkSession / resumeSessionAt → undo/fork semantics (§D6)
   *   passthroughMcp → the deferred-handler MCP when applicable
   *   sdkHooks → the PreToolUse hook that enqueues tool_use_ids
   */
  buildOptions: (args: {
    reopenCritical: ReopenCriticalOptions
    inPlace: InPlaceOptions
    resumeSessionId?: string
    forkSession?: boolean
    resumeSessionAt?: string
    passthroughMcpBinding?: PassthroughMcpBinding
    sdkHooksBinding?: SdkHooksBinding
  }) => Options

  /**
   * Adapter-scoped passthrough toggle + tool set. When passthrough is on,
   * we construct the MCP with deferred handlers AND a PreToolUse hook that
   * populates the runtime's tool_use_id FIFO. When off, neither is wired.
   */
  getPassthroughSpec: () => PassthroughSpec | null
}

/** Late-bound reference to the runtime that the MCP handlers close over. */
export interface RuntimeRef {
  current: SessionRuntime | null
}

/** Output of the passthrough wiring step — a pair of bindings to pass into
 * `buildOptions` so the options builder can attach them to the SDK Options. */
export interface PassthroughMcpBinding {
  mcpServers: Options["mcpServers"]
  allowedTools: string[]
  hasDeferredTools: boolean
}

export interface SdkHooksBinding {
  hooks: Options["hooks"]
}

export interface PassthroughSpec {
  /** Tools the agent is advertising for this session. */
  tools: Array<{ name: string; description?: string; input_schema?: any; defer_loading?: boolean }>
  /** Adapter-provided core tool names (used for auto-defer heuristic). */
  coreToolNames?: readonly string[]
}

// --- Factory ---------------------------------------------------------------

/**
 * Build a `CreateRuntimeFn` that the dispatcher can call. Closed over the
 * injected wiring deps so server.ts constructs one of these once (when a
 * POST /v1/messages request is ready to dispatch) and hands it off.
 */
export function makePersistentCreateRuntime(deps: PersistentWiringDeps): CreateRuntimeFn {
  return async (args: CreateRuntimeArgs): Promise<SessionRuntime> => {
    const inputQueue = createAsyncQueue<SDKUserMessage>()

    // Late-bound reference — the MCP handlers + PreToolUse hook close over
    // this object; we populate `current` after constructing the runtime
    // below. Safe because the handlers fire only after the SDK sees the
    // first user message, which is pushed AFTER the runtime is created.
    const runtimeRef: RuntimeRef = { current: null }

    const passthroughSpec = deps.getPassthroughSpec()
    const passthroughMcpBinding = passthroughSpec
      ? buildPassthroughBinding(passthroughSpec, runtimeRef)
      : undefined
    const sdkHooksBinding = passthroughSpec
      ? buildHooksBinding(runtimeRef)
      : undefined

    const options = deps.buildOptions({
      reopenCritical: args.reopenCritical,
      inPlace: args.inPlace,
      resumeSessionId: args.resumeSessionId,
      forkSession: args.forkSession,
      resumeSessionAt: args.resumeSessionAt,
      passthroughMcpBinding,
      sdkHooksBinding,
    })

    const query = deps.startQuery({ inputQueue, options })

    const runtime = createSessionRuntime({
      profileSessionId: args.profileSessionId,
      optionsHash: "persistent", // dispatcher checks snapshot state, not this
      query,
      inputQueue,
    })

    // Bind the late reference so MCP handlers + hook can now reach runtime.
    runtimeRef.current = runtime

    // Attach the snapshot the dispatcher's drift detection consults.
    attachDispatchState(runtime, snapshotOptions(args.reopenCritical, args.inPlace))

    return runtime
  }
}

// --- Passthrough binding construction --------------------------------------

/**
 * Build the passthrough MCP + deferred handler binding. The MCP handlers
 * reach the runtime via `runtimeRef.current`, which the factory above
 * populates immediately after the runtime is constructed.
 */
export function buildPassthroughBinding(
  _spec: PassthroughSpec,
  _runtimeRef: RuntimeRef,
): PassthroughMcpBinding {
  // The actual MCP construction happens in the server wiring because it
  // needs the @anthropic-ai/claude-agent-sdk `createSdkMcpServer` which
  // pulls in real subprocess behaviour. This stub returns a placeholder
  // that server.ts will replace with `createPassthroughMcpServer(spec.tools,
  // spec.coreToolNames, { deferredMode: {...} })` when integrating.
  //
  // Leaving this as an extension point keeps this module test-friendly:
  // tests pass a no-op binding; production wires the real MCP here.
  throw new Error("buildPassthroughBinding: server.ts must override or supply a deferredMode-compatible MCP binding")
}

/**
 * Build the `sdkHooks.PreToolUse` binding that captures tool_use_ids into
 * the runtime's FIFO without blocking. Replaces the legacy
 * `{ decision: "block" }` path (design §D11).
 */
export function buildHooksBinding(runtimeRef: RuntimeRef): SdkHooksBinding {
  const hook = async (input: { tool_name?: string; tool_use_id?: string }) => {
    // ToolSearch is handled internally by the SDK for deferred-tool loading;
    // return `{}` so the SDK proceeds normally.
    if (input.tool_name === "ToolSearch") return {}

    // Capture the tool_use_id into the runtime's FIFO so the MCP handler
    // can correlate when it fires. If the runtime isn't populated yet
    // (shouldn't happen — the SDK fires hooks only after the first turn
    // starts), silently no-op.
    const runtime = runtimeRef.current
    if (runtime && typeof input.tool_use_id === "string" && typeof input.tool_name === "string") {
      const toolName = stripMcpPrefixLocal(input.tool_name)
      runtime.enqueueToolUseId(toolName, input.tool_use_id)
    }

    // Do NOT return `{ decision: "block" }` — that fabricates "blocked"
    // narrative in the SDK's conversation and poisons subsequent turns
    // (see §1d Scenario C). Returning `{}` lets the SDK proceed to the
    // MCP handler, which blocks on the deferred-handler promise.
    return {}
  }

  return {
    hooks: {
      PreToolUse: [{ matcher: "", hooks: [hook] }],
    } as unknown as Options["hooks"],
  }
}

// --- Helpers ---------------------------------------------------------------

/**
 * Local copy of `stripMcpPrefix` to avoid a circular import from
 * `passthroughTools.ts`. Kept in sync with that module's `PASSTHROUGH_MCP_PREFIX`.
 */
function stripMcpPrefixLocal(toolName: string): string {
  const PASSTHROUGH_MCP_PREFIX = "mcp__oc__"
  if (toolName.startsWith(PASSTHROUGH_MCP_PREFIX)) {
    return toolName.slice(PASSTHROUGH_MCP_PREFIX.length)
  }
  return toolName
}
