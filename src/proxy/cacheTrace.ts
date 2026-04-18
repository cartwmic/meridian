import { appendFileSync, mkdirSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { createHash } from "node:crypto"
import { env, envBool } from "../env"

export interface CacheTrace {
  enabled: boolean
  path?: string
  log: (event: string, data?: Record<string, unknown>) => void
}

export function createCacheTrace(requestId: string): CacheTrace {
  if (!envBool("TRACE_CACHE")) {
    return { enabled: false, log: () => {} }
  }

  const dir = env("TRACE_CACHE_DIR") || tmpdir()
  mkdirSync(dir, { recursive: true })
  const safeRequestId = requestId.replace(/[^a-zA-Z0-9._-]/g, "_")
  const path = join(dir, `meridian-cache-trace-${safeRequestId}.jsonl`)

  return {
    enabled: true,
    path,
    log(event: string, data: Record<string, unknown> = {}) {
      try {
        appendFileSync(path, `${JSON.stringify({
          ts: new Date().toISOString(),
          requestId,
          event,
          ...data,
        })}\n`)
      } catch (error) {
        console.error(`[PROXY] ${requestId} cache trace write failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    },
  }
}

export function summarizeMessages(messages: Array<{ role: string; content: unknown }>): Record<string, unknown> {
  const roleCounts: Record<string, number> = {}
  const blockTypeCounts: Record<string, number> = {}
  const toolNames: string[] = []
  let textChars = 0

  const sample = messages.slice(0, 8).map((message) => {
    roleCounts[message.role] = (roleCounts[message.role] || 0) + 1

    const types = new Set<string>()
    let sampleTextChars = 0

    if (typeof message.content === "string") {
      blockTypeCounts.string = (blockTypeCounts.string || 0) + 1
      types.add("string")
      textChars += message.content.length
      sampleTextChars += message.content.length
    } else if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!block || typeof block !== "object") {
          blockTypeCounts.unknown = (blockTypeCounts.unknown || 0) + 1
          types.add("unknown")
          continue
        }

        const typedBlock = block as { type?: unknown; text?: unknown; name?: unknown }
        const blockType = typeof typedBlock.type === "string" ? typedBlock.type : "unknown"
        blockTypeCounts[blockType] = (blockTypeCounts[blockType] || 0) + 1
        types.add(blockType)

        if (typeof typedBlock.text === "string") {
          textChars += typedBlock.text.length
          sampleTextChars += typedBlock.text.length
        }
        if (blockType === "tool_use" && typeof typedBlock.name === "string") {
          toolNames.push(typedBlock.name)
        }
      }
    } else if (message.content != null) {
      const value = String(message.content)
      blockTypeCounts.other = (blockTypeCounts.other || 0) + 1
      types.add("other")
      textChars += value.length
      sampleTextChars += value.length
    }

    return {
      role: message.role,
      types: [...types],
      textChars: sampleTextChars,
    }
  })

  return {
    count: messages.length,
    roleCounts,
    blockTypeCounts,
    textChars,
    toolNames: [...new Set(toolNames)].slice(0, 8),
    sample,
  }
}

export function summarizeQueryOptions(options: Record<string, unknown>): Record<string, unknown> {
  const allowedTools = Array.isArray(options.allowedTools) ? options.allowedTools : []
  const disallowedTools = Array.isArray(options.disallowedTools) ? options.disallowedTools : []
  const mcpServers = options.mcpServers && typeof options.mcpServers === "object"
    ? Object.keys(options.mcpServers as Record<string, unknown>)
    : []
  const thinking = options.thinking && typeof options.thinking === "object"
    ? options.thinking as { type?: unknown }
    : undefined
  const taskBudget = options.taskBudget && typeof options.taskBudget === "object"
    ? options.taskBudget as { total?: unknown }
    : undefined
  const systemPrompt = options.systemPrompt

  return {
    resume: typeof options.resume === "string" ? options.resume : undefined,
    forkSession: options.forkSession === true,
    resumeSessionAt: typeof options.resumeSessionAt === "string" ? options.resumeSessionAt : undefined,
    maxTurns: typeof options.maxTurns === "number" ? options.maxTurns : undefined,
    includePartialMessages: options.includePartialMessages === true,
    strictMcpConfig: options.strictMcpConfig === true,
    allowedToolCount: allowedTools.length,
    disallowedToolCount: disallowedTools.length,
    mcpServerNames: mcpServers,
    hasHooks: !!(options.hooks && typeof options.hooks === "object" && Object.keys(options.hooks as Record<string, unknown>).length > 0),
    thinkingType: typeof thinking?.type === "string" ? thinking.type : undefined,
    taskBudgetTotal: typeof taskBudget?.total === "number" ? taskBudget.total : undefined,
    betaCount: Array.isArray(options.betas) ? options.betas.length : 0,
    additionalDirectoriesCount: Array.isArray(options.additionalDirectories) ? options.additionalDirectories.length : 0,
    systemPrompt:
      typeof systemPrompt === "string"
        ? { kind: "text", chars: systemPrompt.length }
        : systemPrompt && typeof systemPrompt === "object"
          ? {
              kind: "preset",
              preset: typeof (systemPrompt as { preset?: unknown }).preset === "string"
                ? (systemPrompt as { preset?: string }).preset
                : undefined,
              hasAppend: typeof (systemPrompt as { append?: unknown }).append === "string",
            }
          : undefined,
  }
}

export function summarizeUsage(
  usage: {
    input_tokens?: number
    output_tokens?: number
    cache_read_input_tokens?: number
    cache_creation_input_tokens?: number
  } | undefined
): Record<string, unknown> | undefined {
  if (!usage) return undefined

  return {
    inputTokens: typeof usage.input_tokens === "number" ? usage.input_tokens : undefined,
    outputTokens: typeof usage.output_tokens === "number" ? usage.output_tokens : undefined,
    cacheReadInputTokens: typeof usage.cache_read_input_tokens === "number" ? usage.cache_read_input_tokens : undefined,
    cacheCreationInputTokens: typeof usage.cache_creation_input_tokens === "number" ? usage.cache_creation_input_tokens : undefined,
  }
}

export function digestForTrace(value: string): string {
  return createHash("sha256").update(value).digest("hex")
}

export function previewForTrace(value: string, maxChars = 160): string {
  return value.length <= maxChars ? value : `${value.slice(0, maxChars)}…`
}
