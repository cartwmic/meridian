/**
 * Leaf constants + helpers for passthrough MCP naming.
 *
 * Extracted so both `passthroughTools.ts` (which builds the MCP server) and
 * `session/persistentWiring.ts` (which runs the PreToolUse hook that parses
 * tool names into the runtime FIFO) can share a single source of truth
 * without importing each other. Keeping this module pure data + pure
 * functions means neither side risks a circular import.
 */

export const PASSTHROUGH_MCP_NAME = "oc"
export const PASSTHROUGH_MCP_PREFIX = `mcp__${PASSTHROUGH_MCP_NAME}__`

/**
 * Remove the passthrough MCP prefix from a tool name. Returns the original
 * name unchanged if the prefix isn't present.
 */
export function stripMcpPrefix(toolName: string): string {
  if (toolName.startsWith(PASSTHROUGH_MCP_PREFIX)) {
    return toolName.slice(PASSTHROUGH_MCP_PREFIX.length)
  }
  return toolName
}
