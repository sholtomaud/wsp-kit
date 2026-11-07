/**
 * A provider-neutral conversation. Each provider adapter translates it to its
 * own wire format, and keeps its own native assistant content (`raw`) so it
 * can be replayed unchanged: Anthropic thinking blocks and Gemini thought
 * signatures must come back exactly as they were sent.
 */
import type { Schema } from '../harness/schema.ts';

export interface ToolCall {
  id: string;
  name: string;
  input: unknown;
  /** Set when the provider sent arguments that are not valid JSON. */
  parseError?: string;
}

export interface ToolResult {
  id: string;
  name: string;
  content: string;
  isError: boolean;
}

export type Turn =
  | { role: 'user'; text: string }
  | { role: 'assistant'; text: string; toolCalls: ToolCall[]; raw?: { provider: string; value: unknown } }
  | { role: 'tool'; results: ToolResult[] };

export interface ToolSpec {
  name: string;
  description: string;
  input: Schema;
}

export interface CompletionRequest {
  system: string;
  tools: ToolSpec[];
  turns: Turn[];
  signal?: AbortSignal;
}

export type StopReason = 'tool_use' | 'end' | 'max_tokens' | 'refusal' | 'other';

export interface Completion {
  text: string;
  toolCalls: ToolCall[];
  stopReason: StopReason;
  /** Why the provider stopped, in its own words (refusal explanations, safety blocks). */
  detail?: string;
  raw: unknown;
  usage?: { input: number; output: number };
  /** The model that actually answered (it can differ after a fallback). */
  model?: string;
}

export interface Provider {
  /** Distinguishes whose raw content a turn carries. */
  readonly id: string;
  readonly label: string;
  complete(request: CompletionRequest): Promise<Completion>;
}

export class ProviderError extends Error {
  readonly status?: number;
  readonly retryable: boolean;
  constructor(message: string, status?: number, retryable = false) {
    super(message);
    this.name = 'ProviderError';
    this.status = status;
    this.retryable = retryable;
  }
}

/** Turns an HTTP failure into a message a person can act on. */
export function httpError(provider: string, status: number, body: string): ProviderError {
  let detail = body.slice(0, 400);
  try {
    const j = JSON.parse(body);
    detail = j.error?.message ?? j.message ?? j.error ?? detail;
    if (typeof detail !== 'string') detail = JSON.stringify(detail);
  } catch {
    /* not JSON */
  }
  const hint =
    status === 401 || status === 403 ? ' Check the API key.'
    : status === 404 ? ' Check the model name and base URL.'
    : status === 429 ? ' Rate limited; wait and retry.'
    : '';
  detail = String(detail).trim().replace(/[.\s]*$/, '.');
  return new ProviderError(`${provider} returned ${status}: ${detail}${hint}`, status, status === 429 || status >= 500);
}

export function parseArguments(text: unknown): { input: unknown; parseError?: string } {
  if (typeof text !== 'string') return { input: text ?? {} };
  if (!text.trim()) return { input: {} };
  try {
    return { input: JSON.parse(text) };
  } catch (e) {
    return { input: {}, parseError: `Arguments are not valid JSON: ${(e as Error).message}` };
  }
}
