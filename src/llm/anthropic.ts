/**
 * Claude through the official SDK, called straight from the browser with the
 * user's own key (dangerouslyAllowBrowser: the key never leaves this tab
 * except to api.anthropic.com).
 */
import Anthropic from '@anthropic-ai/sdk';
import { ProviderError, type Completion, type CompletionRequest, type Provider, type StopReason, type Turn } from './types.ts';

export const ANTHROPIC_DEFAULT_MODEL = 'claude-opus-5-5';

export type Effort = 'low' | 'medium' | 'high' | 'xhigh' | 'max';

export interface AnthropicOptions {
  apiKey: string;
  model?: string;
  effort?: Effort;
  maxTokens?: number;
  /** For tests: a fetch to route requests through. */
  fetch?: typeof fetch;
  baseURL?: string;
}

/** Models that take the server-side refusal fallback (`fallbacks: "default"`). */
const FALLBACK_MODELS = /^claude-(fable-5-1|opus-5-5|opus-5|sonnet-5-5)$/;
/** Models that accept output_config.effort. */
const EFFORT_MODELS = /^claude-(fable|mythos|opus|sonnet|haiku)-5|^claude-opus-4-[5-9]|^claude-sonnet-4-6/;

type Block = Anthropic.Beta.BetaContentBlockParam;

export function toAnthropicMessages(turns: Turn[], providerId: string): Anthropic.Beta.BetaMessageParam[] {
  const out: Anthropic.Beta.BetaMessageParam[] = [];
  const push = (role: 'user' | 'assistant', blocks: Block[]) => {
    const last = out[out.length - 1];
    // Tool results and the next user message share one user turn.
    if (last && last.role === role && role === 'user' && Array.isArray(last.content)) {
      (last.content as Block[]).push(...blocks);
    } else out.push({ role, content: blocks });
  };
  for (const turn of turns) {
    if (turn.role === 'user') push('user', [{ type: 'text', text: turn.text }]);
    else if (turn.role === 'tool') {
      push(
        'user',
        turn.results.map((r) => ({ type: 'tool_result', tool_use_id: r.id, content: r.content, is_error: r.isError || undefined }))
      );
    } else if (turn.raw?.provider === providerId) {
      out.push({ role: 'assistant', content: turn.raw.value as Block[] });
    } else {
      const blocks: Block[] = [];
      if (turn.text) blocks.push({ type: 'text', text: turn.text });
      for (const c of turn.toolCalls) blocks.push({ type: 'tool_use', id: c.id, name: c.name, input: (c.input ?? {}) as Record<string, unknown> });
      out.push({ role: 'assistant', content: blocks.length ? blocks : [{ type: 'text', text: '(no content)' }] });
    }
  }
  return out;
}

export class AnthropicProvider implements Provider {
  readonly id = 'anthropic';
  readonly label: string;
  private readonly client: Anthropic;
  private readonly o: AnthropicOptions;

  constructor(o: AnthropicOptions) {
    this.o = o;
    this.label = `Anthropic · ${o.model ?? ANTHROPIC_DEFAULT_MODEL}`;
    this.client = new Anthropic({
      apiKey: o.apiKey,
      dangerouslyAllowBrowser: true,
      maxRetries: 2,
      ...(o.fetch ? { fetch: o.fetch } : {}),
      ...(o.baseURL ? { baseURL: o.baseURL } : {}),
    });
  }

  async complete(req: CompletionRequest): Promise<Completion> {
    const model = this.o.model ?? ANTHROPIC_DEFAULT_MODEL;
    const fallbacks = FALLBACK_MODELS.test(model);
    const params: Anthropic.Beta.MessageCreateParamsNonStreaming = {
      model,
      max_tokens: this.o.maxTokens ?? 16000,
      system: req.system,
      tools: req.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.input as Anthropic.Beta.BetaTool.InputSchema,
        strict: true,
      })),
      messages: toAnthropicMessages(req.turns, this.id),
      ...(EFFORT_MODELS.test(model) ? { output_config: { effort: this.o.effort ?? 'high' } } : {}),
      // Route refusals to a fallback model on the server instead of failing.
      ...(fallbacks ? { betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default' } : {}),
    };
    let response: Anthropic.Beta.BetaMessage;
    try {
      response = await this.client.beta.messages.create(params, { signal: req.signal });
    } catch (e) {
      throw describeError(e);
    }
    const text = response.content
      .filter((b): b is Anthropic.Beta.BetaTextBlock => b.type === 'text')
      .map((b) => b.text)
      .join('\n');
    const toolCalls = response.content
      .filter((b): b is Anthropic.Beta.BetaToolUseBlock => b.type === 'tool_use')
      .map((b) => ({ id: b.id, name: b.name, input: b.input }));
    const stop: StopReason =
      response.stop_reason === 'tool_use' ? 'tool_use'
      : response.stop_reason === 'end_turn' || response.stop_reason === 'stop_sequence' ? 'end'
      : response.stop_reason === 'max_tokens' ? 'max_tokens'
      : response.stop_reason === 'refusal' ? 'refusal'
      : 'other';
    const details = (response as { stop_details?: { category?: string | null; explanation?: string | null } | null }).stop_details;
    return {
      text,
      toolCalls,
      stopReason: stop,
      detail: stop === 'refusal' ? [details?.category, details?.explanation].filter(Boolean).join(': ') || undefined : undefined,
      raw: response.content,
      usage: { input: response.usage.input_tokens, output: response.usage.output_tokens },
      model: response.model,
    };
  }
}

function describeError(e: unknown): Error {
  if (e instanceof Anthropic.APIUserAbortError) return e;
  if (e instanceof Anthropic.AuthenticationError) return new ProviderError('Anthropic rejected the API key (401). Check it in Settings.', 401);
  if (e instanceof Anthropic.PermissionDeniedError) return new ProviderError(`Anthropic: permission denied (403): ${e.message}`, 403);
  if (e instanceof Anthropic.NotFoundError) return new ProviderError(`Anthropic: not found (404) — check the model name. ${e.message}`, 404);
  if (e instanceof Anthropic.RateLimitError) return new ProviderError('Anthropic: rate limited (429). Wait a moment and retry.', 429, true);
  if (e instanceof Anthropic.BadRequestError) return new ProviderError(`Anthropic: bad request (400): ${e.message}`, 400);
  if (e instanceof Anthropic.APIConnectionError) return new ProviderError(`Could not reach api.anthropic.com: ${e.message}`, undefined, true);
  if (e instanceof Anthropic.APIError) return new ProviderError(`Anthropic error ${e.status ?? ''}: ${e.message}`, e.status, (e.status ?? 0) >= 500);
  return e instanceof Error ? e : new Error(String(e));
}
