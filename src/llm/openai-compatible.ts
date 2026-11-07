/**
 * The OpenAI chat-completions dialect, which OpenRouter and llama.cpp's
 * llama-server both speak.
 *
 * llama-server needs `--jinja` for tool calls, and a page served over HTTPS
 * may only reach http://localhost where the browser allows it (Chrome asks
 * for local-network permission; Safari blocks it, so run the site locally).
 */
import { httpError, parseArguments, ProviderError, type Completion, type CompletionRequest, type Provider, type StopReason, type Turn } from './types.ts';

export interface OpenAiCompatibleOptions {
  id: 'openrouter' | 'llamacpp';
  label: string;
  baseUrl: string;
  apiKey?: string;
  model: string;
  maxTokens?: number;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
}

export const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';
export const LLAMACPP_BASE = 'http://localhost:8080/v1';

type Msg = Record<string, unknown>;

export function toOpenAiMessages(system: string, turns: Turn[], providerId: string): Msg[] {
  const out: Msg[] = [{ role: 'system', content: system }];
  for (const turn of turns) {
    if (turn.role === 'user') out.push({ role: 'user', content: turn.text });
    else if (turn.role === 'tool') {
      for (const r of turn.results) out.push({ role: 'tool', tool_call_id: r.id, content: r.isError ? `ERROR: ${r.content}` : r.content });
    } else if (turn.raw?.provider === providerId) out.push(turn.raw.value as Msg);
    else {
      out.push({
        role: 'assistant',
        content: turn.text || null,
        ...(turn.toolCalls.length
          ? { tool_calls: turn.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.input ?? {}) } })) }
          : {}),
      });
    }
  }
  return out;
}

export class OpenAiCompatibleProvider implements Provider {
  readonly id: string;
  readonly label: string;
  private readonly o: OpenAiCompatibleOptions;

  constructor(o: OpenAiCompatibleOptions) {
    this.o = o;
    this.id = o.id;
    this.label = `${o.label} · ${o.model}`;
  }

  async complete(req: CompletionRequest): Promise<Completion> {
    const body = {
      model: this.o.model,
      max_tokens: this.o.maxTokens ?? 8192,
      messages: toOpenAiMessages(req.system, req.turns, this.id),
      tools: req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.input } })),
      tool_choice: 'auto',
    };
    const doFetch = this.o.fetch ?? fetch;
    let res: Response;
    try {
      res = await doFetch(`${this.o.baseUrl.replace(/\/$/, '')}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.o.apiKey ? { Authorization: `Bearer ${this.o.apiKey}` } : {}),
          ...this.o.headers,
        },
        body: JSON.stringify(body),
        signal: req.signal,
      });
    } catch (e) {
      if ((e as Error).name === 'AbortError') throw e;
      const local = this.o.id === 'llamacpp';
      throw new ProviderError(
        `Could not reach ${this.o.baseUrl}: ${(e as Error).message}.${
          local ? ' Is llama-server running (with --jinja)? From an HTTPS page, the browser must also allow local-network access.' : ''
        }`,
        undefined,
        !local
      );
    }
    const text = await res.text();
    if (!res.ok) throw httpError(this.o.label, res.status, text);
    let json: any;
    try {
      json = JSON.parse(text);
    } catch {
      throw new ProviderError(`${this.o.label} sent a response that is not JSON`);
    }
    if (json.error) throw new ProviderError(`${this.o.label}: ${json.error.message ?? JSON.stringify(json.error)}`);
    const choice = json.choices?.[0];
    if (!choice) throw new ProviderError(`${this.o.label} returned no choices`);
    const message = choice.message ?? {};
    const toolCalls = (message.tool_calls ?? []).map((c: any, i: number) => {
      const { input, parseError } = parseArguments(c.function?.arguments);
      return { id: c.id || `call_${Date.now()}_${i}`, name: c.function?.name ?? '', input, parseError };
    });
    // Replay exactly what came back, minus fields some servers reject on input.
    const raw: Msg = { role: 'assistant', content: message.content ?? null };
    if (message.tool_calls?.length) raw.tool_calls = message.tool_calls.map((c: any, i: number) => ({ ...c, id: toolCalls[i].id }));
    const finish = choice.finish_reason;
    const stopReason: StopReason =
      toolCalls.length ? 'tool_use' : finish === 'length' ? 'max_tokens' : finish === 'content_filter' ? 'refusal' : finish === 'stop' ? 'end' : 'other';
    return {
      text: typeof message.content === 'string' ? message.content : '',
      toolCalls,
      stopReason,
      raw,
      usage: json.usage ? { input: json.usage.prompt_tokens ?? 0, output: json.usage.completion_tokens ?? 0 } : undefined,
      model: json.model,
    };
  }
}

export interface OpenRouterModel {
  id: string;
  name: string;
  contextLength: number;
  free: boolean;
}

/** OpenRouter's catalogue, narrowed to models that accept tools. */
export async function listOpenRouterModels(doFetch: typeof fetch = fetch): Promise<OpenRouterModel[]> {
  const res = await doFetch(`${OPENROUTER_BASE}/models?supported_parameters=tools`);
  if (!res.ok) throw httpError('OpenRouter', res.status, await res.text());
  const json = await res.json();
  return (json.data ?? [])
    .filter((m: any) => !m.supported_parameters || m.supported_parameters.includes('tools'))
    .map((m: any) => ({
      id: m.id,
      name: m.name ?? m.id,
      contextLength: m.context_length ?? 0,
      free: Number(m.pricing?.prompt ?? 1) === 0 && Number(m.pricing?.completion ?? 1) === 0,
    }))
    .sort((a: OpenRouterModel, b: OpenRouterModel) => Number(b.free) - Number(a.free) || a.id.localeCompare(b.id));
}
