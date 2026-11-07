/**
 * Gemini's generateContent API, called from the browser with the user's key.
 * Model turns are replayed with their original parts, so thought signatures
 * survive.
 */
import { toOpenApiSubset } from '../harness/schema.ts';
import { httpError, ProviderError, type Completion, type CompletionRequest, type Provider, type StopReason, type Turn } from './types.ts';

export const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta';
export const GEMINI_DEFAULT_MODEL = 'gemini-2.5-flash';

export interface GeminiOptions {
  apiKey: string;
  model?: string;
  fetch?: typeof fetch;
}

type Part = Record<string, any>;

export function toGeminiContents(turns: Turn[], providerId: string): { role: string; parts: Part[] }[] {
  const out: { role: string; parts: Part[] }[] = [];
  const push = (role: string, parts: Part[]) => {
    const last = out[out.length - 1];
    if (last && last.role === role) last.parts.push(...parts);
    else out.push({ role, parts });
  };
  for (const turn of turns) {
    if (turn.role === 'user') push('user', [{ text: turn.text }]);
    else if (turn.role === 'tool') {
      push(
        'user',
        turn.results.map((r) => ({
          functionResponse: {
            name: r.name,
            ...(r.id.startsWith('gemini-') ? {} : { id: r.id }),
            response: r.isError ? { error: r.content } : { result: r.content },
          },
        }))
      );
    } else if (turn.raw?.provider === providerId) out.push({ role: 'model', parts: turn.raw.value as Part[] });
    else {
      const parts: Part[] = [];
      if (turn.text) parts.push({ text: turn.text });
      for (const c of turn.toolCalls) parts.push({ functionCall: { name: c.name, args: c.input ?? {} } });
      out.push({ role: 'model', parts: parts.length ? parts : [{ text: '(no content)' }] });
    }
  }
  return out;
}

export class GeminiProvider implements Provider {
  readonly id = 'gemini';
  readonly label: string;
  private readonly o: GeminiOptions;

  constructor(o: GeminiOptions) {
    this.o = o;
    this.label = `Gemini · ${o.model ?? GEMINI_DEFAULT_MODEL}`;
  }

  async complete(req: CompletionRequest): Promise<Completion> {
    const model = this.o.model ?? GEMINI_DEFAULT_MODEL;
    const body = {
      systemInstruction: { parts: [{ text: req.system }] },
      contents: toGeminiContents(req.turns, this.id),
      tools: [
        {
          functionDeclarations: req.tools.map((t) => {
            const params = toOpenApiSubset(t.input);
            const empty = !Object.keys((params.properties as object) ?? {}).length;
            return { name: t.name, description: t.description, ...(empty ? {} : { parameters: params }) };
          }),
        },
      ],
      toolConfig: { functionCallingConfig: { mode: 'AUTO' } },
    };
    const doFetch = this.o.fetch ?? fetch;
    let res: Response;
    try {
      res = await doFetch(`${GEMINI_BASE}/models/${encodeURIComponent(model)}:generateContent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': this.o.apiKey },
        body: JSON.stringify(body),
        signal: req.signal,
      });
    } catch (e) {
      if ((e as Error).name === 'AbortError') throw e;
      throw new ProviderError(`Could not reach the Gemini API: ${(e as Error).message}`, undefined, true);
    }
    const text = await res.text();
    if (!res.ok) throw httpError('Gemini', res.status, text);
    const json = JSON.parse(text);
    const candidate = json.candidates?.[0];
    if (!candidate) {
      const block = json.promptFeedback?.blockReason;
      return { text: '', toolCalls: [], stopReason: block ? 'refusal' : 'other', detail: block, raw: [] };
    }
    const parts: Part[] = candidate.content?.parts ?? [];
    const toolCalls = parts
      .filter((p) => p.functionCall)
      .map((p, i) => ({ id: p.functionCall.id ?? `gemini-${Date.now()}-${i}`, name: p.functionCall.name, input: p.functionCall.args ?? {} }));
    const finish = candidate.finishReason;
    const stopReason: StopReason =
      toolCalls.length ? 'tool_use'
      : finish === 'STOP' ? 'end'
      : finish === 'MAX_TOKENS' ? 'max_tokens'
      : finish === 'SAFETY' || finish === 'PROHIBITED_CONTENT' || finish === 'BLOCKLIST' ? 'refusal'
      : 'other';
    return {
      text: parts.filter((p) => typeof p.text === 'string' && !p.thought).map((p) => p.text).join(''),
      toolCalls,
      stopReason,
      detail: stopReason === 'refusal' || stopReason === 'other' ? finish : undefined,
      raw: parts,
      usage: json.usageMetadata ? { input: json.usageMetadata.promptTokenCount ?? 0, output: json.usageMetadata.candidatesTokenCount ?? 0 } : undefined,
      model: json.modelVersion,
    };
  }
}
