import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Harness, type AgentEvent } from '../src/harness/agent.ts';
import { S, ToolInputError, validate } from '../src/harness/schema.ts';
import { TOOLS } from '../src/harness/tools.ts';
import { AnthropicProvider, toAnthropicMessages } from '../src/llm/anthropic.ts';
import { GeminiProvider, toGeminiContents } from '../src/llm/gemini.ts';
import { OpenAiCompatibleProvider } from '../src/llm/openai-compatible.ts';
import type { Completion, CompletionRequest, Provider, Turn } from '../src/llm/types.ts';
import { makeWsp } from './helpers.ts';

/** A provider that replays a script of completions and records each request. */
function scripted(steps: Partial<Completion>[]): Provider & { requests: CompletionRequest[] } {
  const requests: CompletionRequest[] = [];
  let i = 0;
  return {
    id: 'scripted',
    label: 'scripted',
    requests,
    async complete(req) {
      requests.push(structuredClone({ ...req, signal: undefined }));
      const s = steps[i++] ?? { text: 'done' };
      return { text: '', toolCalls: [], stopReason: s.toolCalls?.length ? 'tool_use' : 'end', raw: null, ...s };
    },
  };
}

test('every tool schema is strict-mode ready', () => {
  const check = (s: any, path: string) => {
    if (s.type === 'object') {
      assert.equal(s.additionalProperties, false, `${path} needs additionalProperties: false`);
      for (const r of s.required ?? []) assert.ok(r in s.properties, `${path}.${r} is required but not defined`);
      for (const [k, v] of Object.entries(s.properties)) check(v, `${path}.${k}`);
    }
    if (s.type === 'array') check(s.items, `${path}[]`);
    for (const banned of ['minimum', 'maximum', 'minLength', 'maxItems', 'pattern']) assert.ok(!(banned in s), `${path} uses ${banned}`);
  };
  for (const t of TOOLS) {
    assert.match(t.name, /^[a-z_]{1,64}$/);
    check(t.input, t.name);
  }
});

test('the validator coerces sloppy inputs and rejects wrong ones', () => {
  const schema = S.object({ n: S.integer(), flag: S.boolean(), mode: S.string(undefined, ['a', 'b']), list: S.array(S.number()) }, ['n']);
  assert.deepEqual(validate(schema, { n: '3', flag: 'true', list: ['1.5', 2] }), { n: 3, flag: true, list: [1.5, 2] });
  assert.throws(() => validate(schema, {}), ToolInputError);
  assert.throws(() => validate(schema, { n: 1.5 }), /integer/);
  assert.throws(() => validate(schema, { n: 1, mode: 'c' }), /one of/);
  assert.throws(() => validate(schema, { n: 1, extra: 1 }), /not a known parameter/);
});

test('the loop runs tool calls, checkpoints before mutations, and stops when the model does', async () => {
  const wsp = makeWsp();
  const provider = scripted([
    {
      text: 'Looking first.',
      toolCalls: [
        { id: 't1', name: 'model_overview', input: {} },
        { id: 't2', name: 'eval_run', input: { prompt: 'The cat', expected_output: ' sat', max_tokens: 3 } },
      ],
    },
    { toolCalls: [{ id: 't3', name: 'mutate_patch_sparse', input: { target: 'norm', edits: [{ row: 0, col: 0, value: 2 }] } }] },
    { toolCalls: [{ id: 't4', name: 'state_restore', input: { id: 'auto-1' } }] },
    { text: 'Rolled back; nothing helped.' },
  ]);
  const events: AgentEvent[] = [];
  const harness = new Harness({ wsp, provider: () => provider, onEvent: (e) => events.push(e) });
  const before = await wsp.backend.read('norm');
  const final = await harness.send('Make it better');
  assert.equal(final?.text, 'Rolled back; nothing helped.');
  assert.equal(provider.requests.length, 4);
  assert.equal(provider.requests[0].tools.length, TOOLS.length);
  const results = events.filter((e): e is Extract<AgentEvent, { type: 'tool_result' }> => e.type === 'tool_result');
  assert.deepEqual(results.map((r) => r.result.isError), [false, false, false, false]);
  assert.equal(results[2].checkpoint, 'auto-1');
  assert.match(results[2].result.content, /state_restore\(\\"auto-1\\"\) undoes this/);
  assert.deepEqual([...(await wsp.backend.read('norm'))], [...before], 'the restore undid the patch');
  // The model's second request carries the first round of results.
  const tool = provider.requests[1].turns[2];
  assert.equal(tool.role, 'tool');
  assert.equal(tool.role === 'tool' && tool.results.length, 2);
});

test('bad tool calls come back as errors the model can correct', async () => {
  const wsp = makeWsp();
  const provider = scripted([
    {
      toolCalls: [
        { id: 'a', name: 'inspect_summary', input: {} },
        { id: 'b', name: 'no_such_tool', input: {} },
        { id: 'c', name: 'inspect_slice', input: {}, parseError: 'Arguments are not valid JSON' },
        { id: 'd', name: 'inspect_summary', input: { target: 'layer_9' } },
      ],
    },
    { text: 'ok' },
  ]);
  const harness = new Harness({ wsp, provider: () => provider });
  await harness.send('inspect');
  const results = harness.turns[2];
  assert.equal(results.role, 'tool');
  if (results.role !== 'tool') return;
  assert.ok(results.results.every((r) => r.isError));
  assert.match(results.results[0].content, /target is required/);
  assert.match(results.results[1].content, /Unknown tool/);
  assert.match(results.results[2].content, /not valid JSON/);
  assert.match(results.results[3].content, /No layer/);
});

test('auto checkpoints rotate', async () => {
  const wsp = makeWsp();
  const patch = (id: string) => ({ id, name: 'mutate_patch_sparse', input: { target: 'norm', edits: [{ row: 0, col: 1, value: 1 }] } });
  const provider = scripted([{ toolCalls: [patch('1'), patch('2'), patch('3'), patch('4')] }, { text: 'done' }]);
  await new Harness({ wsp, provider: () => provider, keepAuto: 2 }).send('patch');
  assert.deepEqual((await wsp.state.list()).map((c) => c.id).sort(), ['auto-3', 'auto-4']);
});

test('a refusal stops the loop with the reason', async () => {
  const events: AgentEvent[] = [];
  const provider = scripted([{ stopReason: 'refusal', detail: 'cyber' }]);
  await new Harness({ wsp: makeWsp(), provider: () => provider, onEvent: (e) => events.push(e) }).send('x');
  assert.ok(events.some((e) => e.type === 'stopped' && /declined \(cyber\)/.test(e.reason)));
});

const turns: Turn[] = [
  { role: 'user', text: 'hi' },
  { role: 'assistant', text: 'checking', toolCalls: [{ id: 'c1', name: 'model_overview', input: {} }] },
  { role: 'tool', results: [{ id: 'c1', name: 'model_overview', content: '{"dim":8}', isError: false }] },
  { role: 'user', text: 'and?' },
];

test('Anthropic: browser request with strict tools, effort and server-side fallbacks', async () => {
  let sent: { url: string; headers: Headers; body: any } | undefined;
  const fakeFetch = (async (url: string, init: RequestInit) => {
    sent = { url: String(url), headers: new Headers(init.headers), body: JSON.parse(String(init.body)) };
    return new Response(
      JSON.stringify({
        id: 'msg_1',
        type: 'message',
        role: 'assistant',
        model: 'claude-opus-5-5',
        content: [
          { type: 'thinking', thinking: '', signature: 'sig' },
          { type: 'text', text: 'Let me look.' },
          { type: 'tool_use', id: 'tu_1', name: 'inspect_summary', input: { target: 'norm' } },
        ],
        stop_reason: 'tool_use',
        stop_details: null,
        usage: { input_tokens: 10, output_tokens: 5 },
      }),
      { status: 200, headers: { 'content-type': 'application/json' } }
    );
  }) as unknown as typeof fetch;
  const p = new AnthropicProvider({ apiKey: 'sk-test', fetch: fakeFetch });
  const out = await p.complete({ system: 'sys', tools: TOOLS.map(({ name, description, input }) => ({ name, description, input })), turns });
  assert.ok(sent);
  assert.match(sent.url, /\/v1\/messages/);
  assert.ok(sent.headers.get('x-api-key') === 'sk-test' || sent.headers.get('authorization') === 'Bearer sk-test', 'the key is sent');
  assert.equal(sent.headers.get('anthropic-dangerous-direct-browser-access'), 'true');
  assert.match(sent.headers.get('anthropic-beta') ?? '', /server-side-fallback-2026-07-01/);
  assert.equal(sent.body.model, 'claude-opus-5-5');
  assert.equal(sent.body.fallbacks, 'default');
  assert.equal(sent.body.output_config.effort, 'high');
  assert.ok(sent.body.tools.every((t: any) => t.strict === true && t.input_schema.type === 'object'));
  assert.equal(sent.body.thinking, undefined);
  // Tool results and the next user text share one user turn.
  assert.deepEqual(sent.body.messages.map((m: any) => m.role), ['user', 'assistant', 'user']);
  assert.deepEqual(sent.body.messages[2].content.map((b: any) => b.type), ['tool_result', 'text']);
  assert.equal(out.stopReason, 'tool_use');
  assert.deepEqual(out.toolCalls, [{ id: 'tu_1', name: 'inspect_summary', input: { target: 'norm' } }]);
  // The thinking block is kept for replay, untouched.
  const replay = toAnthropicMessages([{ role: 'assistant', text: out.text, toolCalls: out.toolCalls, raw: { provider: 'anthropic', value: out.raw } }], 'anthropic');
  assert.equal((replay[0].content as any[])[0].signature, 'sig');
});

test('Anthropic: models without fallbacks or effort get neither', async () => {
  let body: any;
  const fakeFetch = (async (_url: string, init: RequestInit) => {
    body = JSON.parse(String(init.body));
    return new Response(JSON.stringify({ id: 'm', type: 'message', role: 'assistant', model: 'x', content: [], stop_reason: 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  }) as unknown as typeof fetch;
  await new AnthropicProvider({ apiKey: 'k', model: 'claude-haiku-4-5', fetch: fakeFetch }).complete({ system: 's', tools: [], turns: [{ role: 'user', text: 'x' }] });
  assert.equal(body.fallbacks, undefined);
  assert.equal(body.output_config, undefined);
});

test('OpenAI-compatible (OpenRouter / llama.cpp): tools, tool messages and argument parsing', async () => {
  let body: any;
  let headers: Headers | undefined;
  const fakeFetch = (async (_url: string, init: RequestInit) => {
    body = JSON.parse(String(init.body));
    headers = new Headers(init.headers);
    return new Response(
      JSON.stringify({
        model: 'm',
        choices: [
          {
            finish_reason: 'tool_calls',
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                { id: 'x1', type: 'function', function: { name: 'inspect_summary', arguments: '{"target":"norm"}' } },
                { id: 'x2', type: 'function', function: { name: 'inspect_summary', arguments: '{oops' } },
              ],
            },
          },
        ],
      }),
      { status: 200 }
    );
  }) as unknown as typeof fetch;
  const p = new OpenAiCompatibleProvider({ id: 'openrouter', label: 'OpenRouter', baseUrl: 'https://example/v1', apiKey: 'or-key', model: 'm', fetch: fakeFetch });
  const out = await p.complete({ system: 'sys', tools: TOOLS.slice(0, 2).map(({ name, description, input }) => ({ name, description, input })), turns });
  assert.equal(headers!.get('authorization'), 'Bearer or-key');
  assert.deepEqual(body.messages.map((m: any) => m.role), ['system', 'user', 'assistant', 'tool', 'user']);
  assert.equal(body.messages[2].tool_calls[0].function.arguments, '{}');
  assert.equal(body.tools[0].type, 'function');
  assert.deepEqual(out.toolCalls[0].input, { target: 'norm' });
  assert.match(out.toolCalls[1].parseError ?? '', /not valid JSON/);
});

test('OpenAI-compatible: HTTP errors say what to fix', async () => {
  const fakeFetch = (async () => new Response('{"error":{"message":"No auth"}}', { status: 401 })) as unknown as typeof fetch;
  const p = new OpenAiCompatibleProvider({ id: 'openrouter', label: 'OpenRouter', baseUrl: 'https://x/v1', model: 'm', fetch: fakeFetch });
  await assert.rejects(p.complete({ system: '', tools: [], turns: [{ role: 'user', text: 'x' }] }), /401: No auth\. Check the API key/);
});

test('Gemini: function declarations, responses, and replayed thought signatures', async () => {
  let body: any;
  let url = '';
  const fakeFetch = (async (u: string, init: RequestInit) => {
    url = String(u);
    body = JSON.parse(String(init.body));
    return new Response(
      JSON.stringify({
        candidates: [{ finishReason: 'STOP', content: { role: 'model', parts: [{ functionCall: { name: 'inspect_summary', args: { target: 'norm' } }, thoughtSignature: 'abc' }] } }],
      }),
      { status: 200 }
    );
  }) as unknown as typeof fetch;
  const p = new GeminiProvider({ apiKey: 'g', fetch: fakeFetch });
  const out = await p.complete({ system: 'sys', tools: TOOLS.slice(0, 2).map(({ name, description, input }) => ({ name, description, input })), turns });
  assert.match(url, /models\/gemini-2\.5-flash:generateContent$/);
  assert.equal(body.tools[0].functionDeclarations[0].parameters, undefined, 'no-argument tools omit parameters');
  assert.equal(body.tools[0].functionDeclarations[1].parameters.additionalProperties, undefined);
  assert.deepEqual(body.contents.map((c: any) => c.role), ['user', 'model', 'user']);
  assert.ok(body.contents[2].parts[0].functionResponse);
  assert.equal(out.stopReason, 'tool_use');
  const replay = toGeminiContents([{ role: 'assistant', text: '', toolCalls: out.toolCalls, raw: { provider: 'gemini', value: out.raw } }], 'gemini');
  assert.equal(replay[0].parts[0].thoughtSignature, 'abc');
});
