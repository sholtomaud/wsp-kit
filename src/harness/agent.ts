/**
 * The orchestration loop: natural language in, tool calls out, results back,
 * until the Chief Architect stops calling tools. The person never writes JSON;
 * the schemas ride along with every request.
 */
import type { WSP } from '../wsp.ts';
import { ProviderError, type Completion, type Provider, type ToolCall, type ToolResult, type Turn } from '../llm/types.ts';
import { ToolInputError, validate } from './schema.ts';
import { TOOL_BY_NAME, TOOLS, type ToolDef } from './tools.ts';

export const SYSTEM_PROMPT = `You are the Chief Architect in wsp-kit, a Weight-Space Programming environment. A small Llama-style transformer (RMSNorm, rotary attention with grouped KV heads, SwiGLU MLP) runs locally in the user's browser on WebGPU. You change what it does by editing its weights directly: you decide where and what, and wsp-kit computes the exact floats.

The person types plain-language requests ("make it say Paris after 'The capital of France is'", "stop it naming everyone Lily", "add capacity to layer 2"). Turn them into a plan and carry it out with the tools.

Seeing:
- model_overview gives every tensor, shape and role, plus the exact architecture and tokenizer. Matrices are [rows = outputs, cols = inputs]: in mlp.gate and mlp.up a row is one MLP neuron; in mlp.down a column is one neuron; in tok_embeddings and the classifier a row is one token id (inspect_tokenize maps text to ids).
- You never read whole tensors: use inspect_summary, inspect_find_anomalies, inspect_slice (small blocks) and inspect_neuron.
- eval_run and eval_next_tokens run the network. Measure before and after every change, on the behaviour you are changing and on controls you want to keep.

Changing, in order of preference:
1. Computed edits, the main tool. You state the behaviour; wsp-kit reads the network's own representation at that point and writes a closed-form weight change, with no training loop:
   - mutate_insert_association: "after this prompt, say this" (rank-one edit of one layer's MLP; pass preserve_texts for behaviour to keep).
   - mutate_wire_neuron: one neuron that fires on a context and promotes a token (interpretable, easy to undo).
2. Exact direct edits: mutate_patch_sparse (specific floats), mutate_prune_semantic (zero the neurons a set of texts specifically uses; dry_run first), mutate_graft_node (new neurons/heads that change nothing until edited).
3. Training, only for broad new skills that edits cannot express: write examples with dataset_write, then mutate_micro_train runs backprop on only the tensors or blocks you name. Prefer training new capacity alone.

Working rules:
1. The harness checkpoints before every mutation and tells you the id (auto-N). If the target or a control gets worse, roll back with state_restore. Use state_checkpoint for milestones.
2. Computed edits need a network that already represents its input (a pretrained starter). On a random seed there is little to read; say so, and suggest a starter, wiring a few lookup-style neurons, or training.
3. Be realistic about capacity: compare losses with chanceLoss (ln vocab) and probabilities with the goal you asked for.
4. Prefer the smallest change that works, explain the evidence for each, and finish with before/after numbers and what you would try next. Be brief.`;

export type AgentEvent =
  | { type: 'user'; text: string }
  | { type: 'assistant'; text: string; model?: string }
  | { type: 'tool_call'; call: ToolCall }
  | { type: 'tool_progress'; call: ToolCall; text: string }
  | { type: 'tool_result'; call: ToolCall; result: ToolResult; ms: number; checkpoint?: string }
  | { type: 'stopped'; reason: string }
  | { type: 'error'; message: string };

export interface HarnessOptions {
  wsp: WSP;
  /** Read per request, so the user can switch provider or key mid-conversation. */
  provider: () => Provider;
  /** Model calls per user message. Default 30. */
  maxSteps?: number;
  /** Checkpoint before each mutation. Default true. */
  autoCheckpoint?: boolean;
  /** Auto checkpoints kept. Default 5. */
  keepAuto?: number;
  tools?: ToolDef[];
  onEvent?: (e: AgentEvent) => void;
  system?: string;
}

const MAX_RESULT_CHARS = 12000;

export class Harness {
  readonly turns: Turn[] = [];
  private readonly o: HarnessOptions;
  private readonly tools: ToolDef[];
  private autoIds: string[] = [];
  private autoCounter = 0;
  private busy = false;

  constructor(o: HarnessOptions) {
    this.o = o;
    this.tools = o.tools ?? TOOLS;
  }

  get running() {
    return this.busy;
  }

  reset() {
    this.turns.length = 0;
  }

  private emit(e: AgentEvent) {
    this.o.onEvent?.(e);
  }

  /** One user message, then as many model/tool round-trips as it takes. */
  async send(text: string, signal?: AbortSignal): Promise<Completion | undefined> {
    if (this.busy) throw new Error('The harness is already working on a message');
    this.busy = true;
    try {
      this.turns.push({ role: 'user', text });
      this.emit({ type: 'user', text });
      const maxSteps = this.o.maxSteps ?? 30;
      let last: Completion | undefined;
      for (let step = 0; step < maxSteps; step++) {
        if (signal?.aborted) {
          this.emit({ type: 'stopped', reason: 'Stopped.' });
          return last;
        }
        const provider = this.o.provider();
        last = await this.completeWithRetry(provider, signal);
        this.turns.push({ role: 'assistant', text: last.text, toolCalls: last.toolCalls, raw: { provider: provider.id, value: last.raw } });
        if (last.text) this.emit({ type: 'assistant', text: last.text, model: last.model });
        if (last.stopReason === 'refusal') {
          this.emit({ type: 'stopped', reason: `The model declined${last.detail ? ` (${last.detail})` : ''}.` });
          return last;
        }
        if (!last.toolCalls.length) {
          if (last.stopReason === 'max_tokens') this.emit({ type: 'stopped', reason: 'The reply hit the output token limit.' });
          return last;
        }
        const results: ToolResult[] = [];
        for (const call of last.toolCalls) {
          // Calls run one at a time: they share one GPU queue and may depend on each other.
          results.push(await this.execute(call, signal));
        }
        this.turns.push({ role: 'tool', results });
      }
      this.emit({ type: 'stopped', reason: `Stopped after ${maxSteps} model calls; send another message to continue.` });
      return last;
    } catch (e) {
      if ((e as Error).name === 'AbortError' || signal?.aborted) {
        this.emit({ type: 'stopped', reason: 'Stopped.' });
        this.dropDanglingToolUse();
        return undefined;
      }
      this.emit({ type: 'error', message: (e as Error).message });
      this.dropDanglingToolUse();
      throw e;
    } finally {
      this.busy = false;
    }
  }

  private async completeWithRetry(provider: Provider, signal?: AbortSignal): Promise<Completion> {
    const request = { system: this.o.system ?? SYSTEM_PROMPT, tools: this.tools.map(({ name, description, input }) => ({ name, description, input })), turns: this.turns, signal };
    for (let attempt = 0; ; attempt++) {
      try {
        return await provider.complete(request);
      } catch (e) {
        if (!(e instanceof ProviderError) || !e.retryable || attempt >= 2 || provider.id === 'anthropic') throw e;
        await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
      }
    }
  }

  /** An assistant turn with tool calls but no results can't be sent again. */
  private dropDanglingToolUse() {
    const last = this.turns[this.turns.length - 1];
    if (last?.role === 'assistant' && last.toolCalls.length) this.turns.pop();
  }

  async execute(call: ToolCall, signal?: AbortSignal): Promise<ToolResult> {
    this.emit({ type: 'tool_call', call });
    const started = performance.now();
    const tool = TOOL_BY_NAME.get(call.name) ?? this.tools.find((t) => t.name === call.name);
    let content: string;
    let isError = false;
    let checkpoint: string | undefined;
    try {
      if (!tool) throw new ToolInputError(`Unknown tool "${call.name}". Available: ${this.tools.map((t) => t.name).join(', ')}`);
      if (call.parseError) throw new ToolInputError(call.parseError);
      const input = validate(tool.input, call.input) as Record<string, unknown>;
      if (tool.mutates && (this.o.autoCheckpoint ?? true)) checkpoint = await this.autoCheckpoint();
      const value = await tool.run(this.o.wsp, input, { signal, progress: (text) => this.emit({ type: 'tool_progress', call, text }) });
      const payload = checkpoint ? { ...(value as object), rollback: `state_restore("${checkpoint}") undoes this` } : value;
      content = JSON.stringify(payload);
      if (content.length > MAX_RESULT_CHARS) content = content.slice(0, MAX_RESULT_CHARS) + '… (truncated)';
    } catch (e) {
      isError = true;
      content = (e as Error).message || String(e);
    }
    const result: ToolResult = { id: call.id, name: call.name, content, isError };
    this.emit({ type: 'tool_result', call, result, ms: Math.round(performance.now() - started), checkpoint });
    return result;
  }

  private async autoCheckpoint(): Promise<string> {
    const id = `auto-${++this.autoCounter}`;
    await this.o.wsp.state.checkpoint(id, 'before a mutation');
    this.autoIds.push(id);
    while (this.autoIds.length > (this.o.keepAuto ?? 5)) await this.o.wsp.state.delete(this.autoIds.shift()!);
    return id;
  }
}
