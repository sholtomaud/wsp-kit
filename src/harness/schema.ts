/**
 * The JSON Schema subset the tools use, and a validator for tool inputs.
 *
 * Anthropic's strict tool use guarantees schema-valid inputs; OpenRouter,
 * llama.cpp and Gemini models do not, so every call is checked here before it
 * touches the GPU. Numeric strings and "true"/"false" are coerced, since small
 * local models often quote them.
 */

export type Schema =
  | { type: 'object'; description?: string; properties: Record<string, Schema>; required?: string[]; additionalProperties: false }
  | { type: 'array'; description?: string; items: Schema }
  | { type: 'string'; description?: string; enum?: string[] }
  | { type: 'number' | 'integer'; description?: string }
  | { type: 'boolean'; description?: string };

export class ToolInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ToolInputError';
  }
}

export function validate(schema: Schema, value: unknown, path = 'input'): unknown {
  switch (schema.type) {
    case 'object': {
      if (value === undefined || value === null) value = {};
      if (typeof value !== 'object' || Array.isArray(value)) throw new ToolInputError(`${path} must be an object`);
      const obj = value as Record<string, unknown>;
      const out: Record<string, unknown> = {};
      for (const key of schema.required ?? []) {
        if (obj[key] === undefined || obj[key] === null) throw new ToolInputError(`${path}.${key} is required`);
      }
      for (const [key, v] of Object.entries(obj)) {
        const prop = schema.properties[key];
        if (!prop) throw new ToolInputError(`${path}.${key} is not a known parameter (known: ${Object.keys(schema.properties).join(', ') || 'none'})`);
        if (v === null || v === undefined) continue;
        out[key] = validate(prop, v, `${path}.${key}`);
      }
      return out;
    }
    case 'array': {
      if (!Array.isArray(value)) throw new ToolInputError(`${path} must be an array`);
      return value.map((v, i) => validate(schema.items, v, `${path}[${i}]`));
    }
    case 'string': {
      if (typeof value !== 'string') throw new ToolInputError(`${path} must be a string`);
      if (schema.enum && !schema.enum.includes(value)) throw new ToolInputError(`${path} must be one of ${schema.enum.join(', ')}`);
      return value;
    }
    case 'number':
    case 'integer': {
      const n = typeof value === 'string' && value.trim() !== '' ? Number(value) : value;
      if (typeof n !== 'number' || !Number.isFinite(n)) throw new ToolInputError(`${path} must be a finite number`);
      if (schema.type === 'integer' && !Number.isInteger(n)) throw new ToolInputError(`${path} must be an integer`);
      return n;
    }
    case 'boolean': {
      if (value === 'true') return true;
      if (value === 'false') return false;
      if (typeof value !== 'boolean') throw new ToolInputError(`${path} must be true or false`);
      return value;
    }
  }
}

/** Gemini's function declarations take an OpenAPI subset without additionalProperties. */
export function toOpenApiSubset(schema: Schema): Record<string, unknown> {
  switch (schema.type) {
    case 'object':
      return {
        type: 'object',
        ...(schema.description ? { description: schema.description } : {}),
        properties: Object.fromEntries(Object.entries(schema.properties).map(([k, v]) => [k, toOpenApiSubset(v)])),
        ...(schema.required?.length ? { required: schema.required } : {}),
      };
    case 'array':
      return { type: 'array', ...(schema.description ? { description: schema.description } : {}), items: toOpenApiSubset(schema.items) };
    default:
      return { ...schema };
  }
}

// Builders, to keep tool definitions short.
export const S = {
  object: (properties: Record<string, Schema>, required: string[] = [], description?: string): Schema => ({
    type: 'object',
    ...(description ? { description } : {}),
    properties,
    required,
    additionalProperties: false,
  }),
  string: (description?: string, values?: string[]): Schema => ({ type: 'string', ...(description ? { description } : {}), ...(values ? { enum: values } : {}) }),
  number: (description?: string): Schema => ({ type: 'number', ...(description ? { description } : {}) }),
  integer: (description?: string): Schema => ({ type: 'integer', ...(description ? { description } : {}) }),
  boolean: (description?: string): Schema => ({ type: 'boolean', ...(description ? { description } : {}) }),
  array: (items: Schema, description?: string): Schema => ({ type: 'array', items, ...(description ? { description } : {}) }),
};
