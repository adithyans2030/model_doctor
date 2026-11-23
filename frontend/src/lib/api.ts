// VITE_API_URL unset -> local backend; set to "" -> same origin (behind an ingress/reverse proxy).
const RAW_API_URL = import.meta.env.VITE_API_URL as string | undefined;
export const API_URL = (RAW_API_URL === undefined ? "http://localhost:8000" : RAW_API_URL).replace(/\/$/, "");
const KEY_STORAGE = "md_api_key";
const TOKEN_STORAGE = "md_token";

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_STORAGE) || "";
  } catch {
    return "";
  }
}

export function setToken(token: string) {
  try {
    if (token) localStorage.setItem(TOKEN_STORAGE, token);
    else localStorage.removeItem(TOKEN_STORAGE);
  } catch {
    /* storage unavailable */
  }
}

export function getApiKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

export function setApiKey(key: string) {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable: key lasts for this page only */
  }
}

export class ApiError extends Error {
  status: number;
  code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

type Options = { method?: string; json?: unknown; form?: FormData };

function headers(json: boolean): Record<string, string> {
  const h: Record<string, string> = {};
  const key = getApiKey();
  if (key) h["X-API-Key"] = key;
  const token = getToken();
  if (token) h["Authorization"] = `Bearer ${token}`;
  if (json) h["Content-Type"] = "application/json";
  return h;
}

async function failure(res: Response): Promise<ApiError> {
  if (res.status === 401) window.dispatchEvent(new Event("md:unauthorized"));
  type ErrorBody = { error?: { code?: string; message?: string; details?: { loc: string[]; msg: string }[] } };
  const data = (await res.json().catch(() => null)) as ErrorBody | null;
  const err = data?.error;
  const detail = err?.details?.map((d) => `${d.loc.slice(1).join(".")}: ${d.msg}`).join("; ");
  return new ApiError(res.status, err?.code || "error", (err?.message || res.statusText) + (detail ? ` (${detail})` : ""));
}

export async function api<T>(path: string, opts: Options = {}): Promise<T> {
  let body: BodyInit | undefined;
  if (opts.json !== undefined) body = JSON.stringify(opts.json);
  else if (opts.form) body = opts.form;
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/v1${path}`, {
      method: opts.method || (body ? "POST" : "GET"),
      headers: headers(opts.json !== undefined),
      body,
    });
  } catch {
    throw new ApiError(0, "network", `Can't reach the API at ${API_URL}. Is the backend running?`);
  }
  if (res.status === 204) return undefined as T;
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}

/** POST JSON and read a newline-delimited JSON stream, calling onEvent per line. */
export async function streamNdjson<E>(path: string, json: unknown, onEvent: (e: E) => void, signal?: AbortSignal) {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/v1${path}`, { method: "POST", headers: headers(true), body: JSON.stringify(json), signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    throw new ApiError(0, "network", `Can't reach the API at ${API_URL}. Is the backend running?`);
  }
  if (!res.ok || !res.body) throw await failure(res);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) onEvent(JSON.parse(line) as E);
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as E);
}

export const fmt = {
  pct: (x: number | null | undefined, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`),
  num: (x: number | null | undefined, d = 3) => (x == null ? "–" : x.toFixed(d)),
  int: (x: number | null | undefined) => (x == null ? "–" : x.toLocaleString()),
  /** Readable number of any magnitude (prices, errors): 4 significant digits with separators. */
  sig: (x: number | null | undefined) =>
    x == null ? "–" : Math.abs(x) >= 1000 ? Math.round(x).toLocaleString() : Number(x.toPrecision(4)).toString(),
  date: (s: string) => new Date(s.endsWith("Z") || s.includes("+") ? s : s + "Z").toLocaleString(),
};
