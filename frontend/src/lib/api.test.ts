import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, fmt, streamNdjson } from "./api";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(enc.encode(c)));
      controller.close();
    },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("streamNdjson", () => {
  it("reassembles events split across network chunks", async () => {
    const lines = ['{"type":"meta","mode":"ollama"}\n{"type":"del', 'ta","text":"Hel"}\n', '{"type":"delta","text":"lo"}\n{"type":"done"}'];
    vi.stubGlobal("fetch", vi.fn(async () => new Response(streamOf(lines), { status: 200 })));
    const events: any[] = [];
    await streamNdjson("/assistant/ask/stream", {}, (e) => events.push(e));
    expect(events.map((e) => e.type)).toEqual(["meta", "delta", "delta", "done"]);
    expect(events.filter((e) => e.type === "delta").map((e) => e.text).join("")).toBe("Hello");
  });

  it("raises the API error message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ error: { code: "x", message: "nope" } }), { status: 400 })));
    await expect(streamNdjson("/x", {}, () => {})).rejects.toThrow("nope");
  });
});

describe("api", () => {
  it("returns JSON and maps errors, including validation details", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ ok: 1 }), { status: 200 })));
    expect(await api<{ ok: number }>("/health")).toEqual({ ok: 1 });

    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      error: { code: "validation_error", message: "Request validation failed.", details: [{ loc: ["body", "name"], msg: "too short" }] },
    }), { status: 422 })));
    const err = (await api("/projects", { json: {} }).catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.message).toContain("name: too short");
  });

  it("reports an unreachable backend clearly", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    await expect(api("/health")).rejects.toThrow("Can't reach the API");
  });
});

describe("fmt", () => {
  it("formats numbers of any magnitude", () => {
    expect(fmt.pct(0.1234, 1)).toBe("12.3%");
    expect(fmt.num(null)).toBe("–");
    expect(fmt.sig(1234567.89)).toBe((1234568).toLocaleString());
    expect(fmt.sig(0.000123456)).toBe("0.0001235");
  });
});
