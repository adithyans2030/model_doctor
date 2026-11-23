import type { ReactNode } from "react";

/** Minimal, safe Markdown for chat answers: paragraphs, bullet/numbered lists, **bold**,
 *  `code` and "## headings". Builds React elements (no innerHTML), so model output can
 *  never inject markup. */
function inline(text: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    out.push(tok.startsWith("**")
      ? <strong key={`${key}-${i++}`}>{tok.slice(2, -2)}</strong>
      : <code key={`${key}-${i++}`} className="rounded bg-surface px-1 font-mono text-[12.5px]">{tok.slice(1, -1)}</code>);
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  const flush = () => {
    if (!list) return;
    const k = `l${blocks.length}`;
    const items = list.items.map((it, i) => <li key={i}>{inline(it, `${k}-${i}`)}</li>);
    blocks.push(list.ordered
      ? <ol key={k} className="ml-5 list-decimal space-y-0.5">{items}</ol>
      : <ul key={k} className="ml-5 list-disc space-y-0.5">{items}</ul>);
    list = null;
  };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[*-]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const ordered = !!numbered;
      if (!list || list.ordered !== ordered) { flush(); list = { ordered, items: [] }; }
      list.items.push((bullet || numbered)![1]);
      continue;
    }
    flush();
    if (!line.trim()) continue;
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const k = `p${blocks.length}`;
    blocks.push(heading
      ? <p key={k} className="font-semibold">{inline(heading[1], k)}</p>
      : <p key={k}>{inline(line, k)}</p>);
  }
  flush();
  return <div className="space-y-1.5">{blocks}</div>;
}
