# Phase 6: Dashboard (React + Vite + Tailwind CSS)

**Goal:** a web UI covering the whole workflow (upload → diagnose → explain →
optimise → approve → monitor → ask), built only on the public API.

It was built after Phases 7–9 because it uses their endpoints (registry,
drift, assistant). It was first written in Next.js and then moved to
**React + Vite + Tailwind CSS**: the dashboard is a client-side app talking
to a separate API, so it doesn't need a server framework. The phase number
follows the original roadmap.

## Stack

| Piece | Choice |
|---|---|
| Build tool / dev server | Vite 8 (`npm run dev` on port 3000, the backend's default CORS origin) |
| UI | React 19 + TypeScript (strict) |
| Routing | React Router 7 (`BrowserRouter`) |
| Styling | Tailwind CSS 4 through `@tailwindcss/vite` |
| Charts | hand-built SVG/HTML, no chart library |
| Production | static build served by nginx, which falls back to `index.html` for client-side routes (`frontend/Dockerfile`, `nginx.conf`) |

## Structure

```
frontend/
  index.html                 Vite entry
  vite.config.ts             React + Tailwind plugins, "@" → src alias, port 3000
  nginx.conf                 production static server with SPA fallback
  src/
    main.tsx                 router: / · /projects/:id · /datasets/:id · /models/:id
    index.css                Tailwind import, design tokens, @theme mapping, component classes
    components/
      Layout.tsx             top bar, settings (API key, theme), page outlet
      ui.tsx                 severity badges, finding cards, findings filter, health panel, meters, tabs, stats
      charts.tsx             Bars, DivergingBars, ConfusionMatrix, ThresholdChart, Histogram, Legend
      Assistant.tsx          streaming chat with local-model picker
    pages/
      Home.tsx               hero, create project, project list
      Project.tsx            upload, datasets, model registry + audit trail
      Dataset.tsx            tabs: Overview · Data Doctor · Models · Diagnosis · Assistant
      Model.tsx              tabs: Performance · Explainability · Experiments · Registry · Monitoring · Assistant
      NotFound.tsx
    lib/
      api.ts                 fetch wrapper (base URL, API key, error shape), NDJSON stream reader, formatters
      hooks.ts               useResource (load/reload), useJob (poll a background job)
      types.ts               API types
```

## How Tailwind is set up

- **Design tokens are CSS variables** in `index.css`: surfaces, text, accent,
  chart series, the diverging pair, status colours, and the sequential
  ramp ends.
  - Dark mode is a separately chosen set of values. It follows the OS, and
    the theme toggle (`html[data-theme]`) wins in both directions.
- **`@theme inline`** maps the variables into Tailwind colours, so the
  markup uses `bg-surface`, `text-fg-2`, `border-line`, `bg-series-1`,
  `bg-critical` and so on. Those classes switch automatically with the
  theme, so components never need `dark:` variants.
- **A small set of repeated building blocks** is composed with `@apply` in
  `@layer components`: `.card`, `.btn`, `.btn-primary`, `.btn-danger`,
  `.field`, `.tbl`, `.badge`, `.notice`. Everything else uses utilities in
  the markup.
- Tailwind v4 can't apply variants (such as `file:`) to custom component
  classes, so the file-picker button is styled with plain `file:` utilities.

## Screens

| Screen | Contents |
|---|---|
| Home | the pitch and a numbered workflow strip; create a project; project list |
| Project | CSV upload, datasets, the **registry** (versions, stage, source, macro-F1, balanced accuracy, minority recall) and the audit trail |
| Dataset › Overview | target picker with live warnings, class-distribution bars, schema table, preview |
| Dataset › Data Doctor | data-quality score, missing cells, usable features, findings filterable by severity, feature-relevance bars (leakage shows as ≈1.0), histograms |
| Dataset › Models | algorithms, CV folds, excluded columns; training as a polled job with live progress; leaderboard |
| Dataset › Diagnosis | health hero number and component meters (each with its formula), narrative, findings, recommended fixes |
| Model › Performance | headline stats, findings, confusion matrix, per-class table, threshold chart, native importance |
| Model › Explainability | SHAP and permutation importance, error statements and slices, explain a prediction (diverging bars), group performance |
| Model › Experiments | run quick or standard optimisation, a before → after summary, the trial table (kept/rejected + reason), a link to the new candidate |
| Model › Registry | approve / reject / promote with the critical-findings acknowledgement, history, compare versions |
| Model › Monitoring | upload a batch; drift score, drifted features, prediction-mix shift, batch F1, PSI per feature, drift findings, history |
| Assistant | **streaming** chat; a picker for your **local Ollama models**; each answer shows the provider, the model and the cited check ids; a Stop button |

## Visual conventions (from the data-viz guidelines)

- **Magnitude** (importance, PSI, class share) uses one blue series with
  thin, rounded bars.
- **Polarity** (SHAP contributions) uses the diverging blue/red pair around
  a zero line, with a legend that names both directions.
- **The confusion matrix** uses a one-hue sequential ramp on the
  row-normalised share, with a 2px gap between cells.
- **Severity colours are reserved status colours**, always shown with an
  icon (▲ ◆ ●) and a text label, so meaning never depends on colour alone.
- **Every chart has hover tooltips.** The threshold chart has a snapping
  crosshair, a legend, direct end labels and the best-F1 threshold marked.
- Tables accompany the charts. Grids collapse to one column on phones, and
  wide tables scroll inside their card.

## The assistant panel

- Calls `GET /assistant/providers` to show which provider is active
  (local Ollama, Claude or the rules) and to fill the local-model picker.
  The choice is remembered in `localStorage`.
- Questions go to `POST /assistant/ask/stream`. `streamNdjson()` reads
  the newline-delimited JSON events (`meta` → `delta`… → `done`) and appends
  the text as it arrives, with a blinking cursor. That matters for local
  models on a CPU, where a full answer takes from about 20 seconds
  (gemma2:2b) to over a minute (llama3.1).
- While waiting for the first word it says so ("llama3.1 is thinking
  (local models can take a minute)…"). The Stop button aborts the request.

## Bug found in the browser: the blank Assistant page

The Assistant tab blanked the whole page in Chrome. Server-side tests couldn't
catch this, because it only happens in a browser. Driving real Chrome with
Playwright showed the error:

```
useEffect must not return anything besides a function ... (returned a Promise)
PAGE ERROR: destroy is not a function   (in <Assistant>)
```

The cause was `useEffect(() => end.current?.scrollIntoView(...), [messages])`.
Without braces the arrow function returns `scrollIntoView()`'s result, and
newer Chrome versions return a Promise from it. React treated the Promise as
the effect's cleanup function and crashed when it called it, unmounting the
whole app.

**Fixes:**
- Braces on the effect.
- The same defensive change on the theme effect in `Layout.tsx`.
- **Error boundaries** (`components/ErrorBoundary.tsx`): one around each
  page, reset when you navigate, and one around the assistant. A crash in
  one panel now shows "Something went wrong … Try again" instead of a blank
  screen.

Assistant answers are also rendered with a small, safe Markdown renderer
(`components/Markdown.tsx`: paragraphs, lists, **bold**, `code`, headings,
built as React elements with no `innerHTML`). Local models answer in
Markdown, and before this the raw `**` and `*` characters showed.

## Connecting to the API

`VITE_API_URL` (default `http://localhost:8000`) is read at build/dev time.
For a different backend port:

```powershell
$env:VITE_API_URL="http://localhost:8765"; npm run dev
```

If the server has API keys (`MD_API_KEYS`), paste a key in **Settings**; it
is sent as `X-API-Key`.

## Running

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
npm run typecheck
npm run build        # type-check + production build into dist/
npm run preview      # serve the production build on :3000
```

## Verification done

- `tsc --noEmit` and `vite build` pass; CI runs both. The generated CSS was
  checked to contain the token-based utilities and the arbitrary-value
  classes.
- A live run with the Vite dev server and the API:
  - a deep link (`/models/...`) serves the app shell, and the modules load
  - the provider endpoint lists the 3 local models
  - a real training job completes
  - a **streamed answer from gemma2:2b** arrives with CORS allowed and a
    cited check id
- **Real-browser check (Playwright + Chrome)** against the live servers:
  - all 3 pages and all 10 tabs (dataset: Overview, Data Doctor, Models,
    Diagnosis, Assistant; model: Performance, Explainability, Experiments,
    Registry, Monitoring) render with **no page errors, no console errors
    and no failed API calls**
  - a question asked in the chat was answered by local `gemma2:2b` and
    rendered as formatted Markdown
  - unknown routes show the 404 page
  - screenshots were reviewed for layout

## Limitations

- Client-rendered only. A public deployment should keep API keys behind a
  server-side proxy rather than in the browser.
- No frontend unit tests; type-checking and the production build are the
  gate.
