# Frontend: landing page and dashboard

The frontend is one React app with two parts:

| Route | What it is |
|---|---|
| `/` | Landing page (public, makes no API calls) |
| `/app` | Dashboard: projects |
| `/app/projects/:id` | Upload, datasets, model registry, audit trail |
| `/app/datasets/:id?tab=overview\|data\|models\|diagnosis\|assistant` | Dataset views |
| `/app/models/:id?tab=performance\|explain\|experiments\|approval\|monitoring\|assistant` | Model views |
| `/app/login`, `/app/admin/users` | Accounts (only when the server has `MD_AUTH_REQUIRED`) |

Old URLs (`/projects/:id`, `/datasets/:id`, `/models/:id`, `/login`) redirect to their `/app` equivalents. Tabs live in the query string, so any view can be linked to.

Stack: React 19, Vite 8, TypeScript (strict), Tailwind CSS 4, React Router 7, TanStack Query 5, Recharts (trend lines only), motion (landing page only), lucide-react icons, self-hosted Inter and JetBrains Mono.

## Running it

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000 (needs the API on :8000; see VITE_API_URL below)
npm run lint         # ESLint: typescript-eslint, react-hooks, jsx-a11y
npm run typecheck    # tsc --noEmit
npm test             # Vitest unit/component tests
npm run test:e2e     # Playwright: starts its own API (temp DB) and dev server
npm run build        # typecheck + production build into dist/
npm run preview      # serve dist/ on :3000
```

`VITE_API_URL` (default `http://localhost:8000`) is read at build time. Set it to `""` to call the API on the same origin, behind a reverse proxy.

"Load demo dataset" on `/app` creates a project from `public/demo/churn_demo.csv` (a copy of `sample_data/churn_demo.csv`) using only the public API: create project → upload → set target.

## Design tokens

All tokens are CSS variables in `src/index.css`, mapped into Tailwind through `@theme`. Components use the Tailwind names (`bg-surface`, `text-fg-2`, `border-line`, `text-danger`…) and never need `dark:` variants.

**Dark is the default.** `html[data-theme="light"]` switches to the light set. The choice is stored in `localStorage` (`md_theme`) and applied by an inline script in `index.html` before first paint. Every text/background pair used was checked to meet WCAG AA (4.5:1) in both themes.

### Colour

| Token (Tailwind) | Role | Dark | Light |
|---|---|---|---|
| `bg` | page | `#0b0c0e` | `#f7f7f5` |
| `surface` / `surface-2` / `surface-3` | cards, insets, hover | `#111316` / `#171a1e` / `#1f2328` | `#ffffff` / `#f2f2ef` / `#e9e9e5` |
| `line` / `line-strong` | borders | `#252a30` / `#363c45` | `#e3e3de` / `#cdcdc7` |
| `fg` / `fg-2` / `fg-3` | text: primary, secondary, muted | `#ececef` / `#a9adb6` / `#8b909a` | `#0d0e10` / `#4a4d55` / `#5f636c` |
| `accent` | the one accent: solid fills (primary buttons) | `#2f6fe0` | `#2463d6` |
| `accent-fg` | accent as text (links, active tab) | `#79adff` | `#1d56c0` |
| `success` / `warning` / `danger` / `info` | status text; each has a `-subtle` background | `#4cc27a` / `#e0ac45` / `#f27171` / `#79adff` | `#146f36` / `#855600` / `#c42f2d` / `#1d56c0` |
| `ai` (+ `ai-subtle`, `ai-border`) | the AI explanation panel only | `#b39cf0` | `#6744c4` |
| `series-1..3`, `series-muted` | chart series | blue / orange / green / grey | same hues, light steps |
| `div-pos` / `div-neg` | diverging pair (SHAP polarity) | blue / red | blue / red |
| `seq-0` → `seq-1` | sequential ramp (confusion matrix) | dark → light blue | light → dark blue |

Status colour never carries meaning alone: severities also have their own icon shape and label.

### Type

- **Inter Variable** for the UI: neutral, very legible at 13–14px, with tabular figures.
- **JetBrains Mono Variable** for metrics, ids and code: clear digits and 0/O distinction. Programming ligatures are turned off, so `--build` shows as two hyphens.

| Class | Size / line height | Use |
|---|---|---|
| `text-xs` | 12 / 16 | meta, captions |
| `text-sm` | 13 / 20 | dense UI text, tables |
| `text-base` | 14 / 22 | body |
| `text-md` / `text-lg` | 16 / 26, 18 / 26 | lead paragraphs |
| `text-xl` … `text-5xl` | 22 → 60 | headings; 36–60 only on the landing page |

### Radius, shadow, spacing

- Radius: `sm` 4px (badges), `md` 6px (buttons, inputs), `lg` 8px (cards), `xl` 12px (frames, dialogs).
- Shadows: `shadow-sm` (cards), `shadow-md` (tooltips, popovers), `shadow-lg` (dialogs, screenshot frames). Restrained, with no glow.
- Spacing is Tailwind's 4px scale. Cards use 16px padding. Page sections use 20px gaps in the dashboard and 96–128px on the landing page.
- Layout: dashboard max width 1280px; landing page max width 1200px on a 12-column grid. Both work from 360px with no horizontal page scroll.

## Components

### Primitives: `src/components/ui/`

| Component | Notes |
|---|---|
| `Button`, `buttonClasses()` | `primary / secondary / ghost / danger`, `sm / md / lg`, `loading` (spinner + `aria-busy`), defaults to `type="button"`. `buttonClasses` styles router links as buttons. |
| `Badge` | tones `neutral / accent / success / warning / danger / info / ai`, optional `mono` |
| `Card` | with a `title` it renders a `<section>` labelled by its heading; `actions`, `description`, `flush` |
| `Tabs`, `TabPanel` | WAI-ARIA tabs: roving tabindex, ←/→/Home/End, `aria-controls` links |
| `Table`, `THead`, `TBody`, `TR`, `TH`, `TD` | inside a focusable, labelled scroll region; `numeric` cells right-aligned in mono |
| `Modal`, `ConfirmDialog` | native `<dialog>` (focus trap, inert background, Escape); focus returns to the opener |
| `Tooltip` | shown on hover and focus, hidden with Escape, wired with `aria-describedby` |
| `Skeleton`, `LoadingRegion`, `CardSkeleton`, `TableSkeleton`, `MetricTileSkeleton`, `ChartSkeleton`, `SkeletonText` | skeletons are `aria-hidden`; `LoadingRegion` announces "Loading…" once |
| `EmptyState`, `ErrorState`, `InlineError` | `ErrorState` takes `onRetry` |
| `ToastProvider`, `useToast()` | polite live region, auto-dismiss, dismiss button |
| `CodeBlock` | copy button with an announced "Copied" |
| `MetricTile`, `DeltaBadge`, `deltaSentiment()` | `before → value`, signed delta coloured by whether higher is better |

### Domain components: `src/components/domain/`

`QueryView` (loading, error-with-retry and empty states for any query), `PageHeader` (breadcrumb + meta row), `FindingCard` / `FindingsList` (severity-ranked, evidence → explanation → recommended fix), `SeverityBadge` / `StageBadge`, `HealthPanel` / `ScoreRing`, `FileDrop`.

### Charts: `src/components/charts.tsx`, `TrendChart.tsx`

Hand-built SVG/HTML charts (`Bars`, `DivergingBars`, `Dumbbell`, `SegmentedBar`, `ConfusionMatrix`, `ThresholdChart`, `Histogram`, `ScatterChart`) and one Recharts line (`TrendChart`, used for drift over time and metric by version). Rules:

- Colours come only from tokens.
- Every chart is a labelled `role="img"`, backed by a visually hidden data table (`SrTable`).
- Each has hover tooltips and a legend when there are two or more series.

The AI panel (`Assistant.tsx`) uses the `ai` hue, is labelled "AI explanation · explains deterministic results", and lists the finding ids each answer cites. On the Diagnosis tab those ids link to the finding cards.

## Data layer

- `src/lib/types.ts` types every API response the UI uses, taken from live responses. `ModelDetail` is a discriminated union on `task`. There is no `any` in the codebase, and ESLint enforces it.
- `src/lib/queries.ts` has one TanStack Query hook per resource (`useDataset`, `useDiagnosis`, `useComparison`, …), mutations that invalidate exactly what they change, and `useJob` for polling background jobs.
- `src/lib/api.ts` is the fetch wrapper (base URL, API key, bearer token, error shape) plus the NDJSON stream reader used by the assistant.

No API contract was changed for the redesign.

## Accessibility

- A visible `:focus-visible` ring on every interactive element, and a skip link on both layouts.
- Keyboard: tabs use arrow keys, dialogs trap focus and close with Escape, tooltips appear on focus, and wide tables scroll from the keyboard.
- Charts have accessible names and data tables. Severity is never shown by colour alone.
- `prefers-reduced-motion` turns off landing-page motion and shortens all transitions.
- Checked with axe: the Playwright smoke tests fail on any serious or critical WCAG 2.1 A/AA violation.

## Tests

| Suite | Where | What |
|---|---|---|
| Vitest | `src/**/*.test.tsx` | every primitive (keyboard, ARIA, states), domain components, `QueryView`, API client |
| Playwright smoke | `e2e/smoke.spec.ts` | landing page (sections, CTA, copy, theme persistence, mobile menu, no API calls, axe in both themes); dashboard happy path (load demo → leak finding → overview → diagnosis, axe on each) |
| Playwright workflow | `e2e/workflow.spec.ts` | full classification and regression workflows against a real API |

## Performance

- Routes are lazy-loaded. The landing page loads the main chunk and its own chunk only; the dashboard shell, auth and Recharts load on demand.
- Screenshots are WebP, sized, and lazy-loaded.
- Fonts are self-hosted, split by unicode range.
- nginx serves gzip and caches hashed assets for a year.

Lighthouse on `npm run preview` (landing page, Lighthouse 12.6):

| | Performance | Accessibility | Best practices | SEO |
|---|---|---|---|---|
| Mobile (throttled) | 92 | 100 | 100 | 100 |
| Desktop | 100 | 100 | 100 | 100 |

## Landing page content

Every number on the landing page is defined in `src/pages/landing/content.ts`, next to its source:

- 128 backend tests: `pytest --collect-only`.
- 37 check types: distinct finding ids in the backend.
- 9/9 planted problems: Data Doctor on the demo dataset.
- Churn recall 5% → 49%: leak-free baseline vs the best experiment on the demo dataset.

Demo results are labelled as coming from the synthetic dataset. Screenshots in `public/landing/` are real dashboard views of that run. After a UI change, re-capture them and the Open Graph image (`public/og.png`, 1200×630).
