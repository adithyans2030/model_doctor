import { useCallback, useSyncExternalStore } from "react";

export type Theme = "dark" | "light";
const KEY = "md_theme";
const listeners = new Set<() => void>();

function read(): Theme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

/** Dark by default; the choice is stored and applied before first paint (index.html). */
export function setTheme(theme: Theme) {
  if (theme === "light") document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem(KEY, theme); } catch { /* storage unavailable */ }
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "light" ? "#f7f7f5" : "#0b0c0e");
  listeners.forEach((l) => l());
}

export function useTheme(): [Theme, () => void] {
  const theme = useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => listeners.delete(cb); },
    read,
    () => "dark" as Theme,
  );
  const toggle = useCallback(() => setTheme(theme === "dark" ? "light" : "dark"), [theme]);
  return [theme, toggle];
}
