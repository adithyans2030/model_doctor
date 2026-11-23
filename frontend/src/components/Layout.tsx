import { useEffect, useState } from "react";
import { Link, Outlet, useLocation, useNavigate } from "react-router-dom";
import { ErrorBoundary } from "./ErrorBoundary";
import { API_URL, getApiKey, setApiKey } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { cx } from "./legacy-ui";

type Theme = "system" | "light" | "dark";

function applyTheme(t: Theme) {
  try {
    if (t === "system") {
      delete document.documentElement.dataset.theme;
      localStorage.removeItem("md_theme");
    } else {
      document.documentElement.dataset.theme = t;
      localStorage.setItem("md_theme", t);
    }
  } catch { /* storage unavailable */ }
}

export function Layout() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, status, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const [key, setKey] = useState(getApiKey());
  const [theme, setTheme] = useState<Theme>(() => {
    try { return (localStorage.getItem("md_theme") as Theme) || "system"; } catch { return "system"; }
  });
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <>
      <header className="sticky top-0 z-20 flex items-center gap-4 border-b border-line bg-surface px-4 py-3 sm:px-6">
        <Link to="/" className="flex items-center gap-2 text-base font-bold text-fg hover:no-underline">
          <img src="/favicon.svg" alt="" className="size-6" /> ModelDoctor
        </Link>
        <span className="hidden text-[13px] text-fg-3 sm:inline">Diagnose · Explain · Optimise</span>
        <span className="flex-1" />
        <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" className="hidden text-[13px] sm:inline">API docs</a>
        {user?.role === "admin" && <Link to="/admin/users" className="text-[13px]">Users</Link>}
        {user ? (
          <span className="flex items-center gap-2 text-[13px] text-fg-2">
            <span className="hidden sm:inline">{user.name}</span><span className="badge">{user.role}</span>
            <button className="btn" onClick={() => { logout(); navigate(status?.auth_required ? "/login" : "/"); }}>Sign out</button>
          </span>
        ) : location.pathname !== "/login" && (
          <Link to="/login" className="text-[13px]">Sign in</Link>
        )}
        <button className="btn" onClick={() => setOpen((o) => !o)} aria-expanded={open}>Settings</button>
        {open && (
          <div className="card absolute top-14 right-4 z-30 w-80 shadow-xl">
            <h3>Settings</h3>
            <p className="mt-1 text-xs text-fg-3">API: {API_URL}</p>
            <label className="mt-3 block text-[13px] text-fg-2">API key (only if the server sets MD_API_KEYS)</label>
            <div className="mt-1 flex gap-2">
              <input type="password" className="field flex-1" value={key} onChange={(e) => setKey(e.target.value)} />
              <button className="btn btn-primary" onClick={() => { setApiKey(key.trim()); window.location.reload(); }}>Save</button>
            </div>
            <div className="mt-3 flex items-center gap-2">
              <span className="text-[13px] text-fg-2">Theme</span>
              {(["system", "light", "dark"] as const).map((t) => (
                <button key={t} className={cx("btn", theme === t && "btn-primary")} onClick={() => setTheme(t)}>{t}</button>
              ))}
            </div>
          </div>
        )}
      </header>
      <main className="mx-auto max-w-6xl px-4 pt-6 pb-16">
        {/* keyed by path so navigating away from a crashed page resets it */}
        <ErrorBoundary key={location.pathname}>
          <Outlet />
        </ErrorBoundary>
      </main>
    </>
  );
}
