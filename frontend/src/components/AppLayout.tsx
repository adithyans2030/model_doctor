import { BookOpen, KeyRound, LogOut, Settings, Users } from "lucide-react";
import { Suspense, useState } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { API_URL, getApiKey, setApiKey } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { cx } from "@/lib/cx";
import { ErrorBoundary } from "./ErrorBoundary";
import { Logo } from "./Logo";
import { ThemeToggle } from "./ThemeToggle";
import { Badge, Button, CardSkeleton, LoadingRegion, Modal, buttonClasses } from "./ui";

export function PageSkeleton() {
  return (
    <LoadingRegion label="Loading page" className="space-y-4">
      <div className="h-14" />
      <CardSkeleton lines={5} />
    </LoadingRegion>
  );
}

/** Dashboard shell: top bar, skip link, settings, and a per-page error boundary. */
export function AppLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, status, logout } = useAuth();
  const [settings, setSettings] = useState(false);
  const [key, setKey] = useState(getApiKey());

  return (
    <>
      <a href="#main" className="sr-only-focusable fixed top-2 left-2 z-[70] rounded-md bg-accent px-3 py-2 text-sm text-accent-ink">
        Skip to content
      </a>
      <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur supports-[backdrop-filter]:bg-bg/70">
        <div className="mx-auto flex h-14 max-w-[1280px] items-center gap-3 px-4 sm:px-6">
          <Link to="/app" className="text-fg no-underline hover:no-underline" aria-label="ModelDoctor projects">
            <Logo />
          </Link>
          <nav aria-label="Main" className="ml-4 hidden items-center gap-1 text-sm md:flex">
            <NavLink to="/app" end className={({ isActive }) => cx("rounded-md px-2.5 py-1.5 no-underline hover:bg-surface-2 hover:no-underline", isActive ? "text-fg" : "text-fg-2")}>
              Projects
            </NavLink>
            {user?.role === "admin" && (
              <NavLink to="/app/admin/users" className={({ isActive }) => cx("rounded-md px-2.5 py-1.5 no-underline hover:bg-surface-2 hover:no-underline", isActive ? "text-fg" : "text-fg-2")}>
                Users
              </NavLink>
            )}
          </nav>
          <span className="flex-1" />
          <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer"
            className={buttonClasses({ variant: "ghost", size: "sm", className: "hidden sm:inline-flex" })}>
            <BookOpen size={14} aria-hidden /> API docs
          </a>
          <ThemeToggle />
          <Button variant="ghost" size="sm" className="w-8 px-0" aria-label="Settings" onClick={() => setSettings(true)}>
            <Settings size={15} aria-hidden />
          </Button>
          {user ? (
            <div className="flex items-center gap-2 border-l border-line pl-3">
              <span className="hidden max-w-32 truncate text-sm text-fg-2 sm:inline">{user.name}</span>
              <Badge>{user.role}</Badge>
              {user.role === "admin" && (
                <Link to="/app/admin/users" aria-label="Manage users" className={buttonClasses({ variant: "ghost", size: "sm", className: "w-8 px-0 md:hidden" })}>
                  <Users size={15} aria-hidden />
                </Link>
              )}
              <Button variant="ghost" size="sm" className="w-8 px-0" aria-label="Sign out"
                onClick={() => { logout(); navigate(status?.auth_required ? "/app/login" : "/app"); }}>
                <LogOut size={15} aria-hidden />
              </Button>
            </div>
          ) : location.pathname !== "/app/login" && status?.auth_required !== false && (
            <Link to="/app/login" className={buttonClasses({ size: "sm" })}>Sign in</Link>
          )}
        </div>
      </header>

      <Modal open={settings} onClose={() => setSettings(false)} title="Settings" description={`Connected to ${API_URL}`}
        footer={
          <>
            <Button onClick={() => setSettings(false)}>Cancel</Button>
            <Button variant="primary" onClick={() => { setApiKey(key.trim()); window.location.reload(); }}>Save and reload</Button>
          </>
        }>
        <label htmlFor="api-key" className="flex items-center gap-1.5 text-sm font-medium"><KeyRound size={14} aria-hidden /> API key</label>
        <p id="api-key-help" className="mt-0.5 mb-2 text-xs text-fg-3">Only needed when the server sets MD_API_KEYS. Sent as the X-API-Key header and stored in this browser.</p>
        <input id="api-key" type="password" autoComplete="off" aria-describedby="api-key-help" className="field w-full"
          value={key} onChange={(e) => setKey(e.target.value)} />
      </Modal>

      <main id="main" tabIndex={-1} className="mx-auto max-w-[1280px] px-4 pt-6 pb-20 outline-none sm:px-6">
        {/* keyed by path so navigating away from a crashed page resets it */}
        <ErrorBoundary key={location.pathname}>
          <Suspense fallback={<PageSkeleton />}>
            <Outlet />
          </Suspense>
        </ErrorBoundary>
      </main>
    </>
  );
}
