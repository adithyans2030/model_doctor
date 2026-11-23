import { useState, type FormEvent } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { Button, InlineError } from "@/components/ui";
import { useAuth } from "@/lib/auth";

export default function Login() {
  const { user, status, login, register } = useAuth();
  const navigate = useNavigate();
  const from = (useLocation().state as { from?: string } | null)?.from || "/app";
  const firstUser = status ? !status.has_users : false;
  const [mode, setMode] = useState<"login" | "register">(firstUser ? "register" : "login");
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to={from} replace />;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "login") await login(email, password);
      else await register(email, name, password);
      navigate(from, { replace: true });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const canRegister = !status || status.open_registration || firstUser;
  return (
    <div className="mx-auto mt-8 max-w-sm sm:mt-16">
      <div className="rounded-xl border border-line bg-surface p-6 shadow-md">
        <Logo withName={false} />
        <h1 className="mt-4 text-lg font-semibold">{mode === "login" ? "Sign in to ModelDoctor" : firstUser ? "Create the admin account" : "Create an account"}</h1>
        {firstUser && <p className="mt-1 text-sm text-fg-2">This is the first account, so it becomes the admin.</p>}
        <form className="mt-5 space-y-4" onSubmit={submit}>
          <div>
            <label htmlFor="email" className="mb-1 block text-sm font-medium">Email</label>
            <input id="email" className="field w-full" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          {mode === "register" && (
            <div>
              <label htmlFor="name" className="mb-1 block text-sm font-medium">Name</label>
              <input id="name" className="field w-full" autoComplete="name" required maxLength={120} value={name} onChange={(e) => setName(e.target.value)} />
            </div>
          )}
          <div>
            <label htmlFor="password" className="mb-1 block text-sm font-medium">Password</label>
            <input id="password" className="field w-full" type="password" required minLength={mode === "register" ? 8 : 1}
              aria-describedby={mode === "register" ? "pw-help" : undefined}
              autoComplete={mode === "login" ? "current-password" : "new-password"} value={password} onChange={(e) => setPassword(e.target.value)} />
            {mode === "register" && <p id="pw-help" className="mt-1 text-xs text-fg-3">At least 8 characters. Stored only as an Argon2 hash.</p>}
          </div>
          <InlineError error={error} />
          <Button type="submit" variant="primary" size="lg" className="w-full" loading={busy}>
            {mode === "login" ? "Sign in" : "Create account"}
          </Button>
        </form>
        {canRegister && !firstUser && (
          <p className="mt-5 text-center text-sm text-fg-2">
            {mode === "login" ? "No account? " : "Already registered? "}
            <button type="button" className="cursor-pointer text-accent-fg hover:underline" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); }}>
              {mode === "login" ? "Create one" : "Sign in"}
            </button>
          </p>
        )}
      </div>
    </div>
  );
}
