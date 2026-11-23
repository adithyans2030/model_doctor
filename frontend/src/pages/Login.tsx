import { useState, type FormEvent } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { ErrorBox, cx } from "@/components/legacy-ui";
import { useAuth } from "@/lib/auth";

export default function Login() {
  const { user, status, login, register } = useAuth();
  const navigate = useNavigate();
  const from = (useLocation().state as { from?: string } | null)?.from || "/";
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
    <div className="mx-auto mt-10 max-w-sm">
      <div className="card p-6">
        <h1>{mode === "login" ? "Sign in" : firstUser ? "Create the admin account" : "Create an account"}</h1>
        {firstUser && <p className="mt-1 text-[13px] text-fg-2">This is the first account, so it becomes the admin.</p>}
        <form className="mt-4 space-y-3" onSubmit={submit}>
          <label className="block text-[13px] text-fg-2">Email
            <input className="field mt-1 w-full" type="email" autoComplete="email" required value={email}
              onChange={(e) => setEmail(e.target.value)} />
          </label>
          {mode === "register" && (
            <label className="block text-[13px] text-fg-2">Name
              <input className="field mt-1 w-full" required maxLength={120} value={name} onChange={(e) => setName(e.target.value)} />
            </label>
          )}
          <label className="block text-[13px] text-fg-2">Password
            <input className="field mt-1 w-full" type="password" required minLength={mode === "register" ? 8 : 1}
              autoComplete={mode === "login" ? "current-password" : "new-password"} value={password}
              onChange={(e) => setPassword(e.target.value)} />
          </label>
          {mode === "register" && <p className="text-xs text-fg-3">At least 8 characters. Stored only as an Argon2 hash.</p>}
          <button className={cx("btn btn-primary w-full justify-center")} disabled={busy}>
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
          <ErrorBox error={error} />
        </form>
        {canRegister && !firstUser && (
          <p className="mt-4 text-center text-[13px] text-fg-2">
            {mode === "login" ? "No account? " : "Already registered? "}
            <button className="text-accent hover:underline" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); }}>
              {mode === "login" ? "Create one" : "Sign in"}
            </button>
          </p>
        )}
      </div>
    </div>
  );
}
