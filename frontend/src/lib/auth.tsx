import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { LoadingRegion, Skeleton } from "@/components/ui";
import { api, getToken, setToken } from "./api";
import type { User } from "./types";

type AuthStatus = { auth_required: boolean; open_registration: boolean; has_users: boolean };
type Ctx = {
  user: User | null;
  status: AuthStatus | null;
  ready: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, name: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<Ctx | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        setStatus(await api<AuthStatus>("/auth/status"));
      } catch {
        setStatus(null); // older backend or unreachable: behave as open mode
      }
      if (getToken()) {
        try {
          setUser(await api<User>("/auth/me"));
        } catch {
          setToken("");
        }
      }
      setReady(true);
    })();
    const onUnauthorized = () => {
      if (getToken()) {
        setToken("");
        setUser(null);
      }
    };
    window.addEventListener("md:unauthorized", onUnauthorized);
    return () => window.removeEventListener("md:unauthorized", onUnauthorized);
  }, []);

  const finish = (r: { access_token: string; user: User }) => {
    setToken(r.access_token);
    setUser(r.user);
    setStatus((s) => (s ? { ...s, has_users: true } : s));
  };
  const login = useCallback(async (email: string, password: string) => {
    finish(await api("/auth/login", { json: { email, password } }));
  }, []);
  const register = useCallback(async (email: string, name: string, password: string) => {
    finish(await api("/auth/register", { json: { email, name, password } }));
  }, []);
  const logout = useCallback(() => {
    setToken("");
    setUser(null);
  }, []);

  return <AuthContext.Provider value={{ user, status, ready, login, register, logout }}>{children}</AuthContext.Provider>;
}

export function useAuth(): Ctx {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}

/** Redirects to /login when the server requires sign-in and nobody is signed in. */
export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, status, ready } = useAuth();
  const location = useLocation();
  if (!ready) return <LoadingRegion label="Checking sign-in"><Skeleton className="h-40" /></LoadingRegion>;
  if (status?.auth_required && !user) return <Navigate to="/app/login" replace state={{ from: location.pathname }} />;
  return <>{children}</>;
}

export function canEdit(user: User | null, status: AuthStatus | null) {
  return !user ? !status?.auth_required : user.role !== "viewer";
}

export function isAdmin(user: User | null, status: AuthStatus | null) {
  return !user ? !status?.auth_required : user.role === "admin";
}
