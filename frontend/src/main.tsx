import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode, Suspense, lazy, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { ToastProvider } from "@/components/ui";
import { RequireAuth } from "@/lib/auth";
import { createQueryClient } from "@/lib/queries";
import "./index.css";

// Every page is its own chunk: the landing page never downloads dashboard code and vice versa.
const Landing = lazy(() => import("@/pages/landing/Landing"));
const AppShell = lazy(() => import("@/components/AppShell"));
const Home = lazy(() => import("@/pages/Home"));
const ProjectPage = lazy(() => import("@/pages/Project"));
const DatasetPage = lazy(() => import("@/pages/dataset/DatasetPage"));
const ModelPage = lazy(() => import("@/pages/model/ModelPage"));
const Login = lazy(() => import("@/pages/Login"));
const Users = lazy(() => import("@/pages/Users"));
const NotFound = lazy(() => import("@/pages/NotFound"));

const queryClient = createQueryClient();

/** Old dashboard URLs (before the landing page took "/") keep working. */
function LegacyRedirect({ to }: { to: string }) {
  const { id = "" } = useParams();
  return <Navigate to={to.replace(":id", id)} replace />;
}

const authed = (el: ReactNode) => <RequireAuth>{el}</RequireAuth>;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <BrowserRouter>
            <Routes>
              <Route index element={<Suspense fallback={null}><Landing /></Suspense>} />
              <Route path="app" element={<Suspense fallback={null}><AppShell /></Suspense>}>
                <Route index element={authed(<Home />)} />
                <Route path="login" element={<Login />} />
                <Route path="projects/:id" element={authed(<ProjectPage />)} />
                <Route path="datasets/:id" element={authed(<DatasetPage />)} />
                <Route path="models/:id" element={authed(<ModelPage />)} />
                <Route path="admin/users" element={authed(<Users />)} />
                <Route path="*" element={<NotFound />} />
              </Route>
              <Route path="projects/:id" element={<LegacyRedirect to="/app/projects/:id" />} />
              <Route path="datasets/:id" element={<LegacyRedirect to="/app/datasets/:id" />} />
              <Route path="models/:id" element={<LegacyRedirect to="/app/models/:id" />} />
              <Route path="login" element={<Navigate to="/app/login" replace />} />
              <Route path="admin/users" element={<Navigate to="/app/admin/users" replace />} />
              <Route element={<Suspense fallback={null}><AppShell /></Suspense>}>
                <Route path="*" element={<NotFound />} />
              </Route>
            </Routes>
        </BrowserRouter>
      </ToastProvider>
    </QueryClientProvider>
  </StrictMode>,
);
