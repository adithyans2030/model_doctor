import { AuthProvider } from "@/lib/auth";
import { AppLayout } from "./AppLayout";

/** Dashboard shell: auth (and its /auth/status call) only runs inside /app, never on the landing page. */
export default function AppShell() {
  return (
    <AuthProvider>
      <AppLayout />
    </AuthProvider>
  );
}
