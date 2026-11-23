import { useState } from "react";
import { Navigate } from "react-router-dom";
import { ErrorBox, Loading, Section } from "@/components/ui";
import { api, fmt } from "@/lib/api";
import { isAdmin, useAuth } from "@/lib/auth";
import { useResource } from "@/lib/hooks";
import type { User } from "@/lib/types";

export default function Users() {
  const { user, status } = useAuth();
  const users = useResource<User[]>("/auth/users");
  const [error, setError] = useState<string | null>(null);
  if (!isAdmin(user, status) || !user) return <Navigate to="/" replace />;

  async function update(id: string, patch: Partial<Pick<User, "role" | "is_active">>) {
    setError(null);
    try {
      await api(`/auth/users/${id}`, { method: "PATCH", json: patch });
      users.reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <Section title="Users">
      <p className="mb-3 text-[13px] text-fg-2">
        <b>Viewers</b> can read everything and ask the assistant. <b>Members</b> can also upload, train and run experiments.
        <b> Admins</b> can additionally approve/promote models and manage users.
      </p>
      <ErrorBox error={error || users.error} />
      {!users.data ? <Loading /> : (
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Last sign-in</th></tr></thead>
            <tbody>
              {users.data.map((u) => (
                <tr key={u.id}>
                  <td>{u.name}{u.id === user.id && <span className="badge ml-1.5">you</span>}</td>
                  <td>{u.email}</td>
                  <td>
                    <select className="field" value={u.role} disabled={u.id === user.id}
                      onChange={(e) => update(u.id, { role: e.target.value as User["role"] })}>
                      <option value="viewer">viewer</option><option value="member">member</option><option value="admin">admin</option>
                    </select>
                  </td>
                  <td>
                    <button className="btn" disabled={u.id === user.id} onClick={() => update(u.id, { is_active: !u.is_active })}>
                      {u.is_active ? "Active · deactivate" : "Inactive · reactivate"}
                    </button>
                  </td>
                  <td className="text-xs text-fg-3">{u.last_login_at ? fmt.date(u.last_login_at) : "never"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}
