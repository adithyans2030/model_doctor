import { Navigate } from "react-router-dom";
import { PageHeader } from "@/components/domain/PageHeader";
import { QueryView } from "@/components/domain/QueryView";
import { Badge, Button, Card, InlineError, TBody, TD, TH, THead, TR, Table, TableSkeleton } from "@/components/ui";
import { fmt } from "@/lib/api";
import { isAdmin, useAuth } from "@/lib/auth";
import { useUpdateUser, useUsers } from "@/lib/queries";
import type { Role } from "@/lib/types";

export default function Users() {
  const { user, status } = useAuth();
  const admin = isAdmin(user, status) && !!user;
  const users = useUsers(admin);
  const update = useUpdateUser();
  if (!admin) return <Navigate to="/app" replace />;

  return (
    <div>
      <PageHeader crumbs={[{ label: "Projects", to: "/app" }, { label: "Users" }]} title="Users"
        meta={<span><b className="font-medium text-fg">Viewers</b> read everything and ask the assistant. <b className="font-medium text-fg">Members</b> also upload, train and run experiments. <b className="font-medium text-fg">Admins</b> also approve and promote models and manage users.</span>} />
      <InlineError error={update.error} className="mb-3" />
      <Card flush>
        <QueryView query={users} label="Loading users" skeleton={<TableSkeleton rows={3} cols={5} />}>
          {(list) => (
            <Table label="Users">
              <THead><TR><TH>Name</TH><TH>Email</TH><TH>Role</TH><TH>Status</TH><TH>Last sign-in</TH></TR></THead>
              <TBody>
                {list.map((u) => {
                  const self = u.id === user.id;
                  return (
                    <TR key={u.id}>
                      <TD className="whitespace-nowrap">{u.name}{self && <Badge tone="accent" className="ml-2">you</Badge>}</TD>
                      <TD className="font-mono text-xs">{u.email}</TD>
                      <TD>
                        <select className="field" aria-label={`Role for ${u.name}`} value={u.role} disabled={self}
                          onChange={(e) => update.mutate({ id: u.id, patch: { role: e.target.value as Role } })}>
                          <option value="viewer">viewer</option><option value="member">member</option><option value="admin">admin</option>
                        </select>
                      </TD>
                      <TD className="whitespace-nowrap">
                        <Badge tone={u.is_active ? "success" : "neutral"}>{u.is_active ? "active" : "inactive"}</Badge>
                        <Button size="sm" variant="ghost" className="ml-2" disabled={self} onClick={() => update.mutate({ id: u.id, patch: { is_active: !u.is_active } })}>
                          {u.is_active ? "Deactivate" : "Reactivate"}
                        </Button>
                      </TD>
                      <TD className="whitespace-nowrap text-xs text-fg-3">{u.last_login_at ? fmt.date(u.last_login_at) : "never"}</TD>
                    </TR>
                  );
                })}
              </TBody>
            </Table>
          )}
        </QueryView>
      </Card>
    </div>
  );
}
