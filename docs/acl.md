# raghub — access control

Two layers of authorization ship in raghub, enforced at every
retrieval surface (vector store, graph store, trace corpus, memory
store).

## Workspace RBAC (`workspace_member`)

Every workspace has a `workspace_member(user_id, role)` row per
user. The role determines what the user can do:

| role | can manage users / groups / docs | can ingest | can query |
| --- | --- | --- | --- |
| owner | yes | yes | yes |
| admin | yes | yes | yes |
| member | no | yes | yes |
| viewer | no | no | yes |

`canManageWorkspace(role)` and `canIngest(role)` are the predicates
the API layer uses to gate `/v1/workspaces/members/*` and
`/v1/documents` mutations.

## Document ACL (`document_principal`)

Per-document grants — every chunk retrieval joins
`document_principal` and filters out chunks whose document the
active user's principals can't see.

```
document_principal
  document_id      -- chunks.document_id
  principal_type   -- 'user' | 'role' | 'group'
  principal_id     -- the user / role / group id
  permission       -- 'read' | 'admin'
  granted_by       -- user_id who created the grant
  granted_at       -- epoch ms
```

Default ACL on ingest is `(doc, 'user', owner, 'admin')` plus
`(doc, 'user', admin_user, 'admin')` for every workspace admin.

## Principals

A **principal** is a tuple `{ type, id }` — the resolved access
identity for the active user:

```
principals: [
  { type: 'user',  id: <self> },
  { type: 'role',  id: <roles-the-user-belongs-to> },
  { type: 'group', id: <groups-the-user-belongs-to> },
]
```

`SqliteVecStore` accepts the principal list via
`StoreFilter.principals` and joins `document_principal` to deny
chunks the user can't reach.

## Server enforcement

| layer | enforces |
| --- | --- |
| `JwtAuthMiddleware` | parses bearer token, attaches `claims` to context |
| `requireWorkspaceRole(...)` (in route handlers) | checks `workspace_member` for write actions |
| `StoreFilter.principals` | enforces ACL at the SQLite layer |
| `allowCompanyFilter(user)` | optional RBAC per `User.allowedCompanies` |

There is no escape hatch — even an admin's own chunks only come
back if `document_principal` grants them.

## API surface

```
POST   /v1/documents/:id/principals      grant (read | admin) to (user | role | group)
DELETE /v1/documents/:id/principals      revoke
GET    /v1/documents/:id/principals      list grants on a document

POST   /v1/workspaces/members            invite a user with a role
PATCH  /v1/workspaces/members/:userId    change role
DELETE /v1/workspaces/members/:userId    remove
GET    /v1/workspaces/members            list members
```