# RBAC & ACL

Revex enforces access control at multiple layers: workspace membership,
document-level ACLs, and retrieval-time filters.

## Workspace members

`SqliteWorkspaceMemberStore` stores `workspace_member` rows with a role. Roles:

| Role | Permission |
|---|---|
| `owner` | Full control. |
| `admin` | Invite members, change roles, reset passwords, manage ACLs. |
| `member` | Ingest + query. |
| `viewer` | Read only. |

Helpers:

- `canManageWorkspace(role)` — owner/admin can manage the workspace.
- `canIngest(role)` — owner/admin/member can ingest (viewer cannot).
- `resolveRoleFor(member)`.

API surface: `GET`/`POST /v1/workspaces/members`,
`PATCH`/`DELETE /v1/workspaces/members/:userId`.

## Groups & roles

`SqliteRoleStore` and `SqliteGroupStore` back named roles and groups with
memberships (`role`, `role_member`, `workspace_group`,
`workspace_group_member`). A named role can reference users or groups.

## Document ACL

`SqliteDocumentPrincipalStore` stores `document_principal` rows mapping a
principal (user / role / group) to a permission (`read` | `admin`) per
document. `GrantInput` carries `principalType`, `principalId`, `permission`.

API surface (owner or admin only):

- `GET /v1/documents/:id/principals`
- `POST /v1/documents/:id/principals`
- `DELETE /v1/documents/:id/principals`

## Retrieval-time filtering

`StoreFilter` carries `workspaceId`, `userId`, `collectionId`, `principals`,
and `allowedCompanies`. The vector store enforces ACLs at the SQLite layer via
a parameterized subquery (`SqliteVecStore.buildAclClause`): a user may see only
documents they own, are a member of the workspace for, or have a grant on.

`allowedCompanyFilter(user)` builds the employer-scoped filter for a
`User.allowedCompanies` list. Derived into `rbac_filter` on the orchestrator's
`InvocationState`.

## Authentication

- `BcryptHasher` (configurable rounds) hashes passwords.
- `JwtService` mints/verifies JWTs (`HS256`/`384`/`512`) with claims
  `sub`, `workspace_id`, `is_admin`.
- The API `jwtAuthMiddleware` verifies the `Authorization: Bearer` header and
  reads the `revex_workspace_key` cookie onto the request context
  (`getClaims`, `getPassphrase`).

## Audit log

`SqliteAuditEventStore` records events such as `document.acl.grant`,
`document.acl.revoke`, `settings.update`,
`workspace.member.{add,role_change,remove}`,
`ingest.complete`, `ingest.failure`. Surface: `GET /v1/audit`,
`GET /v1/audit/kinds`, `GET /v1/stats`.