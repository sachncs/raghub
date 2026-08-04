# Tier 13 — `mkdocs build --strict` (Item 71)

After Tier 9 + Tier 11 + Tier 12 land, run `mkdocs build --strict`
and fix any remaining broken cross-references.

---

## Item 71 — `mkdocs build --strict` passes

- **File(s)**: `mkdocs.yml`, any docs file with a broken cross-reference
- **Change**: Fix `nav:` entries, broken internal links, missing files.
- **Test**: `mkdocs build --strict` exits 0.
- **Acceptance criteria**:
  - T3 — pass.
  - O5 — docs match shipped behaviour.
- **Success criteria**:
  - `mkdocs build --strict` exits 0.
  - No warnings emitted.

---

## Tier 13 acceptance gate

- `mkdocs build --strict` exits 0.
