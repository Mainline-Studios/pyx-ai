# Latest Update Logs

Human-readable release notes for Pyx web and API deploys.

## Naming

| File | Meaning |
|------|---------|
| `{version} {Nickname} LATEST.md` | Current release (only one at a time) |
| `{version} {Nickname}.md` | Older releases (remove `LATEST` from the filename when superseded) |

Examples:

- `1.2 Maestro LATEST.md` — live now
- `1.1 Riverside.md` — archived
- `1.05 Schoolhouse Patches.md` — archived after 1.1 ships (drop `LATEST` from the previous file)

## When to update

On every **deploy** or meaningful **release commit**:

1. Rename the previous `* LATEST.md` → drop the word `LATEST` from the filename.
2. Add a new markdown file for the version going out with `LATEST` in the name.
3. Commit the log folder with your deploy or release commit.
4. Push to GitHub.

## Version line

Start each log with:

```markdown
# Pyx {version} — {Nickname}
**Status:** LATEST · **Date:** YYYY-MM-DD
**Preceded by:** [1.0 Schoolhouse](1.0%20Schoolhouse.md) · **Superseded by:** —
```

- **Preceded by** — link to the previous release file (use `—` if this is the first tracked log).
- **Superseded by** — link to the newer release that replaced this one (use `—` while this file is still `LATEST`).

Remove **LATEST** from the status line when archiving. When you ship a new version, update the old file’s **Superseded by** link and set the new file’s **Preceded by** to that old file.
