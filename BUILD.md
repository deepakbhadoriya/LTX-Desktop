# Building MLX Studio for macOS

## Prerequisites

- Node.js 22+
- pnpm 10.30.3 (`npm install -g pnpm@10.30.3`)
- uv ([install](https://docs.astral.sh/uv/))
- git, curl

## Quick Build (Unpacked .app for local testing)

```bash
# Full build — downloads Python, installs MLX deps, builds frontend, packages .app
CSC_IDENTITY_AUTO_DISCOVERY=false bash scripts/local-build.sh --unpack
```

Output: `release/mac-arm64/LTX Desktop.app`

## Launch

```bash
open release/mac-arm64/LTX\ Desktop.app
```

If macOS Gatekeeper blocks it:

```bash
xattr -cr release/mac-arm64/LTX\ Desktop.app
```

## Rebuild Options

```bash
# Skip Python env setup (reuse existing python-embed/)
CSC_IDENTITY_AUTO_DISCOVERY=false bash scripts/local-build.sh --skip-python --unpack

# Clean rebuild from scratch
CSC_IDENTITY_AUTO_DISCOVERY=false bash scripts/local-build.sh --clean --unpack

# Full DMG installer (requires Apple Developer certificate)
bash scripts/local-build.sh
```

## What Each Flag Does

| Flag | Effect |
|---|---|
| `--unpack` | Builds unpacked `.app` only (no DMG, faster) |
| `--skip-python` | Reuses existing `python-embed/` directory |
| `--clean` | Removes `python-embed/`, `release/`, `dist/`, `dist-electron/` before building |
| `CSC_IDENTITY_AUTO_DISCOVERY=false` | Skips code signing (no Apple Developer cert needed) |

## Build Stages

1. **Python environment** — Downloads standalone Python 3.13 (arm64), installs MLX deps (mlx, mlx-video, mflux) into `python-embed/` (~1.25 GB)
2. **pnpm install** — Installs Node.js dependencies
3. **Frontend build** — Vite builds React frontend + Electron main process
4. **Packaging** — electron-builder creates the `.app` bundle

## Troubleshooting

**`packages field missing or empty`** — Wrong pnpm version. Install the correct one: `npm install -g pnpm@10.30.3`

**Gatekeeper blocks the app** — Run `xattr -cr release/mac-arm64/LTX\ Desktop.app`

**Python env is stale** — Delete `python-embed/` and rebuild without `--skip-python`
