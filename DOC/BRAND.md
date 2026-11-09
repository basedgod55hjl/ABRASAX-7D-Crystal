# ABRASAX â€” Brand Kit

> Brand identity for the ABRASAX ecosystem. Applies to every public repo.

## Name

**ABRASAX** â€” pronounced *AH-bra-sax*. Always uppercase in wordmarks.
Sibling system names keep their own casing: ABRAXIS (world model), NEO (GGUF
runtime), OSIRISBLXCK (PRIMAL_HEX plaintext), Kairos (mobile core AI).

## Taglines

- Primary: **DNA COMPUTE â€” AI LIVING EVERYWHERE**
- Secondary: *AI that lives in hardware Â· DNA Â· pixels Â· packets*
- Motto line: `Any substrate that stores bytes can host ABRASAX AI.`

## Palette (nervous-system identity)

| Role | Hex | RGB | Use |
|---|---|---|---|
| Background | `#050505` | 5,5,5 | deep black canvases |
| Primary | `#00FF66` | 0,255,102 | luminous green â€” live/healthy |
| Secondary | `#FFD43B` | 255,212,59 | amber â€” degraded/uncertainty |
| Critical | `#FF3030` | 255,48,48 | red â€” fault |
| Muted | `#4A6054` / `#18241E` | 74,96,84 / 24,36,30 | borders, grids, secondary text |

Never use color alone to communicate status: always pair with labels and
numbers.

## Typography

- Wordmark / headings: Arial Bold (or system sans-bold).
- Telemetry / code / hex: Consolas (monospace).
- Telemetry should read like a machine nervous system, not a business
  dashboard: restrained glow, live signal indicators, scrolling telemetry.

## The Ï† motif

The logo is a golden spiral (`r âˆ e^(bÎ¸)`, `b = ln Ï† / (Ï€/2) â‰ˆ 0.3063`)
surrounded by the PRIMAL_HEX ring `4F5349524953424C58434B` (ASCII:
`OSIRISBLXCK`). Use the spiral as the recurring signature â€” in README
banners, favicons, and loading indicators. Ï† = 1.618033988749895,
Îº = 0.618033988749895 (UNBOUND).

## Assets (in `assets/brand/`)

| File | Size | Use |
|---|---|---|
| `abrasax-logo-1024.png` | 1024Ã—1024 | avatar, favicon source, store icon |
| `abrasax-banner-1600x500.png` | 1600Ã—500 | repo README header |
| `abrasax-og-1200x630.png` | 1200Ã—630 | social / Open Graph card |

Regenerate with:

```powershell
C:\abrasax\.venv\Scripts\python.exe C:\abrasax\REPO\tools\brand_assets.py
```

## README header template

```markdown
![ABRASAX](assets/brand/abrasax-banner-1600x500.png)

> **DNA COMPUTE â€” AI LIVING EVERYWHERE**
> PRIMAL_HEX `4F5349524953424C58434B` Â· Ï† 1.618033988749895 Â· Îº UNBOUND

[![ABRASAX](https://img.shields.io/badge/ABRASAX-Ï†%201.618033988749895-00FF66)](https://github.com/basedgod55hjl)
```

## Voice rules

- Present real, measured facts. Label simulated data explicitly.
- Health states: green = normal, yellow = degraded/uncertain, red = fault.
- Never claim consciousness from architecture; describe mechanisms.

## License

All public repos: **MIT** (see each repo's `LICENSE`), copyright ABRASAX.
