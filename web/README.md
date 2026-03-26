# Observability UI

This is the SvelteKit scaffold for the observability and review UI.

Scope:

- browse chips, pairs, and benchmark runs from this repo
- inspect failures and outliers visually
- evolve toward a human review and annotation surface

Important constraints:

- this UI is for observability in `geogrok-training`, not the product stack
- `reference/geogrok/` is used as reference only
- review artifacts may be compressed for human consumption, but training and
  evaluation continue to use raw manifest-backed reads

Planned reuse from `reference/geogrok/`:

- Kakadu WASM decode worker and viewer concepts from `services/ui`
- review-only HTJ2K artifact generation concepts from `libs/py/pykdu`
- API shape ideas from `libs/py/eopm`

Suggested local workflow once dependencies are installed:

```bash
./scripts/run_obs_dev.sh
```

That starts the UI on `0.0.0.0:5174` by default so it can be opened from
another machine on the same LAN.

Or run the UI side directly:

```bash
cd web
npm install
npm run dev
npm run check
npm run lint
npm run format:check
```

The paired Python API scaffold lives under `src/geogrok/obs/`.

Frontend tooling:

- `svelte-check` validates Svelte template and TypeScript correctness
- `Biome` handles linting, formatting, and import organization
