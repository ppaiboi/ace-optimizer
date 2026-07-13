# ACE vs GEPA vs Baseline — training replay

A tiny Vite + React + TypeScript app that **replays** how three DSPy optimizers
train the *same* program on the ACE paper's FiNER benchmark (XBRL / US-GAAP tag
classification), side by side:

- **Baseline** — the raw `dspy.Predict("question -> answer")`, never trained.
- **GEPA** — reflects on failures and rewrites its instruction each iteration.
- **ACE** — accumulates a reusable playbook of tactics, bullet by bullet.

Press **Play** to fast-forward the training, drag the **scrubber** to jump to any
step, **Replay** to run it again.

All numbers and artifacts are **real** — pulled from our own reproduction
(DeepSeek-V3.1 via AWS Bedrock, train=1000 / val=500 / test=441). See `src/data.ts`.

## Run locally

```bash
npm install
npm run dev      # http://localhost:5173
```

## Deploy to Vercel

Vercel auto-detects Vite (build `npm run build`, output `dist`). The only setting
that matters is the **Root Directory**, because the app lives in `demo/`:

1. Import the repo at <https://vercel.com/new>.
2. Set **Root Directory** → `demo`.
3. Deploy. (Framework preset, build command, and output dir are auto-filled.)

Or from the CLI:

```bash
cd demo
npx vercel        # first run links/creates the project
npx vercel --prod
```

## Updating with the full ACE number

ACE's full-run **test** accuracy is measured last; until it lands, the ACE column
shows `running…`. When the reproduction finishes, set one value in
`src/data.ts` and the ACE bar goes solid:

```ts
export const TEST_ACC = {
  baseline: 0.652,
  gepa: 0.707,
  ace: 0.xx,   // <- drop in the measured number
};
```
