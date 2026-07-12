# Order-Driven Zero-Shot Picking Cell (SO-101 × Cyberwave)

Physical AI agent built during the **Cyberwave Builders Program — Cohort 2** (Jul 2026).

An SO-101 arm receives warehouse orders via a Cyberwave Workflow **webhook** and fulfills them with **open-vocabulary zero-shot manipulation** — BERT NER extracts the target, GroundingDINO grounds it in the overhead camera frame, a calibrated homography projects it to table coordinates, and pick/place skills execute via the Cyberwave SDK. The loop closes with visual verification and fulfillment alerts. New objects require **zero retraining**.

> 🚧 In progress — see [roadmap.md](roadmap.md) and [progress.md](progress.md). Architecture and commands: [runbook.md](runbook.md).

## Loop (Perceive → Reason → Act → Feedback)

```
webhook order ─▶ capture overhead frame ─▶ NER + GroundingDINO + homography
     ▲                                            │
     │                                            ▼
"fulfilled" email/alert ◀─ visual verify ◀─ pick(x,y) / place(bin) via robot.joints.set
```

## Repo layout

`src/` (perception · planning · control · agent_service · wms_mock) · `tests/` · `docs/` (distilled platform + hardware docs) · `hardware/config/` (calibrations) · working files: CLAUDE.md, roadmap, runbook, progress, decisions.

## Publishing (once `gh` is installed)

```bash
sudo apt install gh && gh auth login
gh repo create so101-zeroshot-picking-cell --private --source . --push
```
