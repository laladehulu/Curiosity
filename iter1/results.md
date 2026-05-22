# Results — iter1 (PLACEHOLDER)

Per protocol §"Phase 6: Analysis and writeup → `results.md`".

> **Status: not yet run.** This file is a structural placeholder. It will be
> filled in only after Phase 1–4 complete. Do NOT write results-style prose
> here until real numbers exist. The human author writes the paper from this
> report.

## 1. Summary of findings

_TBD — single paragraph, claims-only, after Phase 5 numbers exist._

## 2. Headline numbers

_TBD — central table from `analysis/headline_comparison.py`; render
`tables/headline.csv` here once Phase 5 produces it._

## 3. Per-task analysis

### 3.1 Pendulum
_TBD._

### 3.2 MountainCarContinuous
_TBD._

### 3.3 BipedalWalker
_TBD (or "scope-reduced; see critique.md §F4" if Box2D failed)._

## 4. Diversity metric agreement

_TBD — do AST, embedding, and behavioral distances agree on which pool is more
diverse? Cross-tab with Spearman rank correlations._

## 5. Seed noise floor

_TBD — per protocol §"Predicted seed-noise issue". Compute the noise floor
(5 different seeds of a single reward) and compare against the between-pool
mechanism effect. Report explicitly, even if it kills the headline._

## 6. Failure modes encountered

_TBD. List scope reductions, install failures, runs that did not converge._

## 7. Pre-registered hypotheses outcomes

Per protocol §"What this paper claims and does not claim":

- F1 (mechanisms indistinguishable): observed / not observed / partial
- F2 (archive wins diversity, loses performance): observed / not observed / partial
- F3 (results task-structure-dependent in unexpected ways): observed / not observed / partial

_Fill in after analysis. Negative or null results are publishable, do not hide._
