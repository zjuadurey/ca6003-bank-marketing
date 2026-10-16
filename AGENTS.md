# Repository instructions for agents

## Read first

Read `PROGRESS.md` for current completion and pending work, then `ENVIRONMENT.md` for environment setup. `requirements.current.txt` is the authoritative dependency entry point (includes `requirements.D.lock.txt`); `.python-version` specifies Python 3.12. Do not combine the historical A/C locks with the current lock or install latest versions opportunistically. Actual runtime versions must be reported, never inferred from a lock alone.

## Evidence and authorship

Preserve A's raw bytes, `config/dataset_lock.json`, frozen split indices and manifest. C formal pipeline lives in `src/preprocessing.py`; D evaluation in `src/d_evaluation.py`; B training exploration in `src/b_discovery.py`. `outputs/integration/CD_import.json` records imported source. Do not attribute D's old reference implementation to C. Historical delivered reports describe past states; current progress is in `PROGRESS.md`.

Explore only `load_partition('train')`. Never tune features, models, groups, thresholds or hyperparameters using holdout results. Keep the original 11-column whitelist, four S/U experiments, logistic regression and threshold 0.5. Report negative and unchanged effects as well as improvements. Preserve original row_id order.

## Running and frozen results

Use a project-local environment with user authorization for installation. Do not modify other environments. See `ENVIRONMENT.md` for ordered commands.

Never remove or overwrite `outputs/D/freeze.json`, `test_started.json` or `test_completed.json` to rerun evaluation. After freeze, notebook replay should recompute validation without overwriting frozen files, and recompute final metrics from saved predictions without fitting models on the test set again. Before changing frozen code, retain its evidence and agree a separately named protocol with the user.

The task owner explicitly authorized completing final calculations in this session before team human sign-off. The runner supports an explicit `--user-authorized` receipt with `human_review=pending`, separate from its original real-human-review receipt. This is not evidence that any member personally reviewed or signed. No future agent should invent permission, names, dates or signatures.

## Documentation and delivery

Keep `PROGRESS.md`, current results and environment instructions consistent with actual runs. `analysis.ipynb` is the integrated notebook; `notebooks/B_discovery.ipynb` is the B chapter. Preserve core code in the integrated notebook. Final artifacts must include data provenance, limitations, GenAI disclosure and accurate member contribution attribution. Public upload, Git push, course submission and messages to teammates need explicit user authorization.
