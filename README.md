# injection-bench

A small, dependency-free harness for evaluating **prompt-injection and jailbreak detectors** the way
they will actually be used: trained on the past, tested on the future, with confidence intervals and
shortcut baselines. It exists because detectors that look strong on author-written examples often fall
apart on real traffic.

What it does that a quick script usually doesn't:

- **Time split with leakage control.** Train on earlier items, test on later ones. Near-duplicate
  prompts (same normalised 300-character prefix) are removed, and no test prompt may duplicate a train prompt.
- **Operating-point metrics.** Recall at fixed false-positive rates (1%, 5%), AUC, and precision at a stated
  prevalence, each with a 95% bootstrap confidence interval. A 0.5 threshold on an arbitrary score says little.
- **Shortcut baselines.** `length-only` and a hand-written `keywords` detector show how much of a score is
  explained by dataset artefacts.
- **Plug in your own detector.** Anything with `score(list[str]) -> list[float]` (and an optional `fit`).

## Quick start

```bash
pip install injection-bench
injectbench download data/                      # public JailbreakHub CSVs (Shen et al., CCS 2024, MIT)
injectbench run --data data/ --detector length --detector keywords --detector hashed-lr --out report.md
```

Your own detector:

```python
# mydet.py
class MyDetector:
    name = "mine"
    def fit(self, texts, labels): ...          # optional
    def score(self, texts): return [...]       # higher = more likely an attack
```

```bash
injectbench run --data data/ --detector mydet:MyDetector
```

Your own data: a JSONL file with `text`, `label` (1 = attack) and `date` fields: `--data my.jsonl`.

## Reference results

JailbreakHub in-the-wild prompts, 1,405 jailbreaks and 13,735 regular prompts (rows without a date are
dropped). Train 7,895 prompts (757 jailbreaks) on the earlier part, test 5,503 prompts (398 jailbreaks, 7.2%
prevalence) on the later part. Reproduce with the command above plus `--detector guardkit:rules --detector guardkit:model`
(needs `pip install guardkit`).

| detector | AUC (95% CI) | recall @1% FPR (95% CI) | recall @5% FPR | precision @1% FPR, 5% prevalence | ms/item |
|---|---|---|---|---|---|
| length-only | 0.587 (0.56-0.61) | 0.020 (0.01-0.04) | 0.080 | 0.096 | 0.0 |
| keywords | 0.653 (0.63-0.68) | 0.053 (0.03-0.07) | 0.214 | 0.217 | 0.04 |
| hashed-lr (trained on the train split) | 0.742 (0.72-0.77) | 0.113 (0.08-0.16) | 0.264 | 0.373 | 0.5 |
| guardkit:rules | 0.511 (0.50-0.52) | 0.000 (0.00-0.00) | 0.048 | 0.000 | 0.6 |
| guardkit:model | 0.739 (0.71-0.76) | 0.118 (0.08-0.18) | 0.261 | 0.383 | 0.8 |

Reading it: length alone gives AUC 0.59, so there is a mild shortcut but not a dominant one. Cheap
bag-of-words detectors reach AUC of about 0.74 and catch only about one in nine jailbreaks at a 1%
false-positive rate. Regular expressions are close to chance on this data.

## Caveats

- "recall @X% FPR" sets the threshold on the **test negatives**. It measures ranking quality at a given
  false-positive budget; it is not a deployable operating point, which you must choose on held-out data.
- The only bundled dataset labels jailbreak intent in user prompts, not document-embedded prompt injection.
  Its "regular" class includes templates that open with "ignore previous instructions". Use `--data` with your
  own labelled data for your threat model.
- Pure-Python metrics are fine for thousands of items, not millions. Bootstrap CIs use 200 resamples by default.
- Reference rows come from one dataset and one split. Do not read them as a ranking of detector families.

MIT licensed.
