# TakeMeter — Discourse Quality Classifier for r/vegan

TakeMeter is a fine-tuned text classifier that labels the *kind* of discourse in a
r/vegan post or comment: a reasoned **argument**, a bold **hot take**, or a personal
**experience**. It compares a fine-tuned `distilbert-base-uncased` model against a
zero-shot `llama-3.3-70b-versatile` baseline on the same held-out test set.

> Design notes, edge-case decision rules, and the AI tool plan live in
> [`planning.md`](planning.md). This README is the final report and stands on its own.

## Community

I chose **r/vegan** because its discourse is active, text-heavy, and varies enormously
in quality and intent. The same topic — dairy, ethics, going vegan around non-vegan
family — gets discussed as careful ethical/health arguments, blunt provocative
declarations, or personal stories and support requests. That variety is exactly what
makes the classification distinction meaningful to people in the community.

## Label Taxonomy

Three mutually exclusive labels:

| Label | Definition | Example |
|---|---|---|
| **argument** | A structured case for/against a position (ethical, environmental, health) backed by specific, verifiable reasoning that could stand on its own without the opinion framing. | "Animal agriculture is ~14.5% of human-caused GHG emissions, and beef needs ~20x the land per gram of protein vs. legumes, so cutting it has an outsized effect." |
| **hot_take** | A bold, confident opinion or moral judgment with little supporting evidence, or evidence that is rhetorical/cherry-picked rather than part of a real argument. | "Anyone who still eats meat in 2026 just doesn't care about the planet, full stop." |
| **experience** | A personal story, emotional expression, or request for support/advice — someone's own vegan journey, family struggles, venting — with little to no general argument. | "Three months in and my parents still leave meat on my plate. How do you all deal with family who won't take it seriously?" |

Second example per label and full definitions: see [`planning.md`](planning.md).

## Data Collection & Labeling

- **Source:** public posts and comments from r/vegan. _(TODO: note subreddit sections / date range you pulled from.)_
- **Labeling process:** each example read individually and labeled against the definitions in `planning.md`. _(TODO: note if/how you used an LLM to pre-label, then reviewed — disclose in AI Usage below.)_
- **Dataset file:** [`data/takemeter_labeled_examples.csv`](data/) — single file, columns `text,label,notes`. The Colab notebook does the 70/15/15 train/val/test split automatically.

### Label distribution

_(TODO: fill in after labeling — count per label. No label should exceed 70%.)_

| Label | Count |
|---|---|
| argument | TODO |
| hot_take | TODO |
| experience | TODO |

### Three difficult-to-label examples

_(TODO: fill in 3 real cases that gave you pause and what you decided. The edge-case decision rules are in `planning.md`; reference them here.)_

1. **Story + argument** — _example text_ → decided **___** because _rule_.
2. **Fact-as-jab** — _example text_ → decided **___** because _rule_.
3. _third case_ → decided **___** because _rule_.

## Fine-Tuning Approach

- **Base model:** `distilbert-base-uncased` (HuggingFace).
- **Training:** sequence classification head over the labeled train split, on a Colab T4 GPU.
- **Hyperparameters:** defaults are 3 epochs, learning rate 2e-5, batch size 16. _(TODO: state the one hyperparameter decision you made and why — e.g., epochs raised to 4 because val loss was still dropping.)_

## Baseline

Zero-shot `llama-3.3-70b-versatile` via Groq, classifying each test example with no
task-specific training. The prompt includes the three label definitions verbatim and
instructs the model to output only the label name.

_(TODO: paste the exact prompt you used and note how results were collected — notebook Section 5.)_

## Evaluation Report

Test set: **32 examples** (true distribution: argument 10, hot_take 7, experience 15).
Full results in [`data/evaluation_results.json`](data/evaluation_results.json); confusion matrix image at [`data/confusion_matrix.png`](data/confusion_matrix.png).

**Headline result: fine-tuning made the classifier *worse*.** The zero-shot Groq
baseline beat the fine-tuned DistilBERT by ~19 points, and the fine-tuned model failed
to learn the `hot_take` class entirely. This did **not** meet my definition of success
(beat baseline AND macro F1 ≥ 0.70) — and the *why* is the most interesting part of the
project.

### Overall accuracy

| Model | Accuracy | Macro F1 |
|---|---|---|
| Zero-shot baseline (llama-3.3-70b) | **0.969** (31/32) | _NEED: from notebook Section 5 output_ |
| Fine-tuned DistilBERT | 0.781 (25/32) | **0.60** |

Δ accuracy = **−0.188** (fine-tuned minus baseline).

### Per-class metrics (fine-tuned)

Computed from the confusion matrix below.

| Label | Precision | Recall | F1 |
|---|---|---|---|
| argument | 1.00 | 1.00 | 1.00 |
| hot_take | 0.00 | 0.00 | **0.00** |
| experience | 0.68 | 1.00 | 0.81 |

`hot_take` F1 is exactly 0 — the model **never once predicted it**. This is the
textbook "one class F1 ≈ 0, others fine" case: the model could not learn that boundary.

_(Baseline per-class metrics: NEED — paste the precision/recall/F1 table the notebook
printed in Section 5 so we can compare class-by-class, not just on accuracy.)_

### Confusion matrix (fine-tuned)

Rows = true label, columns = predicted. (Supplementary image: [`data/confusion_matrix.png`](data/confusion_matrix.png).)

|  | pred argument | pred hot_take | pred experience |
|---|---|---|---|
| **true argument** | 10 | 0 | 0 |
| **true hot_take** | 0 | 0 | **7** |
| **true experience** | 0 | 0 | 15 |

The only off-diagonal mass is a single directional error: **every `hot_take` was
predicted as `experience`.** `argument` is perfectly separated; `experience` catches
everything that isn't `argument` (precision 0.68 because the 7 hot_takes leak in).

### Three wrong predictions, analyzed

All 7 errors are the same failure: **true `hot_take` → predicted `experience`.** Below
I analyze the pattern; I need the actual post texts from the notebook to name 3 specific
ones (see "What I still need" at the bottom).

**The pattern.** The model effectively learned a 2-way rule, not a 3-way one:
*"contains specific evidence/numbers → `argument`; otherwise → `experience`."* It never
carved out `hot_take` at all.

- **Which boundary failed:** `hot_take` vs. `experience`. Both are first-person,
  emotional, and evidence-free on the surface — a confident "anyone who eats meat
  doesn't care about the planet" shares vocabulary and tone with "I can't stop thinking
  about that documentary." The thing that separates them (one is a *judgment/claim*, the
  other is a *personal feeling*) is semantic and subtle, not lexical.
- **Why it's hard:** `argument` has an easy surface signal (numbers, study references,
  causal connectives), so the model locked onto it and got that class perfect. `hot_take`
  has *no* reliable surface signal that distinguishes it from `experience` — so with
  limited data the model folded it into the larger neighboring class.
- **Labeling vs. data problem:** This looks like a **data/class-imbalance problem**, not
  annotation inconsistency. `hot_take` is the minority class (only 7 in test), and with
  ~150 training examples DistilBERT didn't see enough hot_takes to separate them from the
  much larger `experience` class. The boundary itself may also be genuinely thin.
- **What would fix it:** (1) more `hot_take` training examples, ideally hard ones that
  look emotional but are claims; (2) class-weighted loss so the minority class isn't
  swamped; (3) possibly merging `hot_take` and `experience` if the boundary turns out to
  be too thin to learn — though that would abandon the distinction I set out to measure.

> ⚠️ **The other red flag — the baseline is suspiciously high (96.9%).** The spec warns
> that >95% accuracy on a subjective task suggests labels that are *too easy* or test
> leakage. A 70B model nailing 31/32 while a fine-tune can't learn one class points the
> same way: the examples may be too cleanly separable (e.g. if many were LLM-generated or
> templated, each class reads "obvious"). Worth checking whether the dataset reflects
> genuinely messy real r/vegan posts or polished, easy-to-classify ones. _(See reflection.)_

### Sample classifications

_(TODO: 3–5 posts run through the fine-tuned model with predicted label + confidence. Explain why at least one correct prediction is reasonable.)_

| Post (truncated) | Predicted | Confidence |
|---|---|---|
| … | … | … |

## Reflection: learned vs. intended

I intended a three-way distinction along *discourse quality*: reasoned **argument**,
unsupported **hot take**, and personal **experience**. What the model actually learned
was a two-way distinction along *surface form*: **"has evidence markers (numbers, studies,
causal reasoning) vs. doesn't."** It mapped the first bucket to `argument` (perfectly —
F1 1.00) and dumped everything else into `experience`, never instantiating `hot_take` as
a real category.

That gap is the whole point. My `argument` definition happened to align with an easy
lexical signal, so the model captured it cleanly — arguably for the wrong reason (it may
be keying on the *presence of numbers* rather than on whether a claim is actually
reasoned). My `hot_take` definition — "a confident claim *without* evidence" — is defined
by an *absence*, and absences don't give a small model anything to grab onto. To the
model, a hot take and a personal vent are nearly the same object: first-person,
evidence-free, emotionally charged. The distinction I cared about (judgment vs. feeling)
lives in meaning, not wording, and ~150 examples weren't enough to teach it.

So the model overfit to the easy signal (evidence → argument) and missed the subtle one
(claim vs. feeling). Combined with the implausibly strong baseline, the honest read is
that the *task* I posed is learnable by a large zero-shot model but my *dataset* — small,
hot_take-poor, and possibly too clean — couldn't teach it to a small one.

## Spec Reflection

_(TODO: one way the spec/planning helped guide the work, and one way the implementation diverged from the plan and why.)_

## AI Usage

_(TODO: at least 2 specific instances — what you directed the AI to do, what it produced, what you changed/overrode. Disclose any annotation pre-labeling here.)_

1. …
2. …

## Files

- [`planning.md`](planning.md) — design thinking, label definitions, edge-case rules, AI tool plan.
- [`data/`](data/) — labeled dataset CSV.
- [`data/evaluation_results.json`](data/evaluation_results.json) — metrics export from Colab.
- [`data/confusion_matrix.png`](data/confusion_matrix.png) — confusion matrix image from Colab.
