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

_(TODO — fill in from `evaluation_results.json` after running Colab Sections 4–6.)_

### Overall accuracy

| Model | Accuracy | Macro F1 |
|---|---|---|
| Zero-shot baseline (llama-3.3-70b) | TODO | TODO |
| Fine-tuned DistilBERT | TODO | TODO |

### Per-class metrics (fine-tuned)

| Label | Precision | Recall | F1 |
|---|---|---|---|
| argument | TODO | TODO | TODO |
| hot_take | TODO | TODO | TODO |
| experience | TODO | TODO | TODO |

### Confusion matrix (fine-tuned)

Rows = true label, columns = predicted. (Supplementary image: `confusion_matrix.png`.)

|  | pred argument | pred hot_take | pred experience |
|---|---|---|---|
| **true argument** | TODO | TODO | TODO |
| **true hot_take** | TODO | TODO | TODO |
| **true experience** | TODO | TODO | TODO |

### Three wrong predictions, analyzed

_(TODO: pick 3 misclassified test examples. For each: the text, true vs. predicted label, and WHY it failed — which boundary, why that boundary is hard, whether it's a labeling vs. data problem, and what would fix it.)_

1. …
2. …
3. …

### Sample classifications

_(TODO: 3–5 posts run through the fine-tuned model with predicted label + confidence. Explain why at least one correct prediction is reasonable.)_

| Post (truncated) | Predicted | Confidence |
|---|---|---|
| … | … | … |

## Reflection: learned vs. intended

_(TODO: higher-level than the wrong-prediction list. What did the model's decision boundary actually capture vs. what your definitions intended? What did it overfit to — e.g., did "experience" collapse to first-person pronouns, or "argument" to the presence of numbers? What did it miss?)_

## Spec Reflection

_(TODO: one way the spec/planning helped guide the work, and one way the implementation diverged from the plan and why.)_

## AI Usage

_(TODO: at least 2 specific instances — what you directed the AI to do, what it produced, what you changed/overrode. Disclose any annotation pre-labeling here.)_

1. …
2. …

## Files

- [`planning.md`](planning.md) — design thinking, label definitions, edge-case rules, AI tool plan.
- [`data/`](data/) — labeled dataset CSV.
- `evaluation_results.json` — metrics export from Colab _(commit after running)_.
- `confusion_matrix.png` — confusion matrix image from Colab _(commit after running)_.
