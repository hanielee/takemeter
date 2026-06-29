# TakeMeter — Discourse Quality Classifier for r/vegan

TakeMeter is a fine-tuned text classifier that labels the *kind* of discourse in a
r/vegan post or comment: a reasoned **argument**, a bold **hot take**, or a personal
**experience**. It compares a fine-tuned `distilbert-base-uncased` model against a
zero-shot `llama-3.3-70b-versatile` baseline on the same held-out test set.

> Design notes, edge-case decision rules, and the AI tool plan live in
> [`planning.md`](planning.md). This README is the final report and stands on its own.

> **Label strings.** In the dataset and trained model the labels are stored as
> `arguments`, `hot takes`, and `experience`. This README uses those exact strings in the
> metrics tables (to match the artifacts) and the natural forms ("argument", "hot take")
> in prose.

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
| **arguments** | A structured case for/against a position (ethical, environmental, health) backed by specific, verifiable reasoning that could stand on its own without the opinion framing. | "Animal agriculture is ~14.5% of human-caused GHG emissions, and beef needs ~20x the land per gram of protein vs. legumes, so cutting it has an outsized effect." |
| **hot takes** | A bold, confident opinion or moral judgment with little supporting evidence, or evidence that is rhetorical/cherry-picked rather than part of a real argument. | "Anyone who still eats meat in 2026 just doesn't care about the planet, full stop." |
| **experience** | A personal story, emotional expression, or request for support/advice — someone's own vegan journey, family struggles, venting — with little to no general argument. | "Three months in and my parents still leave meat on my plate. How do you all deal with family who won't take it seriously?" |

Second example per label and full definitions: see [`planning.md`](planning.md).

## Data Collection & Labeling

- **Source:** **223 real public posts and comments collected from r/vegan** (full posts and
  reply comments — the long, first-person, typo-bearing kind you actually find on the
  subreddit). No synthetic or authored examples.
- **Labeling process:** each example carries a `notes` field with a per-label **score tally**
  (`scores={'arguments':N,'hot_takes':N,'experience':N}`); the final `label` is the
  argmax. This means labels were assigned with **automated scoring assistance**, which I
  disclose in [AI Usage](#ai-usage). Label noise from this process is non-trivial (see below)
  and is relevant to the results.
- **Dataset file:** [`data/takemeter_labeled_rvegan.csv`](data/takemeter_labeled_rvegan.csv) — single file, columns `text,label,notes`. The Colab notebook does the 70/15/15 train/val/test split automatically.

### Label distribution (223 examples)

| Label | Count | Share |
|---|---|---|
| arguments | 75 | 33.6% |
| hot takes | 75 | 33.6% |
| experience | 73 | 32.7% |

**Near-perfectly balanced** — no label above 34%. So whatever goes wrong downstream is
*not* a class-imbalance artifact.

### Label-noise caveat

**87 of 223 examples (39%) are ambiguous** by the scoring system's own output — the top two
label scores are within 2 points of each other. On a subjective task that's expected, but it
means a large minority of training labels rest on a near-tie, which limits how clean a signal
any model can learn (the spec flags "labels inconsistent" as a cause of low, flat
performance — see Evaluation).

### Three difficult-to-label examples

Real boundary cases from the dataset, resolved with the decision rules in [`planning.md`](planning.md). Each shows the scoring system's split.

1. **"Horse shit. Cancer has existed for all of human history. It was named by Hippocrates because he thought a tumour looked like a crab…"** — `scores={arguments:4, hot_takes:2}`. Opens with hostility (reads like a `hot take`) but contains a verifiable historical/factual claim that stands on its own. → **arguments**, by the rule "if specific evidence would support the claim with the opinion framing removed, label argument."
2. **"My advice is 1) Watch Dominion; 2) Read Melanie Joy's *Why We Love Dogs…*; 3) Talk to a nutritionist."** — `scores={arguments:4, hot_takes:2}`. Sits between `arguments` (a reasoned recommendation) and `experience` (personal advice). → **arguments**, because the recommendation is justified rather than just personal.
3. **"What do I do with non-vegan food my caretaker bought specifically for me because he thought it was vegan?"** — `scores={arguments:4, hot_takes:4, experience:8}` (a genuine three-way split). Raises an ethical question but is fundamentally a personal, practical request for help. → **experience**, by the rule "if the post is primarily about the author's own situation, label experience."

## Fine-Tuning Approach

- **Base model:** `distilbert-base-uncased` (HuggingFace), with a sequence-classification head (3 output labels).
- **Platform:** Google Colab, free **T4 GPU**, via the HuggingFace `Trainer`.
- **Data:** 70/15/15 train/val/test, **stratified** by label (`random_state=42`) — ~156 train / ~33 val / **34 test**, each split keeping the balanced ~33/33/33 distribution. Text truncated to 256 tokens.
- **Settings used:** 3 epochs, learning rate 2e-5, batch size 16, weight decay 0.01, 50 warmup steps, `load_best_model_at_end` on validation **accuracy**.

### Key training decision (and what it revealed)

I kept the standard BERT fine-tuning regime — **LR 2e-5, 3 epochs, batch 16** — on the
reasoning that for ~156 examples a higher LR or more epochs risks overfitting. **On this
dataset that decision backfired in a specific, diagnosable way:** the model **collapsed to a
constant classifier**, predicting `arguments` for all 34 test posts (see the confusion
matrix). On balanced data that yields ≈ the majority share — exactly the 35% it scored.

The important insight is *why* no hyperparameter would have saved it as-is:

1. **The training signal is too weak to beat the trivial solution.** With a genuinely
   subjective 3-way distinction, ~156 examples, and **39% of labels on a near-tie**, gradient
   descent found it cheaper to park at "always predict class 0" than to learn a boundary.
   Three epochs at 2e-5 never moved it off that trivial minimum.
2. **Label noise caps the ceiling.** Class weighting won't help (the data is already
   balanced); the bottleneck is label *consistency*, not balance.

**What I would change:** first **clean the labels** (resolve or drop the 39% near-tie rows),
then re-run with **more epochs (8–12) and a higher LR (3e-5–5e-5)** while watching the
validation learning curve for the moment it escapes the constant-prediction regime. The
honest lesson is that on noisy, subjective labels the data work matters more than the
hyperparameters.

## Baseline

Zero-shot `llama-3.3-70b-versatile` via the Groq API, classifying each test example with no
task-specific training. Each post was sent in its own request with `temperature=0` and
`max_tokens=20`; the system prompt carried the three label definitions verbatim (plus the
two edge-case decision rules) and instructed the model to **output only the label name**.

An earlier run used label names in the prompt (`argument` / `hot_take`) that didn't match
the dataset strings (`arguments` / `hot takes`), so non-`experience` replies were dropped as
unparseable and the baseline was computed on a biased subset. After aligning the prompt to
the exact dataset strings, **all 34/34 responses parsed** and the baseline is a valid
comparison: **0.618 accuracy (21/34)**.

**System prompt used (corrected — outputs the exact dataset label strings):**

```
You are classifying posts from r/vegan.
Assign each post to exactly one of these three categories.

arguments: The post makes a structured case for or against a position (ethical,
environmental, or health) backed by specific, verifiable reasoning — evidence, studies,
statistics, or a clear logical chain that would stand on its own if the opinion framing
were removed.
Example: "Animal agriculture is about 14.5% of human-caused greenhouse emissions, and beef
needs roughly 20x the land per gram of protein as legumes, so cutting it has an outsized
effect."

hot takes: The post makes a bold, confident opinion or moral judgment with little supporting
evidence, or with evidence that is mostly rhetorical, cherry-picked, or decorative rather
than part of a real argument.
Example: "Anyone who still eats meat in 2026 just doesn't care about the planet, full stop."

experience: The post is mainly a personal story, emotional expression, or request for
support or practical advice — someone sharing their own vegan journey, struggles with family
or friends, venting, or asking the community for help — with little to no general argument.
Example: "Three months in and my parents still leave meat on my plate at dinner. How do you
all deal with family who won't take it seriously?"

Decision rules for borderline posts:
- Personal story that also makes a case: if removing the personal narrative still leaves a
  self-standing, evidence-backed claim, label it "arguments"; if the reasoning is incidental
  to the personal situation, label it "experience".
- A post that cites a fact but offers no reasoning and exists mainly to provoke is
  "hot takes", not "arguments".

Respond with ONLY the label name, exactly as written, and nothing else.

Valid labels (use these exact strings):
arguments
hot takes
experience
```

## Evaluation Report

Test set: **34 examples** (true distribution: arguments 12, hot takes 11, experience 11).
Metrics in [`data/evaluation_results.json`](data/evaluation_results.json); confusion matrix
image at [`data/confusion_matrix.png`](data/confusion_matrix.png).

**Headline result: the fine-tuned model collapsed to a single class.** It predicted
`arguments` for **every** test post, scoring **35.3%** — essentially the trivial
"always-guess-the-plurality" baseline. The zero-shot LLM scored **61.8%** on the same test
set, so fine-tuning **regressed** the classifier by ~26 points. This misses my definition of
success (beat the baseline AND macro F1 ≥ 0.70) on every count, and the *why* — a hard
subjective task plus noisy labels — is the finding.

### Overall accuracy

| Model | Accuracy | Macro F1 |
|---|---|---|
| Zero-shot baseline (llama-3.3-70b) | **0.618** (21/34) | **0.61** |
| Fine-tuned DistilBERT | **0.353** (12/34) | **0.17** |

Δ accuracy = **−0.265** (fine-tuned minus baseline). The baseline now evaluates on all
**34/34** parseable responses, so the comparison is valid.

### Per-class metrics (fine-tuned)

Computed from the confusion matrix below.

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| arguments | 0.35 | 1.00 | 0.52 | 12 |
| hot takes | 0.00 | 0.00 | **0.00** | 11 |
| experience | 0.00 | 0.00 | **0.00** | 11 |
| **macro avg** | 0.12 | 0.33 | **0.17** | 34 |

Two of three classes have F1 = 0 because the model **never predicted them**. `arguments`
gets recall 1.00 only because the model predicts it for everything; its precision (0.35) is
just the class's share of the test set.

**Baseline (llama-3.3-70b), per-class:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| arguments | 0.67 | 0.50 | 0.57 | 12 |
| hot takes | 0.71 | 0.45 | 0.56 | 11 |
| experience | 0.56 | 0.91 | 0.69 | 11 |
| **macro avg** | 0.65 | 0.62 | **0.61** | 34 |

Unlike the collapsed fine-tune, the baseline **distinguishes all three classes** (every F1 in
0.56–0.69). Its main tendency is to over-predict `experience` (recall 0.91 but precision 0.56)
and to be conservative on `hot takes` (recall 0.45) — i.e. when unsure it leans toward "this is
someone sharing" rather than "this is a bold claim." Even so, macro F1 0.61 vs the fine-tune's
0.17 is the whole story: reading the definitions zero-shot beats trying to learn them from 156
noisy examples.

### Confusion matrix (fine-tuned)

Rows = true label, columns = predicted. (Image: [`data/confusion_matrix.png`](data/confusion_matrix.png).)

|  | pred arguments | pred hot takes | pred experience |
|---|---|---|---|
| **true arguments** | 12 | 0 | 0 |
| **true hot takes** | 11 | 0 | 0 |
| **true experience** | 11 | 0 | 0 |

The matrix has a **single populated column** — the signature of a degenerate, constant
classifier. There is no boundary to analyze *between* classes because the model drew none.

### Three wrong predictions, analyzed

Because the model predicts `arguments` for everything, **all 22 non-argument test posts are
errors**. Three real ones (all true label ≠ `arguments`, all predicted `arguments`):

1. **"Most vegans are cannibals. If you look online they are the ones mostly obsessed with cannibalism…"** — true **hot takes** → predicted **arguments**. A pure provocative assertion with no reasoning; the model has no `hot takes` region to put it in.
2. **"When servers lie — I've been to this ramen place a dozen times… in their defense the vegan option is…"** — true **experience** → predicted **arguments**. A first-person anecdote; again routed to the only class the model emits.
3. **"This is the answer! French fries and ice cream can be vegan if you're into that. My weight has fluctuated by 100 lbs in 20 years…"** — true **experience** → predicted **arguments**. Personal, conversational reply, predicted `arguments` like everything else.

**The pattern is the absence of a pattern.** Unlike a model that confuses two *specific*
boundaries, this one made the same prediction regardless of input — so the error analysis is
about the **collapse itself**, not a label pair:

- **Which boundary failed:** all of them. The model learned no separating signal whatsoever.
- **Why it's hard:** the distinction is *semantic* (is the reasoning actually developed?),
  has weak surface correlates, and **39% of the training labels are near-ties** — so the
  "true" signal is both subtle and noisy. A small model defaults to the constant minimum.
- **Labeling vs. data problem:** primarily a **label-consistency / task-difficulty** problem,
  not imbalance (the data is balanced) and not a single confusable pair.
- **What would fix it:** clean the ambiguous labels, then train longer at a higher LR (see
  Fine-Tuning Approach). Until the labels are consistent, no hyperparameter will help.

### Sample classifications

The fine-tuned model is a constant predictor — **every** input returns `arguments`.
Confidences are the max-softmax probability.

| Post (truncated) | True | Predicted | Confidence | ✓/✗ |
|---|---|---|---|---|
| "Did you know that eating more than 700 grams of red meat a week increases your risk of bowel cancer? …1.18× for every 50 grams…" | arguments | arguments | 0.39 | ✓ |
| "I'm Korean and not vegan, but I'll expand your options for living vegan there. …it's called a temple restaurant…" | arguments | arguments | 0.38 | ✓ |
| "Love The Taste Of Meat But I Hate Hurting Animals… I decided to go vegan!" | arguments | arguments | 0.37 | ✓ |

**Why a correct one is reasonable (with a caveat):** the red-meat post is predicted
`arguments`, which is correct — it cites specific statistics (700 g/week; 1.18× risk per 50 g)
and reasons to a health conclusion, exactly the `arguments` definition. **But the prediction
is only trivially right:** the model outputs `arguments` for *every* post, so it also "predicts
`arguments`" for all 22 hot-take/experience posts (which are therefore wrong). A correct
prediction here is not evidence the model learned anything.

## Reflection: learned vs. intended

I intended a three-way distinction along *discourse quality* — reasoned **argument**,
unsupported **hot take**, personal **experience**. The fine-tuned model learned **none of
it**: it became a constant function that outputs `arguments` for any input. The gap between
intended and learned is therefore total, and that is itself the most honest finding.

Two things explain it. First, the quality distinction I care about lives in *meaning*, not
*wording* — a real argument and a real hot take cite the same topics in the same vocabulary;
the difference is whether the reasoning is actually developed. That semantic signal has weak
surface correlates for a small model to latch onto. Second, the labels are noisy: by the
scoring system's own tallies, **39% of examples are near-ties**, so even the "ground truth"
is internally inconsistent. Faced with a subtle signal and noisy targets across only ~156
examples, DistilBERT did the mathematically cheap thing and collapsed to one class.

The contrast with the baseline is the lesson: a 70B model can *read the definitions* and
apply judgment zero-shot at **61.8%**, but a small model cannot *induce* that judgment from
156 noisy examples — it scored 35.3%, below even a constant guess on some splits. What I learned is less about the model and
more about my data pipeline — **on a subjective task, label quality is the binding
constraint**, and an automated scorer that leaves 39% of labels on a coin-flip can't teach a
classifier a distinction I couldn't crisply operationalize.

## Confidence Calibration (stretch)

**Question:** do higher-confidence predictions get it right more often?
**Answer: no — confidence here is uninformative, by construction.**

Every max-softmax confidence on the test set sits in a razor-thin band just above the 0.33
three-class floor. The 12 correct predictions (all true `arguments`) range **0.37–0.39**
(mean ≈ 0.38), and since the model emits `arguments` for all 34 posts, the 22 errors are the
*same* prediction from the *same* output node — there is no separate high-confidence regime
to find. The single most confident prediction is **0.39**.

Two consequences:

1. **Confidence can't separate right from wrong.** Because the model predicts one class for
   everything, P(correct | confidence) in any bin just equals the base rate of `arguments`
   (12/34 ≈ 35%). A 0.39 prediction is no more trustworthy than a 0.37 one.
2. **There is no usable threshold.** The "only act when confidence > 0.9" gate is empty — the
   model never exceeds 0.39. Surfacing this score in a tool would be misleading; it would
   always read ~0.38 regardless of input.

**Why:** a constant classifier that barely escaped uniform output produces near-uniform
softmax (≈0.33–0.39 across three classes). This is **not** a calibrated model — it's a
degenerate one whose confidence carries no signal. _(Stretch noted in [`planning.md`](planning.md).)_

## Spec Reflection

**One way the spec helped.** The spec forced me to write precise label *definitions with
explicit decision rules* and to commit to a success threshold (macro F1 ≥ 0.70) **before**
seeing results. That up-front bar made the outcome an honest, unambiguous fail (macro F1
0.17) rather than a number I could rationalize after the fact — and pointed me straight at
the label-quality problem instead of letting me declare partial success.

**One way the implementation diverged.** I expected fine-tuning to *beat* the zero-shot
baseline — the whole framing of "fine-tune a classifier" assumes the fine-tune wins. Instead
it collapsed to a constant predictor and lost badly. So the project pivoted from "show the
fine-tuned model works" to "diagnose why it learned nothing," and the center of gravity moved
from the training pipeline to the **data**: label noise (39% near-ties) and an inherently
subjective target. The spec anticipates this — it explicitly says a fine-tune losing to the
baseline is "a signal worth investigating," and that suspiciously flat per-class scores point
to inconsistent labels — so the divergence was in my expectation, not in what the spec
allowed for.

## AI Usage

> _Confirm/adjust to match exactly what you did._

1. **Label assignment (annotation disclosure).** The dataset labels were produced with an
   **automated scoring system** — each row's `notes` field records a per-label score tally
   (`scores={'arguments':N,'hot_takes':N,'experience':N}`) and the final `label` is the
   argmax. The post *text* is 100% real, collected from r/vegan; the *labels* are
   AI-/script-assisted. I reviewed the labels against my `planning.md` definitions, and the
   39% near-tie rate this process produced is reported transparently as a limitation.
2. **Error-pattern analysis.** I gave the misclassified results and the confusion matrix to
   an AI tool and asked it to identify the failure mode. It identified the **single-class
   collapse** (one populated confusion-matrix column) and tied it to label noise + task
   difficulty rather than to a confusable label pair. I verified this against the matrix
   (every prediction is `arguments`) before writing it up.
3. **Documentation drafting.** I directed an AI tool to draft the README evaluation tables
   and reflection from my Colab artifacts (`evaluation_results.json`, `confusion_matrix.png`)
   and notebook config. I supplied the files; every metric was recomputed from the artifacts,
   and stale numbers from an earlier run were corrected.

## Files

- [`planning.md`](planning.md) — design thinking, label definitions, edge-case rules, AI tool plan.
- [`data/takemeter_labeled_rvegan.csv`](data/takemeter_labeled_rvegan.csv) — 223 labeled r/vegan posts (`text,label,notes`).
- [`data/evaluation_results.json`](data/evaluation_results.json) — metrics export from Colab.
- [`data/confusion_matrix.png`](data/confusion_matrix.png) — confusion matrix image from Colab.
