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

- **Source:** posts and comments reflecting r/vegan discourse. _(⚠️ TODO — the spec requires **real, public** r/vegan posts; document exactly where these came from. See the note at the end of this section.)_
- **Labeling process:** each example labeled against the definitions in `planning.md`, with a free-text `notes` column recording the rationale and any borderline judgment. _(TODO: disclose any LLM assistance in the AI Usage section.)_
- **Dataset file:** [`data/takemeter_rvegan_labeled_examples.csv`](data/takemeter_rvegan_labeled_examples.csv) — single file, columns `text,label,notes`. The Colab notebook does the 70/15/15 train/val/test split automatically.

### Label distribution (208 examples)

| Label | Count | Share |
|---|---|---|
| experience | 100 | 48.1% |
| argument | 65 | 31.2% |
| hot_take | 43 | 20.7% |

No label exceeds 70% ✓. But `hot_take` is the minority class (20.7%), and `experience` is nearly half — an imbalance that turns out to predict the model's failure exactly (see Evaluation).

### Three difficult-to-label examples

These are real boundary cases from the dataset (rationale captured in the `notes` column), resolved with the decision rules in [`planning.md`](planning.md).

1. **"If your ethics only apply to animals that are cute, that's not ethics — that's aesthetics."** — has a genuine logical kernel (a consistency point) but is delivered as a one-line jab with no argument built around it. → **hot_take**, by the rule "evidence/logic that is decorative rather than developed into a case → hot_take."
2. **"Dairy cows are kept perpetually pregnant and their calves are taken away. If that happened to a human it would be called torture. Why does species make a difference?"** — the factual claims are accurate, which pulls toward `argument`, but a rhetorical question is doing the work of the conclusion and no actual case is made. → **hot_take** (the hardest case in the set).
3. **"Doctors get almost no nutrition training in medical school… Get your bloodwork done and find a doctor who knows the research."** — the "doctors get no nutrition training" claim edges toward `argument`, but the post is framed as personal advice from the author's own experience. → **experience**, by the rule "if the post is primarily about the author's own situation and the reasoning is incidental, label Experience."

> 📝 **Data-source note.** The dataset is a mix of genuinely collected r/vegan posts
> (long, first-person, messy, typo-bearing — e.g. the Voodoo Donuts story, the lost-period
> health post, the "Feather Foam" mix-up) and shorter, cleaner exemplar posts. **Document
> this honestly in the data-source line above and in AI Usage:** state which examples were
> collected from r/vegan vs. authored/generated to illustrate a class, and disclose any LLM
> help. The cleaner exemplars are part of why the baseline is so high — see the evaluation.

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
baseline scored a **perfect 100%**; the fine-tuned DistilBERT managed 78.1% and **never
once predicted `hot_take`**. This did **not** meet my definition of success (beat baseline
AND macro F1 ≥ 0.70) — it missed on both counts — and the *why* is the most interesting
part of the project.

### Overall accuracy

| Model | Accuracy | Macro F1 |
|---|---|---|
| Zero-shot baseline (llama-3.3-70b) | **1.000** (32/32) | **1.00** |
| Fine-tuned DistilBERT | 0.781 (25/32) | 0.58 |

Δ accuracy = **−0.219** (fine-tuned minus baseline).

### Per-class metrics

**Fine-tuned DistilBERT:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| argument | 0.71 | 1.00 | 0.83 | 10 |
| hot_take | 0.00 | 0.00 | **0.00** | 7 |
| experience | 0.83 | 1.00 | 0.91 | 15 |
| **macro avg** | 0.52 | 0.67 | 0.58 | 32 |

`hot_take` F1 is exactly 0 — the model **never once predicted it**. This is the textbook
"one class F1 ≈ 0, others fine" case: the model could not learn that boundary.

**Baseline (llama-3.3-70b):** precision/recall/F1 = **1.00 across all three classes**
(argument, hot_take, experience) — a perfect confusion matrix.

### Confusion matrix (fine-tuned)

Rows = true label, columns = predicted. (Supplementary image: [`data/confusion_matrix.png`](data/confusion_matrix.png).)

|  | pred argument | pred hot_take | pred experience |
|---|---|---|---|
| **true argument** | 10 | 0 | 0 |
| **true hot_take** | **4** | 0 | **3** |
| **true experience** | 0 | 0 | 15 |

Every error is a `hot_take`, and the model has **no `hot_take` column at all** — it splits
the 7 hot_takes between `argument` (4) and `experience` (3). `argument` and `experience`
are otherwise perfectly recalled; their precision drops (0.71 / 0.83) only because the
stray hot_takes leak in.

### Three wrong predictions, analyzed

All 7 errors are true `hot_take`s. The revealing part is *which way* each one breaks:

1. **"The egg industry is built on gassing or grinding male chicks alive… That's the product you're buying when you buy eggs."**
   True `hot_take` → **predicted `argument`** (confidence 0.36). The post drops real factual nouns (the chick-culling practice), so the model's "contains facts → argument" heuristic fires. But I labeled it `hot_take` because the fact is delivered as a conclusion-by-shock, with no reasoning built around it. **Boundary: argument vs. hot_take** — the model can't tell *citing* a fact from *reasoning* with one.
2. **"Anyone still eating meat in 2026 is choosing ignorance. The information has been out there for decades."**
   True `hot_take` → **predicted `argument`** (0.34). The appeal-to-evidence phrasing ("the information has been out there") mimics the *surface form* of an argument without making one. Same boundary, same cause.
3. **"Hunters are just murderers with a hunting license. Dress it up however you want."**
   True `hot_take` → **predicted `experience`** (0.36). No facts, pure emotional assertion — so it falls to the model's "no evidence → experience" default. **Boundary: hot_take vs. experience.**

**The pattern.** The model learned a 2-way rule, not a 3-way one: *"factual/evidence-style
language → `argument`; emotional, evidence-free language → `experience`."* `hot_take` —
**a confident claim delivered with feeling but no real reasoning** — straddles both, so it
has no home: the fact-flavored hot_takes get pulled to `argument`, the pure-assertion ones
to `experience`.

- **Why it's hard:** `hot_take` is defined by what it *lacks* (developed reasoning), not by
  any positive surface signal. `argument` (numbers, studies, causal connectives) and
  `experience` (first-person, narrative) each have one; `hot_take` doesn't.
- **Labeling vs. data problem:** **class imbalance**, not annotation noise. `hot_take` is
  the minority class (20.7% overall, only 7 in test); with ~145 training examples, DistilBERT
  never saw enough to carve out a region for it.
- **Confidence tell:** every wrong prediction sits at **0.34–0.36** — barely above the 0.33
  chance floor for three classes. The model isn't confidently wrong on hot_takes; it's
  maximally *uncertain*, which is itself a usable calibration signal (see stretch ideas).
- **What would fix it:** more (and harder) `hot_take` examples; class-weighted loss; or
  accepting the boundary is thin and merging classes — though that abandons the distinction
  I set out to measure.

> ⚠️ **The other red flag — the baseline is *perfect* (100%).** The spec warns that >95%
> accuracy on a subjective task suggests labels that are *too easy* or test leakage. A 70B
> model getting 32/32 while a small fine-tune can't learn one class points the same way: the
> 3-way distinction is easy for a large model to read zero-shot, but hard to *learn* from 208
> imbalanced examples. Worth confirming there's no train/test overlap and that the harder,
> messier real posts (not just the clean exemplars) are well represented in `hot_take`.

### Sample classifications

Run through the fine-tuned model (confidences from the softmax). The wrong rows show the
near-chance confidence discussed above.

| Post (truncated) | True | Predicted | Confidence |
|---|---|---|---|
| "Methane from livestock is ~80x more potent than CO2 over a 20-year window…" | argument | argument | _NEED (high, ~correct)_ |
| "Went to a bbq last weekend. There were zero options for me… drove home and cried a little." | experience | experience | _NEED_ |
| "The egg industry is built on gassing or grinding male chicks alive…" | hot_take | argument | 0.36 |
| "Hunters are just murderers with a hunting license. Dress it up however you want." | hot_take | experience | 0.36 |

**Why a correct one is reasonable:** the methane post is predicted `argument` because it
does exactly what the `argument` label describes — a specific quantified claim (80×, 20-year
window) tied to a causal conclusion. That's the one boundary the model learned cleanly, and
this is a central example of it.

_(NEED: confidence scores for the two correct rows — re-run the Section 4 inference cell on
these texts, or any 2 correctly-predicted test rows, and paste the softmax confidence.)_

## Reflection: learned vs. intended

I intended a three-way distinction along *discourse quality*: reasoned **argument**,
unsupported **hot take**, and personal **experience**. What the model actually learned was
a two-way split along *surface form*: **"factual/evidence-style language → argument;
emotional, first-person, evidence-free language → experience."** `hot_take` was never
instantiated as its own category — its 7 test cases were routed to whichever of the two
learned classes their surface features resembled (4 to `argument`, 3 to `experience`).

That gap is the whole point. `argument` and `experience` each align with an easy positive
signal — numbers/causal connectives for one, first-person narrative for the other — so the
model captured them with perfect recall (arguably for the wrong reason: `argument` may be
keying on the *presence of facts* rather than on whether a claim is actually *reasoned*,
which is exactly why the fact-flavored hot_takes fooled it). My `hot_take` definition — "a
confident claim *delivered without real reasoning*" — is defined by an *absence*, and an
absence gives a small model nothing positive to grab onto. The distinction I cared about
(asserting vs. arguing, judgment vs. feeling) lives in meaning, not wording, and ~145
training examples of a 20%-minority class weren't enough to teach it. The model didn't
learn "take quality"; it learned "does this text look like prose-with-numbers or
prose-with-feelings."

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
