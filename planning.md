# TakeMeter Planning

## Community

I chose r/vegan as the community for this project because it is active, text-heavy, and full of discourse that varies enormously in quality and intent. The same topic — say, dairy, ethics, or going vegan around non-vegan family — gets discussed in very different ways: some posts build careful ethical or health arguments, some are blunt provocative declarations, and many are personal stories or requests for support. That variety makes it a good fit for a classification task, because regulars in the community can usually tell the difference between a reasoned case, a bumper-sticker opinion, and someone just sharing their experience.

## Labels

I will use three mutually exclusive labels:

1. Argument
   - Definition: The post makes a structured case for or against a position (ethical, environmental, or health) backed by specific, verifiable reasoning such as evidence, studies, statistics, or a clear logical chain that could stand on its own if the opinion framing were removed.
   - Example: "The environmental case is hard to dodge: animal agriculture accounts for roughly 14.5% of human-caused greenhouse emissions, and beef in particular needs about 20x the land per gram of protein compared to legumes, so cutting it has an outsized effect."
   - Example: "B12 is the one nutrient worth taking seriously on a vegan diet. It's produced by bacteria, not animals themselves, which is why supplementation (or fortified foods) is the reliable route — animals are just intermediaries that get it the same way."

2. Hot Take
   - Definition: The post makes a bold, confident opinion or moral judgment with little supporting evidence, or with evidence that is mostly rhetorical, cherry-picked, or decorative rather than part of a real argument.
   - Example: "Anyone who still eats meat in 2026 just doesn't care about the planet, full stop."
   - Example: "Dairy is literally poison and the whole industry is a scam."

3. Experience
   - Definition: The post is mainly a personal story, emotional expression, or request for support/practical advice — someone sharing their own vegan journey, struggles with family or friends, venting, or asking the community for help — with little to no general argument.
   - Example: "Three months in and my parents still leave meat on my plate at dinner. How do you all deal with family who won't take it seriously?"
   - Example: "I watched a slaughterhouse documentary last night and I just can't stop thinking about it. Needed to say that somewhere."

## Hard Edge Cases

The hardest case will be a post that blends a personal story with an argument, or one that uses a single fact as a provocative jab rather than as reasoning.

- Story + argument: e.g., "I went vegan after my dad's heart attack, and honestly the science on saturated fat and cardiovascular risk is overwhelming — everyone should at least cut back." This could be Experience or Argument.
  - Decision rule: if removing the personal narrative still leaves a self-standing, evidence-backed claim, label it Argument. If the post is primarily about the person's own situation or feelings and the reasoning is incidental, label it Experience.

- Fact-as-jab: e.g., "Milk causes cancer, look it up." This cites a "claim" but offers no real reasoning and exists mainly to provoke.
  - Decision rule: if the post provides specific evidence that would support the claim even with the opinion framing removed, label it Argument. If the evidence is vague, selective, or mostly there for emphasis, label it Hot Take.

## Data Collection Plan

I will collect public posts and comments from r/vegan and save them in a CSV file with columns for text, label, and notes. I will aim for at least 200 labeled examples, with a target of roughly 70 examples per label to avoid a heavily imbalanced dataset. If one label is underrepresented after the first pass, I will collect additional examples until the distribution is more balanced (no label above 70%).

## Evaluation Metrics

I will evaluate both the zero-shot baseline and the fine-tuned model using overall accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix. Accuracy alone is not enough because the task is subjective and the labels are not perfectly balanced; macro F1 will help show whether the model performs well across all classes rather than only the majority class. The confusion matrix will tell me which specific boundary (e.g., Argument vs. Hot Take) the model struggles with.

## Definition of Success

The classifier will be considered useful if it meaningfully outperforms the zero-shot baseline and achieves at least a macro F1 score of 0.70 on the held-out test set. That would suggest fine-tuning captured a real signal from the labeled data rather than simply memorizing the most common label.

## AI Tool Plan

- Label stress-testing: I will ask an AI tool to generate boundary-case posts using my label definitions and then check whether those examples are hard to classify. If the generated examples expose unclear boundaries, I will revise the definitions before annotation.
- Annotation assistance: I may use an AI tool to pre-label a small batch of posts, but I will review every label myself and correct anything that does not match my definitions.
- Failure analysis: After the model makes mistakes, I will give the misclassified examples to an AI tool and ask it to look for patterns such as sarcasm, short posts, or ambiguous wording. I will verify any pattern by reading the examples myself before including it in the final report.

## Stretch Features

- Confidence calibration (started after the main evaluation): I will check whether the fine-tuned model's softmax confidence is meaningful — specifically, whether correctly-predicted test examples carry higher confidence than incorrectly-predicted ones, and whether the model ever produces a genuinely high-confidence prediction. I will report the confidence ranges for correct vs. wrong predictions and interpret what they mean for using the model as a deployable tool.
