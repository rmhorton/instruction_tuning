# instruction_tuning
Characterization and augmentation of instruction tuning datasets


## The Plan

Fit (linear) predictive models that use semantic embeddings as predictors for a variety of interpretable labels:
* regex patterns
* clusters (semantic, pos_ngram, etc)
* readability score (regression models)

Use the coefficients from the predictive models to re-weight the dimensions of an embedding.
* positive: emphasize the predictive dimensions
* negative: deemphasie the predictive dimensions.

Cluster based on the re-weighted embeddings.
* Positively re-weighting is like fine tuning; it makes unsupervised results more like supervised results.
* Can we show that downweighting particular aspects can give us useful clusters focusing on other aspects?
