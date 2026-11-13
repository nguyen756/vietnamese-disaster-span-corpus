# Data and annotation pipeline

1. Stored web-document snapshots -> sentence splitting, whitespace normalization,
   scope lexicons and boilerplate filtering. Complete original queries and
   classification decisions were not preserved; this is not a complete recrawl.
2. Gold: two original human label sets -> IAA comparison -> disagreement
   resolution/merge -> curator approval. Original references are preserved as
   stand-off labels in the dataset's additional/ directory.
3. Training/development: human-provided guideline -> interactive assistant
   proposals -> deterministic offsets/category/flatness checks -> curator
   inspection, clarification, continuation and reported approval. Historical
   full assistant model/settings/logs are unknown; current P1/P2 are new prompts.
4. Frozen split/exclusion policy -> 9,686 train, 803 validation, 1,000 Gold.
   The 1,563 deferred candidates are not included. No reshuffling or relabeling.
5. PhoBERT/VnCoreNLP alignment -> identical source-unit pooling -> recurrent
   heads -> 20 training epochs -> maximum original Dev F1, earliest ties ->
   strict/original-character and overlap evaluation. Gold is not a selector.
6. Prospective two-model/two-prompt annotation -> unchanged strict compiler ->
   all-row quality scoring -> fixed-K oracle routing/random comparison ->
   actual request/token accounting and common error diagnostics.
7. E4 assisted-reference consistency and E6 two-episode existing-checkpoint
   evaluation use the accepted references, with their dependence disclosed.

Structural validation cannot establish semantic correctness. Source hashes
identify exact research snapshots; live-URL reconstruction is not guaranteed.
