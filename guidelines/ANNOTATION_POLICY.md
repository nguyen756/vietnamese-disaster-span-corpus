# Annotation policy for paper-v1

This release preserves the paper's existing labels. `ANNOTATION_GUIDELINE_V1.md`
is the original supplied guideline, retained unchanged. The following clarifications
were used during the recorded human-in-the-loop annotation process; this document
does not claim a new guideline-v2 pilot or change the evaluated reference spans.

- Mark information explicitly present in each sentence and relevant to disaster
  phenomena, impacts, affected people, time, locations and response.
- Spans use exact source characters, zero-based Unicode code-point indices and
  end-exclusive offsets. `text[start:end]` is the annotated expression in Python.
- Primary spans are flat: no overlapping or nested annotations.
- Select the shortest meaningful complete expression. Complete long DAMAGE spans
  can include affected objects, coordinated objects and their impact description;
  length alone does not make an annotation incorrect.
- Prefer a complete RESPONSE_AGENCY name over an overlapping LOCATION inside it.
- Numerical inundation depths are WARNING_LEVEL under this project convention;
  rainfall amounts are WEATHER_CAUSE. Unrelated speed, voltage or percentages
  must not automatically be treated as warning levels.
- Casualty status is not inferred merely from rescue or burial descriptions.
  There is no separate PERSON class.
- Do not normalize Vietnamese spelling or whitespace before applying the offsets.
  For JavaScript, code-point indexing may differ from UTF-16 string indexing.
- Preserve all test references when evaluating. Some tokenizations cannot express
  every test boundary; report this limitation instead of editing the reference.

These are character-span annotations, not a preselected token/BIO representation.
Align the spans to the tokenizer used by your model and disclose any boundaries
that cannot be represented. `annotation_method` describes label generation, not
a claim of automatic validation or independent human accuracy.
