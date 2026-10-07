# Page receipt clarification diagnostic r2

v1 raw return acknowledges only E80..E97 out of full pageE1..E105 and selects no sources;
validator correctly rejected. Preserve v1. First loss is output acknowledgement, not
raw page delivery. r2 supplies the full expected ID list as an explicit required receipt
in the same page prompt, retaining existing schema, selected/uncertain provenance checks,
model,2048 output cap. One exact same public train150 page, no gold or answer generation,
no grading. Must acknowledge all original pageIDs once with valid local selections.
Receipt is a structural check, NOT proof of semantic completeness. Six-question QA gate
must independently check relevant evidence selection after this structural check passes.
<=2 nominal retry attempts plus thinking-control fallback, usage preserved.
