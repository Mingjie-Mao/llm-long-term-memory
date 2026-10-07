# One-page diagnostic, no answer or accuracy claim

After prerequisite-reader-v18-v1 failed with invalid page selection and a runner bug,
record exactly ONE page of exposed train150 b46e15ed USER sources. Original-history,
public corpus exact body check; no reference answers in the reader. Existing model,
selector schema and prompt unchanged. Preserve raw response even if validator rejects.
This locates the invalid-selection reason before any prompt/schema repair. <=2 provider
attempts plus existing initial thinking-control fallback; usage tracked. No grading,
no extrapolated accuracy. Correct runner failure propagation separately covered by test.
