# v4 preflight correction, same unrun acceptance criteria

v4 stopped before live execution: targeted test revealed the existing resolver already
infers the previous year for a completed yearless March event, yielding2022-03-15, not
future2023-03-15. In the requested last-four-month window of2023-02-26, neither plausible
March year lies within the window. The ambiguity is the missing event year/reference
period; original reported500 cannot be silently asserted as in-window. Correct the
conflict detector accordingly, preserve v4 zero-call preview/snapshot, use new v5 namespace.
All six questions, models,40-call ceiling, original-score reporting and5+1 faithful
mechanism criteria remain as v4. No post-answer acceptance change: v4 made0 paid calls.
