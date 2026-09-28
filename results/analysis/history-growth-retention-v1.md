# Retention as history grows — simulated, zero calls

> Scratch copies of `train150`; the store itself is not written. Growth
> copies other users' history into each namespace, dated after the user's own. It
> competes for rank and budget but never contradicts; answers are not scored.

| history | median memories per user | gold-anchored memory in top 20 | raw turns cover all gold at 4,000 tokens |
|---|---:|---:|---:|
| 1x | 121 | 73.3% | 78.1% |
| 2x | 242 | 72.6% | 71.9% |
| 4x | 484 | 71.9% | 67.8% |
