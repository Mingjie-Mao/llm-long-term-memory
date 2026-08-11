| Variant | n | Accuracy | SS-user | SS-asst | SS-pref | Multi-sess | Temporal | Know-update | Abstention | Evid. recall | Ctx tokens | p95 latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_context` | 50 | **56.0%** | 100.0% | 100.0% | 0.0% | 38.5% | 23.1% | 87.5% | 50.0% | — | 109,260 | 7.9s |
| `naive_rag` | 50 | **54.0%** | 71.4% | 83.3% | 0.0% | 38.5% | 46.2% | 75.0% | 100.0% | 94.0% | 13,057 | 2.1s |
