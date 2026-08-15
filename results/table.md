| Variant | n | Accuracy | SS-user | SS-asst | SS-pref | Multi-sess | Temporal | Know-update | Abstention | Source-session recall | Ctx tokens | p95 API latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_context` | 50 | **56.0%** | 100.0% | 100.0% | 0.0% | 38.5% | 23.1% | 87.5% | 50.0% | — | 109,260 | 7.9s |
| `naive_rag` | 50 | **54.0%** | 71.4% | 83.3% | 0.0% | 38.5% | 46.2% | 75.0% | 100.0% | 94.0% | 13,057 | 2.1s |
| `chronomem_no_temporal` | 50 | **26.0%** | 28.6% | 0.0% | 0.0% | 53.8% | 7.7% | 37.5% | 100.0% | 80.0% | 331 | 7.0s |
| `chronomem` | 50 | **26.0%** | 28.6% | 0.0% | 0.0% | 38.5% | 7.7% | 62.5% | 100.0% | 80.0% | 465 | 1.2s |
