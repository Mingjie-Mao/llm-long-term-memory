# v3.1 phase-3 tune42 comparison

One diagnostic repeat on the inspectable tune split; this is not the dev60 gate.
No question ids, text, answers or judge reasons are reported here.

| arm | accuracy | temporal | multi-session | update | median context | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| `v2-control` | 66.7% | 40.0% | 70.0% | 60.0% | 586 | 4,037 |
| `v3.1-reasoned` | 66.7% | 30.0% | 60.0% | 80.0% | 558 | 4,705 |

Paired difference: +0.0%; exact p=1.0000.
A tune result may guide one candidate; it is not permission to inspect dev60.
