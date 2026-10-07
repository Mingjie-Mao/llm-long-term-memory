# Grounded reader development / regression report

Failure-enriched exposed train150; not overall or unseen-final accuracy.

| cohort | saved readers | graded | correct |
|---|---:|---:|---:|
| reasoning-errors | 9 | 9 | 5 |
| originally-correct | 10 | 10 | 9 |

Grading complete for saved readers: True.
Recorded provider usage: `{"total_requests": 80, "total_tokens": 393059, "failures": 23, "by_role": {"answerer": {"role": "answerer", "calls": 39, "failures": 1, "input_tokens": 370784, "output_tokens": 17819, "p50_ms": 3816.9926246628165, "p95_ms": 8237.789417151362, "max_ms": 9709.09645780921}, "judge": {"role": "judge", "calls": 41, "failures": 22, "input_tokens": 3851, "output_tokens": 605, "p50_ms": 25112.16800007969, "p95_ms": 44294.56137493253, "max_ms": 180005.6612920016}}}`.

| question | cohort | grade | answer |
|---|---|---|---|
| 09ba9854 | reasoning-errors | True | 50 $ |
| 51a45a95 | reasoning-errors | False | I do not know. |
| 6e984301 | reasoning-errors | True | 3 weeks |
| b46e15ed | reasoning-errors | True | 2 months and 3 days |
| eaca4986 | reasoning-errors | False | I do not know. |
| f685340e | reasoning-errors | True | Previously, you played tennis with your friends weekly. Now, you play every other week. |
| f9e8c073 | reasoning-errors | False | I do not know. |
| gpt4_7a0daae1 | reasoning-errors | True | 1 weeks |
| gpt4_a1b77f9c | reasoning-errors | False | I do not know. |
| 0ddfec37_abs | originally-correct | True | I do not know. |
| 1568498a | originally-correct | False | The move made after 27. Kg2 Bd5+ was 29. Rd3 Rh4. |
| 6456829e | originally-correct | True | 8 plants |
| 95228167 | originally-correct | True | Since you are looking at a Gibson Les Paul, keep an eye out for models with coil-tap or coil-splitting features for added tonal versatility, especially if you plan to play in alternate tunings like open D. Also, pay attention to the neck profile and overall weight to make sure it feels comfortable for you. |
| a89d7624 | originally-correct | True | For your upcoming trip to Denver, you should definitely check out top music venues like Red Rocks Park and Amphitheatre, The Ogden Theatre, The Bluebird Theater, The Fillmore Auditorium, and Larimer Lounge. For dining, you've mentioned wanting to eat at The Ship Rock Grille near Red Rocks. |
| af8d2e46 | originally-correct | True | You packed 7 shirts for your trip to Costa Rica. |
| ed4ddc30 | originally-correct | True | You currently have 20 dozen eggs stocked up in your refrigerator. |
| gpt4_2f8be40d | originally-correct | True | 3: wedding, wedding, wedding |
| gpt4_7ddcf75f | originally-correct | True | 3 days |
| gpt4_a2d1d1f6 | originally-correct | True | 3 days |
