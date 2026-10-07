# Grounded reader development / regression report

Failure-enriched exposed train150; not overall or unseen-final accuracy.

| cohort | saved readers | graded | correct |
|---|---:|---:|---:|
| reasoning-errors | 9 | 9 | 8 |
| originally-correct | 10 | 10 | 9 |

Grading complete for saved readers: True.
Recorded provider usage: `{"total_requests": 67, "total_tokens": 272997, "failures": 23, "by_role": {"answerer": {"role": "answerer", "calls": 26, "failures": 1, "input_tokens": 256333, "output_tokens": 12060, "p50_ms": 2333.8827081024647, "p95_ms": 3614.986374974251, "max_ms": 4317.742167040706}, "judge": {"role": "judge", "calls": 41, "failures": 22, "input_tokens": 4016, "output_tokens": 588, "p50_ms": 26553.05404216051, "p95_ms": 39228.73504087329, "max_ms": 58906.565375160426}}}`.

| question | cohort | grade | answer |
|---|---|---|---|
| 09ba9854 | reasoning-errors | True | About 50 $ |
| 51a45a95 | reasoning-errors | True | Likely Target, based on the shopping context in the same conversation. The redemption statement itself does not name the place. |
| 6e984301 | reasoning-errors | True | 3 weeks |
| b46e15ed | reasoning-errors | True | 2 months and 3 days |
| eaca4986 | reasoning-errors | True | C D E F G A B A G F E D C |
| f685340e | reasoning-errors | True | Previously, you played tennis weekly with friends. Now, you play every other week. |
| f9e8c073 | reasoning-errors | False | The records state 3 sessions (period unspecified); 5 sessions (last year). A combined total cannot be determined from these records. |
| gpt4_7a0daae1 | reasoning-errors | True | 1 weeks |
| gpt4_a1b77f9c | reasoning-errors | True | 8 weeks |
| 0ddfec37_abs | originally-correct | True | I do not know. |
| 1568498a | originally-correct | True | 28. Kg3 would be my move. |
| 6456829e | originally-correct | True | 8 plants |
| 95228167 | originally-correct | True | Since you're planning to check out a Gibson Les Paul (considering an upgrade from your Stratocaster), here are a few key things to look for and test out at the music store:  1. **Coil-Tap / Coil-Splitting:** You were previously interested in whether a Les Paul with a coil-tap or coil-splitting feature is worth trying out. Checking for this can give you extra single-coil tones and tonal flexibility. 2. **Weight and Balance:** Les Pauls are known for being a bit heavier than Stratocasters. Pay attention to how it feels on a strap and while sitting down. 3. **Neck Profile:** Check which neck shape (e.g., SlimTaper vs. traditional thicker vintage profiles) feels most comfortable for your fretting hand. 4. **Pickup Response:** Pay attention to how the humbuckers sound across clean and overdriven channels, especially if you plan on experimenting with alternate tunings like open D. |
| a89d7624 | originally-correct | True | Since you are planning a trip to Denver, you might want to check out some of the top music venues and spots recommended previously, such as Red Rocks Park and Amphitheatre (where you can dine at The Ship Rock Grille before a show), The Ogden Theatre, The Bluebird Theater, The Fillmore Auditorium, and Larimer Lounge. |
| af8d2e46 | originally-correct | True | You packed 7 shirts for your trip to Costa Rica. |
| ed4ddc30 | originally-correct | True | 20 dozen eggs |
| gpt4_2f8be40d | originally-correct | False | I do not know. |
| gpt4_7ddcf75f | originally-correct | True | 3 days |
| gpt4_a2d1d1f6 | originally-correct | True | 3 days |
