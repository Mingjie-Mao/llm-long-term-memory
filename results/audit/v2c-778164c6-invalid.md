# LongMemEval-S item audit: `778164c6`

Date: 2026-09-17 (Australia/Sydney)

Decision: **invalid / internally inconsistent question-gold pair**.

## Benchmark fields

- Question asks for a Jamaican snapper dish that the assistant recommended and that
  has fruit in it.
- Gold answer: `Grilled Snapper with Mango Salsa`.

## Source-grounded findings

The answer session states:

1. `Escovitch Fish` is described as Jamaican, made with fried snapper and a pickled
   vegetable sauce.
2. `Grilled Snapper with Mango Salsa` is described as a dish popular in many Caribbean
   countries, with a fruity salsa.
3. When the user asks which dish to try first, the assistant recommends Escovitch Fish.
4. The user then explicitly decides to try Escovitch Fish.

No source candidate satisfies all question qualifiers. The gold item satisfies the
fruit qualifier but was not the recommended item and was not identified as specifically
Jamaican. The actually recommended item satisfies Jamaican + snapper but not fruit.

## Evaluation treatment

- Do not tune the system to emit the gold while suppressing the contradiction.
- Exclude this row from the valid-item accuracy denominator.
- Report it separately as one invalid item.
- A production-quality answer should explain the conflict and identify both items.
