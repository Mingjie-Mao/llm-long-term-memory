# v2c.5 dish-1 preregistration

Date: 2026-09-17 (Australia/Sydney)

## Observed v2c.4 failure

The first v2c.4 row was judge-positive but failed the preregistered strict criterion:
it led with Escovitch Fish and only later named Grilled Snapper with Mango Salsa.
The second v2c.4 repetition is cancelled under the early-stop rule.

The source itself explains the ambiguity. Escovitch is explicitly Jamaican but has
pickled vegetables; Grilled Snapper with Mango Salsa explicitly has fruit but is
described broadly as Caribbean. No candidate literally satisfies every word in the
question. The question's rare discriminating attribute (“has fruit”) identifies the
gold item; the broad category (“Jamaican”) does not.

## Frozen change

When an exact-reference question conflicts with authoritative source evidence and no
candidate satisfies all qualifiers, resolve it using the most specific distinguishing
attribute or relationship rather than a broad category such as cuisine or nationality.
Return the resolved candidate directly. No retrieval, source evidence, model, store,
or non-exact-reference behavior changes.

## Gate

Run `778164c6`. Pass only when the raw hypothesis directly answers Grilled Snapper
with Mango Salsa and does not present Escovitch Fish as the requested answer. A first
pass is directional only; a second independent pass is required for stability.
