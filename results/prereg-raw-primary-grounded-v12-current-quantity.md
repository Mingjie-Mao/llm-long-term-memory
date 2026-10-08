# V12 latest inventory report guard — before v12 outputs

V11 regressed the current-stock question: it copied the January snapshot30 despite
March20 in the same delivered pool. This is current-state selection, not retrieval.
For a narrow explicit current-stock query naming unit, object and storage location,
use completed raw user inventory snapshot expressions (right now/at the moment),
matching the literal item/unit and location, ordered by actual report/session time.
Same latest-time conflicting quantities remain uncertain; plans, future/as-of/past
event statements, other objects/locations cannot establish current stock. This is
snapshot chronology, not fabricated event-date resolution or general balance tracking.
No egg/item/qid hardcoding; common fridge/refrigerator lexical equivalence explicit.
Program output retains source/count/unit/report-time citation. Test latest vs earlier,
wrong role/location/item, ambiguity, and true source-vs-mention-time boundaries.
