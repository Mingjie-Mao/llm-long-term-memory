# Temporal transition replay v1

> Regression evidence over an inspected development store; zero model calls.

Decision: **PASS**

- memories: **3,299**
- changed states: **15**
- first replay writes: **15**
- ambiguous transitions: **3**
- second replay writes: **0**

## Gates

- PASS — `no_active_removal`
- PASS — `known_nike_is_not_current`
- PASS — `known_adidas_is_current`
- PASS — `idempotent_second_replay`

## Changed states

### `mem_21c3f82d1b4e6459`

- content: The user plans to pack a messenger bag inside their duffel bag.
- before: `{"content": "The user plans to pack a messenger bag inside their duffel bag.", "object": "messenger bag", "predicate": "packing_list", "status": "superseded", "superseded_by": "mem_a5305df6171c2f8d", "target_object": null, "update_op": "coexists", "user_id": "a82c026e", "valid_to": "2023-05-25T23:43:00"}`
- after: `{"content": "The user plans to pack a messenger bag inside their duffel bag.", "object": "messenger bag", "predicate": "packing_list", "status": "active", "superseded_by": null, "target_object": null, "update_op": "coexists", "user_id": "a82c026e", "valid_to": null}`

### `mem_2f635055da453625`

- content: The user got rid of their old Adidas sneakers about a month before May 24, 2023.
- before: `{"content": "The user got rid of their old Adidas sneakers about a month before May 24, 2023.", "object": "Adidas sneakers", "predicate": "footwear", "status": "active", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "a82c026e", "valid_to": null}`
- after: `{"content": "The user got rid of their old Adidas sneakers about a month before May 24, 2023.", "object": "Adidas sneakers", "predicate": "footwear", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "a82c026e", "valid_to": "2023-05-23T19:31:00"}`

### `mem_5578a76d73641c27`

- content: The user's book club fizzled out a few months before May 2023.
- before: `{"content": "The user's book club fizzled out a few months before May 2023.", "object": "fizzled out", "predicate": "book_club", "status": "active", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "57f827a0", "valid_to": null}`
- after: `{"content": "The user's book club fizzled out a few months before May 2023.", "object": "fizzled out", "predicate": "book_club", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "57f827a0", "valid_to": "2023-05-21T00:21:00"}`

### `mem_60552b94c1a3218f`

- content: Val left the spacing service after 24 years to be with Odyl.
- before: `{"content": "Val left the spacing service after 24 years to be with Odyl.", "object": "left spacing service", "predicate": "employment_status", "status": "active", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "95228167", "valid_to": null}`
- after: `{"content": "Val left the spacing service after 24 years to be with Odyl.", "object": "left spacing service", "predicate": "employment_status", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "95228167", "valid_to": "2023-05-30T13:27:00"}`

### `mem_6a9ad79b93de0250`

- content: The user attends a comedy writing workshop on Sundays.
- before: `{"content": "The user attends a comedy writing workshop on Sundays.", "object": "Sundays", "predicate": "comedy_writing_workshop", "status": "superseded", "superseded_by": "mem_7d2b6d8606ac490c", "target_object": null, "update_op": "coexists", "user_id": "58ef2f1c", "valid_to": "2023-04-02T16:02:00"}`
- after: `{"content": "The user attends a comedy writing workshop on Sundays.", "object": "Sundays", "predicate": "comedy_writing_workshop", "status": "active", "superseded_by": null, "target_object": null, "update_op": "coexists", "user_id": "58ef2f1c", "valid_to": null}`

### `mem_6f3cc56a023fc60f`

- content: The user attended a study group from late November until January.
- before: `{"content": "The user attended a study group from late November until January.", "object": "attended study group", "predicate": "study_group", "status": "active", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "3a704032", "valid_to": null}`
- after: `{"content": "The user attended a study group from late November until January.", "object": "attended study group", "predicate": "study_group", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "3a704032", "valid_to": "2023-05-27T01:18:00"}`

### `mem_7bad79839d31968c`

- content: The user replaced their bathroom light fixture on Sunday, May 7, 2023.
- before: `{"content": "The user replaced their bathroom light fixture on Sunday, May 7, 2023.", "object": "bathroom light fixture", "predicate": "home_maintenance", "status": "active", "superseded_by": null, "target_object": null, "update_op": "replaces", "user_id": "57f827a0", "valid_to": null}`
- after: `{"content": "The user replaced their bathroom light fixture on Sunday, May 7, 2023.", "object": "bathroom light fixture", "predicate": "home_maintenance", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "replaces", "user_id": "57f827a0", "valid_to": "2023-05-07T00:00:00"}`

### `mem_7d2b6d8606ac490c`

- content: The user completed a stand-up comedy writing workshop at a local comedy club and wrote five new jokes.
- before: `{"content": "The user completed a stand-up comedy writing workshop at a local comedy club and wrote five new jokes.", "object": "completed workshop", "predicate": "comedy_writing_workshop", "status": "active", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "58ef2f1c", "valid_to": null}`
- after: `{"content": "The user completed a stand-up comedy writing workshop at a local comedy club and wrote five new jokes.", "object": "completed workshop", "predicate": "comedy_writing_workshop", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "58ef2f1c", "valid_to": "2023-04-02T16:02:00"}`

### `mem_7eb5e4262d66d4a2`

- content: The user cancelled a FarmFresh subscription on January 5, 2023.
- before: `{"content": "The user cancelled a FarmFresh subscription on January 5, 2023.", "object": "FarmFresh", "predicate": "subscription", "status": "superseded", "superseded_by": "mem_f2425825074736ca", "target_object": null, "update_op": "removes", "user_id": "d905b33f", "valid_to": "2023-05-24T22:49:00"}`
- after: `{"content": "The user cancelled a FarmFresh subscription on January 5, 2023.", "object": "FarmFresh", "predicate": "subscription", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "d905b33f", "valid_to": "2023-01-05T00:00:00"}`

### `mem_8dbcde069a62fce2`

- content: The user replaced their Nike Air Zoom Pegasus 38 shoes around February 10, 2023.
- before: `{"content": "The user replaced their Nike Air Zoom Pegasus 38 shoes around February 10, 2023.", "object": "Nike Air Zoom Pegasus 38", "predicate": "running_shoes", "status": "active", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "6e984301", "valid_to": null}`
- after: `{"content": "The user replaced their Nike Air Zoom Pegasus 38 shoes around February 10, 2023.", "object": "Nike Air Zoom Pegasus 38", "predicate": "running_shoes", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "6e984301", "valid_to": "2023-02-10T00:00:00"}`

### `mem_a5305df6171c2f8d`

- content: The user decided not to pack a portable stove for their trip.
- before: `{"content": "The user decided not to pack a portable stove for their trip.", "object": "portable stove", "predicate": "packing_list", "status": "active", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "a82c026e", "valid_to": null}`
- after: `{"content": "The user decided not to pack a portable stove for their trip.", "object": "portable stove", "predicate": "packing_list", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "a82c026e", "valid_to": "2023-05-25T23:43:00"}`

### `mem_a91e0e9a85a6b9d1`

- content: The user's museum membership expired in December 2022.
- before: `{"content": "The user's museum membership expired in December 2022.", "object": "expired December 2022", "predicate": "museum_membership", "status": "active", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "gpt4_7a0daae1", "valid_to": null}`
- after: `{"content": "The user's museum membership expired in December 2022.", "object": "expired December 2022", "predicate": "museum_membership", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "gpt4_7a0daae1", "valid_to": "2023-03-17T01:36:00"}`

### `mem_ac9ddfe4df29924a`

- content: The user completed a 6-week swimming program and swam 10 laps without stopping on 2023-03-04.
- before: `{"content": "The user completed a 6-week swimming program and swam 10 laps without stopping on 2023-03-04.", "object": "completed", "predicate": "swimming_program", "status": "active", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "6e984301", "valid_to": null}`
- after: `{"content": "The user completed a 6-week swimming program and swam 10 laps without stopping on 2023-03-04.", "object": "completed", "predicate": "swimming_program", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "6e984301", "valid_to": "2023-03-04T00:00:00"}`

### `mem_af327665fd4c7814`

- content: The user has been using Adidas Ultraboost 22 shoes for daily runs since February 10, 2023.
- before: `{"content": "The user has been using Adidas Ultraboost 22 shoes for daily runs since February 10, 2023.", "object": "Adidas Ultraboost 22", "predicate": "running_shoes", "status": "superseded", "superseded_by": "mem_8dbcde069a62fce2", "target_object": null, "update_op": "coexists", "user_id": "6e984301", "valid_to": "2023-02-10T00:00:00"}`
- after: `{"content": "The user has been using Adidas Ultraboost 22 shoes for daily runs since February 10, 2023.", "object": "Adidas Ultraboost 22", "predicate": "running_shoes", "status": "active", "superseded_by": null, "target_object": null, "update_op": "coexists", "user_id": "6e984301", "valid_to": null}`

### `mem_f2425825074736ca`

- content: The user decided to cancel a $50 monthly beauty box subscription.
- before: `{"content": "The user decided to cancel a $50 monthly beauty box subscription.", "object": "beauty box", "predicate": "subscription", "status": "active", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "d905b33f", "valid_to": null}`
- after: `{"content": "The user decided to cancel a $50 monthly beauty box subscription.", "object": "beauty box", "predicate": "subscription", "status": "historical", "superseded_by": null, "target_object": null, "update_op": "removes", "user_id": "d905b33f", "valid_to": "2023-05-24T22:49:00"}`
