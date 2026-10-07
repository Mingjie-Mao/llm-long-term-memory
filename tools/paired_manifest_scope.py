"""Select the frozen manifest before any page/cost or answer processing."""


def select_manifest(instances, question_ids):
    ids = list(question_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate frozen question ids")
    available = {i.question_id: i for i in instances}
    if not set(ids) <= set(available):
        raise ValueError("missing frozen questions")
    return [available[qid] for qid in ids]
