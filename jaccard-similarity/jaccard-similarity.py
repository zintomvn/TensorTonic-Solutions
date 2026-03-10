def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    a = set(set_a)
    b = set(set_b)

    numer = a.intersection(b)
    denom = a.union(b)

    if len(denom) == 0:
        return 0.0
    return len(numer) / len(denom)


    