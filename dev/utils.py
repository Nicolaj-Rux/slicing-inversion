def find_sup_key(dictionary, key):
    """Returns key_sup that satisfies key == key_sup in d, T, L
        and key <= key_sup in K, J, else None"""
    d0, T0, L0, K0, J0 = key
    for key_sup in dictionary:
        d, T, L, K, J = key_sup
        if d0 == d and T0 == T and L0 == L and K0 <= K and J0 <= J:
            return key_sup
    return None
