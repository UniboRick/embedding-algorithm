"""This module provides functions to search for specific structural
patterns among pyramids defined in the 3D graph after the third order
overlaps are added. Each pyramid consists of one apex node and three base nodes."""

# ------------------
# ADDITIONAL HELPERS
# ------------------

def get_apex_and_bases(pyr):
    """
    Extract the apex node and the three base nodes from a pyramid.

    Parameters
    ----------
    pyr : iterable
        A collection of four node labels representing a pyramid. Exactly one
        label must correspond to the apex (a string starting with 'A'), and
        the remaining three labels must be its base nodes.

    Returns
    -------
    tuple
        (apex, bases), where:
        - apex : hashable  
        The unique apex node (string starting with 'A').
        - bases : tuple  
        A 3-tuple containing the three base nodes.

    Notes
    -----
    - The function enforces that exactly one apex node is present; otherwise
    a ValueError is raised.
    - Bases are returned in the order they appear in pyr (except that the
    apex is removed).
    - This function is relied upon throughout the pipeline to validate the
    structural integrity of pyramids.
    """

    apex_candidates = [n for n in pyr if isinstance(n, str) and n.startswith("A")]
    if len(apex_candidates) != 1:
        raise ValueError("Pyramid does not have exactly one apex: {}".format(pyr))
    apex = apex_candidates[0]
    bases = tuple(n for n in pyr if n != apex)
    if len(bases) != 3:
        raise ValueError("Pyramid does not have exactly three base nodes: {}".format(pyr))
    return apex, bases

def indices_pyramids_sharing_two_bases(pyramids):
    """
    Return the indices of all pyramids that share at least one pair of base nodes
    with another pyramid.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    set of int
        A set of indices of pyramids that share **two** base nodes with at least
        one other pyramid. Indices are taken from the original `pyramids` list.

    Notes
    -----
    - For each pyramid, all unordered pairs of base nodes are enumerated using
    ``itertools.combinations``.
    - A mapping from each base pair to the set of pyramid indices containing that
    pair is constructed.
    - Any base pair that appears in ≥2 pyramids identifies all pyramids involved
    in a two-base overlap; all such indices are returned.
    - No ordering is imposed on the result; the returned set may be used directly
    for membership tests or further processing.
    """

    from collections import defaultdict
    from itertools import combinations

    pair_to_indices = defaultdict(set)

    for idx, pyr in enumerate(pyramids):
        _, bases = get_apex_and_bases(pyr)
        for a, b in combinations(bases, 2):
            key = tuple(sorted((a, b)))
            pair_to_indices[key].add(idx)

    involved = set()
    for idxs in pair_to_indices.values():
        if len(idxs) >= 2:
            involved.update(idxs)

    return involved

# ----------------
# SEARCH FUNCTIONS
# ----------------

def find_pyramids(G, apex_list):
    """
    Extract pyramid structures from a graph given the list of apex nodes.

    Parameters
    ----------
    G : networkx.Graph
        The graph containing all nodes and edges of the structure.
    apex_list : iterable
        List of nodes that are known to be apexes. Each apex is expected to have
        exactly three non-apex neighbors forming the pyramid base.

    Returns
    -------
    list of list
        A list of pyramids, where each pyramid is represented as a list
        [apex, b1, b2, b3] containing the apex followed by its three
        base nodes (sorted alphabetically/numerically).

    Notes
    -----
    - For each apex, the function collects all neighbors that are not apexes
    themselves; these must correspond to the three base nodes of the pyramid.
    - A ValueError is raised if an apex does not have exactly three
    non-apex neighbors, as this violates the expected pyramid structure.
    - No geometric assumptions are made here; the function purely inspects the
    graph connectivity.
    """

    pyramids = []
    for apex in apex_list:
        base_neigh = [n for n in G.neighbors(apex) if n not in apex_list]
        
        if len(base_neigh) != 3:
            raise ValueError(f"Apex {apex} has {len(base_neigh)} neighbours, expected 3")
        
        pyramids.append([apex] + sorted(base_neigh))
        
    return pyramids

def find_double_shared_pyramids(pyramids):
    """
    Find all pyramids that share at least one pair of base nodes with another pyramid.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        List of pyramids (in their original input order) that share two base
        nodes with at least one other pyramid.

    Notes
    -----
    - A pyramid is considered “double-shared” if it has at least one pair
    of base nodes that also appears as a base pair in another pyramid.
    - Base pairs are treated as unordered, so (b1, b2) and (b2, b1) are equivalent.
    - Internally, a mapping from base-node pairs to the set of pyramid indices
    containing that pair is built; all pyramids appearing in any pair with
    multiplicity ≥ 2 are returned.
    - If no pyramid shares two base nodes with any other, the function returns
    an empty list.
    """

    from collections import defaultdict
    from itertools import combinations
    
    pair_to_indices = defaultdict(set)

    # Build mapping from unordered base pairs to the set of pyramid indices
    for idx, pyr in enumerate(pyramids):
        _, bases = get_apex_and_bases(pyr)
        for a, b in combinations(bases, 2):
            key = tuple(sorted((a, b)))
            pair_to_indices[key].add(idx)

    # Collect indices of pyramids that share at least one base pair
    involved_indices = set()
    for indices in pair_to_indices.values():
        if len(indices) >= 2:
            involved_indices.update(indices)

    # Return them in input order
    return [pyramids[i] for i in sorted(involved_indices)]

def find_isolated_pyramids(pyramids):
    """
    Find the first pyramid that is 'isolated' according to extended connectivity rules.

    Included are:

    - Type 1: Pyramids whose nodes (apex and bases) do not appear in any other
      pyramid.
    - Type 2: Pyramids that
        * do not belong to the set of pyramids sharing two base nodes with others,
        * share exactly one base node with that two-base-sharing set, and
        * do not share any other node (apex or base) with any pyramid.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list or None
        The first isolated pyramid (Type 1 or Type 2) in input order,
        or None if no isolated pyramid exists.
    """

    from collections import Counter
    
    n = len(pyramids)

    # Node counts for Type 1
    all_nodes = [node for pyr in pyramids for node in pyr]
    counts = Counter(all_nodes)

    # Precompute apex, bases, and node sets
    apex_list = []
    base_sets = []
    node_sets = []
    for pyr in pyramids:
        apex, bases = get_apex_and_bases(pyr)
        apex_list.append(apex)
        base_sets.append(set(bases))
        node_sets.append(set(pyr))

    # Type 1 isolated: all nodes appear exactly once
    type1_indices = set(
        i for i, pyr in enumerate(pyramids)
        if all(counts[node] == 1 for node in pyr)
    )

    # Pyramids that share two bases with others
    two_base_indices = indices_pyramids_sharing_two_bases(pyramids)

    # Type 2 isolated: attached by exactly one base to the two-base set
    type2_indices = set()
    for i in range(n):
        # Skip pyramids already special
        if i in two_base_indices or i in type1_indices:
            continue

        bases_i = base_sets[i]
        nodes_i = node_sets[i]

        # Bases shared with any pyramid in the two-base set
        shared_bases_with_two_base = set()
        for j in two_base_indices:
            shared_bases_with_two_base |= (bases_i & base_sets[j])

        # Must share exactly one base node with that set
        if len(shared_bases_with_two_base) != 1:
            continue

        # All shared nodes with ANY other pyramid
        shared_nodes_global = set()
        for k in range(n):
            if k == i:
                continue
            shared_nodes_global |= (nodes_i & node_sets[k])

        # We want that every shared node is exactly that one base node
        if shared_nodes_global and shared_nodes_global.issubset(shared_bases_with_two_base):
            type2_indices.add(i)

    # Final: type1 ∪ type2, in increasing index order
    final_indices = sorted(type1_indices | type2_indices)

    if not final_indices:
        return []

    # Return the first suitable isolated pyramid
    return pyramids[final_indices[0]]

def find_single_shared_pair(pyramids):
    """
    Find the first pair of pyramids that share exactly one 'clean' base node.

    The function searches for a pair of pyramids (i, j) satisfying all of:

    1) They share exactly one base node.
    2) That shared base node is used as a base only in those two pyramids.
    3) They do not share any other node (no shared apex or extra base).
    4) Any overlap they have with other pyramids is allowed only with
    pyramids that share two bases with others (the two-base-sharing set).

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable pair exists), or a list of length 2:
        [pyr_i, pyr_j] containing the first pair of pyramids that satisfies
        all the conditions above.

    Notes
    -----
    - The helper indices_pyramids_sharing_two_bases(pyramids) is used to
    identify the two-base-sharing set. Intersections with pyramids outside
    this set are forbidden for the candidate pair.
    - The order of pyramids in the returned list follows the original indices
    of the two selected pyramids (i < j).
    - Only the first valid pair found in the nested scan over indices is
    returned; the function does not enumerate all possible pairs.
    """

    from collections import defaultdict

    n = len(pyramids)

    # Precompute apex, bases, and full node sets
    base_sets = []
    node_sets = []
    for pyr in pyramids:
        _, bases = get_apex_and_bases(pyr)
        base_sets.append(set(bases))
        node_sets.append(set(pyr))

    # Indices of pyramids that share two bases with others
    two_base_indices = indices_pyramids_sharing_two_bases(pyramids)

    # Map each base node -> set of pyramid indices that have it as base
    base_to_indices = defaultdict(set)
    for idx, bases in enumerate(base_sets):
        for b in bases:
            base_to_indices[b].add(idx)

    # Search for the first valid pair (i, j)
    for i in range(n):
        if i in two_base_indices:
            continue
        bases_i = base_sets[i]
        nodes_i = node_sets[i]

        for j in range(i + 1, n):
            if j in two_base_indices:
                continue
            bases_j = base_sets[j]
            nodes_j = node_sets[j]

            # 1) They must share exactly ONE base node.
            shared_bases = bases_i & bases_j
            if len(shared_bases) != 1:
                continue

            shared_base = next(iter(shared_bases))

            # 2) That shared base must appear as base in exactly these two pyramids.
            if len(base_to_indices[shared_base]) != 2:
                # There is a third (or more) pyramid using this base -> skip.
                continue

            # 3) They must not share any other node (no shared apex or extra base).
            shared_nodes = nodes_i & nodes_j
            if shared_nodes != {shared_base}:
                continue

            # 4) Check their interactions with OTHER pyramids.
            union_nodes = nodes_i | nodes_j
            ok = True
            for k in range(n):
                if k == i or k == j:
                    continue

                if union_nodes & node_sets[k]:
                    # They are allowed to share nodes with pyramids
                    # in the two-base-sharing set, but not with others.
                    if k not in two_base_indices:
                        ok = False
                        break

            if not ok:
                continue

            # Found the first suitable pair: return it immediately.
            return [pyramids[i], pyramids[j]]

    # No suitable pair found
    return []

def find_shared_triple(pyramids):
    """
    Find the first triple of pyramids that share a common node under restricted
    connectivity conditions.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable triple exists), or a list of length 3:
        [pyr_i, pyr_j, pyr_k] containing the first triple of pyramids that
        satisfies all the conditions below.

    Notes
    -----
    - Only pyramids **not** in the two-base-sharing set (as given by
    indices_pyramids_sharing_two_bases(pyramids)) are considered as
    candidates for the triple.
    - There must exist a node that appears in exactly three such non-core
    pyramids; these three form the candidate triple.
    - Apart from:
    * nodes shared within the triple itself, and
    * any nodes shared with pyramids in the two-base-sharing set,
    the three pyramids must not share any node with any other pyramid.
    - In other words, the triple is either:
    * completely isolated from the rest of the structure, or
    * only connected (via shared nodes) to the two-base-sharing cluster.
    - The function returns the first triple found in this sense and does
    not enumerate all possible triples.
    """

    from collections import defaultdict

    n = len(pyramids)

    # Precompute node sets for each pyramid
    node_sets = [set(pyr) for pyr in pyramids]

    # Indices of pyramids that share two bases with others
    two_base_indices = indices_pyramids_sharing_two_bases(pyramids)

    # Consider only "non-core" pyramids (not in two-base set) for forming triples
    noncore_indices = [i for i in range(n) if i not in two_base_indices]

    # Map each node -> set of non-core pyramid indices that contain it
    node_to_noncore = defaultdict(set)
    for i in noncore_indices:
        for node in node_sets[i]:
            node_to_noncore[node].add(i)

    # For each node, look for nodes that are contained in exactly 3 non-core pyramids
    for node, idx_set in node_to_noncore.items():
        if len(idx_set) != 3:
            continue  # we only want groups of size exactly 3 here

        i, j, k = sorted(idx_set)  # three indices

        # Union of all nodes in the triple
        union_nodes = node_sets[i] | node_sets[j] | node_sets[k]

        # Check interactions with all other pyramids
        ok = True
        for m in range(n):
            if m in (i, j, k):
                continue

            if union_nodes & node_sets[m]:
                # They are allowed to intersect with pyramids in the two-base set,
                # but not with other non-core pyramids.
                if m not in two_base_indices:
                    ok = False
                    break

        if not ok:
            continue

        # Passed all checks: return this first valid triple
        return [pyramids[i], pyramids[j], pyramids[k]]

    # No suitable triple found
    return []

def find_shared_quadruple(pyramids):
    """
    Find the first quadruple of pyramids that share a common node under restricted
    connectivity conditions.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable quadruple exists), or a list of length 4:
        [pyr_i, pyr_j, pyr_k, pyr_l] containing the first quadruple of pyramids
        that satisfies all the conditions below.

    Notes
    -----
    - Only pyramids **not** in the two-base-sharing set (as given by
    indices_pyramids_sharing_two_bases(pyramids)) are considered as
    candidates for the quadruple.
    - There must exist a node that appears in exactly four such non-core
    pyramids; these four form the candidate quadruple.
    - Apart from:
    * nodes shared within the quadruple itself, and
    * any nodes shared with pyramids in the two-base-sharing set,
    the four pyramids must not share any node with any other pyramid.
    - Thus, the quadruple is either:
    * completely isolated from all other non-core pyramids, or
    * only connected (via shared nodes) to the two-base-sharing cluster.
    - The function returns the first quadruple found in this sense and does
    not enumerate all possible quadruples.
    """

    from collections import defaultdict

    n = len(pyramids)

    # Precompute node sets for each pyramid
    node_sets = [set(pyr) for pyr in pyramids]

    # Indices of pyramids that share two bases with others (the "core" structure)
    two_base_indices = indices_pyramids_sharing_two_bases(pyramids)

    # Non-core pyramids are candidates for forming the quadruple
    noncore_indices = [i for i in range(n) if i not in two_base_indices]

    # Map each node -> set of non-core pyramid indices that contain it
    node_to_noncore = defaultdict(set)
    for i in noncore_indices:
        for node in node_sets[i]:
            node_to_noncore[node].add(i)

    # Look for nodes that belong to exactly 4 non-core pyramids
    for node, idx_set in node_to_noncore.items():
        if len(idx_set) != 4:
            continue  # we only want groups of size exactly 4 here

        i, j, k, l = sorted(idx_set)

        # Union of all nodes in the quadruple
        union_nodes = node_sets[i] | node_sets[j] | node_sets[k] | node_sets[l]

        # Check interactions with all other pyramids
        ok = True
        for m in range(n):
            if m in (i, j, k, l):
                continue

            if union_nodes & node_sets[m]:
                # Allowed to intersect with pyramids in the two-base set,
                # but not with other non-core pyramids outside the quadruple.
                if m not in two_base_indices:
                    ok = False
                    break

        if not ok:
            continue

        # Passed all checks: return this first valid quadruple
        return [pyramids[i], pyramids[j], pyramids[k], pyramids[l]]

    # No suitable quadruple found
    return []

def find_shared_five(pyramids):
    """
    Find the first group of ≥5 pyramids that share the same base node under
    restricted connectivity conditions.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable group exists), or a list
        [pyr_i, pyr_j, ..., pyr_k] containing all pyramids in the first
        valid group found, where all pyramids share the same base node.

    Notes
    -----
    - Only pyramids not in the two-base-sharing set (as given by
    indices_pyramids_sharing_two_bases(pyramids)) are considered as
    candidates for the group.
    - The shared node must be a base node that appears as a base in at
    least 5 such non-core pyramids.
    - Apart from:
    * nodes shared within this group, and
    * any nodes shared with pyramids in the two-base-sharing set,
    the pyramids in the group must not share any node with any other pyramid.
    - The base nodes are scanned in sorted order to give a deterministic
    notion of the “first” valid group.
    - The function returns the entire group of pyramids sharing that base if
    it passes all checks; otherwise it continues searching and finally
    returns [] if no group is found.
    """

    from collections import defaultdict

    n = len(pyramids)

    # Precompute node sets and base sets for each pyramid
    node_sets = [set(pyr) for pyr in pyramids]
    base_sets = []
    for pyr in pyramids:
        _, bases = get_apex_and_bases(pyr)
        base_sets.append(set(bases))

    # Indices of pyramids that share two bases with others (the "core" set)
    two_base_indices = indices_pyramids_sharing_two_bases(pyramids)

    # Non-core pyramids can form these 5+ groups
    noncore_indices = [i for i in range(n) if i not in two_base_indices]

    # Map each base node -> set of non-core pyramid indices that have it as base
    base_to_noncore = defaultdict(set)
    for i in noncore_indices:
        for b in base_sets[i]:
            base_to_noncore[b].add(i)

    # Look for bases that belong to 5 or more non-core pyramids
    # (sorted to have a deterministic notion of "first")
    for base in sorted(base_to_noncore.keys()):
        idx_set = base_to_noncore[base]
        if len(idx_set) < 5:
            continue

        group_indices = sorted(idx_set)

        # Union of all nodes in this group
        union_nodes = set()
        for idx in group_indices:
            union_nodes |= node_sets[idx]

        # Check interactions with all other pyramids
        ok = True
        for m in range(n):
            if m in group_indices:
                continue

            if union_nodes & node_sets[m]:
                # Allowed to intersect with pyramids in the two-base set,
                # but not with other non-core pyramids outside this group.
                if m not in two_base_indices:
                    ok = False
                    break

        if not ok:
            continue

        # Passed all checks: return the entire group sharing this base
        return [pyramids[idx] for idx in group_indices]

    # No suitable group found
    return []

def find_single_shared_tree(pyramids):
    """
    Find the first 'tree' of pyramids connected only by single-base overlaps.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable tree exists), or a list
        ``[pyr_i, pyr_j, ..., pyr_k]`` containing the pyramids in the first
        valid tree found (with indices sorted).

    Notes
    -----
    - Only pyramids not in the two-base-sharing set (as given by
    indices_pyramids_sharing_two_bases(pyramids)) can belong to the tree.
    - A valid tree T satisfies:
    1. All pyramids in T are non-core (not in the two-base-sharing set).
    2. For any pair (Pi, Pj) in T:
        - They share at most one node.
        - If they share a node, it is a base node for both.
    3. T is connected when we draw an edge between two pyramids that share
        exactly one base node and no other nodes (the function builds this
        adjacency graph and takes connected components).
    4. Apart from:
        - nodes shared inside T itself, and
        - nodes shared with pyramids in the two-base-sharing set,
        T must not share any node with any other pyramid.
    - Components of size < 3 are discarded; the function returns the **first**
    component satisfying all checks.
    """
    from collections import defaultdict, deque

    n = len(pyramids)

    # Precompute base sets and node sets
    base_sets = []
    node_sets = []
    for pyr in pyramids:
        _, bases = get_apex_and_bases(pyr)
        base_sets.append(set(bases))
        node_sets.append(set(pyr))

    # Indices of pyramids that share two bases with others (core set)
    two_base_indices = indices_pyramids_sharing_two_bases(pyramids)

    # Non-core candidates
    noncore_indices = [i for i in range(n) if i not in two_base_indices]

    # Build adjacency for 'single-base-only' overlaps among non-core pyramids
    base_to_noncore = defaultdict(list)
    for i in noncore_indices:
        for b in base_sets[i]:
            base_to_noncore[b].append(i)

    adjacency = defaultdict(set)

    # For each base, connect pyramids that share exactly that base
    # and no other node.
    for b, idx_list in base_to_noncore.items():
        # consider all pairs of pyramids that contain base b
        for i_pos in range(len(idx_list)):
            i = idx_list[i_pos]
            for j_pos in range(i_pos + 1, len(idx_list)):
                j = idx_list[j_pos]

                shared_nodes = node_sets[i] & node_sets[j]
                # They must share exactly one node, and it must be this base
                if shared_nodes == {b}:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

    # Find connected components of this adjacency graph among non-core indices
    visited = set()
    for start in noncore_indices:
        if start in visited or start not in adjacency:
            continue  # isolated or no edges -> no tree of size >= 2

        # BFS/DFS to get the component
        queue = deque([start])
        component = []
        visited.add(start)

        while queue:
            u = queue.popleft()
            component.append(u)
            for v in adjacency[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        # We want a "bigger" structure: require at least 3 pyramids
        if len(component) < 3:
            continue

        component = sorted(component)

        # --- Check pairwise condition for ALL pairs in the component ---
        ok_pairs = True
        for idx_a in range(len(component)):
            i = component[idx_a]
            for idx_b in range(idx_a + 1, len(component)):
                j = component[idx_b]
                shared = node_sets[i] & node_sets[j]
                if len(shared) > 1:
                    ok_pairs = False
                    break
                if len(shared) == 1:
                    shared_node = next(iter(shared))
                    # must be a base in both pyramids
                    if shared_node not in base_sets[i] or shared_node not in base_sets[j]:
                        ok_pairs = False
                        break
            if not ok_pairs:
                break

        if not ok_pairs:
            continue

        # --- Check isolation from other NON-core pyramids ---
        union_nodes = set()
        for idx in component:
            union_nodes |= node_sets[idx]

        isolated_ok = True
        for m in range(n):
            if m in component:
                continue
            if m in two_base_indices:
                # allowed to share nodes with the two-base set
                continue

            if union_nodes & node_sets[m]:
                # shares at least one node with another non-core pyramid -> reject
                isolated_ok = False
                break

        if not isolated_ok:
            continue

        # Passed all checks: return this component as a tree
        return [pyramids[idx] for idx in component]

    # No suitable tree found
    return []

def find_double_shared_pair(pyramids):
    """
    Find the first pair of pyramids that share exactly two bases only with each other.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable pair exists), or a list of length 2:
        [pyr_i, pyr_j] containing the first pair of pyramids that satisfies
        the conditions below.

    Notes
    -----
    - A valid pair (Pi, Pj) must:
    * share exactly two base nodes with each other, and
    * not share two (or more) base nodes with any third pyramid.
    - Single-base overlaps with other pyramids are allowed and do not invalidate
    the pair.
    - Bases are extracted via get_apex_and_bases(pyr) and compared as sets.
    - The scan is done over unordered index pairs (i < j); the function returns
    the first valid pair found.
    """

    n = len(pyramids)

    # Precompute base sets for all pyramids
    base_sets = []
    for pyr in pyramids:
        _, bases = get_apex_and_bases(pyr)
        base_sets.append(set(bases))

    # Scan all unordered pairs
    for i in range(n):
        Bi = base_sets[i]
        for j in range(i + 1, n):
            Bj = base_sets[j]

            shared_ij = Bi & Bj

            # Condition 1: they must share exactly two base nodes
            if len(shared_ij) != 2:
                continue

            # Condition 2: neither pyramid shares two bases with any third pyramid
            ok = True
            for k in range(n):
                if k == i or k == j:
                    continue
                Bk = base_sets[k]

                # If i or j share >= 2 bases with k, reject this pair
                if len(Bi & Bk) >= 2 or len(Bj & Bk) >= 2:
                    ok = False
                    break

            if not ok:
                continue

            # First suitable pair found
            return [pyramids[i], pyramids[j]]

    # No valid pair found
    return []

def find_double_shared_triplet(pyramids):
    """
    Find the first pair of pyramids that share exactly two bases only with each other.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable pair exists), or a list of length 2:
        [pyr_i, pyr_j] containing the first pair of pyramids that satisfies
        the conditions below.

    Notes
    -----
    - A valid pair (Pi, Pj) must:
    * share exactly two base nodes with each other, and
    * not share two (or more) base nodes with any other pyramid.
    - Single-base overlaps with other pyramids are allowed and do not invalidate
    the pair.
    - Bases are extracted via get_apex_and_bases(pyr) and compared as sets.
    - The scan is done over unordered index pairs (i < j); the function returns
    the first valid pair found.
    """

    n = len(pyramids)

    # Precompute base sets
    base_sets = []
    for pyr in pyramids:
        _, bases = get_apex_and_bases(pyr)
        base_sets.append(set(bases))

    # Precompute:
    # - adjacency_2: edges where exactly 2 bases are shared
    # - neighbors_ge2: neighbors sharing >= 2 bases (used to forbid external 2-base links)
    adjacency_2 = [set() for _ in range(n)]
    neighbors_ge2 = [set() for _ in range(n)]

    for i in range(n):
        Bi = base_sets[i]
        for j in range(i + 1, n):
            Bj = base_sets[j]
            shared = Bi & Bj
            if len(shared) == 2:
                adjacency_2[i].add(j)
                adjacency_2[j].add(i)
                neighbors_ge2[i].add(j)
                neighbors_ge2[j].add(i)
            elif len(shared) > 2:
                # (theoretically 3, if bases are identical) – counts as >= 2
                neighbors_ge2[i].add(j)
                neighbors_ge2[j].add(i)

    # Scan all unordered triplets i < j < k
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triple = {i, j, k}

                # --- 1) Connectivity: each vertex must have at least one
                #         2-base neighbor inside the triple.
                if not (adjacency_2[i] & (triple - {i})):
                    continue
                if not (adjacency_2[j] & (triple - {j})):
                    continue
                if not (adjacency_2[k] & (triple - {k})):
                    continue

                # --- 2) No 2-base link to pyramids outside the triplet
                ok = True
                for v in (i, j, k):
                    # neighbors_ge2[v] contains all pyramids that share >=2 bases with v
                    external_neighbors = neighbors_ge2[v] - triple
                    if external_neighbors:
                        ok = False
                        break

                if not ok:
                    continue

                # First valid triplet found
                return [pyramids[i], pyramids[j], pyramids[k]]

    # No suitable triplet
    return []

def find_double_shared_chain(pyramids):
    """
    Find the first 'double-shared chain' of pyramids.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable chain exists), or a list
        [pyr_i, pyr_j, ..., pyr_k] where the pyramids are ordered along
        the chain.

    Notes
    -----
    - A double-shared chain is a simple path of pyramids (length ≥ 4) in which:
    * An edge between two pyramids means they share exactly two base nodes.
    * Every pyramid in the chain has degree ≤ 2 in this adjacency:
        endpoints have degree 1, internal pyramids degree 2.
    - The adjacency graph is built only from pairs of pyramids that share
    exactly two base nodes. Connected components of this graph are then
    inspected.
    - Inside a candidate component, the following must hold:
    * The component has at least 4 pyramids.
    * It is a simple path: exactly two endpoints (degree 1), all others
        have degree 2.
    * For any node that appears in the component, its multiplicity within
        the component is < 4 (nodes used by 4 or more pyramids in the
        component cause rejection).
    - The function returns the first component that satisfies all constraints,
    ordered along the path starting from one endpoint, or [] if none
    is found.
    """
    from collections import Counter, deque

    n = len(pyramids)

    # --- 1. Precompute base sets and node sets ---
    base_sets = []
    node_sets = []
    for pyr in pyramids:
        _, bases = get_apex_and_bases(pyr)
        base_sets.append(set(bases))
        node_sets.append(set(pyr))

    # --- 2. Build adjacency graph for "double-shared" edges (exactly 2 bases in common) ---
    adjacency_2 = [set() for _ in range(n)]
    for i in range(n):
        Bi = base_sets[i]
        for j in range(i + 1, n):
            Bj = base_sets[j]
            shared = Bi & Bj
            if len(shared) == 2:
                adjacency_2[i].add(j)
                adjacency_2[j].add(i)

    visited = set()

    # --- 3. Explore connected components of the adjacency_2 graph ---
    for start in range(n):
        if start in visited or not adjacency_2[start]:
            continue  # no edges from here or already processed

        # BFS to get the whole connected component in adjacency_2
        queue = deque([start])
        comp = []
        visited.add(start)

        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adjacency_2[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        # We need at least 4 pyramids in the chain
        if len(comp) < 4:
            continue

        # --- 4. Check local degree constraints and path structure ---
        degrees = {idx: len(adjacency_2[idx]) for idx in comp}

        # degree <= 2 for all
        if any(d > 2 for d in degrees.values()):
            continue

        endpoints = [idx for idx, d in degrees.items() if d == 1]
        internals = [idx for idx, d in degrees.items() if d == 2]

        # simple path: exactly two endpoints, rest degree 2
        if len(endpoints) != 2:
            continue
        if len(endpoints) + len(internals) != len(comp):
            continue

        # --- 5. Node-sharing constraint INSIDE THE COMPONENT ONLY ---
        # Count, for each node, how many pyramids in this component contain it
        node_count_in_comp = Counter()
        for idx in comp:
            for node in node_sets[idx]:
                node_count_in_comp[node] += 1

        violates_node_limit = False
        for node, c in node_count_in_comp.items():
            # We only care about shared nodes (c > 1),
            # and they must not appear in 4 or more pyramids of the component.
            if c >= 4:
                violates_node_limit = True
                break

        if violates_node_limit:
            continue

        # --- 6. Build the chain in order, starting from one endpoint ---
        start_node = endpoints[0]
        chain_order = []
        used = set([start_node])
        current = start_node

        while True:
            chain_order.append(current)
            neighbors = [v for v in adjacency_2[current] if v not in used]
            if not neighbors:
                break  # reached the other endpoint
            current = neighbors[0]
            used.add(current)

        # Sanity check: we should have covered the entire component
        if len(chain_order) != len(comp):
            # something weird (cycle or split); skip safely
            continue

        # Return the chain as pyramids in order
        return [pyramids[idx] for idx in chain_order]

    # No suitable chain found
    return []

def find_double_shared_cluster(pyramids):
    """
    Find the first 'double-shared cluster' consisting of one central and three peripheral pyramids.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable cluster exists), or a list
        [central_pyr, P1, P2, P3] containing the central pyramid followed
        by its three peripheral neighbors.

    Notes
    -----
    - A valid cluster has the following structure:
    * One central pyramid C with base set {b1, b2, b3}.
    * Three peripheral pyramids P1, P2, P3.
    - The constraints are:
    * C shares exactly two base nodes with each Pi.
    * The three shared base pairs are precisely the three combinations
        of C's bases: {b1, b2}, {b1, b3}, {b2, b3}, each used once.
    * Each Pi has C as its only double-shared neighbor (degree 1) in the
        "exactly-two-bases-in-common" adjacency graph.
    * C has exactly three such neighbors (P1, P2, P3) and no others.
    - Single-base overlaps with pyramids outside the cluster are allowed and
    do not invalidate the cluster.
    - Bases are obtained via get_apex_and_bases(pyr), and the double-sharing
    adjacency is built by checking for pairs of pyramids that share exactly
    two base nodes.
    """

    from itertools import combinations

    n = len(pyramids)

    # Precompute base sets for all pyramids
    base_sets = []
    for pyr in pyramids:
        _, bases = get_apex_and_bases(pyr)
        base_sets.append(set(bases))

    # Build adjacency for "double-shared" edges (exactly 2 bases in common)
    adjacency_2 = [set() for _ in range(n)]
    for i in range(n):
        Bi = base_sets[i]
        for j in range(i + 1, n):
            Bj = base_sets[j]
            shared = Bi & Bj
            if len(shared) == 2:
                adjacency_2[i].add(j)
                adjacency_2[j].add(i)

    # Search for a central candidate
    for c in range(n):
        Bi = base_sets[c]
        # Central must have exactly 3 double-shared neighbors
        if len(adjacency_2[c]) != 3:
            continue

        neighbors = sorted(adjacency_2[c])  # P1, P2, P3

        # Each neighbor must only have C as double-shared neighbor (degree 1)
        ok_deg = True
        for nb in neighbors:
            if adjacency_2[nb] != {c}:
                ok_deg = False
                break
        if not ok_deg:
            continue

        # Check that the three shared pairs are exactly the 3 combinations
        # of C's bases: {b1,b2}, {b1,b3}, {b2,b3}
        expected_pairs = set(frozenset(p) for p in combinations(Bi, 2))
        found_pairs = set()

        for nb in neighbors:
            shared = Bi & base_sets[nb]
            if len(shared) != 2:
                ok_deg = False
                break
            found_pairs.add(frozenset(shared))

        if not ok_deg:
            continue

        if found_pairs != expected_pairs:
            continue

        # Found the first valid cluster
        return [pyramids[c]] + [pyramids[nb] for nb in neighbors]

    # No cluster found
    return []

def find_double_shared_star(pyramids):
    """
    Find the first 'double-shared star' chain of pyramids.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable star is found), or a list
        [pyr_i, pyr_j, ..., pyr_k] where the pyramids are ordered along
        the chain.

    Notes
    -----
    - Build a graph whose vertices are pyramids and where an (undirected) edge
    between two pyramids means they share **exactly two** base nodes.
    - A valid double-shared star is a connected component of this graph that:
    * forms a simple path (chain) of length ≥ 4:
        - every pyramid has degree ≤ 2,
        - exactly two pyramids have degree 1 (endpoints),
        - all others (if any) have degree 2;
    * has a unique base node (the star node) that belongs to the base
        set of every pyramid in the component.
    - The star node is identified as the unique element in the intersection
    of all base sets of the pyramids in the component.
    - Single-node overlaps with pyramids outside this component are allowed
    and ignored; only the internal structure of the component matters.
    - The function returns the first component satisfying all these conditions,
    arranged in path order starting from one endpoint.
    """

    from collections import deque

    n = len(pyramids)

    # --- 1. Precompute base sets and node sets ---
    base_sets = []
    node_sets = []
    for pyr in pyramids:
        _, bases = get_apex_and_bases(pyr)
        base_sets.append(set(bases))
        node_sets.append(set(pyr))

    # --- 2. Build adjacency graph for "double-shared" edges (exactly 2 bases in common) ---
    adjacency_2 = [set() for _ in range(n)]
    for i in range(n):
        Bi = base_sets[i]
        for j in range(i + 1, n):
            Bj = base_sets[j]
            shared = Bi & Bj
            if len(shared) == 2:
                adjacency_2[i].add(j)
                adjacency_2[j].add(i)

    visited = set()

    # --- 3. Explore connected components of adjacency_2 ---
    for start in range(n):
        if start in visited or not adjacency_2[start]:
            continue  # no edges from here or already processed

        # BFS to get the whole connected component
        queue = deque([start])
        comp = []
        visited.add(start)

        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adjacency_2[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        # Need at least 4 pyramids in the star-chain
        if len(comp) < 4:
            continue

        # --- 4. Check that component is a simple path (chain) ---
        degrees = {idx: len(adjacency_2[idx]) for idx in comp}

        # degree must be <= 2 everywhere
        if any(d > 2 for d in degrees.values()):
            continue

        endpoints = [idx for idx, d in degrees.items() if d == 1]
        internals = [idx for idx, d in degrees.items() if d == 2]

        # simple path: exactly two endpoints, all others degree 2
        if len(endpoints) != 2:
            continue
        if len(endpoints) + len(internals) != len(comp):
            continue

        # --- 5. Star-node condition: a single BASE node common to all pyramids ---
        # Compute intersection of base sets over the component
        common_bases = set(base_sets[comp[0]])
        for idx in comp[1:]:
            common_bases &= base_sets[idx]

        # We require exactly ONE such common base node
        if len(common_bases) != 1:
            continue

        # At this point we know there is a unique star base node
        star_node = next(iter(common_bases))  # not strictly needed, but nice to keep

        # --- 6. Build the chain order starting from one endpoint ---
        start_node = endpoints[0]
        chain_order = []
        used = set([start_node])
        current = start_node

        while True:
            chain_order.append(current)
            neighbors = [v for v in adjacency_2[current] if v not in used]
            if not neighbors:
                break  # reached the other endpoint
            # In a simple path there is exactly one unvisited neighbor
            current = neighbors[0]
            used.add(current)

        # Sanity check: cover the whole component
        if len(chain_order) != len(comp):
            # something weird (cycle, etc.), skip for safety
            continue

        # Return pyramids in chain order
        return [pyramids[idx] for idx in chain_order]

    # No suitable double-shared star found
    return []

def find_double_shared_star_closed(pyramids):
    """
    Find the first 'double-shared star (closed)' cycle of pyramids.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of four node labels.
        Each pyramid must contain exactly one apex node (label starting with 'A')
        and three base nodes.

    Returns
    -------
    list
        Either an empty list (if no suitable closed star is found), or a list
        [pyr_i, pyr_j, ..., pyr_k] where the pyramids are ordered along the loop.

    Notes
    -----
    - Build a graph whose vertices are pyramids and where an undirected edge
    connects two pyramids iff they share exactly two base nodes.
    - A valid double-shared star (closed) is a connected component of this
    graph that:
    * forms a simple cycle (loop) of length ≥ 4:
        - every pyramid in the component has degree exactly 2;
        - the component is connected, so it is a single cycle;
    * has a unique base node (the star node) contained in the base set
        of every pyramid in the component.
    - The star node is obtained as the unique element in the intersection of
    all base sets of pyramids in the component.
    - Single-node overlaps with pyramids outside this component are allowed
    and ignored; only the internal structure of the component matters.
    - The cycle is returned in a consistent order by starting at the pyramid
    with the smallest index and walking around the loop.
    """

    from collections import deque

    n = len(pyramids)

    # --- 1. Precompute base sets and node sets ---
    base_sets = []
    node_sets = []
    for pyr in pyramids:
        _, bases = get_apex_and_bases(pyr)
        base_sets.append(set(bases))
        node_sets.append(set(pyr))  # not strictly needed here, but kept for consistency

    # --- 2. Build adjacency graph for "double-shared" edges (exactly 2 bases in common) ---
    adjacency_2 = [set() for _ in range(n)]
    for i in range(n):
        Bi = base_sets[i]
        for j in range(i + 1, n):
            Bj = base_sets[j]
            shared = Bi & Bj
            if len(shared) == 2:
                adjacency_2[i].add(j)
                adjacency_2[j].add(i)

    visited = set()

    # --- 3. Explore connected components of adjacency_2 ---
    for start in range(n):
        if start in visited or not adjacency_2[start]:
            continue  # already processed or isolated (no 2-base edges)

        # BFS to get the whole connected component
        queue = deque([start])
        comp = []
        visited.add(start)

        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adjacency_2[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        # We need at least 4 pyramids in the closed star
        if len(comp) < 4:
            continue

        # --- 4. Check that the component is a simple cycle ---
        degrees = {idx: len(adjacency_2[idx]) for idx in comp}

        # For a simple cycle: every node must have degree exactly 2
        if any(d != 2 for d in degrees.values()):
            continue

        # Connected + all degree 2 ⇒ the component is a cycle.

        # --- 5. Star-node condition: a single BASE node common to all pyramids ---
        common_bases = set(base_sets[comp[0]])
        for idx in comp[1:]:
            common_bases &= base_sets[idx]

        # We require exactly ONE such common base node
        if len(common_bases) != 1:
            continue

        star_node = next(iter(common_bases))  # not used further, but nice to keep

        # --- 6. Build an ordered loop (cycle) from this component ---
        # Use smallest index as a canonical starting point
        start_node = min(comp)
        cycle_order = [start_node]
        used = set([start_node])
        prev = None
        current = start_node

        while True:
            # Neighbors in the cycle, excluding where we came from
            neighbors = [v for v in adjacency_2[current] if v != prev]

            # In a simple cycle, there should be 1 or 2 neighbors;
            # excluding 'prev', exactly 1 continuation neighbor unless we close the loop.
            next_candidates = [v for v in neighbors if v not in used]

            if next_candidates:
                # Continue around the cycle
                nxt = next_candidates[0]
                cycle_order.append(nxt)
                used.add(nxt)
                prev, current = current, nxt
            else:
                # No unvisited neighbor; we should be back at start to close the loop
                # and have visited all nodes in 'comp'.
                break

        # Sanity check: we must have visited all pyramids in the component
        if len(cycle_order) != len(comp):
            # Something is weird (should not happen for a simple cycle),
            # but let's skip this component to be safe.
            continue

        # Return pyramids in loop order
        return [pyramids[idx] for idx in cycle_order]

    # No suitable closed star found
    return []

