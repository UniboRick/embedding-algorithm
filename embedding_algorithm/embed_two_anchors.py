"""This module contains functions to embed pyramidal structures in 2D. Each structure is composed of multiple pyramids (at least two) sharing at most two base nodes pairwise."""

from .helpers import *
from .embedding import restore_edges

# ---------------------------------------
# TWO PYRAMIDS WITH TWO SHARED BASE NODES
# ---------------------------------------
def double_shared_double_bow_tie(coords, pyramids, force_case = False, start_idx = 0):
    """
    Build the planar embedding of two pyramids sharing 
    two base nodes as a double bow-tie gadget.

    Parameters
    ----------
    coords : dict
        Mapping node -> (x, y, z) for all nodes. Exactly two nodes must have
        z != 0, and they are treated as the left and right apex of the gadget.
    pyramids : list of iterable
        List of pyramids, each given as a collection of node identifiers. One
        pyramid must contain the left apex, another the right apex, and these two
        pyramids must share exactly two base nodes.
    force_case : bool, optional
        If True, bypass the automatic case analysis and force a specific alignment
        of the shared base nodes on the right side. This is mainly used as a
        subroutine in `double_shared_triple_bow_tie`. Default is False.
    start_idx : int, optional
        Initial index for auxiliary/wire nodes, passed through to
        single_bow_tie / complete_bow_tie to keep wire labels (e.g. "W1",
        "W2", …) consistent across gadgets. Default is 0.

    Returns
    -------
    tuple
        (pos, all_nodes, apex_list, orientation, current_idx), where:
        - pos : dict  
        Mapping node -> (x, y) with the final 2D positions for all nodes
        involved in the double bow-tie.
        - all_nodes : set  
        Set of all node labels present in this gadget (including any wire
        nodes created by the helpers).
        - apex_list : list  
        [left_apex, right_apex], the two apex nodes ordered from left to
        right according to their x-coordinate in the original coords.
        - orientation : {"right", "left", "up", "down"}  
        Global orientation of the second bow-tie relative to the first one in
        the final embedding.
        - current_idx : int  
        The next available index for wire-node creation after this gadget is
        fully built.

    Notes
    -----
    - The function assumes:
    - exactly two apex nodes (z ≠ 0) in `coords`, and
    - the corresponding two pyramids share exactly two base nodes.
    - Internally, the first bow-tie is always built from the left-apex pyramid;
    the second bow-tie (right-apex pyramid) is then attached depending on the
    configuration of the two shared base nodes.
    - Several geometric cases are handled (shared nodes on the right, left, top,
    bottom, or opposite tips), sometimes via swapping base nodes and applying
    a ±90° rotation to obtain the correct orientation.
    - Raises ValueError if the configuration of shared nodes does not match
    any of the expected patterns.
    """

    # -----------------------
    # Find the left-most apex
    # -----------------------
    apex_list = []
    for pyr in pyramids:
        apex_list.append(apex_of(pyr, coords))
        
    if len(apex_list) != 2:
        raise ValueError("Expected exactly two apex nodes, found {}".format(apex_list))
    
    left_apex  = min(apex_list, key = lambda n: (coords[n][0], coords[n][1], n))
    right_apex = max(apex_list, key = lambda n: (coords[n][0], coords[n][1], n))
    apex_list  = [left_apex, right_apex]
        
    left_pyramid  = pyramid_with(left_apex, pyramids)
    right_pyramid = pyramid_with(right_apex, pyramids)
    coords_left   = restrict_coords_to_pyramid(left_pyramid, coords)

    # -----------------------
    # Build the first bow-tie
    # -----------------------
    pos = {}
    pos_left, _, _, (rt, rb), (lt, lb), current_idx = single_bow_tie(left_pyramid, coords_left, start_idx)
    pos.update(pos_left)
    all_nodes = set(pos.keys())

    # Find shared nodes and new nodes
    shared_nodes = sorted(n for n in left_pyramid if n in right_pyramid)
    if len(shared_nodes) != 2:
        raise ValueError("Expected exactly two shared nodes between the pyramids")
    
    new_base_node = next(n for n in right_pyramid if n not in left_pyramid and n != right_apex)

    # -----------------------------------------------------
    # Force case to be used in double_shared_triple_bow_tie
    # -----------------------------------------------------
    if force_case:
        n1, n2 = shared_nodes
        x1, x2 = pos[n1][0], pos[n2][0]

        correct_x = max(coord[0] for coord in pos.values())

        if x1 != correct_x:
            correct_node = next(n for n in all_nodes if pos[n][0] == correct_x and n not in shared_nodes)
            swap_with = next((v for v in all_nodes if pos[v][1] == pos[n1][1] and v != n1 and v != n2), correct_node)
            pos, _ = swap_helper(pos, {n1: swap_with})

        if x2 != correct_x:
            correct_node = next(n for n in all_nodes if pos[n][0] == correct_x and n not in shared_nodes)
            swap_with = next((v for v in all_nodes if pos[v][1] == pos[n2][1] and v != n1 and v != n2), correct_node)
            pos, _ = swap_helper(pos, {n2: swap_with})

        new_pos, _, _, current_idx = complete_bow_tie(n1, n2, pos, right_apex, new_base_node, coords, 'right', current_idx)
        pos.update(new_pos)
        all_nodes = set(pos.keys())

        return pos, all_nodes, apex_list, 'right', current_idx

    # -----------------------------------------------
    # Case 1: Both shared nodes are on the right side
    # -----------------------------------------------
    if set(shared_nodes) == {rt, rb}:
        new_pos, _, _, current_idx = complete_bow_tie(rt, rb, pos, right_apex, new_base_node, coords, 'right', current_idx)

        orientation = "right"
        pos.update(new_pos)
        all_nodes = set(pos.keys())

        return pos, all_nodes, apex_list, orientation, current_idx
    
    # ----------------------------------------------
    # Case 2: Both shared nodes are on the left side
    # ----------------------------------------------
    elif set(shared_nodes) == {lt, lb}:
        new_pos, _, _, current_idx = complete_bow_tie(lt, lb, pos, right_apex, new_base_node, coords, 'left', current_idx)

        orientation = "left"
        pos.update(new_pos)
        all_nodes = set(pos.keys())

        return pos, all_nodes, apex_list, orientation, current_idx
    
    # -------------------------------------------------------------
    # Case 3: Both shared nodes are on the top of the first bow-tie
    # -------------------------------------------------------------
    elif set(shared_nodes) == {rt, lt}:

        mapping = {lt : rb}
        pos, swapped  = swap_helper(pos_left, mapping)
        rt, rb = (swapped.get(x, x) for x in (rt, rb))

        new_pos, _, _, current_idx = complete_bow_tie(rt, rb, pos, right_apex, new_base_node, coords, 'right', current_idx)
        
        orientation = "up"
        pos.update(new_pos)
        final_pos = rotate_helper(pos, 90)
        all_nodes = set(final_pos.keys())

        return final_pos, all_nodes, apex_list, orientation, current_idx
    
    # ----------------------------------------------------------------
    # Case 4: Both shared nodes are on the bottom of the first bow-tie
    # ----------------------------------------------------------------
    elif set(shared_nodes) == {rb, lb}:

        mapping = {lb : rt}
        pos, swapped = swap_helper(pos_left, mapping)
        rt, rb = (swapped.get(x, x) for x in (rt, rb))

        new_pos, _, _, current_idx = complete_bow_tie(rt, rb, pos, right_apex, new_base_node, coords, 'right', current_idx)
        
        orientation = "down"
        pos.update(new_pos)
        final_pos = rotate_helper(pos, - 90)
        all_nodes = set(final_pos.keys())

        return final_pos, all_nodes, apex_list, orientation, current_idx

    # ------------------------------------------------------------------
    # Case 5: The shared nodes are on opposite tips of the first bow-tie
    # ------------------------------------------------------------------
    elif set(shared_nodes) == {rt, lb}:
        if coords[right_apex][1] >= coords[left_apex][1]:

            mapping = {lb : rb}
            pos, swapped = swap_helper(pos_left, mapping)
            rt, rb = (swapped.get(x, x) for x in (rt, rb))

            new_pos, _, _, current_idx = complete_bow_tie(rt, rb, pos, right_apex, new_base_node, coords, 'right', current_idx)
            
            orientation = "up"
            pos.update(new_pos)
            final_pos = rotate_helper(pos, 90)
            all_nodes = set(final_pos.keys())

            return pos, all_nodes, apex_list, orientation, current_idx
        
        elif coords[right_apex][1] < coords[left_apex][1]:

            mapping = {lb : rb}
            pos, swapped = swap_helper(pos_left, mapping)
            rt, rb = (swapped.get(x, x) for x in (rt, rb))

            new_pos, _, _, current_idx = complete_bow_tie(rt, rb, pos, right_apex, new_base_node, coords, 'right', current_idx)
            
            orientation = "down"
            pos.update(new_pos)
            final_pos = rotate_helper(pos, - 90)
            all_nodes = set(final_pos.keys())

            return pos, all_nodes, apex_list, orientation, current_idx
    
    # ------------------------------------------------------------------
    # Case 6: The shared nodes are on opposite tips of the first bow-tie
    # ------------------------------------------------------------------
    elif set(shared_nodes) == {rb, lt}:

        if coords[right_apex][1] >= coords[left_apex][1]:

            mapping = {lt : rt}
            pos, swapped = swap_helper(pos_left, mapping)
            rt, rb = (swapped.get(x, x) for x in (rt, rb))

            new_pos, _, _, current_idx = complete_bow_tie(rt, rb, pos, right_apex, new_base_node, coords, 'right', current_idx)
            
            orientation = "up"
            pos.update(new_pos)
            final_pos = rotate_helper(pos, 90)
            all_nodes = set(final_pos.keys())

            return pos, all_nodes, apex_list, orientation, current_idx
        
        elif coords[right_apex][1] < coords[left_apex][1]:

            mapping = {lt : rt}
            pos, swapped = swap_helper(pos_left, mapping)
            rt, rb = (swapped.get(x, x) for x in (rt, rb))

            new_pos, _, _, current_idx = complete_bow_tie(rt, rb, pos, right_apex, new_base_node, coords, 'right', current_idx)
            
            orientation = "down"
            pos.update(new_pos)
            final_pos = rotate_helper(pos, - 90)
            all_nodes = set(final_pos.keys())

            return pos, all_nodes, apex_list, orientation, current_idx

    else:
        raise ValueError(f"Shared nodes do not match any expected configuration")

# -----------------------------------------
# THREE PYRAMIDS WITH TWO SHARED BASE NODES
# -----------------------------------------

# --------------------------
# Additional Helper Function
# --------------------------
def double_shared_vertical(node1, node2, pos, direction, new_pyr, coords, current_idx):
    """
    Place a pyramid sharing two base nodes in a vertical configuration and add
    the corresponding wire nodes.

    Parameters
    ----------
    node1, node2 : hashable
        The two shared base nodes through which the new pyramid is attached.
    pos : dict
        Mapping node -> (x, y) of current 2D positions; updated in place with
        the new layout.
    direction : {"up", "down"}
        Whether the new pyramid should be placed above ("up") or below ("down")
        the shared base edge.
    new_pyr : iterable
        Nodes of the new pyramid, including `node1` and `node2` plus its other
        base nodes (and possibly an apex).
    coords : dict
        Mapping node -> (x, y, z) used to determine left/right ordering of
        the remaining base nodes in the original 3D geometry.
    current_idx : int
        Current index used for naming auxiliary wire nodes (e.g. "W1", "W2", …).

    Returns
    -------
    tuple
        (pos, current_idx), where:
        - pos : dict  
        Updated mapping node -> (x, y) including the placed pyramid and
        the newly created wire nodes.
        - current_idx : int  
        Updated index after adding the two new wire nodes.

    Notes
    -----
    - The helper shift_helper is used to shift the shared base nodes in the
    given direction before placing the rest of the pyramid.
    - The remaining base nodes of new_pyr are vertically aligned above/below
    node1 and node2, preserving the left/right ordering from coords.
    - Two auxiliary wire nodes (labels starting with "W") are created at
    positions forming 60° connections (using √3/2) to maintain an equilateral
    structure in the bow-tie layout.
    - Raises ValueError if direction is not "up" or "down".
    """

    import math
    s = math.sqrt(3) / 2
    v = 1 if direction == "up" else - 1
    if direction not in ("up", "down"):
        raise ValueError(f"Direction must be either 'up' or 'down', got '{direction}'")

    pos, _, current_idx = shift_helper(node1, pos, direction, current_idx)
    pos, _, current_idx = shift_helper(node2, pos, direction, current_idx)

    left_node  = node1 if pos[node1][0] <= pos[node2][0] else node2
    right_node = node2 if left_node == node1 else node1

    remaining_nodes = sorted((n for n in new_pyr if n not in (node1, node2)), key = lambda x: (isinstance(x, str), x))
    left_remaining = remaining_nodes[0] if coords[remaining_nodes[0]][0] <= coords[remaining_nodes[1]][0] else remaining_nodes[1]
    right_remaining = remaining_nodes[1] if left_remaining == remaining_nodes[0] else remaining_nodes[0]

    pos[left_remaining]  = (pos[left_node][0], pos[left_node][1] + v)
    pos[right_remaining] = (pos[right_node][0], pos[right_node][1] + v)

    aux_prefix    = "W"
    num_aux_nodes = 2
    aux_nodes     = [f"{aux_prefix}{i}" for i in range(current_idx + 1, current_idx + 1 + num_aux_nodes)]

    pos[aux_nodes[0]] = (pos[left_node][0] + s, pos[left_node][1] + v / 2)
    pos[aux_nodes[1]] = (pos[right_node][0] - s, pos[right_node][1] + v / 2)
    current_idx += num_aux_nodes

    return pos, current_idx

# -------------
# Main Function
# -------------
def double_shared_triple_bow_tie(coords, pyramids, start_idx = 0):
    """
    Build a triple bow-tie gadget from three 
    pyramids sharing base nodes in pairs.

    Parameters
    ----------
    coords : dict
        Mapping node -> (x, y, z) for all nodes. Nodes with z != 0 are treated
        as apex nodes of pyramids.
    pyramids : list of iterable
        List of three pyramids, each given as a collection of node identifiers.
        Two of them must share two base nodes (forming the initial double bow-tie),
        and the remaining pyramid must share two base nodes with that structure.
    start_idx : int, optional
        Initial index for naming auxiliary wire nodes (e.g. "W1", "W2", …) passed
        down to internal helpers. Default is 0.

    Returns
    -------
    tuple
        (pos, all_nodes, apex_list, direction, current_idx), where:
        - pos : dict  
        Mapping node -> (x, y) giving the final 2D layout of all nodes
        involved in the triple bow-tie gadget.
        - all_nodes : set  
        Set of all node labels present in this gadget (including any wire
        nodes created along the way).
        - apex_list : list  
        Sorted list of all apex nodes (those with z != 0) as inferred
        from coords.
        - direction : {"up", "down"}  
        Relative orientation of the third bow-tie with respect to the initial
        double bow-tie (whether it is attached above or below).
        - current_idx : int  
        The next available index for wire-node creation after this gadget
        is fully built.

    Notes
    -----
    - The two apex nodes with the smallest vertical separation |Δy| are chosen as
    the left and right apex; their pyramids are used to build the initial
    double bow-tie via double_shared_double_bow_tie.
    - The remaining (third) pyramid is assumed to share exactly two nodes with
    the already-built structure; otherwise a ValueError is raised.
    - Before attaching the third pyramid, the two shared nodes are swapped (using
    swap_helper) so that they both lie on a common top or bottom y-level,
    depending on whether the third apex is above or below the initial pair.
    - The third bow-tie is added with double_shared_vertical, which creates
    auxiliary wire nodes ("W…") and preserves the equilateral geometry.
    """

    from itertools import combinations

    # ---------------------------------------------
    # Step 1: Sort the pyramids by their apex nodes
    # ---------------------------------------------
    all_pyr_nodes = set().union(*pyramids)
    apex_list = sorted(list({node for node, (_, _, z) in coords.items() if z != 0 and node in all_pyr_nodes}))

    # Find the two apex nodes which have the smallest |dy|
    min_pair = min(combinations(apex_list, 2), key = lambda pair: compute_abs_dy(pair[0], pair[1], coords))
    apex1, apex2 = min_pair

    # Determine the left and right apex based on their x-coordinates
    if coords[apex1][0] <= coords[apex2][0]:
        left_apex, right_apex = apex1, apex2
    else:
        left_apex, right_apex = apex2, apex1

    left_pyramid  = pyramid_with(left_apex, pyramids)
    right_pyramid = pyramid_with(right_apex, pyramids)
    third_pyramid = next(p for p in pyramids if p != left_pyramid and p != right_pyramid)
    third_apex    = apex_of(third_pyramid, coords)

    # ----------------------------------------------------------------------
    # Step 2: build the first double bow-tie between left and right pyramids
    # ----------------------------------------------------------------------
    left_coords  = restrict_coords_to_pyramid(left_pyramid, coords)
    right_coords = restrict_coords_to_pyramid(right_pyramid, coords)
    double_coords = {**left_coords, **right_coords}
    double_pyrs = [left_pyramid, right_pyramid]

    pos, all_nodes, _, _, current_idx = double_shared_double_bow_tie(double_coords, double_pyrs, True, start_idx)

    # ------------------------------------------------------------------------------
    # Step 3: ensure the shared nodes with the third pyramid are on the same y-level
    # ------------------------------------------------------------------------------
    direction = "up" if coords[third_apex][1] >= coords[left_apex][1] else "down"

    shared_with_third = [n for n in third_pyramid if n in pos] 
    if len(shared_with_third) != 2:
        raise ValueError(f"Expected the remaining pyramid to share exactly two nodes with one of the others")

    correct_y = max(coord[1] for coord in pos.values()) if direction == "up" else min(coord[1] for coord in pos.values())

    n1, n2 = shared_with_third
    y1, y2 = pos[n1][1], pos[n2][1]

    if y1 != correct_y: 
        swap_with = next(v for v in all_nodes if pos[v][0] == pos[n1][0] and v != n1)
        pos, _ = swap_helper(pos, {n1: swap_with})

    if y2 != correct_y:
        swap_with = next(v for v in all_nodes if pos[v][0] == pos[n2][0] and v != n2)
        pos, _ = swap_helper(pos, {n2: swap_with})
        
    # -------------------------------
    # Step 4: Build the third bow-tie
    # -------------------------------
    pos, current_idx = double_shared_vertical(n1, n2, pos, direction, third_pyramid, coords, current_idx)
    all_nodes = set(pos.keys())

    return pos, all_nodes, apex_list, direction, current_idx

# -------------------------------------------------
# CHAIN OF PYRAMIDS WITH TWO SHARED BASE NODES EACH
# ------------------------------------------------- 
def double_shared_chain(pyramids, coords, start_idx = 0):
    """
    Embed a chain of double-shared pyramids as an alternating bow-tie layout.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each given as a collection of node identifiers.
        Consecutive pyramids in the intended chain must share exactly two
        base nodes.
    coords : dict
        Mapping node -> (x, y, z) for all nodes. Nodes with z != 0 are
        treated as apex nodes of the pyramids.
    start_idx : int, optional
        Initial index for naming auxiliary wire nodes (e.g. "W1", "W2", …).
        Default is 0.

    Returns
    -------
    tuple
        (pos, all_nodes, apex_list, current_idx), where:
        - pos : dict  
        Mapping node -> (x, y) giving the final 2D layout of the entire
        chain of pyramids and wire nodes.
        - all_nodes : list  
        Sorted list of all node labels present in the embedding, including
        any wire nodes created along the way.
        - apex_list : list  
        List of apex nodes for all pyramids, in the order extracted at the
        beginning of the function.
        - current_idx : int  
        The next available index for wire-node creation after the full chain
        has been embedded.

    Notes
    -----
    - The first three pyramids (chosen by sorting their apexes by y, then x,
    then label) are embedded as a triple bow-tie using
    double_shared_triple_bow_tie.
    - Remaining pyramids are then attached one by one as a chain:
    - On even steps (by `parity_idx`), the next pyramid is attached
        horizontally via complete_bow_tie, after aligning the two shared
        nodes on the same x-level.
    - On odd steps, the next pyramid is attached vertically via
        double_shared_vertical, after aligning the shared nodes on the
        same y-level.
    - The variable vert_dir controls whether vertical attachments occur
    above ("up") or below ("down") the existing structure.
    - Auxiliary wire nodes (labels starting with "W") are introduced by
    the helper functions to maintain the equilateral geometry of the chain.
    - A ValueError may be raised indirectly by helpers if shared-node
    assumptions are violated (e.g. a candidate pyramid does not share
    exactly two nodes with the previous one).
    """

    # ---------------------------------------------------------
    # Choose the first three pyramids (sorted by y,x and index)
    # ---------------------------------------------------------

    apex_list   = [apex_of(pyr, coords) for pyr in pyramids]
    sorted_apex = sorted(apex_list, key = lambda x: (coords[x][1], coords[x][0], x))
    first_pyrs  = [pyramid_with(apex, pyramids) for apex in sorted_apex[:3]]
    
    first_coords = {}
    for pyr in first_pyrs:
        first_coords.update(restrict_coords_to_pyramid(pyr, coords))

    # ------------------------------
    # Embed the first three pyramids
    # ------------------------------
    pos, _, _, vert_dir, current_idx = double_shared_triple_bow_tie(first_coords, first_pyrs, start_idx)

    # ----------------------------
    # Embed the remaining pyramids
    # ----------------------------
    embedded_pyrs = first_pyrs[:]
    remaining_pyramids = [pyr for pyr in pyramids if pyr not in first_pyrs]
    parity_idx = 0

    while len(embedded_pyrs) < len(pyramids):
        all_nodes = sorted(set(pos.keys()))

        if vert_dir == "down":
            vert_dir = "up"
            parity_idx += 1
            continue        

        if parity_idx % 2 == 0:
            previous_pyr = None
            pyr = None

            for prev_candidate in reversed(embedded_pyrs):
                prev_apex = apex_of(prev_candidate, coords)
                base_prev = base_nodes(prev_candidate, prev_apex)

                for candidate in remaining_pyramids:
                    if candidate is prev_candidate or candidate == prev_candidate:
                        continue

                    cand_apex = apex_of(candidate, coords)
                    base_cand = base_nodes(candidate, cand_apex)
                    shared_nodes = [n for n in base_prev if n in base_cand]

                    if len(shared_nodes) == 2:
                        previous_pyr = prev_candidate
                        pyr = candidate
                        n1, n2 = shared_nodes
                        break

                if pyr is not None:
                    break
            
            if pyr is None:
                raise ValueError("No suitable pyramid found to attach horizontally.")
            
            # Ensure the shared nodes are on the same x-level
            new_apex   = apex_of(pyr, coords)
            new_node   = next(n for n in pyr if n not in previous_pyr and n != new_apex)
            new_coords = {n : coords[n] for n in pyr if n not in previous_pyr}

            previous_apex = apex_of(previous_pyr, coords)
            previous_pos  = {n: pos[n] for n in previous_pyr if n in pos}

            direction = "right" if coords[new_apex][0] >= coords[previous_apex][0] else "left"
            correct_x = max(coord[0] for coord in previous_pos.values()) if direction == "right" else min(coord[0] for coord in previous_pos.values())
            x1, x2    = pos[n1][0], pos[n2][0]

            if x1 != correct_x: 
                swap_with = next(v for v in all_nodes if pos[v][1] == pos[n1][1] and v != n1)
                pos, _    = swap_helper(pos, {n1: swap_with})

            if x2 != correct_x:
                swap_with = next(v for v in all_nodes if pos[v][1] == pos[n2][1] and v != n2)
                pos, _    = swap_helper(pos, {n2: swap_with})

            # Embed the new pyramid
            new_pos, _, _, current_idx = complete_bow_tie(n1, n2, pos, new_apex, new_node, new_coords, direction, current_idx)
            pos.update(new_pos)
            embedded_pyrs.append(pyr)
            remaining_pyramids.remove(pyr)
            parity_idx += 1

        # Find the next pyramid as the one which 
        # shares two nodes with the previous one
        elif parity_idx % 2 == 1:
            previous_pyr = None
            pyr = None

            for prev_candidate in reversed(embedded_pyrs):
                prev_apex = apex_of(prev_candidate, coords)
                base_prev = base_nodes(prev_candidate, prev_apex)

                for candidate in remaining_pyramids:
                    if candidate is prev_candidate or candidate == prev_candidate:
                        continue

                    cand_apex = apex_of(candidate, coords)
                    base_cand = base_nodes(candidate, cand_apex)
                    shared_nodes = [n for n in base_prev if n in base_cand]

                    if len(shared_nodes) == 2:
                        previous_pyr = prev_candidate
                        pyr = candidate
                        n1, n2 = shared_nodes
                        break
                
                if pyr is not None:
                    break

            if pyr is None:
                raise ValueError("No suitable pyramid found to attach vertically.")
            
            # Ensure the shared nodes are on the same y-level
            new_apex   = apex_of(pyr, coords)
            new_node   = next(n for n in pyr if n not in previous_pyr and n != new_apex)
            new_coords = {n : coords[n] for n in pyr if n not in previous_pyr}

            previous_apex = apex_of(previous_pyr, coords)
            previous_pos  = {n: pos[n] for n in previous_pyr if n in pos}

            correct_y = max(coord[1] for coord in previous_pos.values()) if vert_dir == "up" else min(coord[1] for coord in previous_pos.values())
            y1, y2    = pos[n1][1], pos[n2][1]

            if y1 != correct_y: 
                swap_with = next(v for v in all_nodes if pos[v][0] == pos[n1][0] and v != n1)
                pos, _    = swap_helper(pos, {n1: swap_with})

            if y2 != correct_y:
                swap_with = next(v for v in all_nodes if pos[v][0] == pos[n2][0] and v != n2)
                pos, _    = swap_helper(pos, {n2: swap_with})

            # Embed the new pyramid
            pos, current_idx = double_shared_vertical(n1, n2, pos, vert_dir, pyr, coords, current_idx)
            embedded_pyrs.append(pyr)
            remaining_pyramids.remove(pyr)
            parity_idx += 1

    all_nodes = sorted(set(pos.keys()))

    return pos, all_nodes, apex_list, current_idx

# --------------------------------------------------------------------------------
# CLUSTER OF FOUR PYRAMIDS WITH ONE SHARING ALL BASE NODES AND THE OTHER 3 SHARING 
# ALL TWO BASE NODES WITH THE "CENTRAL" ONE (I.E. THE ONE SHARING ALL BASE NODES)
# --------------------------------------------------------------------------------
def double_shared_cluster(pyramids, coords, start_idx = 0):
    """
    Embed four double-shared pyramids in a compact clustered bow-tie layout.

    Parameters
    ----------
    pyramids : list of iterable
        List of four pyramids, each given as a collection of node identifiers.
        Three are first embedded as a triple bow-tie, and the fourth is attached
        to form a compact cluster.
    coords : dict
        Mapping node -> (x, y, z) for all nodes. Nodes with z != 0 are
        treated as apex nodes of the pyramids.
    start_idx : int, optional
        Initial index for naming auxiliary wire nodes (e.g. "W1", "W2", …).
        Default is 0.

    Returns
    -------
    tuple
        (pos, all_nodes, apex_list, current_idx), where:
        - pos : dict  
        Mapping node -> (x, y) giving the final 2D layout of the clustered
        pyramids and wire nodes.
        - all_nodes : list  
        Sorted list of all node labels present in the embedding, including
        any wire nodes created in the process.
        - apex_list : list  
        List of apex nodes for all pyramids, extracted at the beginning and
        sorted by their original (x, y, label).
        - current_idx : int  
        The next available index for wire-node creation after the cluster
        has been fully embedded.

    Notes
    -----
    - The first three pyramids are chosen by sorting their apexes by x, then y,
    then label, and embedded via double_shared_triple_bow_tie.
    - The “third” pyramid in that triple is identified according to its vertical
    position in the embedded layout, depending on whether it was attached
    above or below (vert_dir).
    - The fourth pyramid:
    - is located as the unique pyramid not in the initial triple,
    - shares two nodes with the third pyramid (a common base and a shifted node),
    - is attached using either bottom_right_horizontal_helper or
        top_right_horizontal_helper, depending on vert_dir.
    - One shared node is first shifted (with shift_helper) to the correct
    y-level (direction "up" or "down"), and an existing wire node ("W…") is
    swapped with the other shared node to prepare the geometry.
    - Additional wire nodes (labels starting with "W") are then added to
    route the connection while preserving equilateral spacing
    (using s = √3 / 2).
    - A ValueError is raised if more than one candidate node is found to
    shift between the third and fourth pyramids (violating the expected
    configuration).
    """

    import math
    s = math.sqrt(3) / 2

    # ---------------------------------------------------------
    # Choose the first three pyramids (sorted by x,y and index)
    # ---------------------------------------------------------
    apex_list   = [apex_of(pyr, coords) for pyr in pyramids]
    sorted_apex = sorted(apex_list, key = lambda x: (coords[x][0], coords[x][1], x))
    first_pyrs  = [pyramid_with(apex, pyramids) for apex in sorted_apex[:3]]

    first_coords = {}
    for pyr in first_pyrs:
        first_coords.update(restrict_coords_to_pyramid(pyr, coords))

    # ------------------------------
    # Embed the first three pyramids
    # ------------------------------
    pos, _, _, vert_dir, current_idx = double_shared_triple_bow_tie(first_coords, first_pyrs, start_idx)

    # ------------------------------------------------
    # Find the relative position of the fourth pyramid
    # ------------------------------------------------
    third_apex  = max(sorted_apex[:3], key = lambda x : pos[x][1]) if vert_dir == "up" else min(sorted_apex[:3], key = lambda x: pos[x][1])
    third_pyr   = pyramid_with(third_apex, pyramids)

    fourth_pyr    = next(pyr for pyr in pyramids if pyr not in first_pyrs)
    fourth_apex   = apex_of(fourth_pyr, coords)

    direction = "down" if coords[third_apex][1] >= coords[fourth_apex][1] else "up"
    correct_y = max(coord[1] for coord in pos.values()) if direction == "up" else min(coord[1] for coord in pos.values())

    # -----------------------------------------------
    # Shift the shared node between the third and 
    # fourth pyramids which has already been embedded
    # -----------------------------------------------
    node_to_shift = [node for node in fourth_pyr if node not in third_pyr and node != fourth_apex and node in pos and pos[node][1] == correct_y]
    if len(node_to_shift) == 1: 
        node_to_shift = node_to_shift[0] 
    else: 
        raise ValueError("Multiple nodes found to shift, expected one.")

    pos, _, current_idx = shift_helper(node_to_shift, pos, direction, current_idx)

    common_node = next(node for node in fourth_pyr if node in third_pyr and node != fourth_apex and node != node_to_shift)
    
    # ---------------------------
    # Place the remaining bow-tie
    # ---------------------------
    if vert_dir == "up":
        aux_node = next(node for node in pos if node.startswith("W") and pos[node][1] == pos[common_node][1] - 2 and pos[node][0] == pos[common_node][0])
        pos, _    = swap_helper(pos, {aux_node: common_node})
        backup_pos = pos[common_node]

        new_pos, *_, current_idx = bottom_right_horizontal_helper(node_to_shift, pos[node_to_shift], fourth_pyr, coords, current_idx)
        pos.update(new_pos)

        aux_prefix = "W"
        num_aux = 6
        aux_nodes = [f"{aux_prefix}{i}" for i in range(current_idx + 1, current_idx + num_aux + 1)]
        current_idx += num_aux
        
        pos[aux_nodes[0]] = backup_pos
        pos[aux_nodes[1]] = (pos[aux_node][0] + 1, pos[aux_node][1] - 2)
        pos[aux_nodes[2]] = (pos[aux_node][0] + 1 + s, pos[aux_node][1] - 2.5)
        pos[aux_nodes[3]] = (pos[aux_node][0] + 1 + s, pos[aux_node][1] - 3.5)
        pos[aux_nodes[4]] = (pos[aux_node][0] + 1 + s, pos[aux_node][1] - 4.5)
        pos[aux_nodes[5]] = (pos[common_node][0] + 1, pos[common_node][1])

    elif vert_dir == "down":
        aux_node = next(node for node in pos if node.startswith("W") and pos[node][1] == pos[common_node][1] + 2 and pos[node][0] == pos[common_node][0])
        pos, _    = swap_helper(pos, {aux_node: common_node})
        backup_pos = pos[common_node]

        new_pos, *_, current_idx = top_right_horizontal_helper(node_to_shift, pos[node_to_shift], fourth_pyr, coords, current_idx)
        pos.update(new_pos)

        aux_prefix = "W"
        num_aux = 6
        aux_nodes = [f"{aux_prefix}{i}" for i in range(current_idx + 1, current_idx + num_aux + 1)]
        current_idx += num_aux
        
        pos[aux_nodes[0]] = backup_pos
        pos[aux_nodes[1]] = (pos[aux_node][0] + 1, pos[aux_node][1] + 2)
        pos[aux_nodes[2]] = (pos[aux_node][0] + 1 + s, pos[aux_node][1] + 2.5)
        pos[aux_nodes[3]] = (pos[aux_node][0] + 1 + s, pos[aux_node][1] + 3.5)
        pos[aux_nodes[4]] = (pos[aux_node][0] + 1 + s, pos[aux_node][1] + 4.5)
        pos[aux_nodes[5]] = (pos[common_node][0] + 1, pos[common_node][1])

    all_nodes = sorted(set(pos.keys()))

    return pos, all_nodes, apex_list, current_idx

# ------------------------------------------------------------
# ISOLATED STAR NODES (NODE SHARED BETWEEN 4 OR MORE PYRAMIDS)
# WHERE EACH PAIR OF PYRAMIDS SHARES TWO BASE NODES (NO LOOPS)
# ------------------------------------------------------------

# ---------------------------
# Additional helper functions
# ---------------------------
def external_wire_left_up(start_node_coords, start_idx = 0):
    """
    Generate a left-up external wire segment starting from a given node position.

    This creates a 5-node wire (W-nodes) extending first to the left and then
    upward, following the fixed geometric pattern used for external connections.

    Parameters
    ----------
    start_node_coords : tuple of float
        (x, y) coordinates of the node from which the external wire originates.
    start_idx : int, optional
        Starting index for naming the auxiliary wire nodes (e.g. W1, W2, …).
        Default is 0.

    Returns
    -------
    tuple
        (pos, current_idx), where:
        - pos : dict  
        Mapping wire_node -> (x, y) containing the coordinates of the
        newly created W-nodes.
        - current_idx : int  
        Updated index after creating the 5 new wire nodes.

    Notes
    -----
    - The geometry uses fixed displacements based on equilateral-triangle
    distances:  
    s = √3 / 2 and t = √(1 - (1.5 - s)²), ensuring compatibility
    with the bow-tie structures.
    - Wire nodes are created with labels "W(i)" starting from start_idx + 1.
    - This function only places the wire; it does not connect it to any graph
    structure. Integration must be handled by the caller.
    """

    import math
    s = math.sqrt(3) / 2
    t = math.sqrt(1 - (1.5 - s)** 2)     

    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    pos = {}

    aux_prefix = "W"  
    num_aux_nodes = 5
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x - 1, start_y)
    pos[aux_nodes[1]] = (start_x - 1.5, start_y + s)
    pos[aux_nodes[2]] = (start_x - 1.5 - t, start_y + 1.5)
    pos[aux_nodes[3]] = (start_x - 1.5, start_y + 3 - s)
    pos[aux_nodes[4]] = (start_x - 1, start_y + 3)
    current_idx = start_idx + num_aux_nodes

    return pos, current_idx

def external_wire_left_down(start_node_coords, start_idx = 0):
    """
    Generate a left-down external wire segment starting from a given node position.

    This creates a 5-node wire (W-nodes) extending first to the left and then
    downward, following the fixed geometric pattern used for external connections.

    Parameters
    ----------
    start_node_coords : tuple of float
        (x, y) coordinates of the node from which the external wire originates.
    start_idx : int, optional
        Starting index for naming the auxiliary wire nodes (e.g. W1, W2, …).
        Default is 0.

    Returns
    -------
    tuple
        (pos, current_idx), where:
        - pos : dict  
        Mapping wire_node -> (x, y) containing the coordinates of the
        newly created W-nodes.
        - current_idx : int  
        Updated index after creating the 5 new wire nodes.

    Notes
    -----
    - The geometry uses fixed displacements based on equilateral-triangle
    distances:  
    s = √3 / 2 and t = √(1 - (1.5 - s)²), ensuring compatibility
    with the bow-tie structures.
    - Wire nodes are created with labels "W(i)" starting from start_idx + 1.
    - This function only places the wire; it does not connect it to any graph
    structure. Integration must be handled by the caller.
    """

    import math
    s = math.sqrt(3) / 2
    t = math.sqrt(1 - (1.5 - s)** 2)     

    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    pos = {}

    aux_prefix = "W"  
    num_aux_nodes = 5
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x - 1, start_y)
    pos[aux_nodes[1]] = (start_x - 1.5, start_y - s)
    pos[aux_nodes[2]] = (start_x - 1.5 - t, start_y - 1.5)
    pos[aux_nodes[3]] = (start_x - 1.5, start_y - 3 + s)
    pos[aux_nodes[4]] = (start_x - 1, start_y - 3)
    current_idx = start_idx + num_aux_nodes

    return pos, current_idx

def external_wire_right_up(start_node_coords, start_idx = 0):
    """
    Generate a right-up external wire segment starting from a given node position.

    This creates a 5-node wire (W-nodes) extending first to the right and then
    upward, following the fixed geometric pattern used for external connections.

    Parameters
    ----------
    start_node_coords : tuple of float
        (x, y) coordinates of the node from which the external wire originates.
    start_idx : int, optional
        Starting index for naming the auxiliary wire nodes (e.g. W1, W2, …).
        Default is 0.

    Returns
    -------
    tuple
        (pos, current_idx), where:
        - pos : dict  
        Mapping wire_node -> (x, y) containing the coordinates of the
        newly created W-nodes.
        - current_idx : int  
        Updated index after creating the 5 new wire nodes.

    Notes
    -----
    - The geometry uses fixed displacements based on equilateral-triangle
    distances:  
    s = √3 / 2 and t = √(1 - (1.5 - s)²), ensuring compatibility
    with the bow-tie structures.
    - Wire nodes are created with labels "W(i)" starting from start_idx + 1.
    - This function only places the wire; it does not connect it to any graph
    structure. Integration must be handled by the caller.
    """
    
    import math
    s = math.sqrt(3) / 2
    t = math.sqrt(1 - (1.5 - s)** 2)     

    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    pos = {}
    aux_prefix = "W"  
    num_aux_nodes = 5
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x + 1, start_y)
    pos[aux_nodes[1]] = (start_x + 1.5, start_y + s)
    pos[aux_nodes[2]] = (start_x + 1.5 + t, start_y + 1.5)
    pos[aux_nodes[3]] = (start_x + 1.5, start_y + 3 - s)
    pos[aux_nodes[4]] = (start_x + 1, start_y + 3)
    current_idx = start_idx + num_aux_nodes

    return pos, current_idx

def external_wire_right_down(start_node_coords, start_idx = 0):
    """
    Generate a right-down external wire segment starting from a given node position.

    This creates a 5-node wire (W-nodes) extending first to the right and then
    downward, following the fixed geometric pattern used for external connections.

    Parameters
    ----------
    start_node_coords : tuple of float
        (x, y) coordinates of the node from which the external wire originates.
    start_idx : int, optional
        Starting index for naming the auxiliary wire nodes (e.g. W1, W2, …).
        Default is 0.

    Returns
    -------
    tuple
        (pos, current_idx), where:
        - pos : dict  
        Mapping wire_node -> (x, y) containing the coordinates of the
        newly created W-nodes.
        - current_idx : int  
        Updated index after creating the 5 new wire nodes.

    Notes
    -----
    - The geometry uses fixed displacements based on equilateral-triangle
    distances:  
    s = √3 / 2 and t = √(1 - (1.5 - s)²), ensuring compatibility
    with the bow-tie structures.
    - Wire nodes are created with labels "W(i)" starting from start_idx + 1.
    - This function only places the wire; it does not connect it to any graph
    structure. Integration must be handled by the caller.
    """
    
    import math
    s = math.sqrt(3) / 2
    t = math.sqrt(1 - (1.5 - s)** 2)     

    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    pos = {}
    aux_prefix = "W"  
    num_aux_nodes = 5
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x + 1, start_y)
    pos[aux_nodes[1]] = (start_x + 1.5, start_y - s)
    pos[aux_nodes[2]] = (start_x + 1.5 + t, start_y - 1.5)
    pos[aux_nodes[3]] = (start_x + 1.5, start_y - 3 + s)
    pos[aux_nodes[4]] = (start_x + 1, start_y - 3)
    current_idx = start_idx + num_aux_nodes

    return pos, current_idx

def external_wire_top(start_node_coords, start_idx = 0):
    """
    Generate a top external wire (going right) segment starting from a given node position.

    This creates a 5-node wire (W-nodes) extending upward from the starting point,
    following the standardized geometric pattern used in all external connections.

    Parameters
    ----------
    start_node_coords : tuple of float
        (x, y) coordinates of the node from which the external wire originates.
    start_idx : int, optional
        Starting index for naming the auxiliary wire nodes (e.g. W1, W2, …).
        Default is 0.

    Returns
    -------
    tuple
        (pos, current_idx), where:
        - pos : dict  
        Mapping wire_node -> (x, y) containing the coordinates of the
        newly created W-nodes.
        - current_idx : int  
        Updated index after creating the 5 new wire nodes.

    Notes
    -----
    - The geometry uses displacements derived from equilateral-triangle structure:  
    s = √3 / 2 and t = √(1 - (1.5 - s)²), matching the spacing used in
    all bow-tie and wire gadgets to maintain a consistent lattice.
    - Wire nodes are named "W(i)" starting from start_idx + 1.
    - This function only generates node positions; it does not connect the
    wire to any graph. Integration is the caller's responsibility.
    """
    
    import math
    s = math.sqrt(3) / 2
    t = math.sqrt(1 - (1.5 - s)** 2)     

    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    pos = {}
    aux_prefix = "W"  
    num_aux_nodes = 5
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x, start_y + 1)
    pos[aux_nodes[1]] = (start_x + s, start_y + 1.5)
    pos[aux_nodes[2]] = (start_x + 1.5, start_y + 1.5 + t)
    pos[aux_nodes[3]] = (start_x + 3 - s, start_y + 1.5)
    pos[aux_nodes[4]] = (start_x + 3, start_y + 1)
    current_idx = start_idx + num_aux_nodes

    return pos, current_idx

def external_wire_bottom(start_node_coords, start_idx = 0):
    """
    Generate a top external wire (going right) segment starting from a given node position.

    This creates a 5-node wire (W-nodes) extending upward from the starting point,
    following the standardized geometric pattern used in all external connections.

    Parameters
    ----------
    start_node_coords : tuple of float
        (x, y) coordinates of the node from which the external wire originates.
    start_idx : int, optional
        Starting index for naming the auxiliary wire nodes (e.g. W1, W2, …).
        Default is 0.

    Returns
    -------
    tuple
        (pos, current_idx), where:
        - pos : dict  
        Mapping wire_node -> (x, y) containing the coordinates of the
        newly created W-nodes.
        - current_idx : int  
        Updated index after creating the 5 new wire nodes.

    Notes
    -----
    - The geometry uses displacements derived from equilateral-triangle structure:  
    s = √3 / 2 and t = √(1 - (1.5 - s)²), matching the spacing used in
    all bow-tie and wire gadgets to maintain a consistent lattice.
    - Wire nodes are named "W(i)" starting from start_idx + 1.
    - This function only generates node positions; it does not connect the
    wire to any graph. Integration is the caller's responsibility.
    """

    import math
    s = math.sqrt(3) / 2
    t = math.sqrt(1 - (1.5 - s)** 2)     

    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    pos = {}
    aux_prefix = "W"  
    num_aux_nodes = 5
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x, start_y - 1)
    pos[aux_nodes[1]] = (start_x + s, start_y - 1.5)
    pos[aux_nodes[2]] = (start_x + 1.5, start_y - 1.5 - t)
    pos[aux_nodes[3]] = (start_x + 3 - s, start_y - 1.5)
    pos[aux_nodes[4]] = (start_x + 3, start_y - 1)
    current_idx = start_idx + num_aux_nodes

    return pos, current_idx

# -------------
# Main function
# -------------
def double_shared_star_node(pyramids, coords, start_idx = 0):
    """
    Embed a star-shaped cluster of pyramids sharing a common base node into a planar bow-tie layout.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each given as a collection of node identifiers.
        All pyramids are assumed to share exactly one common base node
        (the star node), and pairwise share two base nodes in a chain-like way.
    coords : dict
        Mapping node -> (x, y, z) for all nodes. Nodes with z != 0 are
        treated as apex nodes of the pyramids.
    start_idx : int, optional
        Initial index for naming auxiliary wire nodes (e.g. "W1", "W2", …).
        Default is 0.

    Returns
    -------
    tuple
        (pos, apex_list, all_nodes, current_idx), where:
        - pos : dict  
        Mapping node -> (x, y) giving the final 2D layout of the star-node
        cluster, including all pyramids and wire nodes.
        - apex_list : list  
        List of apex nodes (one for each pyramid), extracted from coords.
        - all_nodes : list  
        Sorted list of all node labels present in the embedding, including
        any auxiliary wire nodes ("W…").
        - current_idx : int  
        The next available index for wire-node creation after the full
        star cluster has been embedded.

    Notes
    -----
    - The star node is identified as the unique node common to all pyramids.
    - The first pyramid is chosen among those that contain the least-used base
    nodes (by multiplicity in the cluster), with the lowest apex (sorted by
    y, then x); it is embedded via single_bow_tie.
    - The second pyramid is chosen as one that shares exactly two base nodes
    with the first. Its shared base nodes are rearranged using swap_helper
    so that they match the right-top/right-bottom configuration of the first
    bow-tie, then embedded via complete_bow_tie to the right.
    - The third pyramid is attached above the second using
    double_shared_vertical, after ensuring that the star node is the
    upper of the two shared base nodes.
    - Remaining pyramids are attached iteratively:
    - each shares two base nodes with the previously embedded pyramid,
    - the non-star shared node is shifted upward using shift_helper,
    - the pyramid is embedded with top_left_horizontal_helper to the left of
        the existing structure,
    - the star node is kept at the bottom-left corner of the local gadget,
        with swaps performed as needed.
    - At each step, an auxiliary wire node is placed at the star node's
    original position, and an external wire is drawn from that original
    position to the new one via external_wire_left_up, ensuring a clean,
    non-overlapping connection.
    - Several ValueError conditions may be raised indirectly if structural
    assumptions are violated (e.g. pyramid sharing patterns differing from
    the expected star-node configuration).
    """

    from collections import Counter

    # ------------------------------------------------
    # Find the first pyramid as the one with one base 
    # vertex free and the lowest y (and then lowest x)
    # ------------------------------------------------

    # Find apexes and base nodes
    apex_list = [apex_of(pyr, coords) for pyr in pyramids]
    all_base_nodes = [n for pyr in pyramids for n in base_nodes(pyr, apex_list)]
    star_node = next(iter(set.intersection(*(set(pyr) for pyr in pyramids))), None)

    # Select the free base nodes
    base_nodes_count = Counter(all_base_nodes)
    min_count = min(base_nodes_count.values())
    lowest_nodes = [n for n, c in base_nodes_count.items() if c == min_count]

    # Find (unique) pyramids containing any of the lowest-count nodes
    candidate_pyramids = []
    for node in lowest_nodes:
        try:
            p = pyramid_with(node, pyramids)  
            if p not in candidate_pyramids: 
                candidate_pyramids.append(p)
        except ValueError:
            pass

    if not candidate_pyramids:
        raise ValueError("No pyramids found containing the lowest-count base nodes.")

    # Select the first pyramid as the one with the lowest apex (y, then x)
    candidate_apexes = [apex_of(pyr, coords) for pyr in candidate_pyramids]
    selected_apex = min(candidate_apexes, key = lambda n: (coords[n][1], coords[n][0]))
    first_pyr = pyramid_with(selected_apex, candidate_pyramids)

    # Embed the first pyramid
    pos, *_, (rt, rb), _, current_idx = single_bow_tie(first_pyr, coords, start_idx)
    embedded_pyramids = [first_pyr]
    
    # ----------------------------------
    # Find the second pyramid and ensure 
    # the shared nodes are on the right
    # ----------------------------------

    # Identify the second pyramid sharing two base nodes with the first
    second_pyr_candidates = find_double_shared(first_pyr, pyramids, apex_list)
    second_pyr, first_shared = next((p, s) for p, s in second_pyr_candidates if p not in embedded_pyramids)
    new_apex = apex_of(second_pyr, coords)
    new_node = next(n for n in second_pyr if n not in first_shared and n != new_apex)

    # Ensure the shared nodes are in the correct position
    expected = {rt, rb}
    for _ in range(2):
        if first_shared != expected:
            # Identify the wrong shared node (the one not in [rt, rb])
            wrong_shared = next((n for n in first_shared if n not in expected), None)
            if wrong_shared is None:
                raise ValueError("Shared nodes mismatch but no single wrong node detected.")
        
            # Swap with the node on the same y-level
            y_wrong = pos[wrong_shared][1]
            swap_with = next((n for n, (_, y) in pos.items() if y == y_wrong and n != wrong_shared), None)
            pos, swapped_nodes = swap_helper(pos, {wrong_shared: swap_with})

            # Update labels to avoid re-swapping the same targets
            rt = swapped_nodes.get(rt, rt)
            rb = swapped_nodes.get(rb, rb)
            expected = {rt, rb}

    # Embed the second pyramid
    new_pos, _, _, current_idx = complete_bow_tie(rt, rb, pos, new_apex, new_node, coords, "right", current_idx)
    pos.update(new_pos)
    embedded_pyramids.append(second_pyr)

    # Ensure the new node is above the new apex
    if pos[new_node][1] < pos[new_apex][1]:
        pos, _ = swap_helper(pos, {new_node: new_apex})

    # ---------------------------------
    # Find the third pyramid and ensure 
    # the shared nodes are above
    # ---------------------------------
    third_pyr_candidates = find_double_shared(second_pyr, pyramids, apex_list)
    third_pyr, second_shared = next((p, s) for p, s in third_pyr_candidates if p not in embedded_pyramids)
    new_apex = apex_of(third_pyr, coords)
    new_node = next(n for n in third_pyr if n not in second_shared and n != new_apex)

    # Ensure the star node is above
    first_shared_list = list(first_shared)
    not_star_node = next(n for n in first_shared_list if n != star_node)

    if pos[star_node][1] < pos[not_star_node][1]:
        pos, _ = swap_helper(pos, {star_node: not_star_node})

    # Embed the third pyramid
    new_shared_1, new_shared_2 = second_shared
    pos, current_idx = double_shared_vertical(new_shared_1, new_shared_2, pos, "up", third_pyr, coords, current_idx)
    embedded_pyramids.append(third_pyr)

    # -------------------------------------
    # Find and embed the remaining pyramids
    # -------------------------------------
    remaining_pyramids = [pyr for pyr in pyramids if pyr not in embedded_pyramids]

    for _ in range(len(remaining_pyramids)):
        # Find the next pyramid sharing two base nodes with the last embedded pyramid
        previous_pyr = embedded_pyramids[-1]
        next_pyr_candidates = find_double_shared(previous_pyr, pyramids, apex_list)
        next_pyr, next_shared = next((p, s) for p, s in next_pyr_candidates if p not in embedded_pyramids)
        next_apex = apex_of(next_pyr, coords)

        # Shift the non-star shared node up
        node_to_shift = next(n for n in next_shared if n != star_node)
        pos, _, current_idx = shift_helper(node_to_shift, pos, "up", current_idx)

        # Save star node original position for later
        star_node_orig_pos = pos[star_node]

        # Embedding the next pyramid
        node_to_shift_pos = pos[node_to_shift]
        new_pos, _, top_left, bot_left, current_idx = top_left_horizontal_helper(node_to_shift, node_to_shift_pos, next_pyr, coords, current_idx)
        pos.update(new_pos)

        # Adding an auxiliary node where the star node was
        pos[f"W{current_idx + 1}"] = star_node_orig_pos
        current_idx += 1

        # Ensure the star node is at bottom left
        if star_node != bot_left:
            pos, swapped_nodes = swap_helper(pos, {star_node: bot_left})

            # Update labels
            top_left = swapped_nodes.get(top_left, top_left)
            bot_left = swapped_nodes.get(bot_left, bot_left)

        # Swap apex and top left node
        pos, swapped_nodes = swap_helper(pos, {next_apex: top_left})

        # Check if there is a node to the left of the star node's original position to avoid overlaps with the wire
        target_x = star_node_orig_pos[0] - 1
        target_y = star_node_orig_pos[1]
        target_node = next((n for n, (x, y) in pos.items() if x == target_x and y == target_y), None)

        # Move such node back to the star node's original position
        if target_node is not None:
            pos[target_node] = star_node_orig_pos
            current_idx -= 1

        # Add the external wire to connect the star node's original position with the new position
        new_pos, current_idx = external_wire_left_up(star_node_orig_pos, current_idx)
        pos.update(new_pos)

        # Update the embedded pyramids before the next iteration
        embedded_pyramids.append(next_pyr)

    all_nodes = sorted(set(pos.keys()))
    return pos, apex_list, all_nodes, current_idx

# ------------------------------------------------------------
# ISOLATED STAR NODES (NODE SHARED BETWEEN 4 OR MORE PYRAMIDS)
# WHERE EACH PAIR OF PYRAMIDS SHARES TWO BASE NODES WITH LOOPS
# ------------------------------------------------------------
def double_shared_closed_star_node(pyramids, coords, start_idx = 0):
    
   from collections import Counter

   # ---------------------------------------
   # Find the first pyramid as the one whose
   # apex has lowest y (and then lowest x)
   # ---------------------------------------

   # Find apexes and star node
   apex_list = [apex_of(pyr, coords) for pyr in pyramids]
   star_node = next(iter(set.intersection(*(set(pyr) for pyr in pyramids))), None)

   # Find the first pyramid
   first_apex = min(apex_list, key = lambda n: (coords[n][1], coords[n][0]))
   first_pyr = pyramid_with(first_apex, pyramids)

   # ------------------------
   # Embded the first pyramid
   # ------------------------
   pos, *_, (rt, rb), _, current_idx = single_bow_tie(first_pyr, coords, start_idx)
   embedded_pyramids = [first_pyr]

   # ---------------------------------
   # Find and embed the second pyramid
   # ---------------------------------
   candidates = find_double_shared(first_pyr, pyramids, apex_list)
   second_pyr, first_shared = next((p, s) for p, s in candidates if p not in embedded_pyramids)
   new_apex = apex_of(second_pyr, coords)
   new_node = next(n for n in second_pyr if n not in first_shared and n != new_apex)

   # Ensure the shared nodes are in the correct position
   expected = {rt, rb}
   for _ in range(2):
      if first_shared != expected:
         # Identify the wrong shared node (the one not in [rt, rb])
         wrong_shared = next((n for n in first_shared if n not in expected), None)
         if wrong_shared is None:
               raise ValueError("Shared nodes mismatch but no single wrong node detected.")
      
         # Swap with the node on the same y-level
         y_wrong = pos[wrong_shared][1]
         swap_with = next((n for n, (_, y) in pos.items() if y == y_wrong and n != wrong_shared), None)
         pos, swapped_nodes = swap_helper(pos, {wrong_shared: swap_with})

         # Update labels to avoid re-swapping the same targets
         rt = swapped_nodes.get(rt, rt)
         rb = swapped_nodes.get(rb, rb)
         expected = {rt, rb}

   # Embed the second pyramid
   new_pos, _, _, current_idx = complete_bow_tie(rt, rb, pos, new_apex, new_node, coords, "right", current_idx)
   pos.update(new_pos)
   embedded_pyramids.append(second_pyr)

   # Ensure the new node is above the new apex
   if pos[new_node][1] < pos[new_apex][1]:
      pos, _ = swap_helper(pos, {new_node: new_apex})

   # ---------------------------------
   # Find the third pyramid and ensure 
   # the shared nodes are above
   # ---------------------------------
   third_pyr_candidates = find_double_shared(second_pyr, pyramids, apex_list)
   third_pyr, second_shared = next((p, s) for p, s in third_pyr_candidates if p not in embedded_pyramids)
   new_apex = apex_of(third_pyr, coords)
   new_node = next(n for n in third_pyr if n not in second_shared and n != new_apex)

   # Ensure the star node is above
   first_shared_list = list(first_shared)
   not_star_node = next(n for n in first_shared_list if n != star_node)

   if pos[star_node][1] < pos[not_star_node][1]:
      pos, _ = swap_helper(pos, {star_node: not_star_node})

   # Embed the third pyramid
   new_shared_1, new_shared_2 = second_shared
   pos, current_idx = double_shared_vertical(new_shared_1, new_shared_2, pos, "up", third_pyr, coords, current_idx)
   embedded_pyramids.append(third_pyr)

   # -------------------------------------
   # Find and embed the remaining pyramids
   # -------------------------------------
   remaining_pyramids = [pyr for pyr in pyramids if pyr not in embedded_pyramids]
   last_pyr = candidates[1][0]
   first_node = next(n for n in last_pyr if n in first_pyr and n != star_node)
   first_node_pos = pos[first_node]

   for _ in range(len(remaining_pyramids)):
      # Find the next pyramid sharing two base nodes with the last embedded pyramid
      previous_pyr = embedded_pyramids[-1]
      next_pyr_candidates = find_double_shared(previous_pyr, pyramids, apex_list)
      next_pyr, next_shared = next((p, s) for p, s in next_pyr_candidates if p not in embedded_pyramids)
      next_apex = apex_of(next_pyr, coords)

      # Shift the non-star shared node up
      node_to_shift = next(n for n in next_shared if n != star_node)
      pos, _, current_idx = shift_helper(node_to_shift, pos, "up", current_idx)

      # Save star node original position for later
      star_node_orig_pos = pos[star_node]

      # Embedding the next pyramid
      node_to_shift_pos = pos[node_to_shift]
      new_pos, _, top_left, bot_left, current_idx = top_left_horizontal_helper(node_to_shift, node_to_shift_pos, next_pyr, coords, current_idx)
      pos.update(new_pos)

      # Adding an auxiliary node where the star node was
      pos[f"W{current_idx + 1}"] = star_node_orig_pos
      current_idx += 1

      # Ensure the star node is at bottom left
      if star_node != bot_left:
         pos, swapped_nodes = swap_helper(pos, {star_node: bot_left})

         # Update labels
         top_left = swapped_nodes.get(top_left, top_left)
         bot_left = swapped_nodes.get(bot_left, bot_left)

      # Swap apex and top left node
      pos, swapped_nodes = swap_helper(pos, {next_apex: top_left})

      # Check if there is a node to the left of the star node's original position to avoid overlaps with the wire
      target_x = star_node_orig_pos[0] - 1
      target_y = star_node_orig_pos[1]
      target_node = next((n for n, (x, y) in pos.items() if x == target_x and y == target_y), None)

      # Move such node back to the star node's original position
      if target_node is not None:
         pos[target_node] = star_node_orig_pos
         current_idx -= 1

      # Add the external wire to connect the star node's original position with the new position
      new_pos, current_idx = external_wire_left_up(star_node_orig_pos, current_idx)
      pos.update(new_pos)

      # Update the embedded pyramids before the next iteration
      embedded_pyramids.append(next_pyr)

      if next_pyr == last_pyr:
         # Place a wire at the old position of the first node
         aux_node = f"W{current_idx + 1}"
         pos[aux_node] = first_node_pos
         current_idx += 1
         _, new_pos, current_idx = restore_edges(aux_node, first_node, pos, current_idx, is_odd = True)
         pos.update(new_pos)

   all_nodes = sorted(pos.keys())
   return pos, apex_list, all_nodes, current_idx

