"""This module implements the main embedding algorithm and its helper functions."""

from .helpers import *
from .generation import *
from .structure_search import *
from .embed_one_anchor import *
from .embed_two_anchors import *

# -------------------------
# FUNCTION TO RESTORE EDGES 
# -------------------------
def restore_edges(start_node, end_node, pos, start_idx = 0, tol = 0.05, angle_step_deg = 5, max_turn_deg = 165, max_steps_factor = 15, verbose = False, is_odd = False):
    """
    Route a collision-free chain of unit-length wire nodes between two existing nodes.

    Parameters
    ----------
    start_node : hashable
        Label of the node where the wire should start. Must already exist in pos.
    end_node : hashable
        Label of the node where the wire should end. Must already exist in pos.
    pos : dict
        Mapping node -> (x, y) giving current 2D positions. 
    start_idx : int, optional
        Initial index used to generate names for new wire nodes (e.g. "W1",
        "W2", …). The function ensures no collision with existing keys in pos.
    tol : float, optional
        Relative tolerance on edge length. Both the intermediate wire edges and
        the final edge to end_node are constrained to lie in the interval
        [1 - tol, 1 + tol]. Default is 0.05.
    angle_step_deg : float, optional
        Angular step (in degrees) used when rotating the heading around the
        direct line to the target. Default is 5 degrees.
    max_turn_deg : float, optional
        Maximum allowed deviation (in degrees) from the straight direction to
        the target when exploring candidate headings. Default is 165 degrees.
    max_steps_factor : int or float, optional
        Factor controlling the maximum number of routing steps. The total number
        of allowed steps is max(10, ceil(d0) * max_steps_factor), where
        d0 is the initial distance between start and end. Default is 10.
    verbose : bool, optional
        If True, print step-by-step debug information about the routing process.
        Default is False.
    is_odd : bool, optional
        If True, enforce that the number of added wire nodes is odd; otherwise, 
        enforce even number. Default is False.

    Returns
    -------
    tuple
        (new_nodes, new_pos, start_idx), where:
        - new_nodes : list of str  
        Ordered list of labels of the newly created wire nodes ("W…") from
        start_node toward end_node.
        - new_pos : dict  
        Mapping wire_node -> (x, y) containing only the positions of the
        new wire nodes (a subset of what has been added to pos).
        - start_idx : int  
        Updated index after adding all new wire nodes.

    Notes
    -----
    - All new edges (between consecutive wire nodes, and between the last wire
    node and end_node) are constrained to have length ≈ 1 within the
    tolerance interval [1 - tol, 1 + tol].
    - A proximity guard enforces that each new wire node stays at distance
    greater than 1 + tol from any existing node, except:
    - its immediate predecessor in the wire, and
    - the end_node, which it may approach only if it is the **final**
      wire node.
    - The function enforces that the number of added wire nodes is even; this
    is used for parity constraints in the larger embedding pipeline.
    - Names for wire nodes are generated as "W{idx}", starting from
    start_idx + 1 and skipping any labels already present in pos.
    - If start_node and end_node are already at unit distance (within tol),
    a ValueError is raised and no new nodes are added.
    - If no valid step can be found at some iteration, or if the maximum number
    of steps is exceeded, a ValueError is raised describing the failure.
    - The function uses a greedy, step-by-step approach, exploring candidate
    headings by rotating around the direct line to the target in increments of
    angle_step_deg, up to a maximum deviation of max_turn_deg.
    - The is_odd parameter allows enforcing an odd number of wire nodes instead
      of even (it will be only used as specific subroutines in other functions).
    """

    import numpy as np
    import math

    # ---------------------------
    # Additional helper functions
    # ---------------------------
    def rotate_vec(v, angle_rad):
        """
        Rotate a 2D vector by a given angle.

        Parameters
        ----------
        v : array-like of float
            The 2D vector to rotate, given as (vx, vy).
        angle_rad : float
            Rotation angle in radians. Positive values rotate counterclockwise.

        Returns
        -------
        numpy.ndarray
            The rotated 2D vector.

        Notes
        -----
        - The rotation uses the standard 2D rotation matrix.
        - Input v may be any sequence convertible to a NumPy array.
        """

        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])
    
    def next_free_name(idx):
        """
        Return the first available wire-node name of the form 'W{idx}'.

        Parameters
        ----------
        idx : int
            Starting index from which to search for the next unused wire-node
            label ("W{idx}").

        Returns
        -------
        tuple
            (name, next_idx), where:
            - name : str  
            The first label "W{k}" not already present in the external pos dict.
            - next_idx : int  
            The next index to use for subsequent naming attempts.

        Notes
        -----
        - This function assumes the existence of a global or closure-scoped
        dictionary pos mapping node names to coordinates.
        - Names are generated sequentially as "W{idx}", increasing  idx until an
        unused label is found.
        """

        while True:
            name = f"W{idx}"
            if name not in pos:
                return name, idx + 1
            idx += 1

    # Check if the target nodes exist
    if start_node not in pos or end_node not in pos:
        raise ValueError("start_node and end_node must exist in pos.")

    # Compute the distance between start and end nodes
    p_start = np.asarray(pos[start_node], dtype = float)
    p_end   = np.asarray(pos[end_node], dtype = float)
    d0      = float(np.linalg.norm(p_end - p_start))

    # If they're too close, they're already connected, so no action is needed
    if (1.0 - tol) <= d0 <= (1.0 + tol):
        raise ValueError(f"Start and end are alredy connected: d = {d0:.4f}", "no wire nodes added.")

    new_nodes = []
    new_pos = {}
    current = p_start.copy()

    step_len = 1.0
    R_forbid = 1.0 + tol
    max_steps = max(10, int(math.ceil(d0 / step_len) * max_steps_factor))
    target_parity = 1 if is_odd else 0

    # --------------------
    # Verbose debug output
    # --------------------
    if verbose:
        print("Initial distance =", d0, "Maximum number of allowed steps (i.e. new allowed wire nodes) =", max_steps)

    # --------------------------------------------------------------------------------
    # # Step 1: Compute the forward direction toward the end, then generate a set of
    # rotated headings (0°, ± step, ±2 * step, …) up to max_turn_deg around it.
    # These angles define all candidate directions we will test for the next wire node
    # --------------------------------------------------------------------------------
    for step in range(max_steps):
        to_end = p_end - current
        dist_current_to_end = float(np.linalg.norm(to_end))
        if dist_current_to_end == 0.0:
            break

        base_dir = to_end / dist_current_to_end

        # Angles to try around the direction to the end
        angles_deg = [0]
        k = 1
        while k * angle_step_deg <= max_turn_deg:
            angles_deg.append(k * angle_step_deg)
            angles_deg.append(- k * angle_step_deg)
            k += 1

        candidate_raw = []

        # -----------------------------------------------------------------------------
        # Step 2: For each candidate angle, rotate the base direction, propose a new
        # point at distance step_len, compute its distance to the end, and determine
        # whether it could serve as the final wire node; also identify the previous
        # node in the chain for neighbor checks and ensure an even number of wire nodes
        # -----------------------------------------------------------------------------
        for ang_deg in angles_deg:
            ang_rad = math.radians(ang_deg)
            dir_vec = rotate_vec(base_dir, ang_rad)
            cand = current + step_len * dir_vec
            cand_tuple = (float(cand[0]), float(cand[1]))
            dist_to_end   = float(np.linalg.norm(cand - p_end))

            can_be_final = ((1.0 - tol) <= dist_to_end <= (1.0 + tol) and (len(new_nodes) + 1) % 2 == target_parity)
            prev_name = start_node if len(new_nodes) == 0 else new_nodes[-1]

            # -----------------------------------------------------------------------
            # Step 3: Proximity guard – reject this candidate if it would come within
            # R_forbid of any node other than its direct predecessor (and, unless it
            # is final, the end node); otherwise mark it as geometrically valid.
            # -----------------------------------------------------------------------
            ok = True
            for existing_name, p in pos.items():
                p_arr = np.asarray(p, dtype=float)

                # The direct predecessor is allowed to be at distance ~ 1
                if existing_name == prev_name:
                    continue

                # Distance from candidate to this existing node
                dist = float(np.linalg.norm(p_arr - cand))

                if existing_name == start_node:
                    # Only the FIRST wire node may be near start
                    # (and that first node has prev_name == start_node)
                    if dist <= R_forbid:
                        ok = False
                        break

                elif existing_name == end_node:
                    # The end node may only be near the FINAL wire node
                    if can_be_final:
                        continue
                    if dist <= R_forbid:
                        ok = False
                        break

                else:
                    # Any other node (including older wire nodes) must be farther than R_forbid
                    if dist <= R_forbid:
                        ok = False
                        break

            if not ok:
                continue
            # ---------------------
            # Proximity guard ended
            # ---------------------

            # --------------------------------------------------------------------
            # Step 4: If this candidate lands within the final edge window (1±tol)
            # and preserves the even-node rule, commit it immediately and finish;
            # otherwise record it as a regular candidate for later selection.
            # --------------------------------------------------------------------
            if can_be_final:
                name, start_idx = next_free_name(start_idx)
                pos[name] = cand_tuple
                new_pos[name] = cand_tuple
                new_nodes.append(name)

                # --------------------
                # Verbose debug output
                # --------------------
                if verbose:
                    print("Final wire node", name, "at", cand_tuple, "dist_to_end =", dist_to_end)

                return new_nodes, new_pos, start_idx

            candidate_raw.append((dist_to_end, cand_tuple, ang_deg))

        # ---------------------------------------------------------------
        # If the wire gets trapped, and no candidates are feasible, abort
        # ---------------------------------------------------------------
        if not candidate_raw:
            raise ValueError(f"Routing failed from {start_node} to {end_node}: "f"no feasible step at iteration {step}.")


        # ---------------------------------------------------------------------------
        # Step 5: Choose the best candidate among those that passed the guard.
        # If still far from the end, prefer moves that reduce the remaining distance;
        # if too close, prefer moves that push us back toward ~1 unit away;
        # otherwise (in the normal band), pick the move that keeps us closest
        # to the ideal distance of 1. The top-ranked candidate becomes the next step.
        # ---------------------------------------------------------------------------
        if dist_current_to_end > 1.0 + tol:
            # Far from end: prefer candidates that reduce the distance
            closer = [c for c in candidate_raw if c[0] < dist_current_to_end]
            pool = closer if closer else candidate_raw
            pool.sort(key = lambda t: t[0])
        elif dist_current_to_end < 1.0 - tol:
            # Too close: move away, but roughly aim for distance ~1 later
            further = [c for c in candidate_raw if c[0] > dist_current_to_end]
            pool = further if further else candidate_raw
            pool.sort(key = lambda t: abs(t[0] - 1.0))
        else:
            # Around the right distance: aim to stay near distance ~1
            pool = sorted(candidate_raw, key = lambda t: abs(t[0] - 1.0))

        best_dprime, best_cand_tuple, best_ang = pool[0]

        # --------------------
        # Verbose debug output
        # --------------------
        if verbose:
            print("Step", step, "→ choose angle", best_ang, "pos", best_cand_tuple, "remaining d' =", best_dprime)

        # ---------------------------------------------------------------------
        # Step 6: Commit the selected candidate as the next wire node—assign it
        # a fresh name, record its position, append it to the chain, and update
        # 'current' so the next routing iteration begins from this new point.
        # ---------------------------------------------------------------------
        name, start_idx = next_free_name(start_idx)
        pos[name] = best_cand_tuple
        new_pos[name] = best_cand_tuple
        new_nodes.append(name)
        current = np.asarray(best_cand_tuple, dtype = float)

    # ------------------------------------------------------------
    # If we exit the loop without returning, we exceeded max_steps
    # ------------------------------------------------------------
    raise ValueError(f"Routing failed from {start_node} to {end_node}: "f"max steps {max_steps} exceeded.")

# -------------------
#    THE ALGORITHM
# -------------------

# ---------------------------
# Additional helper functions
# ---------------------------

def search_and_remap(G, apex_list, coords, max_tries = 50):
    """
    Iteratively embed all pyramid structures of a 3D 
    graph into 2D bow-tie and wire gadgets combinations.

    Parameters
    ----------
    G : networkx.Graph
        The original 3D unit-disk graph containing all nodes and edges.
    apex_list : iterable
        List of apex node labels present in G. Each apex must belong to
        exactly one pyramid.
    coords : dict
        Mapping node -> (x, y, z) giving the 3D coordinates of every node
        in the graph.
    max_tries : int, optional
        Maximum number of outer iterations to attempt while searching and
        embedding pyramid patterns. Used as a safety cap to avoid infinite
        loops if some configuration cannot be matched. Default is 50.

    Returns
    -------
    tuple
        (all_pos, current_idx, all_embedded), where:
        - all_pos : list of dict  
        List of position maps. Each element is a dict node -> (x, y)
        containing the 2D embedding of one detected structure (isolated
        pyramids, chains, clusters, trees, stars, etc.) together with
        its auxiliary wire nodes.
        - current_idx : int  
        The next available index for naming wire nodes (e.g. "W{current_idx+1}")
        after all embedded structures have been processed (or the loop stops).
        - all_embedded : bool  
        True if all detected pyramids were successfully embedded before
        reaching max_tries, False if the loop aborted early.

    Notes
    -----
    - The function first calls find_pyramids(G, apex_list) to extract all
    pyramids from the graph, then repeatedly scans the remaining (not yet
    embedded) pyramids and tries to match them to increasingly complex
    patterns.
    - After each successful pattern match, the corresponding embedding helper
    returns a position map and an updated current_idx; the pyramids used
    in that structure are marked as embedded and excluded from subsequent
    searches.
    - The process stops when all pyramids have been embedded or when
    loop_counter > max_tries. In the latter case, all_embedded is set
    to False and some pyramids may remain unembedded.
    """

    # ------------------------------
    # Find all pyramids in the graph 
    # and initialize variables
    # ------------------------------
    pyrs_to_embed = find_pyramids(G, apex_list)
    embedded_pyramids = []
    all_pos = []
    current_idx = 0 
    loop_counter = 0
    all_embedded = True

    # ------------------------------------
    # Loop until all pyramids are embedded
    # ------------------------------------
    while len(embedded_pyramids) < len(pyrs_to_embed):
        pyrs_to_check = [pyr for pyr in pyrs_to_embed if pyr not in embedded_pyramids]

        # ----------------------------
        # Search for isolated pyramids
        # ----------------------------
        isolated_pyr = find_isolated_pyramids(pyrs_to_check)
        if isolated_pyr:
            pos, *_, current_idx = single_bow_tie(isolated_pyr, coords, current_idx)
            all_pos.append(pos)
            embedded_pyramids.append(isolated_pyr)
            continue

        # ----------------------------
        # Search for pairs of pyramids 
        # sharing only one base node 
        # ---------------------------- 
        single_shared_pair = find_single_shared_pair(pyrs_to_check)
        if single_shared_pair:
            pos, *_, current_idx = single_shared_double_bow_tie(coords, single_shared_pair, current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(single_shared_pair)
            continue

        # -------------------------------
        # Search for triplets of pyramids 
        # sharing all the same base node
        # -------------------------------
        single_shared_triple = find_shared_triple(pyrs_to_check)
        if single_shared_triple:
            pos, *_, current_idx = three_pyramids_one_shared(coords, single_shared_triple, current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(single_shared_triple)
            continue

        # --------------------------
        # Search for four pyramids 
        # sharing the same base node
        # --------------------------
        single_shared_quartet = find_shared_quadruple(pyrs_to_check)
        if single_shared_quartet:
            pos, *_, current_idx = four_pyramids_one_shared(coords, single_shared_quartet, current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(single_shared_quartet)
            continue

        # --------------------------------
        # Search for five or more pyramids 
        # sharing the same base node
        # --------------------------------
        single_shared_group = find_shared_five(pyrs_to_check)
        if single_shared_group:
            pos, *_, current_idx = N_pyramids_one_shared(coords, single_shared_group, current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(single_shared_group)
            continue

        # --------------------------------------
        # Find a tree-like structure of pyramids 
        # sharing always one base node for each 
        # pair of adjacent pyramids
        # --------------------------------------
        single_shared_tree_like = find_single_shared_tree(pyrs_to_check)
        if single_shared_tree_like:
            pos, *_, current_idx = single_shared_tree(coords, single_shared_tree_like, current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(single_shared_tree_like)
            continue

        # ----------------------------
        # Search for pairs of pyramids 
        # sharing two base nodes
        # ----------------------------
        double_shared_pair = find_double_shared_pair(pyrs_to_check)
        if double_shared_pair:
            pos, *_, current_idx = double_shared_double_bow_tie(coords, double_shared_pair, start_idx = current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(double_shared_pair)
            continue

        # -----------------------------------
        # Search for triplets of pyramids 
        # sharing two one base nodes pairwise 
        # -----------------------------------
        double_shared_triplet = find_double_shared_triplet(pyrs_to_check)
        if double_shared_triplet:
            pos, *_, current_idx = double_shared_triple_bow_tie(coords, double_shared_triplet, current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(double_shared_triplet)
            continue

        # ----------------------------------
        # Search for chains of pyramids each 
        # sharing two base nodes pairwise
        # ----------------------------------
        double_sharing_chain = find_double_shared_chain(pyrs_to_check)
        if double_sharing_chain:
            pos, *_, current_idx = double_shared_chain(double_sharing_chain, coords, current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(double_sharing_chain)
            continue

        # -------------------------------------------
        # Search for clusters of pyramids with one 
        # central pyramid sharing two base nodes with 
        # three different pyramids (all pairs taken)
        # -------------------------------------------
        double_sharing_cluster = find_double_shared_cluster(pyrs_to_check)
        if double_sharing_cluster:
            pos, *_, current_idx = double_shared_cluster(double_sharing_cluster, coords, current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(double_sharing_cluster)
            continue

        # ------------------------------------
        # Search for pyramids sharing two base 
        # nodes, while sharing a star node
        # ------------------------------------
        double_sharing_star = find_double_shared_star(pyrs_to_check)
        if double_sharing_star:
            pos, *_, current_idx = double_shared_star_node(double_sharing_star, coords, current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(double_sharing_star)
            continue

        # --------------------------------------------
        # Search for pyramids sharing two base nodes,
        # while sharing a star node and forming a loop
        # --------------------------------------------
        closed_double_sharing_star = find_double_shared_star_closed(pyrs_to_check)
        if closed_double_sharing_star:
            pos, *_, current_idx = double_shared_closed_star_node(closed_double_sharing_star, coords, current_idx)
            all_pos.append(pos)
            embedded_pyramids.extend(closed_double_sharing_star)
            continue

        loop_counter += 1
        if loop_counter > max_tries:
            all_embedded = False
            break

    return all_pos, current_idx, all_embedded

def check_no_extra_edges(old_comp, new_comp, shared_node, G):
    """
    Check that the new component attaches to the existing layout only via
    the shared node.

    More precisely, this verifies that no node in `new_comp` other than
    `shared_node` is adjacent (in `G`) to any node in `old_comp` other than
    `shared_node`.

    Parameters
    ----------
    old_comp : dict
        Positions of the original layout, mapping node -> (x, y). This
        dictionary must contain `shared_node`.
    new_comp : dict
        Positions of the new component being attached, mapping node -> (x, y).
        This dictionary must also contain `shared_node`.
    shared_node : hashable
        The unique node that belongs to both `old_comp` and `new_comp`
        and is allowed to connect the two components.
    G : networkx.Graph
        Graph built from the combined nodes of `old_comp` and `new_comp`.
        Its edges represent potential connections between all nodes.

    Returns
    -------
    tuple
        (ok, bad_edges)

        ok : bool
            True if no forbidden cross-edges exist; False otherwise.
        bad_edges : list of tuple
            List of forbidden edges (u, v) where:
              * u is a non-shared node in `new_comp`, and
              * v is a non-shared node in `old_comp`,
            or vice versa.

    Notes
    -----
    - The function does not modify `old_comp`, `new_comp`, or `G`.
    - Only edges that directly connect non-shared nodes across the two
      components are considered "forbidden". Edges involving `shared_node`
      are allowed.
    """

    final_nodes = set(old_comp.keys())
    comp_nodes  = set(new_comp.keys())

    if shared_node not in final_nodes or shared_node not in comp_nodes:
        raise ValueError("The shared node must be present in both the old and new component")
    
    # Nodes we care about for the "no cross edges" condition
    final_non_shared = final_nodes - {shared_node}
    comp_non_shared  = comp_nodes  - {shared_node}

    bad_edges = []

    for u, v in G.edges():
        # Forbidden if a non-shared component node connects to a
        # non-shared final_pos node (in either direction).
        if (u in comp_non_shared and v in final_non_shared) or \
           (v in comp_non_shared and u in final_non_shared):
            bad_edges.append((u, v))

    ok = (len(bad_edges) == 0)
    return ok, bad_edges

def find_shift_direction(new_apex, old_apex, coords):
    """
    Determine the cardinal shift direction from an old apex to a new apex.

    Given two nodes and their coordinates, this function returns the
    direction in which the new apex lies relative to the old one,
    restricted to horizontal or vertical displacements.

    Parameters
    ----------
    new_apex : hashable
        Identifier of the new apex node in `coords`.
    old_apex : hashable
        Identifier of the reference (old) apex node in `coords`.
    coords : dict
        Dictionary mapping node identifiers to 2D coordinates, e.g.
        {id: (x, y), ...}. Must contain entries for both `new_apex`
        and `old_apex`.

    Returns
    -------
    str or None
        One of {"right", "left", "up", "down"} indicating where
        `new_apex` lies with respect to `old_apex` along a purely
        horizontal or vertical line. Returns None if the orientation
        is neither horizontal nor vertical.

    Notes
    -----
    - The function relies on `orientation(new_apex, old_apex, coords)`
      to decide whether the points are horizontally or vertically aligned.
    - The input `coords` dictionary is not modified.
    """

    direction = orientation(new_apex, old_apex, coords)
    shift_dir = None
    if direction == "horizontal":
        shift_dir = "right" if coords[new_apex][0] > coords[old_apex][0] else "left"
    elif direction == "vertical":
        shift_dir = "up" if coords[new_apex][1] > coords[old_apex][1] else "down"

    return shift_dir

def attempt(final_pos, candidate_pos, shared_node, label = None, verbose = False):
    """
    Try to attach a candidate component to the current layout and check for
    forbidden edges.

    This function merges the positions in `candidate_pos` into `final_pos`,
    constructs the corresponding unit-disk graph, and uses
    `check_no_extra_edges` to verify that the new component connects to the
    existing layout only via `shared_node`.

    Parameters
    ----------
    final_pos : dict
        Current layout, mapping node identifiers to 2D coordinates,
        e.g. {id: (x, y), ...}. Must contain `shared_node`.
    candidate_pos : dict
        Positions of the component to be attached, also mapping
        node identifiers to 2D coordinates and containing `shared_node`.
    shared_node : hashable
        The unique node common to both `final_pos` and `candidate_pos`,
        which is allowed to connect the two components.
    label : str, optional
        Descriptive label used in the verbose printout (e.g. a strategy name
        or orientation tag). If None, no label is printed.
    verbose : bool, optional
        If True, prints whether the attempt succeeded or failed, prefixed
        by `label` when provided. Default is False.

    Returns
    -------
    tuple
        (ok, candidate_final)

        ok : bool
            True if attaching `candidate_pos` to `final_pos` introduces no
            forbidden cross-edges; False otherwise.
        candidate_final : dict
            Merged dictionary containing all positions from `final_pos`
            and `candidate_pos`. Returned regardless of the value of `ok`.

    Notes
    -----
    - The unit-disk graph is constructed via
      `unit_disk_edges(list(candidate_final.keys()), candidate_final)`,
      so edges are determined purely by geometric proximity.
    - The original dictionaries `final_pos` and `candidate_pos` are not
      modified; a shallow copy of `final_pos` is created and updated.
    - Forbidden edges are defined and detected by `check_no_extra_edges`,
      which ensures that only `shared_node` connects the two components.
    """

    candidate_final = final_pos.copy()
    candidate_final.update(candidate_pos)

    G_candidate = unit_disk_edges(list(candidate_final.keys()), candidate_final)
    ok, _ = check_no_extra_edges(final_pos, candidate_pos, shared_node, G_candidate)

    if verbose and label is not None:
        print(f"{label}: {'OK' if ok else 'failed'}")

    return ok, candidate_final

def check_local_edges(original_graph, final_pos):
    """
    Verify that all currently embeddable edges of the original graph are
    correctly realised in the 2D layout.

    An edge (u, v) of the original graph is considered realised in the
    layout if, in the geometric graph built from `final_pos` via
    `unit_disk_edges`, there exists either:
      - a direct edge (u, v), or
      - a path from u to v whose internal nodes are all wire nodes
        (labels starting with 'W') and whose number is even.

    Parameters
    ----------
    original_graph : networkx.Graph
        The original abstract graph (e.g. 3D connectivity) whose edges
        we want to realise in the 2D layout.
    final_pos : dict
        Current 2D layout, mapping node identifiers to coordinates,
        e.g. {id: (x, y), ...}. This may include additional wire nodes
        with labels starting with 'W' in addition to the original nodes
        of `original_graph`.

    Returns
    -------
    tuple
        (all_ok, missing_edges)

        all_ok : bool
            True if every edge (u, v) in `original_graph` whose endpoints
            are both present in `final_pos` is realised in the geometric
            graph as defined above. False if at least one such edge is not
            realised.
        missing_edges : list of tuple
            List of edges (u, v) from `original_graph` that are NOT realised
            in the current layout, according to the direct-or-even-wire-path
            criterion. Empty if everything is consistent.

    Notes
    -----
    - The geometric graph is constructed via
      `unit_disk_edges(list(final_pos.keys()), final_pos)`, so adjacency
      depends purely on geometric distances.
    - Only edges whose endpoints are already embedded (present in
      `final_pos`) are checked. Edges involving nodes not yet placed in
      the layout are ignored.
    - The helper `even_W_path_exists(G, u, v)` is assumed to:
        * accept direct edges (u, v) as valid realisations, and
        * accept paths between u and v with an even number of internal
          wire nodes (labels starting with 'W').

    """

    # Build the geometric graph from current positions
    G = unit_disk_edges(list(final_pos.keys()), final_pos)

    # We only care about original edges whose endpoints are already embedded
    embedded_nodes = set(original_graph.nodes()) & set(final_pos.keys())

    missing_edges = []

    for u, v in original_graph.edges():
        # Skip edges whose endpoints are not both present in final_pos
        if u not in embedded_nodes or v not in embedded_nodes:
            continue

        # even_W_path_exists must implement:
        #  - direct edge => ok
        #  - even number of 'W*' nodes in between => ok
        ok, _ = even_W_path_exists(G, u, v)
        if not ok:
            missing_edges.append((u, v))

    all_ok = (len(missing_edges) == 0)
    return all_ok, missing_edges

def relative_directions(node1, node2, final_pos, tol = 1e-9):
    """
    Compute the cardinal directions in which two nodes should move to get
    closer to each other.

    Given two nodes and their coordinates in `final_pos`, this function
    returns, for each node, the list of directions ("up", "down", "left",
    "right") that would move it closer to the other node along the x and/or
    y axes.

    Parameters
    ----------
    node1, node2 : hashable
        Node identifiers (keys in `final_pos`) whose relative directions
        are to be computed.
    final_pos : dict
        Mapping from node identifiers to 2D coordinates, e.g.
        {id: (x, y), ...}. Must contain entries for both `node1` and `node2`.
    tol : float, optional
        Numerical tolerance used to decide whether the x or y coordinates
        are effectively equal. If |dx| <= tol, no horizontal direction is
        suggested; similarly for |dy| <= tol. Default is 1e-9.

    Returns
    -------
    dict
        A dictionary of the form:
            {node1: [dir_1, dir_2, ...], node2: [dir_1, dir_2, ...]}
        where each list contains zero, one, or two directions chosen from
        {"up", "down", "left", "right"}:
          - If the nodes are horizontally separated, each node gets a
            "left"/"right" direction pointing towards the other.
          - If the nodes are vertically separated, each node gets an
            "up"/"down" direction pointing towards the other.
          - If they are coincident within tolerance in one axis, no direction
            is returned for that axis.

    Notes
    -----
    - Both nodes may receive up to two directions (one horizontal and one
      vertical) if they differ in both x and y.
    - If the two nodes are coincident in both coordinates within `tol`, both
      direction lists will be empty.
    """

    x1, y1 = final_pos[node1]
    x2, y2 = final_pos[node2]

    dx = x2 - x1
    dy = y2 - y1

    dirs1 = []
    dirs2 = []

    # Horizontal directions
    if abs(dx) > tol:
        if dx > 0:
            dirs1.append("right")
            dirs2.append("left")
        else:
            dirs1.append("left")
            dirs2.append("right")

    # Vertical directions
    if abs(dy) > tol:
        if dy > 0:
            dirs1.append("up")
            dirs2.append("down")
        else:
            dirs1.append("down")
            dirs2.append("up")

    return {node1: dirs1, node2: dirs2}

def try_shift_one_node(node, directions, final_pos, current_idx, unit_disk_edges):
    """
    Attempt to move a node along one of several directions, accepting only
    moves that create a valid local wiring pattern.

    For each direction in `directions`, this function tries to move `node`
    using `shift_helper`. The move is accepted only if, in the resulting
    geometric graph:

      1. At least one new wire node ('W*') is created by the shift.
      2. Among the newly created wire nodes, the one with the highest
         numeric suffix (e.g. 'W7' > 'W3') is the **only** neighbor of
         `node`.

    The first direction that yields a valid configuration is accepted, and
    the corresponding updated layout and wire index are returned.

    Parameters
    ----------
    node : hashable
        Identifier of the node to be moved. Must be present in `final_pos`.
    directions : list of str
        Candidate directions in which to move the node, e.g.
        ["right", "up"]. Directions are passed directly to `shift_helper`.
    final_pos : dict
        Current 2D layout, mapping node identifiers to coordinates, e.g.
        {id: (x, y), ...}. This dictionary is not modified in-place; a copy
        is used for each attempted move.
    current_idx : int
        Current wire index passed to `shift_helper` to generate new wire
        node labels (e.g. 'W0', 'W1', ...). Updated if a move is accepted.
    unit_disk_edges : callable
        Function with signature `unit_disk_edges(nodes, pos) -> networkx.Graph`,
        used to construct the geometric graph from node positions.

    Returns
    -------
    tuple
        (success, new_pos, new_idx)

        success : bool
            True if a valid move was found for at least one direction in
            `directions`. False if all directions failed or produced invalid
            wiring.
        new_pos : dict
            Updated layout including the moved node and any newly created
            wire nodes, if `success` is True. Otherwise, the original
            `final_pos` (unchanged).
        new_idx : int
            Updated wire index after a successful shift (as returned by
            `shift_helper`). If no valid move was found, this is the
            unchanged `current_idx`.

    Notes
    -----
    - `shift_helper(node, pos, direction, current_idx)` is expected to:
        * return an updated position dictionary,
        * possibly add new 'W*' wire nodes, and
        * return an updated wire index.
      It may raise `ValueError` for invalid moves; such directions are
      treated as failed attempts.
    - Newly created wire nodes are detected by comparing the node sets
      before and after the shift, and selecting those whose labels start
      with 'W' and end with an integer suffix.
    - The "last wire" condition enforces that the moved node has exactly
      one neighbor in the new geometric graph, and that this neighbor is
      the wire node with the largest numeric suffix among those created
      in this specific attempt.
    """

    import re

    def find_new_last_wire(before_keys, after_keys):
        """
        Identify the newly created wire node with the highest numeric index.

        Given the sets of node labels before and after an update, this function
        finds nodes that were added in the update and selects among them those
        that represent wire nodes with labels of the form "W<int>". It then
        returns the wire node with the largest integer suffix.

        Parameters
        ----------
        before_keys : iterable
            Collection of node labels present before the update
            (e.g. `final_pos.keys()`).
        after_keys : iterable
            Collection of node labels present after the update
            (e.g. `tmp_pos.keys()`).

        Returns
        -------
        str or None
            The label of the newly added wire node with the maximum numeric
            suffix, e.g. "W7". Returns None if no new wire nodes were created.

        Notes
        -----
        - A node is considered a wire node if:
            * it is a string, and
            * it starts with 'W', and
            * the remainder of the label is an integer (matched by "W(\\d+)$").
        - Only nodes that appear in `after_keys` but not in `before_keys`
        are considered "newly added".
        """

        new_nodes = set(after_keys) - set(before_keys)
        wires = []
        for n in new_nodes:
            if isinstance(n, str) and n.startswith("W"):
                # extract numeric part after 'W'
                m = re.match(r"W(\d+)$", n)
                if m:
                    wires.append((int(m.group(1)), n))
        if not wires:
            return None
        # max by numeric index
        wires.sort()
        return wires[-1][1]

    for direction in directions:
        # work on a copy to avoid polluting final_pos on failure
        tmp_pos = final_pos.copy()

        try:
            tmp_pos, _, new_idx = shift_helper(node, tmp_pos, direction, current_idx)
        except ValueError:
            # shift_helper can fail, treat as invalid move
            continue

        # Build graph from candidate layout
        G = unit_disk_edges(list(tmp_pos.keys()), tmp_pos)

        # Find the last wire created in THIS attempt
        last_wire = find_new_last_wire(final_pos.keys(), tmp_pos.keys())
        if last_wire is None:
            # No new wire created -> does not satisfy your "last wire" condition
            continue

        if node not in G:
            continue

        neighbors = list(G.neighbors(node))

        # Check: exactly one neighbor and it's last_wire
        if len(neighbors) == 1 and neighbors[0] == last_wire:
            # Accept this candidate
            return True, tmp_pos, new_idx

    # If all directions failed, keep original layout
    return False, final_pos, current_idx

def shift_pair_with_wire_check(node1, node2, dirs, final_pos, current_idx, unit_disk_edges):
    """
    Attempt to shift a pair of nodes, validating each move with the
    "last wire only" condition.

    This function takes two nodes and their suggested movement directions
    (typically obtained via `relative_directions`), and tries to move them
    one after the other using `try_shift_one_node`. Each attempted move is
    accepted only if it satisfies the local wiring constraint enforced by
    `try_shift_one_node` (i.e. the moved node is connected exclusively to
    the newest wire node created by that shift).

    Parameters
    ----------
    node1, node2 : hashable
        Identifiers of the two nodes to consider. They should be present
        in `final_pos`.
    dirs : dict
        Dictionary of the form `{node1: [...], node2: [...]}`, usually the
        output of `relative_directions(node1, node2, final_pos)`. Each value
        is a list of candidate directions (e.g. ["right", "up"]) in which
        the corresponding node may try to move.
    final_pos : dict
        Current 2D layout, mapping node identifiers to coordinates, e.g.
        {id: (x, y), ...}.
    current_idx : int
        Current wire index, passed through to `try_shift_one_node` (and
        ultimately to `shift_helper`) to generate new wire node labels.
    unit_disk_edges : callable
        Function with signature `unit_disk_edges(nodes, pos) -> networkx.Graph`,
        used by `try_shift_one_node` to construct the geometric graph
        after each candidate move.

    Returns
    -------
    tuple
        (new_pos, new_idx)

        new_pos : dict
            Layout after attempting to move both nodes. If a move fails to
            satisfy the wiring constraint, the corresponding node remains
            in its original position and the layout is unchanged for that
            node.
        new_idx : int
            Wire index after processing both nodes. It is incremented only
            when a move is successfully accepted (as propagated by
            `try_shift_one_node`).

    Notes
    -----
    - Node1 is processed first, then node2, always using the most recent
      layout and wire index produced so far.
    - If `dirs` does not contain an entry for a given node, or if the list
      of directions for that node is empty, no move is attempted for that
      node.
    - This function delegates all validity checks for individual moves to
      `try_shift_one_node`, which enforces the "last wire only" neighbor
      condition.
    """

    # Start from current layout
    pos = final_pos
    idx = current_idx

    # 1) Try shifting node1
    dirs1 = dirs.get(node1, [])
    if dirs1:
        success, pos, idx = try_shift_one_node(node1, dirs1, pos, idx, unit_disk_edges)
        # if not success, pos and idx stay as they were

    # 2) Try shifting node2, using the possibly updated layout/index
    dirs2 = dirs.get(node2, [])
    if dirs2:
        success, pos, idx = try_shift_one_node(node2, dirs2, pos, idx, unit_disk_edges)

    return pos, idx

def place_new_node_around(old_node, new_node, coords, pos, unit_disk_edges, angle_step_deg = 5):
    """
    Place `new_node` at distance 1 around `old_node` (or along an existing
    wire chain) so that it has exactly one geometric neighbor.

    The function first tries to place `new_node` at unit distance from
    `old_node`:
      - along a preferred cardinal direction inferred from 3D coordinates
        via `find_shift_direction`, then
      - along the remaining cardinal directions, and
      - along points sampled on the unit circle (every `angle_step_deg`).

    A placement is accepted if, in the graph built by `unit_disk_edges`,
    `new_node` has exactly one neighbor, which is the chosen anchor
    (`old_node` or a wire node).

    If no valid placement is found directly around `old_node`, the function
    searches existing wire chains: it looks for wire nodes 'W*' at even
    graph distance from `old_node` and degree 2, and repeats the same
    placement procedure around those anchors to guarantee an even number
    of wire nodes between `old_node` and `new_node`.

    Parameters
    ----------
    old_node : hashable
        Existing node, already present in `pos`.
    new_node : hashable
        Node to place; must not yet be in `pos`.
    coords : dict
        3D coordinates mapping node -> (x, y, z), used only by
        `find_shift_direction`.
    pos : dict
        Current 2D layout mapping node -> (x, y). Not modified in-place.
    unit_disk_edges : callable
        Function (nodes, pos_dict) -> networkx.Graph encoding geometric
        connectivity.
    angle_step_deg : float, optional
        Angular step (in degrees) for the circular search (default 5).

    Returns
    -------
    dict
        A new positions dictionary including `new_node` at a valid location.

    Raises
    ------
    ValueError
        If no position satisfying the single-neighbor and even-wire
        constraints can be found.
    """

    import math
    import collections

    if old_node not in pos:
        raise ValueError(f"old_node {old_node!r} is not present in pos")

    # ----------------
    # Helper functions
    # ----------------

    def candidate_ok(candidate_pos, expected_neighbor):
        """
        Test whether `new_node` is attached only to a specific neighbor
        in a candidate layout.

        The geometric graph is built from `candidate_pos` via
        `unit_disk_edges`. The placement is considered valid if:
          - both `new_node` and `expected_neighbor` are present in the graph, and
          - `new_node` has exactly one neighbor, which is `expected_neighbor`.

        Parameters
        ----------
        candidate_pos : dict
            Candidate 2D layout mapping node -> (x, y), including `new_node`.
        expected_neighbor : hashable
            Node that `new_node` is allowed to connect to.

        Returns
        -------
        bool
            True if the above conditions are satisfied, False otherwise.
        """

        G = unit_disk_edges(list(candidate_pos.keys()), candidate_pos)

        if new_node not in G or expected_neighbor not in G:
            return False

        neighbors = list(G.neighbors(new_node))
        return len(neighbors) == 1 and neighbors[0] == expected_neighbor

    def try_around_anchor(anchor_node, base_dirs):
        """
        Try to place `new_node` around a given anchor using cardinal directions
        and a circular scan.

        Starting from `anchor_node`'s position, this helper:
        1. Tests positions at unit distance along the directions in `base_dirs`
            (mapped via `dir_to_vec`).
        2. If none work, samples points on the unit circle (radius = 1) around
            the anchor every `angle_step_deg` degrees.

        A candidate is accepted as soon as `candidate_ok` reports that `new_node`
        has exactly one neighbor, namely `anchor_node`.

        Parameters
        ----------
        anchor_node : hashable
            Node around which `new_node` is to be placed.
        base_dirs : list of str
            Ordered list of cardinal directions (e.g. ["right", "left", ...])
            to test first.

        Returns
        -------
        dict or None
            A new positions dictionary including `new_node` at a valid location,
            or None if no valid placement is found around this anchor.
        """

        x_anchor, y_anchor = pos[anchor_node]

        # 1) Cardinal directions
        for d in base_dirs:
            dx, dy = dir_to_vec[d]
            x_new = x_anchor + dx
            y_new = y_anchor + dy

            candidate_pos = pos.copy()
            candidate_pos[new_node] = (x_new, y_new)

            if candidate_ok(candidate_pos, anchor_node):
                return candidate_pos  # success

        # 2) Circle search around the anchor
        radius = 1.0
        angle = 0.0
        while angle < 360.0 + 1e-9:
            theta = math.radians(angle)
            x_new = x_anchor + radius * math.cos(theta)
            y_new = y_anchor + radius * math.sin(theta)

            candidate_pos = pos.copy()
            candidate_pos[new_node] = (x_new, y_new)

            if candidate_ok(candidate_pos, anchor_node):
                return candidate_pos  # success

            angle += angle_step_deg

        # No valid placement around this anchor
        return None

    # Preferred cardinal directions
    pref_dir = find_shift_direction(new_node, old_node, coords)
    dir_to_vec = {"right": (1.0, 0.0), "left": (-1.0, 0.0), "up": (0.0, 1.0), "down": (0.0, -1.0)}

    # Provide a default order if pref_dir is not one of the known ones
    all_dirs = ["right", "left", "up", "down"]
    if pref_dir in all_dirs:
        base_dirs = [pref_dir] + [d for d in all_dirs if d != pref_dir]
    else:
        base_dirs = all_dirs

    # Try around old_node first (cardinals + circle)
    result = try_around_anchor(old_node, base_dirs)
    if result is not None:
        return result 

    # Fallback: attach to existing wire chain
    # Build the current graph once
    G = unit_disk_edges(list(pos.keys()), pos)

    if old_node not in G:
        raise ValueError(f"old_node {old_node!r} is not in the geometric graph")

    # BFS over W-nodes starting from old_node
    visited = {old_node}
    queue = collections.deque([(old_node, 0)])
    candidates = []

    while queue:
        node, dist = queue.popleft()

        for nbr in G.neighbors(node):
            if nbr in visited:
                continue
            visited.add(nbr)

            # Move along wire nodes only
            if isinstance(nbr, str) and nbr.startswith("W"):
                new_dist = dist + 1

                # Candidate wire: even distance (2nd, 4th, ...) and degree 2
                if new_dist % 2 == 0 and G.degree[nbr] == 2:
                    candidates.append((new_dist, nbr))

                queue.append((nbr, new_dist))

    # Sort candidates by increasing distance
    candidates.sort(key=lambda t: t[0])

    # For each candidate wire node, try to place new_node around it
    for _, wire_node in candidates:
        result = try_around_anchor(wire_node, base_dirs)
        if result is not None:
            return result

    # If everything fails, give up
    raise ValueError(
        f"Could not place node {new_node!r} around {old_node!r} "
        "with the required edge constraints, even via existing wire chains.")

def special_wire(start_node, end_node, pos, start_idx = 0):
    """
    Insert a two-segment "special" wire between two nodes at distance 2
    (horizontally or vertically) using equilateral-triangle geometry.

    For a pair of nodes whose coordinates differ by:
      - dx = 2, dy = 0  (horizontal), or
      - dx = 0, dy = 2  (vertical),
    the function proposes two possible detours made of two wire nodes "W*"
    placed at distance 1 from each other and from the endpoints, forming
    a bent path of unit-length edges.

    It then checks, for each side, whether the two candidate positions are
    free of existing nodes (within a radius 1). If exactly one side is free,
    or both are free (in which case the first is chosen), two new wire nodes
    W<idx+1>, W<idx+2> are added there. If no side is free, the layout is
    left unchanged.

    Parameters
    ----------
    start_node, end_node : hashable
        Endpoints of the segment to be wired, already present in `pos`.
    pos : dict
        Current 2D layout mapping node -> (x, y). May be updated with
        new "W*" nodes.
    start_idx : int, optional
        Starting index used to label new wire nodes as "W<start_idx+1>",
        "W<start_idx+2>", etc. Default is 0.

    Returns
    -------
    tuple
        (pos, new_idx) where:
          - pos : dict
              Updated layout (possibly unchanged if no valid wire can
              be placed).
          - new_idx : int
              Updated wire index (incremented by 2 if wires are added,
              otherwise equal to `start_idx`).
    """

    import math
    s = 0.5 * math.sqrt(3)

    x0, y0 = pos[start_node]
    x1, y1 = pos[end_node]

    dx = compute_abs_dx(start_node, end_node, pos)
    dy = compute_abs_dy(start_node, end_node, pos)

    def side_is_free(candidates, pos, start_node, end_node, radius = 1.0, tol = 1e-9):
        """
        Check whether a set of candidate positions is free from nearby nodes.

        A side is considered "free" if no existing node (other than
        `start_node` and `end_node`) lies within distance `radius` of
        any point in `candidates` (up to a tolerance `tol`).

        Parameters
        ----------
        candidates : list of tuple
            List of (x, y) coordinates to test.
        pos : dict
            Current layout mapping node -> (x, y).
        start_node, end_node : hashable
            Endpoints that are ignored in the collision check.
        radius : float, optional
            Exclusion radius around each candidate point (default 1.0).
        tol : float, optional
            Numerical tolerance for the radius comparison (default 1e-9).

        Returns
        -------
        bool
            True if no interfering node is found near any candidate point,
            False otherwise.
        """

        r2 = (radius - tol) ** 2
        for cx, cy in candidates:
            for name, (x, y) in pos.items():
                if name in (start_node, end_node):
                    continue
                d2 = (x - cx)**2 + (y - cy)**2
                if d2 < r2:
                    return False
        return True

    # Case 1: horizontal
    if dy == 0 and dx == 2:

        below_candidates = [(x0 + 0.5, y0 - s),(x0 + 1.5, y0 - s)]
        above_candidates = [(x0 + 0.5, y0 + s),(x0 + 1.5, y0 + s)]

        below_free = side_is_free(below_candidates, pos, start_node, end_node)
        above_free = side_is_free(above_candidates, pos, start_node, end_node)

        if below_free and not above_free:
            chosen = below_candidates
        elif above_free and not below_free:
            chosen = above_candidates
        elif below_free and above_free:
            chosen = below_candidates
        else:
            return pos, start_idx

        new_pos = {f"W{start_idx + 1}": chosen[0],f"W{start_idx + 2}": chosen[1]}
        pos = {**pos, **new_pos}
        start_idx += 2

    # Case 2: vertical
    elif dx == 0 and dy == 2:
        y_base = y0
        x_base = x0

        right_candidates = [(x_base + s, y_base + 0.5),(x_base + s, y_base + 1.5)]
        left_candidates = [(x_base - s, y_base + 0.5),(x_base - s, y_base + 1.5)]

        right_free = side_is_free(right_candidates, pos, start_node, end_node)
        left_free  = side_is_free(left_candidates,  pos, start_node, end_node)

        if right_free and not left_free:
            chosen = right_candidates
        elif left_free and not right_free:
            chosen = left_candidates
        elif right_free and left_free:
            chosen = right_candidates
        else:
            return pos, start_idx

        new_pos = {f"W{start_idx + 1}": chosen[0], f"W{start_idx + 2}": chosen[1]}
        pos = {**pos, **new_pos}
        start_idx += 2
        
    return pos, start_idx

def has_only_allowed_edge(G_copy, component_nodes, final_pos, allowed_pair):
    """
    Check that there is exactly one allowed edge from a component to the
    existing layout.

    Scans all edges (u, v) with u in `component_nodes` and v in `final_pos`
    and verifies that:
      - there is exactly one such edge, and
      - it matches `allowed_pair = (u_allowed, v_allowed)`.

    Parameters
    ----------
    G_copy : networkx.Graph
        Graph containing both component and final nodes.
    component_nodes : iterable
        Nodes belonging to the component being attached.
    final_pos : dict
        Existing layout mapping node -> (x, y); only its keys matter here.
    allowed_pair : tuple
        The only permitted cross-edge (u_allowed, v_allowed).

    Returns
    -------
    bool
        True if and only if the unique edge from component to final nodes
        is exactly `allowed_pair`.
    """

    u_allowed, v_allowed = allowed_pair
    found_allowed = False

    for u in component_nodes:
        for v in G_copy.neighbors(u):
            if v in final_pos:
                # Wrong edge: directly reject
                if not (u == u_allowed and v == v_allowed):
                    return False
                # Same allowed edge twice: also reject
                if found_allowed:
                    return False
                found_allowed = True

    # We want *exactly* one allowed edge
    return found_allowed

def attempt_adjacent(final_pos, candidate_pos, allowed_pair, label = None, verbose = False):
    """
    Try to merge a candidate component into the current layout, enforcing a
    single allowed cross-edge.

    The function builds the geometric graph from `final_pos ∪ candidate_pos`
    and accepts the merge only if the only edge between `candidate_pos`
    nodes and `final_pos` nodes is `allowed_pair`.

    Parameters
    ----------
    final_pos : dict
        Current layout mapping node -> (x, y).
    candidate_pos : dict
        Positions of the candidate component to attach.
    allowed_pair : tuple
        The unique permitted cross-edge (u_allowed, v_allowed).
    label : str, optional
        Label used in the verbose printout.
    verbose : bool, optional
        If True, print whether the attempt succeeded or failed.

    Returns
    -------
    tuple
        ok : bool
            True if the adjacency condition is satisfied, False otherwise.
        candidate_final : dict
            Merged positions dict (final_pos ∪ candidate_pos).
    """

    candidate_final = final_pos.copy()
    candidate_final.update(candidate_pos)

    G_candidate = unit_disk_edges(sorted(candidate_final.keys()), candidate_final)
    component_nodes = set(candidate_pos.keys())

    ok = has_only_allowed_edge(G_candidate, component_nodes, final_pos, allowed_pair)

    if verbose and label is not None:
        print(f"{label}: {'OK' if ok else 'failed'}")

    return ok, candidate_final

# -------------
# Main function
# -------------

def run_embedding_algorithm(num_nodes, seed = None, border = None, radius = 1.0, height = 1.0, plot_2d = False, plot_3d = False, time_check = False, verbose = False):
    """
    Run the full embedding pipeline for a random 3D UDG instance and
    construct a 2D layout that preserves its topology.

    The algorithm:
      1. Generates `num_nodes` base points via a Sobol sequence inside
         `border`, and builds the 2D unit disk graph (UDG).
      2. Builds the corresponding 3D UDG with pyramid apex nodes above
         triple overlaps of disks.
      3. Uses `search_and_remap` to embed all pyramids into 2D components.
      4. Iteratively merges components and non-pyramid nodes into a single
         layout, restoring missing edges with wire chains, handling:
           - components sharing a node,
           - components adjacent by a single edge,
           - direct neighbors of the current layout,
           - fully disconnected components and isolated nodes.
      5. Performs a final topological check to ensure that all original
         edges of the 3D graph are preserved in the 2D embedding.

    Parameters
    ----------
    num_nodes : int
        Number of base nodes to generate with the Sobol sequence.
    seed : int or None, optional
        Random seed passed to the Sobol generator (and point generation)
        for reproducibility. Default is None.
    border : shapely.geometry.Polygon or None, optional
        Polygon defining the region where Sobol points are drawn. If None,
        a default hexagon is used.
    radius : float, optional
        Disk radius used to build the 2D UDG and to detect overlaps
        (default 1.0).
    height : float, optional
        z-coordinate for apex nodes in the 3D UDG (default 1.0).
    plot_2d : bool, optional
        If True, plot the initial 2D UDG instance (default False).
    plot_3d : bool, optional
        If True, plot the 3D UDG with apex nodes (default False).
    time_check : bool, optional
        If True, print timing information of the whole embedding process (default False).
    verbose : bool, optional
        If True, print status messages during component attachment
        attempts (default False).

    Returns
    -------
    tuple
        (sobol_points, G_2d, G_3d, coords_3d, final_pos, all_nodes, apex_list, total_time)

        sobol_points : dict
            Mapping node -> (x, y) of the generated Sobol points.
        G_2d : networkx.Graph
            2D unit disk graph on the Sobol points.
        G_3d : networkx.Graph
            3D graph including base nodes and apex nodes.
        coords_3d : dict
            Mapping node -> (x, y, z) of all nodes in `G_3d`.
        final_pos : dict
            Mapping node -> (x, y) giving the final 2D embedding.
        all_nodes : list
            Sorted list of all nodes present in `final_pos`.
        apex_list : list
            List of apex-node identifiers created in the 3D construction.
        total_time : float
            Total time (in seconds) taken by the embedding process. 
            None if `time_check` is False.

    Raises
    ------
    ValueError
        If pyramids cannot be embedded, if components or edges cannot be
        attached/restored under the constraints, or if the final embedding
        fails the topological consistency check.
    """
        
    import math
    import time

    start_time = time.time()

    # -----------------------
    # Generating the instance
    # -----------------------
    sobol_points = generate_sobol_points(num_nodes, seed = seed, border = border)
    G_2d, edges_2d = build_udg(sobol_points, radius = radius)

    if plot_2d:
        plot_udg(G_2d, sobol_points, edges_2d, radius = radius, border = border)

    G_3d, coords_3d, _, apex_list = build_3d_udg(sobol_points, radius = radius, height = height)

    if plot_3d:
        plot_3d_udg(G_3d, coords_3d)

    # -----------------------------------
    # First, call the function to search 
    # for all the pyramids and embed them
    # -----------------------------------
    all_pos, current_idx, all_embedded = search_and_remap(G_3d, apex_list, coords_3d)

    if not all_embedded:   
        raise ValueError("Not all pyramids could be embedded.")

    # -------------------------------
    # Setup for final embedding phase
    # -------------------------------

    # We need to have all the pyramids in the graph
    pyramids = find_pyramids(G_3d, apex_list)
    non_pyr_nodes = [n for n in G_3d.nodes() if not any(n in pyr for pyr in pyramids)]

    # ---------------------------------------------------
    # Case 1: At least one pyramid was found and embedded
    # ---------------------------------------------------
    if all_pos:
        # Take as first component to embed the largest one
        first_component = max(all_pos, key = lambda p: len(p))
        final_pos = first_component
        final_nodes = sorted(set(final_pos.keys()))
        final_apex_list = [node for node in final_nodes if node.startswith("A")]

        # Lists to keep track of checked pyramids and components
        checked_pyramids = [pyramid_with(apex, pyramids) for apex in final_apex_list]
        pyramids_to_check = [p for p in pyramids if p not in checked_pyramids]

        checked_components = [first_component]
        components_to_check = [c for c in all_pos if c not in checked_components]
    
    # -----------------------------------------------
    # Case 2: No pyramids were found (empty all_pos)
    # We pick a seed node to start from, i.e. the one 
    # with lowest (y, x) coords, so the "corner node"
    # -----------------------------------------------
    else:
        if not G_3d.nodes:
            raise ValueError("Graph is empty; nothing to embed.")
        
        seed = min(G_3d.nodes(), key = lambda n: (coords_3d[n][1], coords_3d[n][0], n))

        x_seed, y_seed, _ = coords_3d[seed]
        final_pos = {seed: (x_seed, y_seed)}
        final_nodes = [seed]
        final_apex_list = [seed] if seed.startswith("A") else []

        # No pyramids embedded, so these are empty
        checked_pyramids = []
        pyramids_to_check = pyramids[:]

        # No pyramid-based components from all_pos
        checked_components = []
        components_to_check = []

        # We’ve already placed `seed`, so it is no longer a "non-pyr node to embed"
        if seed in non_pyr_nodes:
            non_pyr_nodes.remove(seed)
    
    loop_counter = 0
    while components_to_check or non_pyr_nodes:

        if loop_counter > 50:
            raise ValueError("Maximum number of iterations reached.")

        # ------------------------------------------------
        # At the beginning of each loop, check for missing 
        # edges and try to restore them if any is found
        # ------------------------------------------------
        ok, missing = check_local_edges(G_3d, final_pos)
        if missing:
            for m in missing:
                directions = relative_directions(m[0], m[1], final_pos)
                final_pos, current_idx = shift_pair_with_wire_check(m[0], m[1], directions, final_pos, current_idx, unit_disk_edges)
                final_copy = final_pos.copy()

                # -----------------------------------------------
                # Special case when two nodes are two units apart
                # -----------------------------------------------
                dx = compute_abs_dx(m[0], m[1], final_copy)
                dy = compute_abs_dy(m[0], m[1], final_copy)
                if dx == 2 and dy == 0:
                    final_copy, current_idx = special_wire(m[0], m[1], final_copy, current_idx)
                    continue
                elif dx == 0 and dy == 2:
                    final_copy, current_idx = special_wire(m[0], m[1], final_copy, current_idx)
                    continue

                _, new_wire_pos, current_idx = restore_edges(m[0], m[1], final_copy, current_idx)

                if new_wire_pos:
                    final_pos.update(new_wire_pos)
                    loop_counter += 1
                    continue
                else:
                    raise ValueError(f"❌ Could not restore edge between {m[0]} and {m[1]}")

        # -----------------------------------------------------
        # First, check if there is a new component that shares
        # one node with the current layout and try to attach it
        # -----------------------------------------------------
        sharing_component = next((c for c in components_to_check if any(label in list(final_pos.keys()) for label in c.keys())), None)
        if sharing_component:

            # ----------------------------------------
            # Find the shared node between the current 
            # component and the existing layout
            # ----------------------------------------
            shared_nodes = set(final_pos.keys()) & set(sharing_component.keys())
            if len(shared_nodes) != 1:
                raise ValueError(f"Expected exactly one shared node between the components, got {shared_nodes}")
            
            shared_node = next(iter(shared_nodes))

            # ------------------------------------------------
            # Find the pyramids containing the shared node and 
            # find their relative positions via their apexes
            # ------------------------------------------------
            new_pyr = pyramid_with(shared_node, pyramids_to_check)
            old_pyr = pyramid_with(shared_node, checked_pyramids)

            new_apex = apex_of(new_pyr, coords_3d)
            old_apex = apex_of(old_pyr, coords_3d)

            shift_dir = find_shift_direction(new_apex, old_apex, coords_3d)

            # --------------------------------------------
            # Move the common node in the existing layout 
            # outside the structure to make safe space for 
            # the new component attached to it
            # --------------------------------------------
            final_pos, _, current_idx = shift_helper(shared_node, final_pos, shift_dir, current_idx)
            pivot = final_pos[shared_node]

            # -----------------------------------------------
            # Translate the component's coordinates so that 
            # its shared_node matches final_pos, we work on
            # a copy since translate_helper modifies in-place
            # -----------------------------------------------
            target_x, target_y = final_pos[shared_node]
            base_new_pos = translate_helper(sharing_component.copy(), shared_node, target_x, target_y)

            # ---------------------------------------------------------------------------
            # We try different approaches to attach the component, while preserving the
            # original graph structure: first we try to directly attach it (so just a
            # rigid translation), then we try rotations (0°, 90°, 180°, 270°) around the
            # shared node, and for each rotation we also try reflections across vertical,
            # horizontal, and point axes. If any of these cases lead to a valid embedding
            # embedding (no edge crossings, all edges within unit distance), we accept 
            # that configuration. If none work, we report a failure for this component.
            # ---------------------------------------------------------------------------
            axes = ("vertical", "horizontal", "point")
            success = False

            for angle in (0, 90, 180, 270):
                if angle == 0:
                    rotated = base_new_pos
                    base_label = "Direct attachment"
                else:
                    rotated = rotate_helper(base_new_pos, angle, pivot = pivot)
                    base_label = f"Rotation {angle}°"

                ok, candidate_final = attempt(final_pos, rotated, shared_node, base_label, verbose)
                if ok:
                    final_pos = candidate_final
                    success = True
                    break

                for axis in axes:
                    candidate = reflect_positions(rotated, shared_node, axis = axis)
                    ok, candidate_final = attempt(final_pos, candidate, shared_node, f"{base_label} + reflection {axis}", verbose)
                    if ok:
                        final_pos = candidate_final
                        success = True
                        break

            if not success:
                raise ValueError(f"❌ No valid attachment found for component with shared node {shared_node}")

            if success:
                checked_components.append(sharing_component)
                components_to_check.remove(sharing_component)
                loop_counter += 1
                continue
            
        # -----------------------------------------------------
        # Second, check if there is a new component that has any 
        # node which is adjacent to any node already present in
        # the current layout and try to attach it
        # -----------------------------------------------------

        # ----------------------------------------------------------
        # Helper function to identify wire nodes and filter them out
        # in the adjacency check (because they are not present in G)
        # ----------------------------------------------------------
        def is_wire_node(node):
            return isinstance(node, str) and node.startswith("W")

        adjacent_component = next((c for c in components_to_check if any(set(G_3d.neighbors(label)) & set(final_pos.keys()) for label in c.keys() if not is_wire_node(label) and label in G_3d)), None)
        if adjacent_component:
            final_copy = final_pos.copy()

            # -------------------------------
            # Find the adjacent pair of nodes
            # -------------------------------
            adjacent_pair = None
            valid_final_nodes = {n for n in final_pos.keys() if not is_wire_node(n) and n in G_3d}

            for label in adjacent_component.keys():
                if is_wire_node(label) or label not in G_3d:
                    continue
                for neigh in G_3d.neighbors(label):
                    if neigh in valid_final_nodes:
                        adjacent_pair = (label, neigh)  # (node in new component, node in final_pos)
                        break
                if adjacent_pair is not None:
                    break

            # ------------------------------------------------------
            # Find the position in which the adjacent node is placed
            # ------------------------------------------------------
            direction = find_shift_direction(adjacent_pair[0], adjacent_pair[1], coords_3d) 
            v = + 1 if direction in ("right", "up") else - 1
            x_target = final_pos[adjacent_pair[1]][0] + v if direction in ("right", "left") else final_pos[adjacent_pair[1]][0]
            y_target = final_pos[adjacent_pair[1]][1] + v if direction in ("up", "down") else final_pos[adjacent_pair[1]][1]

            base_new_pos = translate_helper(adjacent_component.copy(), adjacent_pair[0], x_target, y_target)
            final_copy.update(base_new_pos)
            G_copy = unit_disk_edges(sorted(final_copy.keys()), final_copy)

            # -----------------------------------------------
            # Checking if the adjacent node's only edge with 
            # the current layout is the one with its neighbor
            # -----------------------------------------------
            component_nodes = set(adjacent_component.keys())
            allowed_pair = (adjacent_pair[0], adjacent_pair[1])
            placed = False

            # ---------------------------------------------------------------
            # First try to attach the new component next to the adjacent node
            # ---------------------------------------------------------------
            final_copy = final_pos.copy()
            base_new_pos = translate_helper(adjacent_component.copy(), adjacent_pair[0], x_target, y_target)
            final_copy.update(base_new_pos)

            for trial in range(2):
                # Check current candidate (final_copy) against the current layout base
                G_copy = unit_disk_edges(sorted(final_copy.keys()), final_copy)
                layout_nodes = set(final_pos.keys())  # Existing layout is still final_pos here
                only_edge = has_only_allowed_edge(G_copy, component_nodes, layout_nodes, allowed_pair)

                if only_edge:
                    final_pos = final_copy
                    checked_components.append(adjacent_component)
                    components_to_check.remove(adjacent_component)
                    loop_counter += 1
                    placed = True
                    break

                # ------------------------------
                # If first attempt failed, shift
                # and rebuild final_copy once
                # ------------------------------
                if trial == 0:
                    # Shift the neighbor + path in the candidate layout
                    final_copy, _, current_idx = shift_helper(adjacent_pair[1], final_copy, direction, current_idx)

                    # Recompute target for the component relative to the *shifted* neighbor
                    x_target = (final_copy[adjacent_pair[1]][0] + (v if direction in ("right", "left") else 0))
                    y_target = (final_copy[adjacent_pair[1]][1] + (v if direction in ("up", "down") else 0))

                    base_new_pos = translate_helper(adjacent_component.copy(), adjacent_pair[0], x_target, y_target)
                    final_copy.update(base_new_pos)

            if placed:
                continue

            # -------------------------------------------------
            # If direct attachment failed, try a sequence of 
            # rotations and reflections, as in the sharing case
            # -------------------------------------------------
            axes = ("vertical", "horizontal", "point")
            success = False
            pivot_node = adjacent_pair[0]
            pivot_tuple = final_copy[pivot_node]

            # Re-anchor the component for rot/reflect w.r.t. the ORIGINAL layout
            x_target = (final_pos[adjacent_pair[1]][0] + (v if direction in ("right", "left") else 0))
            y_target = (final_pos[adjacent_pair[1]][1] + (v if direction in ("up", "down") else 0))

            base_new_pos = translate_helper(adjacent_component.copy(), adjacent_pair[0], x_target, y_target)

            for angle in (0, 90, 180, 270):
                if angle == 0:
                    rotated = base_new_pos
                    base_label = "Direct attachment"
                else:
                    rotated = rotate_helper(base_new_pos, angle, pivot = pivot_tuple)
                    base_label = f"Rotation {angle}°"

                ok, candidate_final = attempt_adjacent(final_pos, rotated, allowed_pair, base_label, verbose)
                if ok:
                    final_pos = candidate_final
                    success = True
                    break

                for axis in axes:
                    candidate = reflect_positions(rotated, pivot_node, axis = axis)
                    ok, candidate_final = attempt_adjacent(final_pos, candidate, allowed_pair, f"{base_label} + reflection {axis}", verbose)
                    
                    if ok:
                        final_pos = candidate_final
                        success = True
                        break

                if success:
                    break

            if not success:
                raise ValueError(
                    f"❌ No valid attachment (rot/reflect) found for adjacent component "
                    f"with edge {allowed_pair}"
                )

            checked_components.append(adjacent_component)
            components_to_check.remove(adjacent_component)
            loop_counter += 1
            continue
        
        # ---------------------------------------------------------------------
        # If there are no components attachable to the current layout and no 
        # edges to restore, add direct neighbors of nodes in the current layout
        # ---------------------------------------------------------------------
        direct_nodes = [n for n in non_pyr_nodes if set(G_3d.neighbors(n)) & set(final_pos.keys())]
        couples = []

        if direct_nodes:
            for node in direct_nodes:
                original_neighbors = list(G_3d.neighbors(node))
                present_neighbor = next((nbr for nbr in original_neighbors if nbr in final_pos), None)

                if present_neighbor is None:
                    continue

                couples.append((node, present_neighbor))

            for new_node, old_node in couples:
                final_pos = place_new_node_around(old_node, new_node, coords_3d, final_pos, unit_disk_edges)
                non_pyr_nodes.remove(new_node)
            
            loop_counter += 1
            continue
        
        # --------------------------------------------
        # If all above fails, search for disconnected 
        # components and embed them separately
        # --------------------------------------------
        placed_nodes = set(G_3d.nodes()) & set(final_pos.keys())
        isolated_nodes = [n for n in G_3d.nodes() if n not in placed_nodes and not (set(G_3d.neighbors(n)) & placed_nodes)]
        if isolated_nodes:

            def euclidean(p, q):
                """p and q are (x, y) tuples"""
                return math.hypot(p[0] - q[0], p[1] - q[1])

            placed_nodes = set(G_3d.nodes()) & set(final_pos.keys())

            # ------------------------------------------------
            # Find the node in the new isolated component that 
            # is closest to any of the already placed nodes
            # ------------------------------------------------

            best_node = None          # Isolated node closest to placed_nodes
            best_neighbor = None      # Placed node it's closest to
            best_dist = float("inf")  # That minimal distance

            for n in isolated_nodes:
                if n not in coords_3d:
                    continue
                p = coords_3d[n]

                local_best_dist = float("inf")
                local_best_neighbor = None

                for m in placed_nodes:
                    if m not in coords_3d:
                        continue
                    d = euclidean(p, coords_3d[m])
                    if d < local_best_dist:
                        local_best_dist = d
                        local_best_neighbor = m

                if local_best_dist < best_dist:
                    best_dist = local_best_dist
                    best_node = n
                    best_neighbor = local_best_neighbor

            # ---------------------------------------------------------------
            # Find relative direction of the closest node of the disconnected 
            # component with respect to its closest placed node and place it
            # ---------------------------------------------------------------
            direction = find_shift_direction(best_node, best_neighbor, coords_3d)
            v = + 2 if direction in ("right", "up") else - 2
            x_target = final_pos[best_neighbor][0] + v if direction in ("right", "left") else final_pos[best_neighbor][0]
            y_target = final_pos[best_neighbor][1] + v if direction in ("up", "down") else final_pos[best_neighbor][1]
            
            # Find if best_node is contained in any component in components_to_check
            isolated_component = next((c for c in components_to_check if best_node in c.keys()), None)

            if not isolated_component:
                final_copy = final_pos.copy()
                final_copy[best_node] = (x_target, y_target)

                # ----------------------------------------
                # Check if the new position is valid, i.e. 
                # the new node should have no edges
                # ----------------------------------------
                G_copy = unit_disk_edges(sorted(final_copy.keys()), final_copy)
                neighbors = list(G_copy.neighbors(best_node))
                if neighbors:

                    max_tries = 20
                    tries = 0
                    
                    # ------------------------------------------------
                    # Keep shifting the new node in the same direction
                    # until it has no neighbors or max tries reached
                    # ------------------------------------------------
                    while neighbors and tries < max_tries:
                        if direction in ("right", "left"):
                            x_target += v
                        else:
                            y_target += v

                        final_copy[best_node] = (x_target, y_target)
                        G_copy = unit_disk_edges(sorted(final_copy.keys()), final_copy)
                        neighbors = list(G_copy.neighbors(best_node))
                        tries += 1
                    
                    if neighbors:
                        raise ValueError(f"❌ Could not place isolated node {best_node} without edges after {max_tries} attempts.")
                    
                    final_pos = final_copy
                    loop_counter += 1

                    if best_node in non_pyr_nodes:
                        non_pyr_nodes.remove(best_node)

                    continue
                
                else:
                    final_pos = final_copy
                    loop_counter += 1
                    non_pyr_nodes.remove(best_node)
                    continue
            
            # -------------------------------------
            # If the closest node is part of a new 
            # component, place the entire component
            # -------------------------------------
            else:
                max_tries = 10
                tries = 0
                has_edges = True

                # ----------------------------------------
                # Check if any node of the new component 
                # touches any node in the original layout
                # and keep shifting it until it doesn't
                # ----------------------------------------
                while has_edges and tries < max_tries:

                    final_copy = final_pos.copy()
                    base_new_pos = translate_helper(isolated_component.copy(), best_node, x_target, y_target)
                    final_copy.update(base_new_pos)
                    G_copy = unit_disk_edges(sorted(final_copy.keys()), final_copy)

                    has_edges = any(nbr in final_pos for label in isolated_component.keys() for nbr in G_copy.neighbors(label))
                    if has_edges:
                        if direction in ("right", "left"):
                            x_target += v
                        else:
                            y_target += v 
                        tries += 1

                if not has_edges:
                    final_pos = final_copy
                    loop_counter += 1
                    components_to_check.remove(isolated_component)
                    checked_components.append(isolated_component)
                    continue

                else:
                    raise ValueError(f"❌ Could not place component containing {best_node} without edges after {max_tries} attempts.")
        
        # -------------------------------------------
        # If no other action was performed, increment 
        # loop counter to avoid infinite loops
        # -------------------------------------------
        loop_counter += 1

    all_nodes = sorted(list(final_pos.keys()))

    # -------------------------------------------
    # Before the return, do a topological check 
    # to see if all original edges are preserved
    # -------------------------------------------
    G_final = unit_disk_edges(all_nodes, final_pos)
    ok, _ = topology_check(G_3d, G_final)

    end_time = time.time()

    if ok:
        if time_check:
            total_time = end_time - start_time
        else:
            total_time = None
        
        return sobol_points, G_2d, G_3d, coords_3d, final_pos, all_nodes, apex_list, total_time
    else:
        raise ValueError("❌ Final embedding does not preserve all original edges.")

