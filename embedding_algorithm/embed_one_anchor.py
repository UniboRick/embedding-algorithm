"""This module contains functions to embed pyramidal structures in 2D. Each structure is composed of multiple pyramids (at least two) sharing only one base node pairwise."""

from .helpers import *
from .embedding import restore_edges

# ----------------------------------
# TWO PYRAMIDS SHARING ONE BASE NODE
# ----------------------------------
def single_shared_double_bow_tie(coords, pyramids, start_idx = 0):
    """
    Embed two pyramids in 2D as a double-bow-tie structure when they share
    exactly one node. The function first embeds one pyramid as a bow-tie,
    then attaches the second pyramid to it using the appropriate geometric
    helper.

    Parameters
    ----------
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to identify apex nodes.
    pyrA : iterable
        Collection of node identifiers forming the first pyramid.
    pyrB : iterable
        Collection of node identifiers forming the second pyramid.
    start_idx : int, optional
        Starting index for generating auxiliary node labels
        (prefix "W"). Updated internally as helper functions add nodes.

    Returns
    -------
    tuple
        - pos : dict  
          Dictionary mapping all embedded nodes (original + auxiliary)
          to their 2D coordinates.
        - all_nodes : set  
          Set of all nodes involved in the final embedding.
        - apex_list : list  
          List containing the apex nodes of the two pyramids in left-to-right order.
        - current_idx : int  
          Updated index after all auxiliary nodes have been generated.

    Notes
    -----
    - The function assumes exactly one shared node between pyrA and pyrB.
    - The two pyramids are identified as left and right based on the
      x/y ordering of their apex nodes.
    - The left pyramid is embedded first using single_bow_tie.
    - The second pyramid is attached using the correct geometric helper
      selected by pick_bow_tie_helper, based on:
        - which tip of the bow-tie is shared,
        - the orientation (vertical/horizontal),
        - the relative positions of the apex nodes.
    - The output coordinate dictionary pos contains the entire embedded
      double bow-tie, including auxiliary "W" nodes produced by both steps.
    """

    # -----------------------------------------------
    # Step 1: Find the left-most pyramid and embed it
    # -----------------------------------------------

    apex_nodes = []
    for pyr in pyramids:
        apex_nodes.extend(n for n in pyr if coords[n][2] != 0)

    sorted_apex_nodes = sorted(apex_nodes, key = lambda n: (coords[n][0], coords[n][1], n))

    if len(sorted_apex_nodes) != len(pyramids):
        raise ValueError("The number of apex nodes does not match the number of pyramids")
    
    left_apex  = sorted_apex_nodes[0]
    right_apex = sorted_apex_nodes[1]

    left_pyramid  = pyramid_with(left_apex, pyramids)
    right_pyramid = pyramid_with(right_apex, pyramids)

    left_coords  = restrict_coords_to_pyramid(left_pyramid, coords)
    right_coords = restrict_coords_to_pyramid(right_pyramid, coords)

    apex_list = []
    apex_list.append(left_apex)
    apex_list.append(right_apex)

    pos, _, _, (rt, rb), (lt, lb), current_idx = single_bow_tie(left_pyramid, left_coords, start_idx)

    # ---------------------------------------------------------
    # Step 2: Call the right helper to embed the second pyramid
    # ---------------------------------------------------------
    shared = set(left_pyramid) & set(right_pyramid)
    if len(shared) != 1:
        raise ValueError("There should be exactly one shared node among the pyramids") 
    
    shared_node = next(iter(shared))
    helper_fn, _ = pick_bow_tie_helper(shared_node, rt, rb, lt, lb, left_apex, right_apex, coords)
    new_pos, _, _, _, current_idx = helper_fn(shared_node, pos[shared_node], right_pyramid, right_coords, current_idx)

    pos.update(new_pos)
    all_nodes = set(pos.keys())

    return pos, all_nodes, apex_list, current_idx

# -----------------------------------------
# THREE PYRAMIDS SHARING ONE SAME BASE NODE
# -----------------------------------------
def three_pyramids_one_shared(coords, pyramids, start_idx = 0):
    """
    Embed three pyramids in 2D that share exactly one common node, forming a
    triple bow-tie-like configuration: one left pyramid and two right pyramids
    (top-right and bottom-right).

    Parameters
    ----------
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to identify apex nodes.
    pyrA : iterable
        Collection of node identifiers forming the first pyramid.
    pyrB : iterable
        Collection of node identifiers forming the second pyramid.
    pyrC : iterable
        Collection of node identifiers forming the third pyramid.
    start_idx : int, optional
        Starting index for generating auxiliary node labels
        (prefix "W"). Updated internally as new auxiliary nodes are added.

    Returns
    -------
    tuple
        - pos : dict  
          Dictionary mapping all embedded nodes (original and auxiliary)
          to their 2D coordinates.
        - apex_list : list  
          List of the three apex nodes in the order:
          left apex, top-right apex, bottom-right apex.
        - all_nodes : set  
          Set of all node identifiers present in the final 2D embedding.
        - current_idx : int  
          Updated index after all auxiliary nodes have been generated.

    Notes
    -----
    - The three pyramids must share exactly one node in common; this
      shared node is used as the central junction of the construction.
    - Exactly three apex nodes (z ≠ 0) are expected in coords.
    - The apex nodes are ordered by x-coordinate to identify the left apex
      and the two right apexes, which are then ordered by y-coordinate into
      top-right and bottom-right.
    - The left pyramid is first embedded as a bow-tie using single_bow_tie.
    - The shared node is then forced to lie on one of the two right tips of
      the left bow-tie using the internal helper ensure_shared_on_right.
    - Four auxiliary nodes are created to route connections and then
      top_right_horizontal_helper and bottom_right_horizontal_helper
      are used to attach the top-right and bottom-right pyramids.
    """

    # ---------------
    # Helper function
    # ---------------
    def ensure_shared_on_right(pos, shared, apex, rt, rb):
        """
        Ensure that the shared node is one of the two right-hand tips
        (rt or rb) of the left bow-tie, swapping coordinates if needed.

        Parameters
        ----------
        pos : dict
            Dictionary mapping node identifiers to 2D coordinates for the
            already embedded left bow-tie.
        shared : hashable
            The shared node that must become one of the right-hand tips.
        apex : hashable
            Apex node of the left pyramid.
        rt : hashable
            Current right-top tip node.
        rb : hashable
            Current right-bottom tip node.

        Returns
        -------
        tuple
            Updated pair (rt, rb) after possibly swapping the shared node
            into one of these positions.

        Notes
        -----
        - If shared is already one of (rt, rb), nothing is changed.
        - If the apex coincides with one of the right tips, that tip is
          preserved as apex and the other is considered for swapping.
        - Otherwise, one of the two right tips is chosen at random for swapping
          with the shared node.
        """

        import random

        if shared in (rt, rb):
            return rt, rb 

        if apex == rt:
            target = rt
        elif apex == rb:
            target = rb
        else:
            target = random.choice((rt, rb))

        pos[shared], pos[target] = pos[target], pos[shared]

        if target == rt:
            rt = shared
        else:                         
            rb = shared

        return rt, rb

    # ---------------------------------------------
    # Step 1: Sort the pyramids by their apex nodes
    # ---------------------------------------------
    shared = set(pyramids[0]) & set(pyramids[1]) & set(pyramids[2])
    if len(shared) != 1:
        raise ValueError("There should be exactly one shared node among the pyramids")

    all_pyr_nodes = set().union(*pyramids)
    apex_nodes = {node for node, (_, _, z) in coords.items() if z != 0 and node in all_pyr_nodes}
    if len(apex_nodes) != 3:
        raise ValueError("Expected exactly three apex nodes")

    sorted_by_x = sorted(apex_nodes, key = lambda node: coords[node][0])
    left_apex = sorted_by_x[0]
    top_right_apex, bottom_right_apex = sorted(sorted_by_x[1:], key = lambda n: coords[n][1], reverse = True)

    left_pyramid         = pyramid_with(left_apex, pyramids)
    bottom_right_pyramid = pyramid_with(bottom_right_apex, pyramids)
    top_right_pyramid    = pyramid_with(top_right_apex, pyramids)

    # Remove the shared node 'E' from the pyramids
    shared_node = next(iter(shared))
    bottom_right_pyramid = [n for n in bottom_right_pyramid if n != shared_node]
    top_right_pyramid = [n for n in top_right_pyramid if n != shared_node]
    
    coords_l_pyr  = restrict_coords_to_pyramid(left_pyramid, coords)
    coords_br_pyr = restrict_coords_to_pyramid(bottom_right_pyramid, coords)
    coords_tr_pyr = restrict_coords_to_pyramid(top_right_pyramid, coords)

    pos_left, _, apex_left, (rt, rb), _, current_idx = single_bow_tie(left_pyramid, coords_l_pyr, start_idx)
    rt, rb = ensure_shared_on_right(pos_left, shared_node, apex_left, rt, rb)
    start_x, start_y = pos_left[shared_node]

    # -------------------------------
    # Step 2: Build the first bow-tie
    # -------------------------------
    pos = {} 
    apex_list = []
    apex_list.append(apex_left)

    aux_prefix       = "W"  
    num_aux_nodes    = 4
    aux_nodes_full   = [f"{aux_prefix}{i}" for i in range(current_idx + 1, current_idx + num_aux_nodes + 1)]
    aux_nodes_sorted = sorted(aux_nodes_full, key = lambda x: int(x[1:]))
    current_idx += num_aux_nodes

    # -------------------------------------------------------------
    # Step 3: Build the other bow-ties depending on the shared node
    # -------------------------------------------------------------
    if shared_node == rb:
        pos[aux_nodes_sorted[0]] = (start_x + 1, start_y)
        pos[aux_nodes_sorted[1]] = (start_x + 2, start_y)
        pos[aux_nodes_sorted[2]] = (start_x, start_y - 1)
        pos[aux_nodes_sorted[3]] = (start_x, start_y - 2)

        pos_tr, apex_tr, _, _, current_idx = top_right_horizontal_helper(aux_nodes_sorted[1], pos[aux_nodes_sorted[1]], top_right_pyramid, coords_tr_pyr, current_idx)
        pos_br, apex_br, _, _, current_idx = bottom_right_horizontal_helper(aux_nodes_sorted[3], pos[aux_nodes_sorted[3]], bottom_right_pyramid, coords_br_pyr, current_idx)

        pos = {**pos, **pos_tr, **pos_br, **pos_left}
        apex_list.append(apex_tr)
        apex_list.append(apex_br)

        all_nodes = set(pos.keys())

        return pos, apex_list, all_nodes, current_idx
    else:
        pos[aux_nodes_sorted[0]] = (start_x + 1, start_y)
        pos[aux_nodes_sorted[1]] = (start_x + 2, start_y)
        pos[aux_nodes_sorted[2]] = (start_x, start_y + 1)
        pos[aux_nodes_sorted[3]] = (start_x, start_y + 2)

        pos_tr, apex_tr, _, _, current_idx = top_right_horizontal_helper(aux_nodes_sorted[3], pos[aux_nodes_sorted[3]], top_right_pyramid, coords_tr_pyr, current_idx)
        pos_br, apex_br, _, _, current_idx = bottom_right_horizontal_helper(aux_nodes_sorted[1], pos[aux_nodes_sorted[1]], bottom_right_pyramid, coords_br_pyr, current_idx)

        pos = {**pos, **pos_tr, **pos_br, **pos_left}
        apex_list.append(apex_tr)
        apex_list.append(apex_br)

        all_nodes = set(pos.keys())

        return pos, apex_list, all_nodes, current_idx

# ------------------------------------------------
# THREE PYRAMIDS SHARING ONE SAME BASE NODE WITH A
# GIVEN START PYRAMID (USED IN SINGLE SHARED TREE)
# ------------------------------------------------

# ------------------
# Additional Helpers
# ------------------
def choose_helper_triple(direction, anchor, is_top):
    """
    Select the appropriate geometric helper for embedding a pyramid in a
    triple-pyramid configuration, based on its placement relative to the
    shared node and the 2x2 bow-tie grid.

    Parameters
    ----------
    direction : {"right", "left", "up", "down"}
        Direction in which the new pyramid extends from the shared node.
    anchor : {"below", "above", "left", "right"}
        Location of the shared node relative to the 2x2 grid used to attach
        the new bow-tie. This determines which local quadrant or edge aligns
        with the shared node before expansion.
    is_top : bool
        If True, the pyramid occupies the “upper” slot of the corresponding
        direction-based pair (e.g., top-right vs. bottom-right).  
        If False, it occupies the “lower” slot.

    Returns
    -------
    tuple
        A pair (helper_fn, next_direction) where:
        - helper_fn is the appropriate geometric helper function
          (e.g., top_right_vertical_helper).
        - next_direction is the canonical direction in which the helper
          will place the new bow-tie relative to the anchor point.

    Notes
    -----
    - This function is used only in the triple-pyramid embedding routine.
    - The mapping is exhaustive and deterministic: the triple key
      (direction, anchor, is_top) selects exactly one helper.
    - The returned next_direction is used to propagate embedding
      orientation into subsequent helper calls.
    """

    helpers = {
                # ───────── Right + Above ─────────
                ("right", "above", True ) : (top_right_vertical_helper,   "up"),
                ("right", "above", False) : (top_right_horizontal_helper, "right"),

                # ───────── Right + Below ─────────
                ("right", "below", True ): (bottom_right_horizontal_helper, "right"),
                ("right", "below", False): (bottom_right_vertical_helper,   "down"),

                # ───────── Left + Above ─────────
                ("left",  "above", True ) : (top_left_vertical_helper,   "up"),
                ("left",  "above", False) : (top_left_horizontal_helper, "left"),

                # ───────── Left + Below ─────────
                ("left",  "below", True ) : (bottom_left_horizontal_helper, "left"),
                ("left",  "below", False) : (bottom_left_vertical_helper,   "down"),

                # ───────── Up + Right ─────────
                ("up",    "right", True ) : (top_right_vertical_helper,   "up"),
                ("up",    "right", False) : (top_right_horizontal_helper, "right"),

                # ───────── Up + Left ─────────
                ("up",    "left",  True ) : (top_left_vertical_helper,   "up"),
                ("up",    "left",  False) : (top_left_horizontal_helper, "left"),

                # ───────── Down + Right ─────────
                ("down",  "right", True ) : (bottom_right_horizontal_helper, "right"),
                ("down",  "right", False) : (bottom_right_vertical_helper,   "down"),

                # ───────── Down + Left ─────────
                ("down",  "left",  True ) : (bottom_left_horizontal_helper, "left"),
                ("down",  "left",  False) : (bottom_left_vertical_helper,   "down"),

              }

    return helpers[(direction, anchor, is_top)]

def dispatch_pyramid_triple(is_top, aux_idx, aux_nodes, pos, pyr, coords_local, current_idx, direction, anchor):
    """
    Dispatch the correct geometric helper for embedding a pyramid in the
    triple-pyramid configuration, and execute it on the chosen auxiliary node.

    Parameters
    ----------
    is_top : bool
        Whether the pyramid occupies the “upper” slot relative to its
        direction/anchor pair (used by choose_helper_triple).
    aux_idx : int
        Index selecting which auxiliary node (from aux_nodes) is used as
        the anchor for the new bow-tie.
    aux_nodes : list of hashable
        List of auxiliary node identifiers created for the 2x2 bow-tie grid.
    pos : dict
        Dictionary mapping existing node identifiers to their 2D coordinates.
        This includes the left bow-tie and the central 2x2 structure.
    pyr : iterable
        Collection of node identifiers forming the pyramid to embed.
    coords_local : dict
        Mapping of nodes in pyr to their 3D coordinates. Used by the helper
        to determine apex ordering and top/bottom assignment.
    current_idx : int
        Current index for generating auxiliary “W” nodes.
    direction : {"right", "left", "up", "down"}
        Direction in which the triple-pyramid structure is growing.
    anchor : {"below", "above", "left", "right"}
        Location of the shared node relative to the intermediate 2x2 grid,
        used for selecting the correct geometric helper.

    Returns
    -------
    tuple
        The full output of the chosen helper, followed by:
        - edge_direction : str  
          The canonical direction in which the helper extended the bow-tie.

        Concretely, the return tuple is:

        (pos_new, apex, tip1, tip2, new_idx, edge_direction)
    
    Notes
    -----
    - This function is a thin wrapper: it delegates both selection and
      execution of the correct geometric helper.
    - choose_helper_triple determines which helper to use and the
      corresponding canonical direction.
    - The selected helper is then called with the appropriate anchor node and
      returns its standard bow-tie embedding result.
    - edge_direction is returned to inform the next embedding step.
    """

    helper, edge_direction = choose_helper_triple(direction, anchor, is_top)
    res = helper(aux_nodes[aux_idx], pos[aux_nodes[aux_idx]], pyr, coords_local, current_idx)
    return (*res, edge_direction)

def place_aux_triple(shared_xy, anchor, direction, aux_nodes, pos):
    """
    Position the four auxiliary nodes of the triple-pyramid configuration
    around the shared node, using a fixed set of direction-dependent offsets.

    Parameters
    ----------
    shared_xy : tuple of float
        The 2D coordinates (x, y) of the shared node to which the 2x2
        auxiliary grid will be anchored.
    anchor : {"below", "above", "left", "right"}
        Describes the location of the shared node relative to the internal
        2x2 bow-tie grid. Used to select the appropriate offset pattern.
    direction : {"right", "left", "up", "down"}
        The global direction of expansion for the triple-pyramid embedding.
        Combined with anchor to choose one of eight placement patterns.
    aux_nodes : list of hashable
        The four auxiliary node identifiers that form the 2x2 grid.
        Nodes are placed in the order given.
    pos : dict
        Dictionary mapping node identifiers to their 2D coordinates.
        Updated **in place** with the positions of the auxiliary nodes.

    Notes
    -----
    - Exactly four auxiliary nodes are expected.
    - The offsets define a rigid 2x2 “L-shaped” placement around the
      shared node, depending on the orientation of the triple-pyramid.
    - The pattern is deterministic: the key (anchor, direction) selects
      one of eight hard-coded offset lists.
    - This function only updates pos; it does not return anything.

    Example (conceptual)
    --------------------
    For anchor="below" and direction="right", the auxiliary nodes are
    placed as:
        (shared.x + 1, shared.y    )
        (shared.x + 2, shared.y    )
        (shared.x    , shared.y - 1)
        (shared.x    , shared.y - 2)
    """

    OFFSETS = {
                ("below", "right"): [(+ 1, 0), (+ 2, 0), (0, - 1), (0, - 2)],
                ("above", "right"): [(0, + 1), (0, + 2), (+ 1, 0), (+ 2, 0)],
                ("below", "left") : [(- 1, 0), (- 2, 0), (0, - 1), (0, - 2)],
                ("above", "left") : [(0, + 1), (0, + 2), (- 1, 0), (- 2, 0)],
                ("right","up")   :  [(0, + 1), (0, + 2), (+ 1, 0), (+ 2, 0)],
                ("left","up")    :  [(0, + 1), (0, + 2), (- 1, 0), (- 2, 0)],
                ("right","down") :  [(+ 1, 0), (+ 2, 0), (0, - 1), (0, - 2)],
                ("left","down")  :  [(- 1, 0), (- 2, 0), (0, - 1), (0, - 2)],
              }

    for node, (dx, dy) in zip(aux_nodes, OFFSETS[(anchor, direction)]):
        pos[node] = (shared_xy[0] + dx, shared_xy[1] + dy)

# -------------
# Main Function
# ------------- 
def three_star_node_tree(start_pyr, node1, node2, start_pos, pyr1, pyr2, coords, direction, start_idx = 0):
    """
    Embed a local configuration of three pyramids sharing a single node in 2D,
    arranged around an already embedded “starting” pyramid.

    Parameters
    ----------
    start_pyr : iterable
        Collection of node identifiers forming the already embedded starting
        pyramid.
    node1 : hashable
        First node of the edge in the starting pyramid that defines the
        attachment orientation.
    node2 : hashable
        Second node of the edge in the starting pyramid that defines the
        attachment orientation.
    start_pos : dict
        Dictionary mapping nodes of start_pyr to their existing 2D
        coordinates, e.g. {id: (x, y), ...}.
    pyr1 : iterable
        Collection of node identifiers forming the first new pyramid that
        will be attached.
    pyr2 : iterable
        Collection of node identifiers forming the second new pyramid that
        will be attached.
    coords : dict
        Dictionary mapping all node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. Used to identify apex nodes and relative
        positions (top/bottom, left/right).
    direction : {"right", "left", "up", "down"}
        Direction in which the structure is growing from the starting pyramid.
        Determines how the auxiliary 2x2 grid is attached around the shared node.
    start_idx : int, optional
        Initial index for generating auxiliary node labels (prefix "W").
        Updated internally as new auxiliary nodes are added.

    Returns
    -------
    tuple
        - pos : dict  
          Mapping from node identifiers (original and auxiliary) to their
          resulting 2D coordinates.
        - all_nodes : list  
          Sorted list of all node identifiers present in this local embedding.
        - apex_list : list  
          List of the three apex nodes: starting pyramid apex and the two
          apexes of the attached pyramids.
        - current_idx : int  
          Updated index after all auxiliary nodes have been generated.
        - directions : dict  
          Dictionary mapping each apex (except the starting one under key
          "Old") to the "edge direction" returned by the helper that embedded
          its pyramid. Includes the key "Old" for the original direction.

    Notes
    -----
    - The three pyramids must share **exactly one common node**, used as
      the central "star" node.
    - Exactly one apex is assumed per pyramid (z ≠ 0 in coords).
    - Four auxiliary nodes are created with labels "W{start_idx+1}" to
      "W{start_idx+4}" and placed around the shared node using
      place_aux_triple.
    - The relative vertical ordering of apex1 and apex2 determines
      which pyramid is treated as the "top" one.
    - dispatch_pyramid_triple is then used twice to embed the two new
      pyramids on the appropriate auxiliary nodes, updating pos and
      recording the resulting directions.
    """

    # Find the shared node among the pyramids
    shared = set(start_pyr) & set(pyr1) & set(pyr2)
    if len(shared) != 1:
        raise ValueError("There should be exactly one shared node among the pyramids")
    
    shared_node = next(iter(shared))

    Pyr1 = [n for n in pyr1 if n != shared_node]
    Pyr2 = [n for n in pyr2 if n != shared_node]

    coords1 = restrict_coords_to_pyramid(Pyr1, coords)
    coords2 = restrict_coords_to_pyramid(Pyr2, coords)

    # Find the apex nodes for each pyramid
    start_apex = next(n for n in start_pyr if coords[n][2] != 0)
    apex1      = next(n for n in Pyr1 if coords[n][2] != 0)
    apex2      = next(n for n in Pyr2 if coords[n][2] != 0)
    apex_list  = [start_apex, apex1, apex2]

    aux_prefix     = "W"  
    num_aux_nodes  = 4
    aux_nodes      = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    current_idx    = start_idx + num_aux_nodes

    directions = {"Old" : direction}
    pos = {}

    if direction in ("right", "left"):
        lower_node, _ = ((node1, node2) if start_pos[node1][1] < start_pos[node2][1] else (node2, node1))
        anchor = 'below' if shared_node == lower_node else 'above'
        place_aux_triple(start_pos[shared_node], anchor, direction, aux_nodes, pos)

        apex1_is_top = coords[apex1][1] > coords[apex2][1]
        pyr_for      = {apex1: (Pyr1, coords1), apex2: (Pyr2, coords2)}

        for apex, is_top in [(apex1, apex1_is_top), (apex2, not apex1_is_top)]:
            p, *_ , current_idx, new_direction = dispatch_pyramid_triple(is_top, 1 if is_top else 3, aux_nodes, pos, pyr_for[apex][0], pyr_for[apex][1], current_idx, direction, anchor)
            pos.update(p)
            directions[apex] = new_direction

    elif direction in ("up", "down"):
        left_node, _ = ((node1, node2) if start_pos[node1][0] < start_pos[node2][0] else (node2, node1))
        anchor = 'left' if shared_node == left_node else 'right'
        place_aux_triple(start_pos[shared_node], anchor, direction, aux_nodes, pos)

        apex1_is_top = coords[apex1][1] > coords[apex2][1]
        pyr_for      = {apex1: (Pyr1, coords1), apex2: (Pyr2, coords2)}

        for apex, is_top in [(apex1, apex1_is_top), (apex2, not apex1_is_top)]:
            p, *_ , current_idx, new_direction = dispatch_pyramid_triple(is_top, 1 if is_top else 3, aux_nodes, pos, pyr_for[apex][0], pyr_for[apex][1], current_idx, direction, anchor)
            pos.update(p)
            directions[apex] = new_direction

    all_nodes = sorted(set(pos.keys()))

    return pos, all_nodes, apex_list, current_idx, directions

# ----------------------------------------
# FOUR PYRAMIDS SHARING ONE SAME BASE NODE
# ----------------------------------------
def four_pyramids_one_shared(coords, pyramids, start_idx = 0):
    """
    Embed four pyramids in 2D that share a single common node, forming a
    symmetric “cross” configuration around the shared node.

    Parameters
    ----------
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to identify apex nodes.
    pyrA, pyrB, pyrC, pyrD : iterable
        Collections of node identifiers, each describing one pyramid in 3D.
        All four pyramids must share the same unique node.
    start_idx : int, optional
        Starting index for generating auxiliary node labels (prefix "W").
        Eight auxiliary nodes are created around the shared node.

    Returns
    -------
    tuple
        - pos : dict  
          Mapping of all nodes (shared node, pyramid nodes, and auxiliary
          nodes) to their 2D coordinates.
        - sorted_apex : list  
          List of the four apex nodes ordered by increasing x-coordinate.
        - all_nodes : list  
          Sorted list of all node identifiers present in the final embedding.
        - current_idx : int  
          Updated index after all auxiliary nodes have been generated.

    Notes
    -----
    - The function assumes exactly four pyramids and exactly one shared node
      common to all of them.
    - Exactly four apex nodes (z ≠ 0) are expected and are split into left
      and right pairs, then further into bottom/top to identify:
      bottom_left_apex, top_left_apex, bottom_right_apex,
      and top_right_apex.
    - The shared node is placed at the origin, and eight auxiliary nodes are
      positioned on a fixed 2x2 cross-shaped scaffold around it.
    - Each pyramid is then embedded by attaching it to one side of this
      scaffold using the appropriate horizontal helper:
      bottom_left_horizontal_helper, top_left_horizontal_helper,
      top_right_horizontal_helper, and bottom_right_horizontal_helper.
    - The returned layout represents a fully connected, planar embedding
      of the four pyramids around the shared node.
    """

    # ---------------------------------------------
    # Step 1: Sort the pyramids by their apex nodes
    # ---------------------------------------------
    shared = set(pyramids[0]) & set(pyramids[1]) & set(pyramids[2]) & set(pyramids[3])
    if len(shared) != 1:
        raise ValueError("There should be exactly one shared node among the pyramids")

    all_pyr_nodes = set().union(*pyramids)
    apex_nodes  = {node for node, (_, _, z) in coords.items() if z != 0 and node in all_pyr_nodes}
    sorted_apex = sorted(apex_nodes, key = lambda node: coords[node][0])
    left_apex   = sorted_apex[:2]
    right_apex  = sorted_apex[2:]

    left_apex_sorted  = sorted(left_apex, key = lambda node: coords[node][1])
    right_apex_sorted = sorted(right_apex, key = lambda node: coords[node][1])

    bottom_left_apex  = left_apex_sorted[0]
    top_left_apex     = left_apex_sorted[1]
    bottom_right_apex = right_apex_sorted[0]
    top_right_apex    = right_apex_sorted[1]

    bottom_left_pyramid  = pyramid_with(bottom_left_apex, pyramids)
    bottom_right_pyramid = pyramid_with(bottom_right_apex, pyramids)
    top_left_pyramid     = pyramid_with(top_left_apex, pyramids)
    top_right_pyramid    = pyramid_with(top_right_apex, pyramids)

    # Remove the shared node 'E' from the pyramids
    shared_node = next(iter(shared))
    bottom_left_pyramid = [n for n in bottom_left_pyramid if n != shared_node]
    bottom_right_pyramid = [n for n in bottom_right_pyramid if n != shared_node]
    top_left_pyramid = [n for n in top_left_pyramid if n != shared_node]
    top_right_pyramid = [n for n in top_right_pyramid if n != shared_node]

    coords_bl_pyr = restrict_coords_to_pyramid(bottom_left_pyramid, coords)
    coords_tl_pyr = restrict_coords_to_pyramid(top_left_pyramid, coords)
    coords_br_pyr = restrict_coords_to_pyramid(bottom_right_pyramid, coords)
    coords_tr_pyr = restrict_coords_to_pyramid(top_right_pyramid, coords)

    # -----------------------
    # Step 2: Build the graph
    # -----------------------
    pos = {}
    start_x = 0
    start_y = 0
    pos[shared_node] = (start_x, start_y)

    aux_prefix       = "W"  
    num_aux_nodes    = 8
    aux_nodes_full   = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    aux_nodes_sorted = sorted(aux_nodes_full, key = lambda x: int(x[1:]))
    current_idx      = start_idx + num_aux_nodes
    
    pos[aux_nodes_sorted[0]] = (start_x - 1, start_y)
    pos[aux_nodes_sorted[1]] = (start_x - 2, start_y)
    pos[aux_nodes_sorted[2]] = (start_x, start_y + 1)
    pos[aux_nodes_sorted[3]] = (start_x, start_y + 2)
    pos[aux_nodes_sorted[4]] = (start_x + 1, start_y)
    pos[aux_nodes_sorted[5]] = (start_x + 2, start_y)
    pos[aux_nodes_sorted[6]] = (start_x, start_y - 1)
    pos[aux_nodes_sorted[7]] = (start_x, start_y - 2)

    pos_bl, _, _, _, current_idx = bottom_left_horizontal_helper(aux_nodes_sorted[1], pos[aux_nodes_sorted[1]], bottom_left_pyramid, coords_bl_pyr, current_idx)
    pos_tl, _, _, _, current_idx = top_left_horizontal_helper(aux_nodes_sorted[3], pos[aux_nodes_sorted[3]], top_left_pyramid, coords_tl_pyr, current_idx)
    pos_tr, _, _, _, current_idx = top_right_horizontal_helper(aux_nodes_sorted[5], pos[aux_nodes_sorted[5]], top_right_pyramid, coords_tr_pyr, current_idx)
    pos_br, _, _, _, current_idx = bottom_right_horizontal_helper(aux_nodes_sorted[7], pos[aux_nodes_sorted[7]], bottom_right_pyramid, coords_br_pyr, current_idx)

    pos = {**pos, **pos_bl, **pos_tl, **pos_tr, **pos_br}

    all_nodes = sorted(set(pos.keys()))

    return pos, sorted_apex, all_nodes, current_idx

# ------------------------------------------------
# FOUR PYRAMIDS SHARING ONE SAME BASE NODE WITH A
# GIVEN START PYRAMID (USED IN SINGLE SHARED TREE)
# ------------------------------------------------

# ------------------
# Additional Helpers
# ------------------
def place_aux_quadruple(shared_xy, direction, aux_nodes, pos):
    """
    Position the six auxiliary nodes used in the four-pyramid (quadruple)
    configuration around the shared node, according to the global expansion
    direction.

    Parameters
    ----------
    shared_xy : tuple of float
        The 2D coordinates (x, y) of the shared central node to which the
        auxiliary scaffold will be anchored.
    direction : {"right", "left", "up", "down"}
        Direction in which the overall four-pyramid structure expands.
        Determines which of the four predefined offset patterns is used.
    aux_nodes : list of hashable
        List of six auxiliary node identifiers created for this configuration.
        They will be placed in the order they appear in this list.
    pos : dict
        Dictionary mapping node identifiers to their 2D coordinates.
        Updated in place with the coordinates of the six auxiliary nodes.

    Notes
    -----
    - This helper is used exclusively in the four-pyramids-one-shared
      embedding routine.
    - Exactly six auxiliary nodes are expected; offsets map each of them
      to a specific location relative to the shared node.
    - Offsets are designed so that:
        * one auxiliary pair sits along the expansion direction,
        * one pair sits above,
        * and one pair sits below (or left/right in the vertical case),
      forming a symmetric 3-arm scaffold around the shared node.
    - The function modifies pos directly and does not return a value.
    """

    OFFSETS = { 
                # aux_node[1] right, aux_node[3] top, aux_node[5] bottom
                ("right"): [(+ 1, 0), (+ 2, 0), (0, + 1), (0, + 2), (0, - 1), (0, - 2)],

                # aux_node[1] left, aux_node[3] top, aux_node[5] bottom
                ("left") : [(- 1, 0), (- 2, 0), (0, + 1), (0, + 2), (0, - 1), (0, - 2)],

                # aux_node[1] top, aux_node[3] right, aux_node[5] left
                ("up")   : [(0, + 1), (0, + 2), (+ 1, 0), (+ 2, 0), (- 1, 0), (- 2, 0)],

                # aux_node[1] bottom, aux_node[3] right, aux_node[5] left
                ("down") : [(0, - 1), (0, - 2), (+ 1, 0), (+ 2, 0), (- 1, 0), (- 2, 0)]
              }

    for node, (dx, dy) in zip(aux_nodes, OFFSETS[(direction)]):
        pos[node] = (shared_xy[0] + dx, shared_xy[1] + dy)

def choose_helper_quadruple(direction, anchor):
    """
    Select the appropriate embedding helper function for constructing one of the
    four pyramids in the quadruple-pyramid configuration (four pyramids sharing
    exactly one node).

    Parameters
    ----------
    direction : {"right", "left", "up", "down"}
        Global expansion direction of the quadruple structure. Determines the
        dominant axis along which the auxiliary scaffold was constructed.
    anchor : {"above", "below", "left", "right"}
        Spatial relation between the shared node and the pyramid currently being
        embedded, relative to the central coordinate frame of the quadruple
        layout.

        Examples:
        - "above" : pyramid sits above the shared node
        - "right" : pyramid sits to the right of the shared node, etc.

    Returns
    -------
    tuple
        A pair (helper_function, new_direction) where:
        - helper_function is one of the specialised bow-tie construction
          helpers (e.g., top_right_horizontal_helper).
        - new_direction is the direction that the pyramid extends after
          embedding (important for later propagation steps).

    Notes
    -----
    - This function is symmetric with respect to the four main cardinal
      directions and uses geometry-based rules to select the correct helper.
    - Used exclusively by the four-pyramids-one-shared embedding routine.
    - The returned new_direction is used by downstream embedding steps to
      correctly propagate bow-tie expansions.
    """

    helpers = {
                # ───────── Right ─────────
                ("right", "above") : (top_left_vertical_helper,     "up"),
                ("right", "below") : (bottom_right_vertical_helper, "down"),
                ("right", "right") : (top_right_horizontal_helper,  "right"),

                # ───────── Left ─────────
                ("left",  "above") : (top_right_vertical_helper,   "up"),
                ("left",  "below") : (bottom_left_vertical_helper, "down"),
                ("left",  "left" ) : (top_left_horizontal_helper,  "left"),

                # ───────── Up ─────────
                ("up",    "right") : (top_right_horizontal_helper,   "right"),
                ("up",    "left" ) : (bottom_left_horizontal_helper, "left"),
                ("up",    "above") : (top_left_vertical_helper,      "up"),

                # ───────── Down ─────────
                ("down",  "right") : (bottom_right_horizontal_helper, "right"),
                ("down",  "left" ) : (top_left_horizontal_helper,     "left"),
                ("down",  "below") : (bottom_left_vertical_helper,   "down"),

              }

    return helpers[(direction, anchor)]

def dispatch_pyramid_quadruple(start_node, pos, pyr, coords_local, current_idx, direction, anchor):
    """
    Dispatch a pyramid in the four-pyramids-one-shared configuration to the
    appropriate geometric helper, based on global direction and anchor location.

    Parameters
    ----------
    start_node : hashable
        The node in the current (shared) coordinate frame from which the new
        pyramid must be grown. Acts as the entry point for the helper.
    pos : dict
        Dictionary mapping node identifiers to their 2D positions. Updated by
        the helper function with the newly embedded nodes.
    pyr : iterable
        List of node identifiers forming the pyramid (excluding the shared node).
    coords_local : dict
        Dictionary of 3D coordinates restricted to this pyramid, used to
        identify apex and geometric ordering.
    current_idx : int
        Current index used for naming newly generated auxiliary nodes.
    direction : {"right", "left", "up", "down"}
        Global expansion direction of the quadruple structure.
    anchor : {"above", "below", "left", "right"}
        Relative placement of this pyramid around the shared node, determining
        how the bow-tie must be attached.

    Returns
    -------
    tuple
        The tuple returned by the selected bow-tie helper:
        (pos_new, apex, high_node, low_node, new_idx, edge_direction), where:
        - pos_new : dict of new 2D positions for this pyramid,
        - apex : the apex node of the embedded pyramid,
        - high_node / low_node : the two non-apex nodes ordered by
          height or left-right position (depending on helper),
        - new_idx : updated index after auxiliary nodes,
        - edge_direction : direction in which this pyramid extends after
          embedding.

    Notes
    -----
    - This is the high-level dispatcher used specifically in the
      four-pyramids-one-shared configuration.
    - Internally calls choose_helper_quadruple to select the correct helper
      based on direction and anchor.
    - The output edge_direction is required by downstream expansion logic.
    """

    helper, edge_direction = choose_helper_quadruple(direction, anchor)
    res = helper(start_node, pos[start_node], pyr, coords_local, current_idx)
    return (*res, edge_direction)

# -------------
# Main Function
# ------------- 
def four_star_node_tree(start_pyr, start_pos, pyr1, pyr2, pyr3, coords, direction, start_idx = 0):
    """
    Embed four pyramids in 2D that share a single common node, starting from an
    already embedded pyramid and extending the structure in a chosen direction.

    Parameters
    ----------
    start_pyr : iterable
        Collection of node identifiers forming the already embedded starting
        pyramid. One of its nodes must be the shared node.
    start_pos : dict
        Dictionary mapping nodes of start_pyr to their existing 2D
        coordinates, e.g. {id: (x, y), ...}.
    pyr1, pyr2, pyr3 : iterable
        Collections of node identifiers forming the three additional pyramids.
        Each must share exactly one node with start_pyr and with each other
        (the same shared node).
    coords : dict
        Dictionary mapping all node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to identify apex
        nodes and to determine relative geometric ordering.
    direction : {"right", "left", "up", "down"}
        Global direction in which the overall four-pyramid structure expands
        from the starting pyramid. Determines how the shared node is shifted,
        how auxiliary nodes are placed, and how the remaining pyramids are
        attached.
    start_idx : int, optional
        Initial index used for naming newly generated auxiliary nodes
        (prefix "W"). Updated internally as new nodes are added.

    Returns
    -------
    tuple
        - pos : dict  
          Mapping from node identifiers (shared node, auxiliary nodes,
          and nodes of the three attached pyramids) to their final 2D
          coordinates. Original nodes from start_pos other than the
          shared node are removed to avoid duplication.
        - all_nodes : list  
          Sorted list of all node identifiers present in this local embedding.
        - apex_list : list  
          List of the four apex nodes (one for each pyramid).
        - current_idx : int  
          Updated index after generating all auxiliary nodes.
        - directions : dict  
          Dictionary mapping each apex to the “edge direction” produced by the
          helper used to embed its pyramid, plus the key "Old" storing
          the original global "direction".

    Notes
    -----
    - All four pyramids must share **exactly one common node**, which is used
      as the central “star” node of the configuration.
    - The shared node is first shifted using shift_helper in the global
      direction, then six auxiliary nodes are placed around it with
      place_aux_quadruple.
    - The three non-starting pyramids are classified (e.g. top/right/bottom
      or top/left/right, depending on direction) using their apex positions.
    - Each of these pyramids is then embedded by calling
      dispatch_pyramid_quadruple, which selects the correct geometric
      helper and returns its local layout and effective edge direction.
    - The final pos dictionary contains only the shared node from
      start_pos plus all auxiliary and newly embedded nodes; other nodes
      from the original starting pyramid are discarded at the end.
    """

    # Find the shared node among the pyramids
    shared = set(start_pyr) & set(pyr1) & set(pyr2) & set(pyr3)
    if len(shared) != 1:
        raise ValueError("There should be exactly one shared node among the pyramids")
    
    # Shifting the shared node according to the direction
    shared_node = next(iter(shared))
    pos, _, current_idx = shift_helper(shared_node, start_pos, direction, start_idx)

    # Placing auxiliary nodes according to the direction
    aux_prefix     = "W"    
    num_aux_nodes  = 6
    aux_nodes      = [f"{aux_prefix}{i}" for i in range(current_idx + 1, current_idx + num_aux_nodes + 1)]
    current_idx    += num_aux_nodes
    place_aux_quadruple(pos[shared_node], direction, aux_nodes, pos)

    pyramids   = [start_pyr, pyr1, pyr2, pyr3]
    apex_list  = [n for pyr in pyramids for n in pyr if coords[n][2] != 0]
    start_apex = next(apex for apex in apex_list if apex in start_pyr)
    new_apex   = [apex for apex in apex_list if apex != start_apex]

    pyramids = [start_pyr, pyr1, pyr2, pyr3]
    directions = {"Old" : direction}

    if direction == "right":
        right_apex_list  = sorted(new_apex, key = lambda x: coords[x][0])
        top_apex_list    = sorted(new_apex, key = lambda x: coords[x][1], reverse = True)
        bottom_apex_list = sorted(new_apex, key = lambda x: coords[x][1])

        top_apex    = top_apex_list[0]
        right_apex  = right_apex_list[0]
        bottom_apex = bottom_apex_list[0]

        # Ensures to not select the same apex twice causing errors
        if right_apex in (top_apex, bottom_apex):
            right_apex = right_apex_list[1] if right_apex_list[1] not in (top_apex, bottom_apex) else right_apex_list[2]

        right_pyramid  = pyramid_with(right_apex, pyramids) 
        right_pyramid  = [n for n in right_pyramid if n != shared_node]

        top_pyramid    = pyramid_with(top_apex, pyramids)
        top_pyramid    = [n for n in top_pyramid if n != shared_node]

        bottom_pyramid = pyramid_with(bottom_apex, pyramids)
        bottom_pyramid = [n for n in bottom_pyramid if n != shared_node]

        right_coords  = restrict_coords_to_pyramid(right_pyramid, coords)
        top_coords    = restrict_coords_to_pyramid(top_pyramid, coords)
        bottom_coords = restrict_coords_to_pyramid(bottom_pyramid, coords)

        attributes = [
                        (right_apex,  right_pyramid,  right_coords,  aux_nodes[1], "right"),
                        (top_apex,    top_pyramid,    top_coords,    aux_nodes[3], "above"),
                        (bottom_apex, bottom_pyramid, bottom_coords, aux_nodes[5], "below"),
                     ]

        for apex, pyr, pyr_coords, start_node, anchor in attributes:
            p, *_ , current_idx, new_direction = dispatch_pyramid_quadruple(start_node, pos, pyr, pyr_coords, current_idx, direction, anchor)
            pos.update(p)
            directions[apex] = new_direction

    elif direction == "left":
        left_apex_list   = sorted(new_apex, key = lambda x: coords[x][0])
        top_apex_list    = sorted(new_apex, key = lambda x: coords[x][1], reverse = True)
        bottom_apex_list = sorted(new_apex, key = lambda x: coords[x][1])

        left_apex   = left_apex_list[0]
        top_apex    = top_apex_list[0]
        bottom_apex = bottom_apex_list[0]

        if left_apex in (top_apex, bottom_apex):
            left_apex = left_apex_list[1] if left_apex_list[1] not in (top_apex, bottom_apex) else left_apex_list[2]

        left_pyramid = pyramid_with(left_apex, pyramids) 
        left_pyramid = [n for n in left_pyramid if n != shared_node]

        top_pyramid = pyramid_with(top_apex, pyramids)
        top_pyramid = [n for n in top_pyramid if n != shared_node]

        bottom_pyramid = pyramid_with(bottom_apex, pyramids)
        bottom_pyramid = [n for n in bottom_pyramid if n != shared_node]

        left_coords   = restrict_coords_to_pyramid(left_pyramid, coords)
        top_coords    = restrict_coords_to_pyramid(top_pyramid, coords)
        bottom_coords = restrict_coords_to_pyramid(bottom_pyramid, coords)

        attributes = [
                        (left_apex,   left_pyramid,   left_coords,  aux_nodes[1],  "left"),
                        (top_apex,    top_pyramid,    top_coords,    aux_nodes[3], "above"),
                        (bottom_apex, bottom_pyramid, bottom_coords, aux_nodes[5], "below"),
                     ]

        for apex, pyr, pyr_coords, start_node, anchor in attributes:
            p, *_ , current_idx, new_direction = dispatch_pyramid_quadruple(start_node, pos, pyr, pyr_coords, current_idx, direction, anchor)
            pos.update(p)
            directions[apex] = new_direction

    elif direction == "up":
        top_apex_list    = sorted(new_apex, key = lambda x: coords[x][1])
        right_apex_list  = sorted(new_apex, key = lambda x: coords[x][0], reverse = True)
        left_apex_list   = sorted(new_apex, key = lambda x: coords[x][0])

        right_apex  = right_apex_list[0]
        left_apex   = left_apex_list[0]
        top_apex    = top_apex_list[0]

        if top_apex in (right_apex, left_apex):
            top_apex = top_apex_list[1] if top_apex_list[1] not in (right_apex, left_apex) else top_apex_list[2]

        top_pyramid = pyramid_with(top_apex, pyramids)
        top_pyramid = [n for n in top_pyramid if n != shared_node]

        right_pyramid  = pyramid_with(right_apex, pyramids) 
        right_pyramid  = [n for n in right_pyramid if n != shared_node]

        left_pyramid = pyramid_with(left_apex, pyramids) 
        left_pyramid = [n for n in left_pyramid if n != shared_node]

        top_coords    = restrict_coords_to_pyramid(top_pyramid, coords)
        right_coords  = restrict_coords_to_pyramid(right_pyramid, coords)
        left_coords   = restrict_coords_to_pyramid(left_pyramid, coords)

        attributes = [
                        (top_apex,    top_pyramid,    top_coords,    aux_nodes[1], "above"),
                        (right_apex,  right_pyramid,  right_coords,  aux_nodes[3], "right"),
                        (left_apex,   left_pyramid,   left_coords,   aux_nodes[5], "left"),
                     ]

        for apex, pyr, pyr_coords, start_node, anchor in attributes:
            p, *_ , current_idx, new_direction = dispatch_pyramid_quadruple(start_node, pos, pyr, pyr_coords, current_idx, direction, anchor)
            pos.update(p)
            directions[apex] = new_direction

    elif direction == "down":
        bottom_apex_list = sorted(new_apex, key = lambda x: coords[x][1])
        right_apex_list  = sorted(new_apex, key = lambda x: coords[x][0], reverse = True)
        left_apex_list   = sorted(new_apex, key = lambda x: coords[x][0])

        right_apex  = right_apex_list[0]
        left_apex   = left_apex_list[0]
        bottom_apex = bottom_apex_list[0]

        if bottom_apex in (right_apex, left_apex):
            bottom_apex = bottom_apex_list[1] if bottom_apex_list[1] not in (right_apex, left_apex) else bottom_apex_list[2]

        bottom_pyramid = pyramid_with(bottom_apex, pyramids)
        bottom_pyramid = [n for n in bottom_pyramid if n != shared_node]

        right_pyramid  = pyramid_with(right_apex, pyramids) 
        right_pyramid  = [n for n in right_pyramid if n != shared_node]

        left_pyramid = pyramid_with(left_apex, pyramids) 
        left_pyramid = [n for n in left_pyramid if n != shared_node]

        bottom_coords = restrict_coords_to_pyramid(bottom_pyramid, coords)
        right_coords  = restrict_coords_to_pyramid(right_pyramid, coords)
        left_coords   = restrict_coords_to_pyramid(left_pyramid, coords)

        attributes = [
                        (bottom_apex, bottom_pyramid, bottom_coords, aux_nodes[1], "below"),
                        (right_apex,  right_pyramid,  right_coords,  aux_nodes[3], "right"),
                        (left_apex,   left_pyramid,   left_coords,   aux_nodes[5], "left"),
                     ]

        for apex, pyr, pyr_coords, start_node, anchor in attributes:
            p, *_ , current_idx, new_direction = dispatch_pyramid_quadruple(start_node, pos, pyr, pyr_coords, current_idx, direction, anchor)
            pos.update(p)
            directions[apex] = new_direction

    pos = {k: v for k, v in pos.items() if k not in start_pos or k == shared_node}
    all_nodes = sorted(set(pos.keys()))
    
    return pos, all_nodes, apex_list, current_idx, directions

# --------------------------------------------------
# FIVE (OR MORE) PYRAMIDS SHARING ONE SAME BASE NODE
# --------------------------------------------------

# ---------------------------
# Additional Helper Functions
# ---------------------------
def extend_structure_short(start_node, start_coords, start_idx = 0):
    """
    Create a short “arm” of auxiliary nodes extending to the right of a
    starting node, forming a small 3-branch local scaffold.

    Parameters
    ----------
    start_node : hashable
        Identifier of the node from which the extension starts.
    start_coords : dict
        Dictionary mapping node identifiers to their existing 2D coordinates,
        e.g. {id: (x, y), ...}. Must contain start_node.
    start_idx : int, optional
        Starting index for generating auxiliary node labels (prefix "W").
        Six auxiliary nodes are created.

    Returns
    -------
    tuple
        - all_nodes : set  
          Set of all node identifiers present after the extension
          (original + auxiliary).
        - pos : dict  
          Dictionary mapping all nodes to their 2D coordinates, including the
          six new auxiliary nodes.
        - top_node : hashable  
          Auxiliary node at the upper end of the vertical branch
          (furthest above the start node).
        - bottom_node : hashable  
          Auxiliary node at the lower end of the vertical branch
          (furthest below the start node).
        - central_node : hashable  
          Auxiliary node lying on the main horizontal arm, used as a central
          attachment point for further extensions.
        - current_idx : int  
          Updated index after consuming six auxiliary node labels.

    Notes
    -----
    - The start node is kept at its original position.
    - Two auxiliary nodes form a short horizontal arm to the right of the
      start node.
    - Four auxiliary nodes form two vertical branches (up and down) attached
      to the far end of the horizontal arm.
    - This helper is intended as a compact, reusable scaffold for building
      more complex local structures.
    """

    start_x = start_coords[start_node][0]
    start_y = start_coords[start_node][1]

    pos = {**start_coords}
    all_nodes = [start_node]

    aux_prefix       = "W"  
    num_aux_nodes    = 6
    aux_nodes_full   = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    aux_nodes_sorted = sorted(aux_nodes_full, key = lambda x: int(x[1:]))

    pos[aux_nodes_sorted[0]] = (start_x + 1, start_y)
    pos[aux_nodes_sorted[1]] = (start_x + 2, start_y)
    pos[aux_nodes_sorted[2]] = (start_x + 2, start_y + 1)
    pos[aux_nodes_sorted[3]] = (start_x + 2, start_y + 2)
    pos[aux_nodes_sorted[4]] = (start_x + 2, start_y - 1)
    pos[aux_nodes_sorted[5]] = (start_x + 2, start_y - 2)

    all_nodes = set(pos.keys())

    central_node = aux_nodes_sorted[1]
    top_node = aux_nodes_sorted[3]
    bottom_node = aux_nodes_sorted[5]
    current_idx = start_idx + num_aux_nodes

    return all_nodes, pos, top_node, bottom_node, central_node, current_idx

def extend_structure_long(start_node, start_coords, start_idx = 0):
    """
    Create a long horizontal extension from a starting node, with a vertical
    3-branch scaffold at the far end. This function is similar to
    extend_structure_short but produces a longer arm (four units wide)
    before branching.

    Parameters
    ----------
    start_node : hashable
        Identifier of the node from which the structure extends.
    start_coords : dict
        Mapping from node identifiers to their 2D coordinates, e.g.
        {id: (x, y), ...}. Must contain start_node.
    start_idx : int, optional
        Starting index for generating auxiliary node labels (prefix "W").
        Eight auxiliary nodes are created.

    Returns
    -------
    tuple
        - all_nodes : set  
          Set of all nodes in the resulting structure (original + auxiliary).
        - pos : dict  
          Coordinates of all nodes after adding the long extension.
        - top_node : hashable  
          Topmost node of the vertical branch at the end of the extension.
        - bottom_node : hashable  
          Bottommost node of that vertical branch.
        - central_node : hashable  
          Node at the end of the horizontal arm (the branch junction).
        - current_idx : int  
          Updated index after allocating eight auxiliary node labels.

    Notes
    -----
    - Produces a 4-unit-long horizontal arm extending rightward from
      start_node.
    - At the far end of this arm, a symmetric vertical branch of two nodes
      above and two below is created.
    - Useful as a scaffold for embedding multi-pyramid or multi-gadget
      structures in a clean and collision-free region of the plane.
    """

    start_x = start_coords[start_node][0]
    start_y = start_coords[start_node][1]

    pos = {**start_coords}
    all_nodes = [start_node]

    aux_prefix       = "W"  
    num_aux_nodes    = 8
    aux_nodes_full   = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    aux_nodes_sorted = sorted(aux_nodes_full, key = lambda x: int(x[1:]))

    pos[aux_nodes_sorted[0]] = (start_x + 1, start_y)
    pos[aux_nodes_sorted[1]] = (start_x + 2, start_y)
    pos[aux_nodes_sorted[2]] = (start_x + 3, start_y)
    pos[aux_nodes_sorted[3]] = (start_x + 4, start_y)
    pos[aux_nodes_sorted[4]] = (start_x + 4, start_y + 1)
    pos[aux_nodes_sorted[5]] = (start_x + 4, start_y + 2)
    pos[aux_nodes_sorted[6]] = (start_x + 4, start_y - 1)
    pos[aux_nodes_sorted[7]] = (start_x + 4, start_y - 2)

    all_nodes = set(pos.keys())

    central_node = aux_nodes_sorted[3]
    top_node = aux_nodes_sorted[5]
    bottom_node = aux_nodes_sorted[7]
    current_idx = start_idx + num_aux_nodes

    return all_nodes, pos, top_node, bottom_node, central_node, current_idx
    
# -------------
# Main Function
# -------------
def N_pyramids_one_shared(coords, pyramids, start_idx = 0):
    """
    Embed an arbitrary number of pyramids in 2D that share a single common node,
    arranging them along a chain of bow-ties and auxiliary extensions.

    Parameters
    ----------
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to identify
        apex nodes and to determine their left-right ordering.
    pyramids : list of iterable
        List of pyramids, each given as a collection of node identifiers.
        All pyramids must share exactly one common node (the shared node).
    start_idx : int, optional
        Starting index used for naming auxiliary nodes (prefix "W").
        Updated internally as new auxiliary nodes are created.

    Returns
    -------
    tuple
        - pos : dict  
          Mapping from node identifiers (original + auxiliary) to 2D
          coordinates representing the full embedding.
        - all_nodes : list  
          List of all node identifiers present in the final embedding.
        - apex_list : list  
          List of apex nodes (one per pyramid) in the order they were
          embedded: left-most first, then all others as attached.
        - current_idx : int  
          Updated index after generating all auxiliary nodes.

    Notes
    -----
    - Step 1: The apex of each pyramid (z ≠ 0 in coords) is extracted
      and all apex nodes are sorted by their (x, y) coordinates. The
      left-most apex identifies the initial “left” pyramid.
    - Step 2: The left-most pyramid is embedded first as a bow-tie using
      single_bow_tie, producing an initial 2D layout and identifying
      its right-top (rt) and right-bottom (rb) tips.
    - Step 3: The unique shared node among all pyramids is then enforced
      to coincide with either rt or rb. If necessary, labels are
      swapped using swap_helper so that the shared node sits on one
      of these right-hand tips.
    - Step 4: A sequence of short and long horizontal “extension arms”
      is created to the right of the shared node using
      extend_structure_short and extend_structure_long. Their
      pattern depends on the total number of additional pyramids and
      provides attachment anchors (top/bottom nodes) for new bow-ties.
      If the number of remaining pyramids is odd, two extra auxiliary
      nodes are removed to keep the scaffold consistent.
    - Step 5: Each remaining pyramid is attached in turn to an appropriate
      top or bottom anchor node via the vertical helpers:
      top_left_vertical_helper, top_right_vertical_helper,
      bottom_left_vertical_helper, or bottom_right_vertical_helper.
      The choice alternates between left/right and top/bottom so that
      the structure grows in a balanced, zig-zag manner.
    - The final embedding preserves the shared node as a common junction
      while placing all pyramids along a clean, non-overlapping 2D layout.
    """

    # ---------------------------------------------
    # Step 1: Sort the pyramids by their apex nodes
    # ---------------------------------------------
    apex_nodes = []
    for pyr in pyramids:
        apex_nodes.extend(n for n in pyr if coords[n][2] != 0)

    sorted_apex_nodes = sorted(apex_nodes, key = lambda n: (coords[n][0], coords[n][1], n))

    if len(sorted_apex_nodes) != len(pyramids):
        raise ValueError("The number of apex nodes does not match the number of pyramids")
    
    left_apex    = sorted_apex_nodes[0]
    left_pyramid = pyramid_with(left_apex, pyramids)
    left_coords  = restrict_coords_to_pyramid(left_pyramid, coords)

    # ----------------------------------------------------------
    # Step 2: Build the first bow-tie from the left-most pyramid
    # ----------------------------------------------------------
    pos = {}
    apex_list = []
    pos, nodes, left_apex, (rt, rb), _, current_idx = single_bow_tie(left_pyramid, left_coords, start_idx)
    all_nodes = set(nodes)
    apex_list.append(left_apex)

    # -------------------------------------------------
    # Step 3: Ensure the shared node is either rb or rt
    # -------------------------------------------------
    shared = set(pyramids[0]).intersection(*pyramids[1:])
    if len(shared) != 1:
        raise ValueError(f"Expected the pyramids to share exactly one node")
    
    shared_node = next(iter(shared))

    if shared_node not in (rt, rb):
        if rt == left_apex:
            target = rt
        elif rb == left_apex:
            target = rb
        else: 
            y_shared = pos[shared_node][1]
            target   = min((rt, rb), key = lambda n: abs(pos[n][1] - y_shared))

        pos, _ = swap_helper(pos, {shared_node: target})

        # Update the label of the shared node
        if target == rt:
            rt = shared_node
        else:
            rb = shared_node

    # -----------------------------------------
    # Step 4: Build the extensions to the right
    # -----------------------------------------
    other_pyramids = [p for p in pyramids if p != left_pyramid]
    number_extensions = (len(other_pyramids) + 1) // 2

    # Creating the sequence of extensions based on the number of pyramids
    sequence = []
    for i in range(number_extensions):
        if i < 2:
            sequence.append(1)
        else:
            sequence.append(2 if i % 2 == 0 else 1)

    top_nodes = []
    bottom_nodes = []
    cursor_node = shared_node

    # Calling the extension helpers based on the sequence
    for code in sequence:
        if code == 1:              
            all_n, pos_n, top, bot, cent, current_idx = extend_structure_short(cursor_node, pos, current_idx)
        else:                     
            all_n, pos_n, top, bot, cent, current_idx = extend_structure_long(cursor_node, pos, current_idx)

        pos = {**pos, **pos_n}
        all_nodes.update(all_n)
        top_nodes.append(top)
        bottom_nodes.append(bot)

        cursor_node = cent

    # Removing two auxiliary nodes if the number of pyramids is odd
    if len(other_pyramids) % 2 == 1: 
        w_labels = sorted((n for n in pos if n.startswith("W")), key = lambda x: int(x[1:]))

        # Removing the highest labels 
        for orphan in w_labels[-2:]:
            pos.pop(orphan, None)           
            if isinstance(all_nodes, set):
                all_nodes.discard(orphan)
            else:
                try:                           
                    all_nodes.remove(orphan)
                except ValueError:
                    pass
        
        # Removing the last bottom node since no bow-tie will be attached to it
        bottom_nodes = bottom_nodes[:-1]
        current_idx -= 2

    # ---------------------------------
    # Step 5: Build all others bow-ties
    # ---------------------------------
    TOP_HELPERS    = [top_left_vertical_helper,  top_right_vertical_helper]
    BOTTOM_HELPERS = [bottom_left_vertical_helper, bottom_right_vertical_helper]
    attachments    = len(other_pyramids)  # How many bow-ties to add
    current_idx 

    for n_attach in range(attachments):
        pair_idx  = n_attach // 2
        use_top   = (n_attach % 2 == 0)

        if use_top:
            anchor = top_nodes[pair_idx]
            helper = TOP_HELPERS[pair_idx % 2]
        else:
            anchor = bottom_nodes[pair_idx]
            helper = BOTTOM_HELPERS[pair_idx % 2]

        # Pop the next pyramid to attach
        next_pyr   = other_pyramids.pop(0)
        coords_pyr = restrict_coords_to_pyramid(next_pyr, coords)

        # Build the bow-tie
        next_pyr  = [n for n in next_pyr if n != shared_node]
        coords_pyr = {n: coords[n] for n in next_pyr}
        pos_new, apex, _, _, current_idx = helper(anchor, pos[anchor], next_pyr, coords_pyr, current_idx)

        # Merge results
        pos.update(pos_new)
        all_nodes.update(pos_new.keys())
        apex_list.append(apex)

    all_nodes = list(all_nodes)

    return pos, all_nodes, apex_list, current_idx

# -------------------------------------------------------
# FIVE (OR MORE) PYRAMIDS SHARING ONE SAME BASE NODE WITH 
# A GIVEN STARTING PYRAMID (USED IN SINGLE SHARED TREE)
# -------------------------------------------------------
def N_star_node_tree(start_pyr, start_pos, new_pyr, new_coords, direction, start_idx = 0):
    """
    Embed an arbitrary number of pyramids that share a single common node,
    starting from an already embedded pyramid and growing a star-like structure
    in a chosen global direction.

    Parameters
    ----------
    start_pyr : iterable
        Collection of node identifiers forming the already embedded starting
        pyramid. One of these nodes must be the shared node.
    start_pos : dict
        Dictionary mapping nodes of start_pyr to their existing 2D
        coordinates, e.g. {id: (x, y), ...}.
    new_pyr : list of iterable
        List of additional pyramids to be attached, each given as a collection
        of node identifiers. All pyramids in new_pyr must share exactly
        one node with start_pyr and with each other (the same shared node).
    new_coords : dict
        Dictionary mapping all node identifiers (from start_pyr and
        new_pyr) to their 3D coordinates, e.g. {id: (x, y, z)}.
        The z-coordinate is used to identify apex nodes and to determine
        relative “top/bottom” placement.
    direction : {"right", "left", "up", "down"}
        Global direction in which the entire star-shaped structure should
        extend from the starting pyramid. The construction is first carried out
        assuming growth to the right, and then rotated/reflected if needed.
    start_idx : int, optional
        Starting index used to generate auxiliary node labels (prefix "W").
        Updated internally as new auxiliary nodes are created.

    Returns
    -------
    tuple
        - pos: dict  
        Final mapping from node identifiers (original + auxiliary) to 2D
        coordinates, after all rotations/reflections and pruning of the
        unneeded starting nodes (only the shared node from start_pyr is
        kept in the local gadget).
        - all_nodes : set  
        Set of all node identifiers present in this local embedding.
        - apex_list : list  
        List of apex nodes (one per pyramid), starting with the apex of
        start_pyr followed by the apexes of the attached pyramids in
        attachment order.
        - current_idx : int  
        Updated index after all auxiliary nodes have been allocated.
        - directions : dict  
        Dictionary mapping each apex node to a qualitative edge direction
        (e.g. "up", "down", or the original global direction),
        plus the key "Old" storing the initial global "direction".

    Notes
    -----
    - A single shared node across all pyramids is required; otherwise a
    ValueError is raised.
    - The starting apex is recorded, and the remaining pyramids are attached
    along a chain of “extension gadgets” placed to the right of the
    shared node:
    
    * A sequence of extension codes is built based on the number of new
        pyramids. Each code selects either:
        - a short extension arm via extend_structure_short,
        - a long extension arm via extend_structure_long, or
        - a simple horizontal segment (code "H").
    * Each extension returns a top node, a bottom node, and a central
        node; these nodes act as anchors for subsequent bow-tie attachments.

    - If the number of new pyramids is even, the last two auxiliary nodes
    are removed (together with the last unused bottom anchor) to keep the
    scaffold consistent with the number of attachments.

    - The remaining pyramids are then attached one by one:

    * Attachments alternate between top and bottom anchors within each
        extension pair.
    * Vertical helpers from the sets
        TOP_HELPERS = [top_left_vertical_helper, top_right_vertical_helper]
        and
        BOTTOM_HELPERS = [bottom_left_vertical_helper, bottom_right_vertical_helper]
        are used, alternating left/right to create a balanced zig-zag pattern.
    * The last pyramid is attached horizontally to the final central node
        using top_right_horizontal_helper or
        bottom_right_horizontal_helper, so that the chain of gadgets
        terminates cleanly.

    - For each attached pyramid, a bow-tie embedding is built (excluding the
    shared node), its apex is recorded in apex_list, and a logical
    direction ("up" or "down" for non-final attachments, or the
    global direction for the last one) is stored in directions.

    - After the rightward construction is complete, the layout is adapted
    to the requested global direction:
    * "right": no change.
    * "left": positions are reflected horizontally around the shared
        node using reflect_positions.
    * "up": the entire gadget is rotated +90° around the shared node
        via rotate_helper, and direction labels in directions are
        updated ("up" → "left", "down" → "right").
    * "down": the gadget is rotated −90° around the shared node and
        direction labels are updated accordingly
        ("up" → "right", "down" → "left").

    - Finally, nodes originally in start_pos other than the shared node
    are removed from the local gadget, and positions are merged so that
    the returned pos only contains the shared node from the starting
    pyramid plus the newly constructed auxiliary and bow-tie nodes.
    """

    # --------------------
    # Find the shared node
    # --------------------
    all_pyrs = [start_pyr] + new_pyr
    shared = set(all_pyrs[0]).intersection(*all_pyrs[1:])
    if len(shared) != 1:
        raise ValueError(f"Expected the pyramids to share exactly one node")
    
    shared_node = next(iter(shared))
    start_apex  = next(n for n in start_pyr if new_coords[n][2] != 0)

    apex_list  = [start_apex]
    directions = {"Old" : direction}

    # ---------------------------------
    # Build the extensions to the right
    # ---------------------------------
    number_extensions = (len(new_pyr) + 1) // 2

    # Creating the sequence of extensions based on the number of pyramids
    sequence = []
    for i in range(number_extensions):
        if i < 2:
            sequence.append(1)
        else:
            sequence.append(2 if i % 2 == 0 else 1)

    if len(new_pyr) % 2 == 1:
        sequence[-1] = "H"

    top_nodes, bottom_nodes = [], []
    pos          = start_pos.copy()
    current_idx  = start_idx
    cursor_node  = shared_node
    all_nodes    = set(pos.keys())

    # Calling the extension helpers based on the sequence
    for code in sequence:
        if code == 1:              
            all_n, pos_n, top, bot, cent, current_idx = extend_structure_short(cursor_node, pos, current_idx)
            
        elif code == 2:                     
            all_n, pos_n, top, bot, cent, current_idx = extend_structure_long(cursor_node, pos, current_idx)

        elif code == "H":   
            x0, y0  = pos[cursor_node]

            aux1    = f"W{current_idx + 1}"
            aux2    = f"W{current_idx + 2}"
            current_idx += 2

            pos_n   = {aux1: (x0 + 1, y0), aux2: (x0 + 2, y0)}
            all_n   = {aux1, aux2}
            top     = aux1                 
            bot     = aux2              
            cent    = aux2

        else:
            raise ValueError(f"Unknown code {code!r}")

        pos.update(pos_n)
        all_nodes.update(all_n)
        top_nodes.append(top)
        bottom_nodes.append(bot)
        cursor_node = cent

    # Removing two auxiliary nodes if the number of pyramids is even
    if len(new_pyr) % 2 == 0: 
        w_labels = sorted((n for n in pos if n.startswith("W")), key = lambda x: int(x[1:]))

        # Removing the highest labels 
        for orphan in w_labels[-2:]:
            pos.pop(orphan, None)           
            if isinstance(all_nodes, set):
                all_nodes.discard(orphan)
            else:
                try:                           
                    all_nodes.remove(orphan)
                except ValueError:
                    pass
        
        # Removing the last bottom node since no bow-tie will be attached to it
        bottom_nodes = bottom_nodes[:-1]
        current_idx -= 2

    # -------------------------
    # Build all others bow-ties
    # -------------------------
    TOP_HELPERS    = [top_left_vertical_helper,  top_right_vertical_helper]
    BOTTOM_HELPERS = [bottom_left_vertical_helper, bottom_right_vertical_helper]
    attachments    = len(new_pyr)  # How many bow-ties to add
    current_idx 

    for n_attach in range(attachments):
        pair_idx  = n_attach // 2
        use_top   = (n_attach % 2 == 0)

        if use_top and n_attach != attachments - 1:
            anchor = top_nodes[pair_idx]
            helper = TOP_HELPERS[pair_idx % 2]

        elif not use_top and n_attach != attachments - 1:
            anchor = bottom_nodes[pair_idx]
            helper = BOTTOM_HELPERS[pair_idx % 2]

        elif use_top and n_attach == attachments - 1:
            anchor = cent
            helper = (top_right_horizontal_helper)

        elif not use_top and n_attach == attachments - 1:
            anchor = cent
            helper = (bottom_right_horizontal_helper)


        # Pop the next pyramid to attach
        next_pyr   = new_pyr.pop(0)
        coords_pyr = restrict_coords_to_pyramid(next_pyr, new_coords)

        # Build the bow-tie
        next_pyr  = [n for n in next_pyr if n != shared_node]
        coords_pyr = {n: new_coords[n] for n in next_pyr}
        pos_new, apex, _, _, current_idx = helper(anchor, pos[anchor], next_pyr, coords_pyr, current_idx)

        if n_attach != attachments - 1:
            directions[apex] = "up" if use_top else "down"
        else:
            directions[apex] = direction

        # Merge results
        pos.update(pos_new)
        all_nodes.update(pos_new.keys())
        apex_list.append(apex)
    
    all_nodes = sorted(list(all_nodes))

    pos_to_remove = {k: v for k, v in start_pos.items() if k != shared_node}
    filtered_pos = {k: v for k, v in pos.items() if k not in pos_to_remove}
    
    if direction == "right":
        pass 
    elif direction == "left":
        filtered_pos = reflect_positions(filtered_pos, shared_node)
    elif direction == "up":
        filtered_pos = rotate_helper(filtered_pos, 90, pivot = pos[shared_node])

        for key, value in directions.items():
            if value == "up":
                directions[key] = "left"
            elif value == "down":
                directions[key] = "right"

    elif direction == "down":
        filtered_pos = rotate_helper(filtered_pos, - 90, pivot = pos[shared_node])

        for key, value in directions.items():
            if value == "up":
                directions[key] = "right"
            elif value == "down":
                directions[key] = "left"
    
    pos = {**filtered_pos, **pos_to_remove}
    pos = {k: v for k, v in pos.items() if k not in start_pos or k == shared_node}
    all_nodes = set(pos.keys())

    return pos, all_nodes, apex_list, current_idx, directions
  
# ---------------------------------
# SINGLE SHARED TREE-LIKE STRUCTURE
# ---------------------------------

# ---------------------------
# Additional Helper Functions
# ---------------------------

def make_pyramid_graph(pyramids):
    """
    Build an adjacency graph in which each node represents a pyramid and
    edges connect pyramids that share exactly one base node.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each given as a collection of node identifiers.

    Returns
    -------
    dict
        A dictionary adj mapping each pyramid index i to a set of
        indices of pyramids that share a base node with it. Formally:

            adj[i] = { j | pyramid_i and pyramid_j share ≥1 base node }
    Notes
    -----
    - The function works by building a reverse map node_to_pyrs listing
      which pyramids contain each node.
    - For each node that belongs to *exactly two* pyramids, an undirected
      edge is added between those pyramids.
    - This graph is useful for:
        * detecting linear chains of pyramids,
        * identifying extremal pyramids (degree-1 nodes),
        * guiding traversal order in embedding routines.
    - Star configurations (nodes shared by ≥3 pyramids) do not
      generate edges here; they are treated by higher-level routines.
    """

    node_to_pyrs = {}
    for idx, pyr in enumerate(pyramids):
        for n in pyr:
            node_to_pyrs.setdefault(n, []).append(idx)

    adj = {i: set() for i in range(len(pyramids))}
    for p_list in node_to_pyrs.values():          
        if len(p_list) == 2:                      
            a, b = p_list
            adj[a].add(b)
            adj[b].add(a)

    return adj
    
def distance_to_nearest_branch(adj, branch_set):
    """
    Compute, for each pyramid in the adjacency graph, the shortest distance
    (in number of pyramid-to-pyramid hops) to reach any pyramid belonging
    to the given branch set.

    Parameters
    ----------
    adj : dict
        Adjacency dictionary where keys are pyramid indices and values are
        sets of neighbouring pyramid indices. Typically produced by
        make_pyramid_graph.
    branch_set : iterable
        Collection of pyramid indices that form the reference “branch”.
        Distances from all pyramids are computed to the nearest member
        of this set.

    Returns
    -------
    dict
        A dictionary dist mapping each pyramid index to the integer
        length of the shortest path to any pyramid in branch_set.
        Distance is measured in number of hops between adjacent pyramids.
        For pyramids already in branch_set, the distance is 0.

    Notes
    -----
    - The algorithm performs a standard multi-source breadth-first search
      (BFS), initialising the queue with all branch pyramids at distance 0.
    - Unreachable pyramids (rare in a well-formed pyramid graph) would keep
      a distance of None, but in typical embedding graphs the adjacency
      structure is connected.
    - Useful for identifying the “farthest” extremal pyramids in a chain,
      e.g. in the starting-pyramid selection routine.
    """

    from collections import deque

    dist = {v: None for v in adj}
    q    = deque((b, 0) for b in branch_set)

    for b in branch_set:
        dist[b] = 0

    while q:
        node, d = q.popleft()
        for nb in adj[node]:
            if dist[nb] is None:
                dist[nb] = d + 1
                q.append((nb, d + 1))

    return dist
    
def furthest_extremal(pyramids):
    """
    Return the extremal pyramid (degree-1 node in the pyramid adjacency graph)
    that lies farthest from any branch pyramid.

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each given as a collection of node identifiers.
        Used to build the pyramid adjacency graph with make_pyramid_graph.

    Returns
    -------
    tuple
        - far_extremal : int  
          Index of the extremal pyramid (degree-1) that maximises the
          distance to the nearest branch (degree ≥ 3) pyramid.
        - distance : int  
          The number of pyramid-hops from this extremal piramid to the
          nearest branch pyramid.

    Notes
    -----
    - The adjacency graph is built using make_pyramid_graph, where
      two pyramids are connected if they share exactly one base node.
    - Extremal pyramids are those with degree 1 in this graph; they
      represent endpoints of chains.
    - Branch pyramids are those with degree ≥ 3 and represent
      branching points in the pyramid structure.
    - If no*branch pyramids exist, the structure is a simple linear chain.
      In this case, the first extremal pyramid is returned and the distance
      is 0.
    - Otherwise, a multi-source BFS (via distance_to_nearest_branch)
      computes distances from all branch nodes, and the extremal with the
      largest distance is selected.
    """

    adj     = make_pyramid_graph(pyramids)
    degrees = {i: len(nbs) for i, nbs in adj.items()}

    extremals = [i for i, deg in degrees.items() if deg == 1] 
    branches  = [i for i, deg in degrees.items() if deg >= 3]

    # chain with no branches → pick either end
    if not branches:
        return extremals[0], 0

    dist = distance_to_nearest_branch(adj, branches)
    far_extremal = max(extremals, key = lambda i: dist[i])

    return far_extremal, dist[far_extremal]

def other_pyramid_and_apex(node_to_pyrs, shared_node, cur_pyr, pyramids, coords):
    """
    Given a shared base node that belongs to exactly two pyramids, return the
    other pyramid (i.e., the one that is not the current pyramid) together
    with its apex.

    Parameters
    ----------
    node_to_pyrs : dict
        Mapping node → [pyr_idx, pyr_idx] listing all pyramids that
        contain each node. For shared base nodes involved in chain-like
        structures, the list must contain exactly two pyramid indices.
    shared_node : hashable
        The base node shared between cur_pyr and one other pyramid.
    cur_pyr : int
        Index of the currently processed pyramid. The function returns
        information about the other pyramid.
    pyramids : list of iterable
        List of all pyramids, each given as a collection of node identifiers.
    coords : dict
        Mapping of all nodes to their 3D coordinates (x, y, z). The apex
        of a pyramid is identified as the unique node with z ≠ 0.

    Returns
    -------
    tuple
        (other_idx, other_pyr, other_apex) where:
        - other_idx : int  
          Index of the pyramid (other than cur_pyr) that contains
          shared_node.
        - other_pyr : iterable  
          The node set of that pyramid.
        - other_apex : hashable  
          The apex node of other_pyr (the unique node with z ≠ 0).

    Notes
    -----
    - This helper assumes the shared base node belongs to exactly two
      pyramids; otherwise, a higher-level star-node handler should be used.
    - Commonly used in chain traversal, where each pair of adjacent pyramids
      shares exactly one base node.
    """

    idx1, idx2 = node_to_pyrs[shared_node] # Exactly two owners
    other_idx  = idx1 if idx1 != cur_pyr else idx2 # Skip the current one
    other_pyr  = pyramids[other_idx]
    # the apex is the only node with z-coordinate ≠ 0
    other_apex = next(n for n in other_pyr if coords[n][2] != 0)
    return other_idx, other_pyr, other_apex
 
def find_undesired_edges(nodes, pos, original_edges, debug = False):
    """
    Find geometric edges that do not belong to the intended interaction graph.

    Parameters
    ----------
    nodes : iterable of hashable
        Sequence of node identifiers present in the layout.
    pos : dict
        Mapping node -> (x, y) giving the 2D coordinates of each node.
    original_edges : iterable of tuple
        List or set of intended (u, v) edges. Each edge is a pair of node labels.
    debug : bool, optional
        If True, print diagnostic information about geometric edges,
        original edges, and undesired edges. Default is False.

    Returns
    -------
    list of tuple
        Sorted list of geometric edges (u, v) that appear in the unit-disk graph
        but are not part of original_edges.

    Notes
    -----
    - Edges involving nodes whose labels start with "W" (e.g. wire or
    non-physical nodes) are automatically ignored.
    - Geometric edges are computed from a unit-disk graph built with the given
    positions via unit_disk_edges.
    - Returned edges are always given as sorted tuples to ensure consistency.
    """

    # -------------------------------------------------------
    # Build the geometric graph and collect endpoint–endpoint
    # edges (ignore every edge involving a 'W...' label)
    # -------------------------------------------------------
    G = unit_disk_edges(nodes, pos)
    geom_ep_edges = {tuple(sorted((u, v))) for u, v in G.edges() if not u.startswith("W") and not v.startswith("W")}

    # ------------------------------------------
    # Remove those that are part of the intended 
    # interaction graph (original_edges)
    # ------------------------------------------
    norm_original = {tuple(sorted(e)) for e in original_edges}
    undesired     = geom_ep_edges - norm_original

    if debug:
        print("▶ Geometric endpoint-endpoint edges:", geom_ep_edges)
        print("▶ Declared original edges:          ", norm_original)
        print("▶ Reported as undesired:            ", undesired)

    return sorted(undesired)

def find_angle_orient(old_direction, new_direction, node1, node2, shared_node, coords):
    """
    Compute the rotation needed to reorient a chain segment when the direction changes.

    Parameters
    ----------
    old_direction : {"up", "down", "left", "right"}
        The direction of the previous segment in the chain.
    new_direction : {"up", "down", "left", "right"}
        The direction of the next segment being attached.
    node1, node2 : hashable
        The two nodes defining the previous segment; used to determine which
        endpoint is left/right or lower/higher.
    shared_node : hashable
        The node through which the new segment attaches to the old one.
    coords : dict
        Mapping `node -> (x, y)` giving each node's 2D coordinates.

    Returns
    -------
    tuple
        (angle, axis), where:
        - angle : int  
        Either +90 or -90 degrees, indicating the required rotation.
        - axis : {"horizontal", "vertical"}  
        The axis around which the rotation is conceptually applied.

    Notes
    -----
    - The function infers geometric orientation (left/right or lower/higher)
    from the nodes' coordinates to determine how a turn should be interpreted.
    - A rotation is only applied if `old_direction` and `new_direction`
    imply a turn; straight continuations may still require a flip depending
    on the shared endpoint.
    - Angles are always returned as ±90° since all allowed turns are quarter-turns.
    """

    angle = 0
    axis  = "vertical"

    if old_direction == "right":
        lower_node, higher_node = ((node1, node2) if coords[node1][1] < coords[node2][1] else (node2, node1))
        if new_direction == "right":
            if   shared_node == higher_node: angle, axis = 90, "vertical"
            elif shared_node == lower_node:  angle, axis = - 90, "vertical"

        if new_direction == "up": 
            angle, axis = - 90, "horizontal"

        if new_direction == "down":
            angle, axis = 90, "horizontal"

    elif old_direction == "left":
        lower_node, higher_node = ((node1, node2) if coords[node1][1] < coords[node2][1] else (node2, node1))
        if new_direction == "left":
            if   shared_node == higher_node: angle, axis = - 90, "vertical"
            elif shared_node == lower_node:  angle, axis = 90, "vertical"

        if new_direction == "up": 
            angle, axis = - 90, "horizontal"

        if new_direction == "down":
            angle, axis = 90, "horizontal"

    elif old_direction == "up":
        left_node, right_node = ((node1, node2) if coords[node1][0] < coords[node2][0] else (node2, node1))
        if new_direction == "up":
            if   shared_node == right_node: angle, axis = - 90, "horizontal"
            elif shared_node == left_node:  angle, axis = 90, "horizontal"

        if new_direction == "right":
            angle, axis = 90, "vertical"
        
        if new_direction == "left":
            angle, axis = - 90, "vertical"

    elif old_direction == "down":
        left_node, right_node = ((node1, node2) if coords[node1][0] < coords[node2][0] else (node2, node1))
        if new_direction == "down":
            if   shared_node == right_node: angle, axis = 90, "horizontal"
            elif shared_node == left_node:  angle, axis = - 90, "horizontal"

        if new_direction == "right":
            angle, axis = - 90, "vertical"
        
        if new_direction == "left":
            angle, axis = 90, "vertical"

    return angle, axis

def bow_tie_to_rotate(undesired_edges, new_pos, new_pyramids, shared_node, pos_copy, all_nodes, coords):
    """
    Identify the subset of nodes belonging to the misaligned bow-tie pyramid that
    must be rotated, and determine the pivot node for the rotation.

    Parameters
    ----------
    undesired_edges : iterable of tuple
        List/set of geometric edges (u, v) that should not appear in the layout.
    new_pos : dict
        Mapping `node -> (x, y)` of the current (partially constructed) positions
        of nodes relevant to the bow-tie structure.
    new_pyramids : list of iterable
        Collection of pyramids already placed, each given as an iterable of node
        labels.
    shared_node : hashable
        Node that belongs to both the currently processed pyramid and its neighbor.
    pos_copy : dict
        Full mapping `node -> (x, y)` for all nodes, including those not yet
        overwritten in `new_pos`. Used to infer neighbors in the geometric graph.
    all_nodes : iterable
        Complete set of nodes present in the structure.
    coords : dict
        Mapping `node -> (x, y, z)` used to identify apex nodes (those with z ≠ 0).

    Returns
    -------
    tuple
        (pos_to_rotate, pivot_node), where:
        - pos_to_rotate : dict  
        Mapping `node -> (x, y)` of the nodes inside the misoriented pyramid
        (excluding the shared node) whose coordinates must be rotated.
        - pivot_node : hashable  
        The auxiliary wire-node ("W…") that serves as the rotation pivot.
        It is selected as the unique wire-neighbor connected to the apex.

    Notes
    -----
    - A “wrong” node is detected as any node appearing in both `new_pos` and in an
    undesired geometric edge; this identifies the misaligned pyramid.
    - The apex is the unique node of the pyramid whose z-coordinate is nonzero.
    - Auxiliary wire nodes (labels starting with "W") are included to ensure
    the correct geometric rotation around the pivot.
    - Only nodes that already appear in `new_pos` are included in `pos_to_rotate`,
    ensuring that the rotation affects only placed nodes.
    """

    undesired_nodes = set().union(*undesired_edges)
    wrong_node      = next(n for n in new_pos.keys() if n in undesired_nodes)
    wrong_pyramid   = next(pyr for pyr in new_pyramids if wrong_node in pyr)
    wrong_pyramid   = [n for n in wrong_pyramid if n != shared_node]
    wrong_apex      = next(n for n in wrong_pyramid if coords[n][2] != 0)

    G_new = unit_disk_edges(all_nodes, pos_copy)

    aux_in_wrong_pyramid = [nbr for nid in wrong_pyramid for nbr in G_new.neighbors(nid) if nbr.startswith("W")]
    aux_in_wrong_pyramid = sorted(set(aux_in_wrong_pyramid), key = lambda w: int(w[1:]))
    pivot_node           = next(n for n in aux_in_wrong_pyramid if G_new.has_edge(n, wrong_apex))

    full_wrong_pyramid = wrong_pyramid + aux_in_wrong_pyramid
    pos_to_rotate      = {n: pos_copy[n] for n in full_wrong_pyramid if n in new_pos and n != shared_node}

    return pos_to_rotate, pivot_node

def find_star_nodes(pyramids):
    """
    Identify nodes that belong to more than two pyramids (star nodes).

    Parameters
    ----------
    pyramids : list of iterable
        List of pyramids, each represented as a collection of node identifiers.
        Nodes appearing in multiple pyramids may indicate shared or star nodes.

    Returns
    -------
    dict
        Mapping node -> count for all nodes that occur in more than two
        pyramids. The count indicates how many pyramids each star node belongs to.

    Notes
    -----
    - Star nodes are defined as nodes appearing in **three or more** pyramids.
    - Internally, the function flattens all pyramid node lists and counts
    occurrences using collections.Counter.
    - The returned dictionary can contain multiple star nodes if the structure
    has several branching points.
    """

    from collections import Counter
    from itertools import chain
    
    counts = Counter(chain.from_iterable(pyramids))
    star_nodes = {n : k for n, k in counts.items() if k > 2}

    return star_nodes

# -------------
# Main Function
# -------------

def single_shared_tree(coords, pyramids, start_idx = 0):
    """
    Embed a collection of pyramids that share nodes into 2D, constructing a
    tree-like arrangement of bow-ties and star-node gadgets while preserving
    the original 3D adjacency structure. Each pair of pyramids always shares
    at most one base node.

    This is one of the main high-level embedding routines: it automatically detects
    shared (star) nodes, chooses a good starting pyramid, grows branches, and
    handles complex star nodes of arbitrary degree by delegating to specialised
    subroutines.

    Parameters
    ----------
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to identify apex
        nodes of pyramids and for geometric decisions (e.g. top vs bottom).
    pyramids : list of iterable
        List of pyramids, each given as a collection of node identifiers
        (typically four nodes: three base nodes + one apex). Pyramids may
        share one or more base nodes, forming “star” configurations and
        chains.
    start_idx : int, optional
        Starting index for generating auxiliary node labels (prefix "W").
        Updated internally as new auxiliary nodes are introduced.

    Returns
    -------
    tuple
        - pos : dict  
        Mapping from all node identifiers (original + auxiliary) to their
        final 2D coordinates. 
        - all_nodes : list  
        Sorted list of all node identifiers present in the final embedding.
        - apex_list : list  
        List of apex nodes, in the order in which their pyramids have been
        embedded.
        - current_idx : int  
        Updated auxiliary-node index after all constructions, suitable for
        creating further gadgets without label collisions.

    Function overview
    ------------------
    1. Shared and star nodes.
    - Count how many pyramids each node belongs to and classify:
        * shared nodes: nodes appearing in at least two pyramids;
        * star nodes: nodes shared by three or more pyramids, as detected
            by find_star_nodes.
    - Distinguish extremal pyramids (with exactly one shared node) from
        more central ones.

    2. Starting pyramid selection.
    - If no extremal pyramids exist (closed loop), pick the first pyramid.
    - Otherwise, select as starting pyramid the extremal one that is
        “farthest” from a branching region, using furthest_extremal.
    - Embed this starting pyramid as a bow-tie with single_bow_tie,
        obtaining an initial 2D layout and the four corner nodes
        (rt, rb, lt, lb).

    3. Branch and direction bookkeeping.
    - Maintain:
        * embedded_pyramids: mapping from branch-local indices to
            global pyramid indices;
        * directions: the current direction ("horizontal",
            "right", "left", "up", "down") for each branch;
        * pyr_by_branch: which pyramids belong to which branch;
        * visited: set of pyramids already embedded;
        * shared_nodes_embedded: shared nodes fully processed;
        * pyr_to_revisit and node_to_save: to resume branches where
            two shared nodes are available.

    4. Main embedding loop.
    - While some branch still has a non-None direction:
        * Take the current branch and its last embedded pyramid.
        * Identify the available shared base nodes of this pyramid that
            have not yet been processed.
        * If no shared nodes remain and there is no pyramid to revisit,
            the branch is closed and its direction is set to None.
        * Otherwise:
            - If the shared node is a **star node** (shared by ≥ 3 pyramids),
                delegate to:
                    · three_star_node_tree for 3-star nodes,
                    · four_star_node_tree for 4-star nodes,
                    · N_star_node_tree for general N-star nodes.
                These subroutines embed multiple pyramids at once around the
                star node and return:
                    · new local coordinates,
                    · the new apex nodes,
                    · updated directions for the newly created branches.
                The routine then:
                    · updates pos, apex_list, directions,
                    pyr_by_branch, and pyr_to_revisit, and
                    · chooses which pyramid stays on the current branch and
                    which ones spawn new branches.
            - Otherwise (the shared node is not star-like), use the current
                branch direction and previous attachment nodes (node1, node2)
                to determine the appropriate bow-tie helper via
                pick_bow_tie_helper or find_case and embed the next
                pyramid on the same branch.

    5. Edge validation and local corrections.
    - After each embedding step (star node gadget or single bow-tie), the
        routine checks that no *undesired* edges have been created:
        * Compute the set of *original* edges from the 3D pyramids
            (via all 2-combinations of the nodes of each pyramid).
        * Use find_undesired_edges on the current pos to detect any
            extra edges induced by the unit-disk metric.
    - If undesired edges are found, perform local geometric corrections:
        * For 3-/4-/N-star gadgets:
            · Use find_angle_orient to determine an angle and/or axis,
            · Use bow_tie_to_rotate and rotate_helper to adjust
                positions around a pivot node.
        * For single bow-tie attachments:
            · Use find_angle_orient again,
            · then rotate with rotate_helper and reflect with
                reflect_positions as needed.
    - After correction, reinsert the shared node at its fixed 2D position
        and update pos.

    6. Branch splitting (two shared nodes).
    - When an embedded pyramid has two unused shared nodes:
        * The algorithm splits the evolution into branches by:
            · looking at the relative positions of the candidate next
                apexes apex_a and apex_b (from the two neighbouring pyramids),
            · deciding which neighbour pyramid continues the current branch
                and which one spawns a new branch with direction determined
                from the relative 2D positions of the shared nodes.
        * The pyramid that continues the current branch is embedded
            immediately, the other is added to pyr_to_revisit.

    7. Loop closure and edge restoration.
    - During the embedding, if some node’s position is changed relative to
        an earlier embedding (e.g., due to rotations), the original position
        is stored in equivalent_positions.
    - After the main loop:
        * For each such node, a new auxiliary node is created at the
            original position.
        * restore_edges is called to rebuild missing edges so that
            unclosed loops are properly reconnected in the 2D layout.

    8. Final output.
    - All accumulated positions in pos are collected.
    - all_nodes is set to the sorted list of keys of pos.
    - apex_list contains the apex for every embedded pyramid, in the
        order they were added.
    - current_idx holds the next free index for auxiliary nodes.
    """

    from collections import Counter
    from itertools import chain
    from itertools import combinations

    # -------------------------------------------------------------------
    # Step 1: Find the star nodes and the shared nodes among the pyramids
    # -------------------------------------------------------------------
    counts = Counter(chain.from_iterable(pyramids))
    shared_nodes = {n : k for n, k in counts.items() if k >= 2}
    shared_nodes = sorted(shared_nodes)
    star_nodes = find_star_nodes(pyramids)
    
    # ---------------------------------------------------------
    # Step 2: Identify the "type" of pyramids: 
    #  - Extremal pyramids (pyramids with only one shared node)
    #  - Connect pyramids (pyramids with two shared nodes)
    #  - Branch pyramids (pyramids with all three shared nodes)
    # ---------------------------------------------------------
    extremal_pyramids = [pyr for pyr in pyramids if len([n for n in pyr if n in shared_nodes]) == 1]
    # connect_pyramids  = [pyr for pyr in pyramids if len([n for n in pyr if n in shared_nodes]) == 2]
    # branch_pyramids   = [pyr for pyr in pyramids if len([n for n in pyr if n in shared_nodes]) == 3]
    num_branches = len(extremal_pyramids) - 1
    directions   = {str(i): None for i in range(1, num_branches + 1)}

    # ------------------------------------------------------------------------------------------------
    # Step 3: Identify the starting pyramid, i.e. the one that is "farthest" from a branch pyramid,
    # that is the one that needs to traverse the most connect pyramids route to reach a branch pyramid
    # ------------------------------------------------------------------------------------------------

    # If we have a close loop with no extremal pyramids, we 
    # simply pick the first pyramid, since it doesn't matter 
    if not extremal_pyramids:
        start_idx_pyr = 0

    elif len(extremal_pyramids) != 0:
        start_idx_pyr, _ = furthest_extremal(pyramids)

    start_pyramid = pyramids[start_idx_pyr]
    start_coords  = restrict_coords_to_pyramid(start_pyramid, coords)

    # ---------------------------------------------------------------------------------------
    # Step 4: Initialize the first direction (horizontal by default) and the starting pyramid
    # ---------------------------------------------------------------------------------------
    pos = {}
    apex_list = []
    equivalent_positions = {}
    start_pos, _, start_apex, (rt, rb), (lt, lb), current_idx = single_bow_tie(start_pyramid, start_coords, start_idx)

    pos.update(start_pos)
    apex_list.append(start_apex)

    embedded_pyramids = {"1" : start_idx_pyr} 
    
    branch_id_counter = 1                     
    current_branch_id = "1"                
    pyr_by_branch = {current_branch_id: [start_idx_pyr]}
    directions[current_branch_id] = 'horizontal'  
    visited = {start_idx_pyr}

    # -------------------------------------------
    # Step 5: Add all other pyramids to the graph
    # -------------------------------------------

    # Node → list of pyramid indices that contain it
    node_to_pyrs = {n: [] for n in shared_nodes}
    for idx, pyr in enumerate(pyramids):
        for n in pyr:
            if n in node_to_pyrs:
                node_to_pyrs[n].append(idx)

    # For each pyramid index, the one (or two) shared nodes it owns
    pyr_to_shared = {idx: [n for n in pyr if n in shared_nodes] for idx, pyr in enumerate(pyramids)}
    
    first_connect_done = False
    node1 = node2 = None
    shared_nodes_embedded = []
    pyr_to_revisit = []
    node_to_save = []
    
    # -------------------------------------------------
    # The while ends when all diretions are None, which 
    # is set when there are no more pyramids to embed
    # -------------------------------------------------
    while not all(v is None for v in directions.values()):
        pos_copy_for_loop = pos.copy()
        cur_idx  = next(reversed(embedded_pyramids))
        cur_pyr  = embedded_pyramids[cur_idx]
        cur_apex = next(n for n in pyramids[cur_pyr] if coords[n][2] != 0)
        cur_idx  = int(cur_idx) 

        shared_base_nodes = pyr_to_shared[cur_pyr]
        available_shared_nodes = [n for n in shared_base_nodes if n not in shared_nodes_embedded] 

        # ----------------------------------------------------------------------------------
        # If there are no available shared nodes and no pyramids to revisit, we end the loop
        # ----------------------------------------------------------------------------------
        if not available_shared_nodes and not pyr_to_revisit:
            directions = {i: None for i in directions}
            continue
        
        # -----------------------------------------------------------------------------
        # If there are no available shared nodes, on a given branch, but still pyramids 
        # to revisit, we change branch and start again until it ends again
        # -----------------------------------------------------------------------------
        elif not available_shared_nodes and len(pyr_to_revisit) != 0:
            cur_idx_old = cur_idx
            cur_idx  = str(pyr_to_revisit.pop(0))
            cur_pyr  = embedded_pyramids[cur_idx]
            cur_apex = next(n for n in pyramids[cur_pyr] if coords[n][2] != 0)
            cur_idx  = int(cur_idx_old)

            shared_base_nodes = pyr_to_shared[cur_pyr]
            available_shared_nodes = [n for n in shared_base_nodes if n not in shared_nodes_embedded]
            current_branch_id = str(int(current_branch_id) + 1)

            if not available_shared_nodes:
                break

            node1 = available_shared_nodes[0]
            node2 = node_to_save.pop(0)

        shared_node = available_shared_nodes[0]
        idx_candidates = node_to_pyrs[shared_node] 

        # --------------------------------------------
        # If we met a star node, we need to handle it 
        # differently by calling appropriate functions
        # --------------------------------------------
        if shared_node in star_nodes:
            start_pyr = pyramids[cur_pyr]
            shared_nodes_embedded.append(shared_node)

            # --------------------------------------------------------------------------
            # Case 1: If the star node is a three-star node, so shared by three pyramids
            # --------------------------------------------------------------------------
            if star_nodes[shared_node] == 3:
                idx1, idx2 = (i for i in idx_candidates if i != cur_pyr)
                pyr1, pyr2 = pyramids[idx1], pyramids[idx2]

                # Call the three-star node tree function
                new_pos, _, new_apex, current_idx, new_dirs = three_star_node_tree(start_pyr, node1, node2, pos, pyr1, pyr2, coords, directions[current_branch_id], current_idx)
                apex_list += [a for a in new_apex if a != cur_apex]

                # Embed both pyramids
                for off, idx in enumerate((idx1, idx2), 1):
                    embedded_pyramids[str(cur_idx + off)] = idx

                # --------------------------------------------------------------------------------
                # Checking if the new position is valid, i.e. if it doesn't create undesired edges
                # --------------------------------------------------------------------------------
                pyramids_list = [pyramids[i] for i in embedded_pyramids.values()]
                original_edges = list(chain.from_iterable(combinations(pyr, 2) for pyr in pyramids_list))

                pos_copy = pos.copy()
                pos_copy.update(new_pos)
                all_nodes = set(pos_copy.keys())

                undesired_edges = find_undesired_edges(all_nodes, pos_copy, original_edges)
                if len(undesired_edges) != 0:
                    angle, _ = find_angle_orient(directions[current_branch_id], orientation, node1, node2, shared_node, coords)

                    pos_to_rotate, pivot_node = bow_tie_to_rotate(undesired_edges, new_pos, new_pyramids, shared_node, pos_copy, all_nodes, coords)

                    new_pos.update({shared_node: pos[shared_node]})
                    rotated_pos = rotate_helper(pos_to_rotate, angle, pos_copy[pivot_node])
                    new_pos.update(rotated_pos)
                    undesired_edges = []

                pos.update(new_pos)

                # Distinguishing the two new directions
                dir_a, dir_b = list(new_dirs.values())[1:3]   
                keep_dir, new_dir = (dir_a, dir_b) if dir_a == directions[current_branch_id] else (dir_b, dir_a)

                directions[current_branch_id] = keep_dir
                branch_id_counter += 1
                new_branch_id = str(branch_id_counter)
                directions[new_branch_id] = new_dir

                # Assign each pyramid to the right branch
                same_idx, other_idx = ((idx1, idx2) if new_dirs[apex_of(pyr1, coords)] == keep_dir else (idx2, idx1))
                pyr_by_branch.setdefault(current_branch_id, []).append(same_idx)
                pyr_by_branch.setdefault(new_branch_id    , []).append(other_idx)
                pyr_to_revisit.append(other_idx)

                # Assigning node1 and node2 for next iteration
                same_pyr      = pyramids[same_idx]
                same_apex     = apex_of(same_pyr, coords)
                node1, node2  = [n for n in same_pyr if n not in (same_apex, shared_node)]

            # ------------------------------------------------------------------------
            # Case 2: If the star node is a four-star node, so shared by four pyramids
            # ------------------------------------------------------------------------
            elif star_nodes[shared_node] == 4:
                # New pyramids
                idx1, idx2, idx3 = (i for i in idx_candidates if i != cur_pyr)
                pyr1, pyr2, pyr3 = pyramids[idx1], pyramids[idx2], pyramids[idx3]

                # Calling the four-star node tree function
                new_pos, _, new_apex, current_idx, new_directions = four_star_node_tree(start_pyr, pos, pyr1, pyr2, pyr3, coords, directions[current_branch_id], current_idx)
                apex_list += [apex for apex in new_apex if apex != cur_apex]

                # Updating the embedded pyramids with the new indices
                for offset, idx in zip(range(1, 4), [idx1, idx2, idx3]):
                    next_key = str(cur_idx + offset)
                    embedded_pyramids[next_key] = idx

                # --------------------------------------------------------------------------------
                # Checking if the new position is valid, i.e. if it doesn't create undesired edges
                # --------------------------------------------------------------------------------
                pyramids_list = [pyramids[i] for i in embedded_pyramids.values()]
                original_edges = list(chain.from_iterable(combinations(pyr, 2) for pyr in pyramids_list))

                pos_copy = pos.copy()
                pos_copy.update(new_pos)
                all_nodes = set(pos_copy.keys())

                undesired_edges = find_undesired_edges(all_nodes, pos_copy, original_edges)
                if len(undesired_edges) != 0:
                    angle, _ = find_angle_orient(directions[current_branch_id], orientation, node1, node2, shared_node, coords)

                    pos_to_rotate, pivot_node = bow_tie_to_rotate(undesired_edges, new_pos, new_pyramids, shared_node, pos_copy, all_nodes, coords)

                    new_pos.update({shared_node: pos[shared_node]})
                    rotated_pos = rotate_helper(pos_to_rotate, angle, pos_copy[pivot_node])
                    new_pos.update(rotated_pos)
                    undesired_edges = []

                pos.update(new_pos)

                # Assigning the directions to the corresponding branches
                values = list(new_directions.values())
                directions[current_branch_id] = next(value for value in values if value == directions[current_branch_id])

                for idx, pyr in zip((idx1, idx2, idx3), (pyr1, pyr2, pyr3)):
                    apex = next(n for n in pyr if coords[n][2] != 0)
                    dir_val = new_directions[apex]

                    if dir_val == directions[current_branch_id]:
                        continue

                    branch_id_counter += 1
                    new_branch_id = str(branch_id_counter)
                    directions[new_branch_id] = dir_val

                    pyr_by_branch.setdefault(new_branch_id, []).append(idx)

                # Identifying the pyramids to revisit
                idx_to_not_revisit = next(idx for idx in (idx1, idx2, idx3) if new_directions[next(n for n in pyramids[idx] if coords[n][2] != 0)] == directions[current_branch_id])
                idx_to_revisit = [idx for idx in (idx1, idx2, idx3) if idx != idx_to_not_revisit]
                pyr_to_revisit.extend(idx_to_revisit)

                pyr_by_branch.setdefault(current_branch_id, []).append(idx_to_not_revisit)

                # Assigning node1 and node2
                same_branch_pyramid = pyramids[idx_to_not_revisit]
                same_branch_apex    = next(n for n in same_branch_pyramid if coords[n][2] != 0)
                node1, node2 = [n for n in same_branch_pyramid if n != same_branch_apex and n != shared_node]

            # ------------------------------------------------------------------
            # Case 3: If the star node is a N-star node, so shared by N pyramids
            # ------------------------------------------------------------------
            elif star_nodes[shared_node] > 4:
                new_idx = [i for i in idx_candidates if i != cur_pyr]
                new_pyramids = [pyramids[i] for i in new_idx]

                # Calling the N-star node tree function
                new_pos, _, new_apex, current_idx, new_directions = N_star_node_tree(start_pyr, pos, new_pyramids.copy(), coords, directions[current_branch_id], current_idx)
                apex_list += [apex for apex in new_apex if apex != cur_apex]

                # Updating the embedded pyramids with the new indices
                for offset, idx in zip(range(1, len(new_idx) + 1), new_idx):
                    next_key = str(cur_idx + offset)
                    embedded_pyramids[next_key] = idx

                # --------------------------------------------------------------------------------
                # Checking if the new position is valid, i.e. if it doesn't create undesired edges
                # --------------------------------------------------------------------------------
                pyramids_list = [pyramids[i] for i in embedded_pyramids.values()]
                original_edges = list(chain.from_iterable(combinations(pyr, 2) for pyr in pyramids_list))

                pos_copy = pos.copy()
                pos_copy.update(new_pos)
                all_nodes = set(pos_copy.keys())

                undesired_edges = find_undesired_edges(all_nodes, pos_copy, original_edges)
                if len(undesired_edges) != 0:
                    angle, _ = find_angle_orient(directions[current_branch_id], orientation, node1, node2, shared_node, coords)

                    pos_to_rotate, pivot_node = bow_tie_to_rotate(undesired_edges, new_pos, new_pyramids, shared_node, pos_copy, all_nodes, coords)

                    new_pos.update({shared_node: pos[shared_node]})
                    rotated_pos = rotate_helper(pos_to_rotate, angle, pos_copy[pivot_node])
                    new_pos.update(rotated_pos)
                    undesired_edges = []

                pos.update(new_pos)

                # Remove the "Old" direction
                first_key = next(iter(new_directions))
                del new_directions[first_key]

                values = list(new_directions.values())
                directions[current_branch_id] = values[0]

                # Assigning the directions to the corresponding branches
                for idx, pyr in zip(new_idx[1:], new_pyramids[1:]):
                    apex = next(n for n in pyr if coords[n][2] != 0)
                    dir_val = new_directions[apex]

                    branch_id_counter += 1
                    new_branch_id = str(branch_id_counter)
                    directions[new_branch_id] = dir_val

                    pyr_by_branch.setdefault(new_branch_id, []).append(idx)

                # Identifying the pyramids to revisit
                idx_to_not_revisit = new_idx[0]
                idx_to_revisit = new_idx[1:]
                pyr_to_revisit.extend(idx_to_revisit)
                pyr_by_branch.setdefault(current_branch_id, []).append(idx_to_not_revisit)

                # Assigning node1 and node2
                same_branch_pyramid = pyramids[idx_to_not_revisit]
                same_branch_apex    = next(n for n in same_branch_pyramid if coords[n][2] != 0)
                node1, node2        = [n for n in same_branch_pyramid if n != same_branch_apex and n != shared_node]
            
        next_idx = idx_candidates[0] if idx_candidates[0] != cur_pyr else idx_candidates[1]
        next_idx = next_idx if next_idx not in visited else None

        if next_idx is None:
            raise ValueError("Next pyramid could not be found!")

        if not extremal_pyramids and next_idx is None:
            directions = {i: None for i in directions}
            continue

        next_pyr    = pyramids[next_idx]
        next_apex   = next(n for n in next_pyr if coords[n][2] != 0)
        next_coords = restrict_coords_to_pyramid(next_pyr, coords)
        
        if (not first_connect_done) and len(available_shared_nodes) == 1:
            helper_fn, orientation = pick_bow_tie_helper(shared_node, rt, rb, lt, lb, start_apex, next_apex, coords)
            new_pos, _, node1, node2, current_idx = helper_fn(shared_node, pos[shared_node], next_pyr, next_coords, current_idx)

            pos.update(new_pos)
            apex_list.append(next_apex)

            next_key = str(cur_idx + 1)
            embedded_pyramids[next_key] = next_idx
            directions[current_branch_id] = orientation 
            first_connect_done = True    
            shared_nodes_embedded.append(shared_node)
            pyr_by_branch.setdefault(current_branch_id, []).append(next_idx)

            visited.add(next_idx)    

        if (not first_connect_done) and len(available_shared_nodes) == 2:
            helper_fn, orientation = pick_bow_tie_helper(shared_node, rt, rb, lt, lb, start_apex, next_apex, coords)
            new_pos, _, node1, node2, current_idx = helper_fn(shared_node, pos[shared_node], next_pyr, next_coords, current_idx)

            pos.update(new_pos)
            apex_list.append(next_apex)

            next_key = str(cur_idx + 1)
            embedded_pyramids[next_key] = next_idx
            directions[current_branch_id] = orientation 
            first_connect_done = True    
            shared_nodes_embedded.append(shared_node)
            pyr_by_branch.setdefault(current_branch_id, []).append(next_idx)

            visited.add(next_idx)

        elif first_connect_done and len(available_shared_nodes) == 1:
            if node1 is not None and node2 is not None and shared_node in (node1, node2):
                helper_fn, orientation = find_case(directions[current_branch_id], node1, node2, cur_apex, next_apex, shared_node, coords)
                new_pos, _, node1, node2, current_idx = helper_fn(shared_node, pos[shared_node], next_pyr, next_coords, current_idx)

                next_key = str(cur_idx + 1)
                embedded_pyramids[next_key] = next_idx

                # --------------------------------------------------------------------------------
                # Checking if the new position is valid, i.e. if it doesn't create undesired edges
                # --------------------------------------------------------------------------------
                pyramids_list = [pyramids[i] for i in embedded_pyramids.values()]
                original_edges = list(chain.from_iterable(combinations(pyr, 2) for pyr in pyramids_list))

                pos_copy = pos.copy()
                pos_copy.update(new_pos)
                all_nodes = set(pos_copy.keys())

                undesired_edges = find_undesired_edges(all_nodes, pos_copy, original_edges)
                if len(undesired_edges) != 0:
                    angle, axis = find_angle_orient(directions[current_branch_id], orientation, node1, node2, shared_node, coords)

                    new_pos.update({shared_node: pos[shared_node]})
                    new_pos = rotate_helper(new_pos, angle, pos[shared_node])
                    new_pos = reflect_positions(new_pos, shared_node, axis)
                    undesired_edges = []

                pos.update(new_pos)
                apex_list.append(next_apex)

                directions[current_branch_id] = orientation
                shared_nodes_embedded.append(shared_node)

                pyr_by_branch.setdefault(current_branch_id, []).append(next_idx)
                visited.add(next_idx)

        elif first_connect_done and len(available_shared_nodes) == 2:
            pyr_to_revisit.append(cur_idx)
            shared_a, shared_b = available_shared_nodes
            idx_a, _, apex_a = other_pyramid_and_apex(node_to_pyrs, shared_a, cur_pyr, pyramids, coords)
            idx_b, _, apex_b = other_pyramid_and_apex(node_to_pyrs, shared_b, cur_pyr, pyramids, coords)

            branch_id_counter += 1
            new_branch_id = str(branch_id_counter)

            if directions[current_branch_id] in ('right', 'left'):

                if compute_abs_dy(apex_a, cur_apex, coords) > compute_abs_dy(apex_b, cur_apex, coords):

                    directions[new_branch_id] = 'up' if pos[shared_a][1] > pos[shared_b][1] else 'down'
                    pyr_by_branch.setdefault(new_branch_id, []).append(idx_a)
                    pyr_by_branch.setdefault(current_branch_id, []).append(idx_b)
                    next_idx = idx_b
                    shared_node = shared_b                

                elif compute_abs_dy(apex_a, cur_apex, coords) < compute_abs_dy(apex_b, cur_apex, coords):

                    directions[new_branch_id] = 'up' if pos[shared_b][1] > pos[shared_a][1] else 'down'
                    pyr_by_branch.setdefault(new_branch_id, []).append(idx_b)
                    pyr_by_branch.setdefault(current_branch_id, []).append(idx_a)
                    next_idx = idx_a
                    shared_node = shared_a

            elif directions[current_branch_id] in ('up', 'down'):

                if compute_abs_dx(apex_a, cur_apex, coords) > compute_abs_dx(apex_b, cur_apex, coords):

                    directions[new_branch_id] = 'right' if pos[shared_a][0] > pos[shared_b][0] else 'left'
                    pyr_by_branch.setdefault(new_branch_id, []).append(idx_a)
                    pyr_by_branch.setdefault(current_branch_id, []).append(idx_b)
                    next_idx = idx_b
                    shared_node = shared_b

                elif compute_abs_dx(apex_a, cur_apex, coords) < compute_abs_dx(apex_b, cur_apex, coords):

                    directions[new_branch_id] = 'right' if pos[shared_b][0] > pos[shared_a][0] else 'left'
                    pyr_by_branch.setdefault(new_branch_id, []).append(idx_b)
                    pyr_by_branch.setdefault(current_branch_id, []).append(idx_a)
                    next_idx = idx_a
                    shared_node = shared_a

            next_idx    = next_idx if next_idx not in visited else None
            next_pyr    = pyramids[next_idx]
            next_apex   = next(n for n in next_pyr if coords[n][2] != 0)
            next_coords = restrict_coords_to_pyramid(next_pyr, coords)

            helper_fn, orientation = find_case(directions[current_branch_id], node1, node2, cur_apex, next_apex, shared_node, coords)
            new_pos, _, node1, node2, current_idx = helper_fn(shared_node, pos[shared_node], next_pyr, next_coords, current_idx)

            next_key = str(cur_idx + 1)
            embedded_pyramids[next_key] = next_idx

            # --------------------------------------------------------------------------------
            # Checking if the new position is valid, i.e. if it doesn't create undesired edges
            # --------------------------------------------------------------------------------
            pyramids_list = [pyramids[i] for i in embedded_pyramids.values()]
            original_edges = list(chain.from_iterable(combinations(pyr, 2) for pyr in pyramids_list))

            pos_copy = pos.copy()
            pos_copy.update(new_pos)
            all_nodes = set(pos_copy.keys())

            undesired_edges = find_undesired_edges(all_nodes, pos_copy, original_edges)
            if len(undesired_edges) != 0:
                angle, axis = find_angle_orient(directions[current_branch_id], orientation, node1, node2, shared_node, coords)

                new_pos.update({shared_node: pos[shared_node]})
                new_pos = rotate_helper(new_pos, angle, pos[shared_node])
                new_pos = reflect_positions(new_pos, shared_node, axis)
                undesired_edges = []

            pos.update(new_pos)
            apex_list.append(next_apex)

            directions[current_branch_id] = orientation
            shared_nodes_embedded.append(shared_node)
            visited.add(next_idx)
            node_to_save.append(shared_node)

            continue
    
        # ------------------------------------------------------------
        # Update the equivalent positions if needed (needed for loops)
        # ------------------------------------------------------------
        for node in pos_copy_for_loop:
            if node in pos and pos[node] != pos_copy_for_loop[node]:
                equivalent_positions[node] = pos_copy_for_loop[node]


    # -----------------------------------------
    # Restore missing edges from unclosed loops
    # -----------------------------------------
    for i, (key, (x, y)) in enumerate(equivalent_positions.items(), start = current_idx + 1):
        aux_nodes = []
        new_node = f"W{i}"
        pos[new_node] = (x, y)
        aux_nodes.append(new_node)
        current_idx += 1

        new_pos, current_idx, _ = restore_edges(key, new_node, pos, current_idx, is_odd = True)
        pos.update(new_pos)

    all_nodes = sorted(set(pos.keys()))

    return pos, all_nodes, apex_list, current_idx

