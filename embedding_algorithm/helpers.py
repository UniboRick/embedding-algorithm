"""This module contains helper functions for geometric transformations, node search and connectivity checks. These functions are used internally by the main embedding functions."""

def rotate_helper(pos, angle_deg, pivot = None):
    """
    Rotate a set of 2D points around a pivot.

    Parameters
    ----------
    pos : dict
        Dictionary mapping node identifiers to 2D coordinates, e.g.
        {id: (x, y), ...}.
    angle_deg : float
        Rotation angle in degrees. Positive values correspond to a
        counterclockwise rotation.
    pivot : tuple of float, optional
        Point (px, py) around which the coordinates are rotated.
        If None (default), the geometric center of all points in pos
        is used as the pivot.

    Returns
    -------
    dict
        A new dictionary with the same keys as pos, where each coordinate
        has been rotated by angle_deg degrees around pivot.

    Notes
    -----
    - The rotation is performed using the standard 2D rotation matrix.
    - The function does not modify the input dictionary; it returns
      a new one with rotated coordinates.
    """

    import math

    if pivot is None:
        xs, ys = zip(*pos.values())
        pivot = (sum(xs)/len(xs), sum(ys)/len(ys))

    cx, cy   = pivot
    theta    = math.radians(angle_deg)
    ct,  st  = math.cos(theta), math.sin(theta)

    new_pos = {}
    for n, (x, y) in pos.items():
        x0, y0 = x - cx, y - cy
        new_pos[n] = (x0*ct - y0*st + cx,
                      x0*st + y0*ct + cy)
    return new_pos

def translate_helper(target_pos, attachment_node, x_to_reach, y_to_reach):
    """
    Translate a subset of 2D points so that a set chosen nodes reaches a target position.

    Parameters
    ----------
    target_pos : dict
        Dictionary mapping node identifiers to 2D coordinates, e.g.
        {id: (x, y), ...}. This dictionary is modified in place.
    attachment_node : hashable
        Identifier of the node whose position will be moved to
        (x_to_reach, y_to_reach).
    x_to_reach : float
        Target x-coordinate for the attachment node.
    y_to_reach : float
        Target y-coordinate for the attachment node.

    Returns
    -------
    dict
        The same dictionary target_pos with updated coordinates after
        applying the translation.

    Notes
    -----
    - The translation vector is computed as the difference between the
      desired target point and the current position of attachment_node.
    - All nodes in target_nodes are shifted by the same translation vector.
    - The input dictionary is modified in place and also returned.
    """

    dx = x_to_reach - target_pos[attachment_node][0]
    dy = y_to_reach - target_pos[attachment_node][1]
    target_nodes = list(target_pos.keys())

    for node in target_nodes:
        target_pos[node] = (target_pos[node][0] + dx, target_pos[node][1] + dy)

    return target_pos

def swap_helper(pos, mapping):
    """
    Swap node identifiers in a coordinate dictionary according to a mapping.

    Parameters
    ----------
    pos : dict
        Dictionary mapping node identifiers to 2D coordinates, e.g.
        {id: (x, y), ...}.
    mapping : dict
        Dictionary specifying pairs of node identifiers to be swapped.
        Each key-value pair (a, b) means that a becomes b and
        b becomes a. The mapping is considered symmetric.

    Returns
    -------
    tuple of dict
        - pos_new: a new dictionary with updated node identifiers.
        - swapped_nodes: a dictionary mapping each original identifier
          to its new identifier.

    Notes
    -----
    - A full symmetric mapping is constructed internally, ensuring that
      swaps apply bidirectionally.
    - If the resulting mapping assigns two nodes to the same identifier,
      a ValueError is raised.
    - The input dictionary pos is not modified; a new one is returned.
    """

    full_map = {}
    for a, b in mapping.items():
        full_map[a] = b
        full_map[b] = a

    pos_new = {}
    swapped_nodes = {}

    for old_id, coord in pos.items():
        new_id = full_map.get(old_id, old_id)

        if new_id in pos_new:
            raise ValueError(f"Duplicate target label {new_id!r}")

        pos_new[new_id] = coord
        swapped_nodes[old_id] = new_id 

    return pos_new, swapped_nodes

def shift_helper(target, coords, direction, start_idx = 0):
    """
    Create a positional shift of a target node by inserting temporary auxiliary
    nodes and swapping them with the target.

    Parameters
    ----------
    target : hashable
        Identifier of the node to be shifted.
    coords : dict
        Dictionary mapping node identifiers to 2D coordinates,
        e.g. {id: (x, y), ...}. This dictionary is copied internally.
    direction : {'right', 'left', 'up', 'down'}
        Direction in which the target node should be shifted.
        The shift is implemented by creating auxiliary nodes and then
        swapping one of them with the target.
    start_idx : int, optional
        Starting index used for naming the auxiliary nodes. New node labels
        will be W{start_idx+1}, W{start_idx+2}, etc.

    Returns
    -------
    tuple
        - pos: dictionary with updated coordinates after the shift.
        - all_nodes: set of all node identifiers present after insertion
          of auxiliary nodes.
        - current_idx: integer index marking the next available suffix
          for future auxiliary node creation.

    Notes
    -----
    - Two auxiliary nodes are created with prefix 'W' and placed in
      positions determined by the chosen shift direction.
    - One auxiliary node swaps labels with the target node, effectively
      moving the target to the shifted coordinate.
    - The input dictionary is not modified; a new one is returned.
    """

    pos = dict(coords) 
    start_x, start_y = pos[target]

    aux_prefix       = "W"  
    num_aux_nodes    = 2
    aux_nodes_full   = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    aux_nodes_sorted = sorted(aux_nodes_full, key = lambda x: int(x[1:]))

    if direction == "right":
        pos[aux_nodes_sorted[1]] = (start_x + 1, start_y)
        pos[aux_nodes_sorted[0]] = (start_x + 2, start_y)
    elif direction == "left":
        pos[aux_nodes_sorted[1]] = (start_x - 1, start_y)
        pos[aux_nodes_sorted[0]] = (start_x - 2, start_y)
    elif direction == "up":
        pos[aux_nodes_sorted[1]] = (start_x, start_y + 1)
        pos[aux_nodes_sorted[0]] = (start_x, start_y + 2)
    elif direction == "down":
        pos[aux_nodes_sorted[1]] = (start_x, start_y - 1)
        pos[aux_nodes_sorted[0]] = (start_x, start_y - 2)

    pos, _ = swap_helper(pos, {aux_nodes_sorted[0]: target})
    all_nodes = set(pos.keys())
    current_idx = start_idx + num_aux_nodes

    return pos, all_nodes, current_idx

def unit_disk_edges(nodes, pos, radius = 0.525):
    """
    Construct a Unit Disk Graph (UDG) from a set of nodes and their 2D coordinates.

    Parameters
    ----------
    nodes : iterable
        Collection of node identifiers to include in the graph.
    pos : dict
        Dictionary mapping node identifiers to 2D coordinates,
        e.g. {id: (x, y), ...}.
    radius : float, optional
        Distance threshold used to determine adjacency.
        An undirected edge is added between two nodes if the Euclidean
        distance between their coordinates is less than or equal to radius.
    Returns
    -------
    networkx.Graph
        A graph where nodes are connected according to the unit disk rule.

    Notes
    -----
    - The function creates a new NetworkX graph.
    - All possible node pairs are checked for adjacency.
    - The input coordinate dictionary is not modified.
    """

    import itertools
    import math
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u,v in itertools.combinations(nodes, 2):
        if math.dist(pos[u], pos[v]) <= 2 * radius:
            G.add_edge(u,v)

    return G

def even_W_path_exists(G, u, v):
    """
    Determine whether there exists a parity-even path from u to v 
    using only auxiliary nodes whose labels start with 'W'.

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph in which the search is performed.
    u : hashable
        Starting node identifier.
    v : hashable
        Target node identifier.

    Returns
    -------
    tuple
        - bool: True if an allowed path exists, False otherwise.
        - list or None: The sequence of intermediate 'W' nodes 
          forming the path, or None if no valid path exists.

    Notes
    -----
    - If u and v are directly connected by an edge, the function 
      returns True with an empty path.
    - Paths are constrained to move only through nodes whose labels start 
      with 'W'.
    - The algorithm tracks parity (even/odd depth) to ensure the path has 
      even parity when reaching v.
    - Breadth-first search (BFS) is used, extending states with explicit 
      parity tracking.
    """

    import collections

    if G.has_edge(u, v):
        return True, []

    queue   = collections.deque([(u, 0, [])])
    visited = {(u, 0)}

    while queue:
        node, parity, path = queue.popleft()

        for nbr in G.neighbors(node):
            # 1. reached the target
            if nbr == v:
                if parity == 0:                  
                    return True, path
                continue                          

            if not nbr.startswith("W"):
                continue

            new_parity = 1 - parity               
            state      = (nbr, new_parity)

            if state not in visited:
                visited.add(state)
                queue.append((nbr, new_parity, path + [nbr]))

    return False, None

def topology_check(original_graph, new_graph):
    """
    Verify that the 2D embedded graph correctly realises all edges of 
    the original 3D graph via valid connectivity paths.

    Parameters
    ----------
    original_graph : networkx.Graph
        The reference graph whose edges must be topologically realised.
    new_graph : networkx.Graph
        The graph representing the 2D embedding, potentially containing 
        auxiliary 'W' nodes that mediate valid paths.

    Returns
    -------
    tuple
        - bool: True if all edges of the original graph are correctly 
          realised in the embedded graph, False otherwise.
        - list of tuple: The list of edges (u, v) that are not 
          realised correctly. Empty list if the topology is valid.

    Notes
    -----
    - For each edge in original_graph, the function checks for an 
      allowable connection in new_graph using even_W_path_exists.
    - All edges are examined; the function does not stop after the first 
      failure.
    - No output is printed; all information is conveyed through the 
      returned boolean and the list of missing edges.
    """
    
    missing_edges = []

    for u, v in original_graph.edges():
        ok, _ = even_W_path_exists(new_graph, u, v)
        if not ok:
            missing_edges.append((u, v))

    all_ok = (len(missing_edges) == 0)
    return all_ok, missing_edges

def pyramid_with(node, choices):
    """
    Retrieve the pyramid structure containing a given node.

    Parameters
    ----------
    node : hashable
        Node identifier to search for within the set of pyramid choices.
    choices : iterable
        Collection of pyramid descriptors. Each element must be a structure
        (e.g., tuple, list, or custom object) that supports membership tests
        via node in p.

    Returns
    -------
    object
        The first pyramid descriptor from choices that contains node.

    Notes
    -----
    - The function scans all pyramid descriptors in choices and returns
      the first one whose contents include node.
    - If no pyramid contains the specified node, a ValueError is raised.
    """

    try:
        return next(p for p in choices if node in p)
    except StopIteration:                     
        raise ValueError(f"No pyramid contains node {node!r}")

def restrict_coords_to_pyramid(pyramid, coords):
    """
    Extract the coordinate dictionary restricted to the nodes of a given pyramid.

    Parameters
    ----------
    pyramid : iterable
        Collection of node identifiers defining the pyramid.
    coords : dict
        Dictionary mapping node identifiers to 2D or 3D coordinates,
        e.g. {id: (x, y), ...}.

    Returns
    -------
    dict
        A dictionary containing only the coordinates of the nodes
        listed in pyramid.

    Notes
    -----
    - The function assumes that every node in pyramid exists in coords.
    - The returned dictionary is a new object and does not modify coords.
    """        

    return {node: coords[node] for node in pyramid}

def compute_abs_dy(node1, node2, coords):
    """
    Compute the absolute vertical distance between two nodes.

    Parameters
    ----------
    node1 : hashable
        First node identifier.
    node2 : hashable
        Second node identifier.
    coords : dict
        Dictionary mapping node identifiers to 2D coordinates,
        e.g. {id: (x, y), ...}.

    Returns
    -------
    float
        Absolute difference between the y-coordinates of the two nodes.
    """

    return abs(coords[node1][1] - coords[node2][1])
    
def compute_abs_dx(node1, node2, coords):
    """
    Compute the absolute horizontal distance between two nodes.

    Parameters
    ----------
    node1 : hashable
        First node identifier.
    node2 : hashable
        Second node identifier.
    coords : dict
        Dictionary mapping node identifiers to 2D coordinates,
        e.g. {id: (x, y), ...}.

    Returns
    -------
    float
        Absolute difference between the x-coordinates of the two nodes.
    """
    
    return abs(coords[node1][0] - coords[node2][0])

def compute_dy(node1, node2, pos):
    """
    Compute the signed vertical displacement from node2 to node1.

    Parameters
    ----------
    node1 : hashable
        Node whose coordinate is used as the minuend.
    node2 : hashable
        Node whose coordinate is used as the subtrahend.
    pos : dict
        Dictionary mapping node identifiers to 2D coordinates.

    Returns
    -------
    float
        Signed difference pos[node1].y - pos[node2].y.
    """

    return pos[node1][1] - pos[node2][1]
    
def compute_dx(node1, node2, pos):
    """
    Compute the signed horizontal displacement from node2 to node1.

    Parameters
    ----------
    node1 : hashable
        Node whose coordinate is used as the minuend.
    node2 : hashable
        Node whose coordinate is used as the subtrahend.
    pos : dict
        Dictionary mapping node identifiers to 2D coordinates.

    Returns
    -------
    float
        Signed difference pos[node1].x - pos[node2].x.
    """

    return pos[node1][0] - pos[node2][0]

def base_nodes(pyr, apex_node):
    """
    Extract the base nodes of a pyramid by removing its apex node.

    Parameters
    ----------
    pyr : iterable
        Collection of node identifiers that form the full pyramid.
    apex_node : hashable
        Node identifier corresponding to the pyramid's apex.

    Returns
    -------
    set
        Set of nodes in pyr that are not part of apex_node.

    Notes
    -----
    - The function performs a simple set difference: all nodes in pyr
      except those listed in apex_node.
    - The returned set is a new object and does not modify the inputs.
    """

    return {n for n in pyr if n not in apex_node}

def pick_bow_tie_helper(shared_node, rt, rb, lt, lb, left_apex, right_apex, coords):
    """
    Select the appropriate bow-tie transformation helper based on geometry
    and on which tip of the bow-tie is shared with the next structure.

    Parameters
    ----------
    shared_node : hashable
        The node that is shared with the adjacent bow-tie.
    rt : hashable
        Right-top tip node of the current bow-tie.
    rb : hashable
        Right-bottom tip node of the current bow-tie.
    lt : hashable
        Left-top tip node of the current bow-tie.
    lb : hashable
        Left-bottom tip node of the current bow-tie.
    left_apex : hashable
        Apex node on the left side of the bow-tie.
    right_apex : hashable
        Apex node on the right side of the bow-tie.
    coords : dict
        Dictionary mapping node identifiers to 3D coordinates,
        e.g. {id: (x, y, z), ...}.

    Returns
    -------
    tuple
        A pair (helper_function, direction_string) indicating which
        transformation helper to use and in which direction the next
        structure should be attached.

    Notes
    -----
    - The function first determines the side (left/right) and level 
      (top/bottom) of the shared node relative to the bow-tie structure.
    - It then decides whether the bow-tie should be treated as vertically 
      or horizontally oriented, based on the absolute displacement between 
      the two apex nodes.
    - The combination (side, level, vertical) is used as a key in a 
      lookup table mapping to the correct transformation helper.
    - A ValueError is raised if shared_node is not one of the four
      tips of the bow-tie.
    """

    # Locate the anchor on the current bow-tie
    if   shared_node == rt: side, lvl = "right", "top"
    elif shared_node == rb: side, lvl = "right", "bottom"
    elif shared_node == lt: side, lvl = "left" , "top"
    elif shared_node == lb: side, lvl = "left" , "bottom"
    else:
        raise ValueError("The shared node is not on any tip of the bow-tie")

    # Decide vertical vs horizontal from geometry
    vertical = compute_abs_dy(left_apex, right_apex, coords) > compute_abs_dx(left_apex, right_apex, coords)

    # Map (side, lvl, vertical) → helper + direction 
    table = {
            ("right", "top",    False): (top_right_horizontal_helper,    "right"),
            ("right", "bottom", False): (bottom_right_horizontal_helper, "right"),
            ("left",  "top",    False): (top_left_horizontal_helper,     "left"),
            ("left",  "bottom", False): (bottom_left_horizontal_helper,  "left"),
            ("right", "top",    True):  (top_right_vertical_helper,    "up"),
            ("left",  "top",    True):  (top_left_vertical_helper,     "up"),
            ("right", "bottom", True):  (bottom_right_vertical_helper, "down"),
            ("left",  "bottom", True):  (bottom_left_vertical_helper,  "down"),
            }

    return table[(side, lvl, vertical)]

def find_case(direction, node1, node2, cur_apex, next_apex, shared_node, coords):
    """
    Determine the correct transformation helper for attaching a new structure
    given the direction of attachment and the relative position of the shared node.

    Parameters
    ----------
    direction : {'right', 'left', 'up', 'down'}
        Direction in which the next structure will be attached.
    node1 : hashable
        First endpoint of the side connecting the two structures.
    node2 : hashable
        Second endpoint of the side connecting the two structures.
    cur_apex : hashable
        Apex node of the current structure.
    next_apex : hashable
        Apex node of the next structure to be attached.
    shared_node : hashable
        The node shared between the current structure and the next one.
    coords : dict
        Dictionary mapping node identifiers to 3D coordinates,
        e.g. {id: (x, y, z), ...}.

    Returns
    -------
    tuple
        A pair (helper_function, direction_string) specifying which
        helper function should be used and in which direction the shift
        should be applied.

    Notes
    -----
    - Based on direction, the function determines whether the shared
      node corresponds to the top/bottom or left/right part of the side
      spanned by node1 and node2.
    - It classifies the current configuration as vertically or horizontally
      oriented by comparing the absolute y- and x-displacements between the
      two apex nodes.
    - A lookup table maps the tuple (side, level, vertical) to the
      appropriate helper and direction.
    - The function assumes that shared_node matches one of the inferred
      side/level cases.
    """

    if direction == "right":
        lower_node, higher_node = ((node1, node2) if coords[node1][1] <= coords[node2][1] else (node2, node1))
        if   shared_node == higher_node: side, lvl = "right", "top"
        elif shared_node == lower_node:  side, lvl = "right", "bottom"
        
    elif direction == "left":
        lower_node, higher_node = ((node1, node2) if coords[node1][1] <= coords[node2][1] else (node2, node1))
        if   shared_node == higher_node: side, lvl = "left", "top"
        elif shared_node == lower_node:  side, lvl = "left", "bottom"

    elif direction == "up":
        left_node, right_node = ((node1, node2) if coords[node1][0] <= coords[node2][0] else (node2, node1))            
        if   shared_node == left_node:  side, lvl = "left", "top"
        elif shared_node == right_node: side, lvl = "right", "top"

    elif direction == "down":
        left_node, right_node = ((node1, node2) if coords[node1][0] <= coords[node2][0] else (node2, node1))            
        if   shared_node == left_node:  side, lvl = "left", "bottom"
        elif shared_node == right_node: side, lvl = "right", "bottom"
        
    vertical = compute_abs_dy(cur_apex, next_apex, coords) > compute_abs_dx(cur_apex, next_apex, coords)

    table = {
                ("right", "top",    False): (top_right_horizontal_helper,    "right"),
                ("right", "bottom", False): (bottom_right_horizontal_helper, "right"),
                ("left",  "top",    False): (top_left_horizontal_helper,     "left"),
                ("left",  "bottom", False): (bottom_left_horizontal_helper,  "left"),
                ("right", "top",    True):  (top_right_vertical_helper,      "up"),
                ("left",  "top",    True):  (top_left_vertical_helper,       "up"),
                ("right", "bottom", True):  (bottom_right_vertical_helper,   "down"),
                ("left",  "bottom", True):  (bottom_left_vertical_helper,    "down"),
            }

    return table[(side, lvl, vertical)]

def complete_bow_tie(start_node1, start_node2, start_pos, apex, new_node, new_nodes_coords, orientation, start_idx = 0):
    """
    Complete a bow-tie structure by adding one new node, one apex node and 
    two auxiliary nodes, using two already-embedded nodes as anchors.

    Parameters
    ----------
    start_node1 : hashable
        First node already embedded in 2D, forming one endpoint of the base.
    start_node2 : hashable
        Second node already embedded in 2D, forming the other endpoint
        of the base.
    start_pos : dict
        Dictionary mapping existing node identifiers to their 2D coordinates,
        e.g. {id: (x, y), ...}.
    apex : hashable
        Identifier of the apex node to be added.
    new_node : hashable
        Identifier of the second new node to be added (the non-apex new node).
    new_nodes_coords : dict
        Dictionary mapping the 3D coordinates of the new nodes, used only
        to determine which of the two new nodes lies “above/below” or
        “left/right” relative to the apex.
    orientation : {'right', 'left', 'up', 'down'}
        Direction from the starting nodes in which the new bow-tie is built.
    start_idx : int, optional
        Starting index used to generate unique labels for auxiliary nodes
        (prefix "W"). Two auxiliary nodes are created.

    Returns
    -------
    tuple
        - new_pos : dict  
          Dictionary mapping the newly added nodes (auxiliary nodes, new node,
          and apex) to their new 2D coordinates.
        - new_nodes : set  
          Set containing the identifiers of all newly created nodes.
        - apex : hashable  
          The apex node added to the graph (returned unchanged for convenience).
        - current_idx : int  
          Updated index after consuming two auxiliary node labels.

    Notes
    -----
    - This function completes a bow-tie when two base nodes already exist
      in 2D, so only one new node + one apex (and auxiliary nodes) must be
      positioned.
    - Two auxiliary nodes are placed first; then new_node and apex are
      positioned at offsets determined by the bow-tie geometry.
    - The relative vertical or horizontal ordering between new_node and
      apex is inferred from their 3D coordinates in new_nodes_coords.
    - The input start_pos is not modified; the function returns a new
      coordinate dictionary for the added nodes only.
    """

    import math
    s = math.sqrt(3) / 2
    current_idx = start_idx

    if orientation in ('right', 'left'):
        start_x = start_pos[start_node1][0]
        start_y = (start_pos[start_node1][1] + start_pos[start_node2][1]) / 2

    elif orientation in ('up', 'down'):
        start_x = (start_pos[start_node1][0] + start_pos[start_node2][0]) / 2
        start_y = start_pos[start_node1][1]

    else:
        raise ValueError("Invalid geometry. Check orientation and starting nodes")
    
    new_pos = {}

    aux_prefix    = "W"  
    num_aux_nodes = 2
    aux_nodes     = [f"{aux_prefix}{i}" for i in range(current_idx + 1, current_idx + num_aux_nodes + 1)]

    if orientation == 'right':
        new_pos[aux_nodes[0]] = (start_x + s, start_y)
        new_pos[aux_nodes[1]] = (start_x + s + 1, start_y)
        new_pos[new_node]     = (start_x + 2 * s + 1, start_y + 0.5) if new_nodes_coords[new_node][1] >= new_nodes_coords[apex][1] else (start_x + 2 * s + 1, start_y - 0.5)
        new_pos[apex]         = (start_x + 2 * s + 1, start_y + 0.5) if new_nodes_coords[apex][1] > new_nodes_coords[new_node][1] else (start_x + 2 * s + 1, start_y - 0.5)

    elif orientation == 'left':
        new_pos[aux_nodes[0]] = (start_x - s, start_y)
        new_pos[aux_nodes[1]] = (start_x - s - 1, start_y)
        new_pos[new_node]     = (start_x - 2 * s - 1, start_y + 0.5) if new_nodes_coords[new_node][1] >= new_nodes_coords[apex][1] else (start_x - 2 * s - 1, start_y - 0.5)
        new_pos[apex]         = (start_x - 2 * s - 1, start_y + 0.5) if new_nodes_coords[apex][1] > new_nodes_coords[new_node][1] else (start_x - 2 * s - 1, start_y - 0.5)

    elif orientation == 'up':
        new_pos[aux_nodes[0]] = (start_x, start_y + s)
        new_pos[aux_nodes[1]] = (start_x, start_y + s + 1)
        new_pos[new_node]     = (start_x + 0.5, start_y + 2 * s + 1) if new_nodes_coords[new_node][0] >= new_nodes_coords[apex][0] else (start_x - 0.5, start_y + 2 * s + 1)
        new_pos[apex]         = (start_x + 0.5, start_y + 2 * s + 1) if new_nodes_coords[apex][0] > new_nodes_coords[new_node][0] else (start_x - 0.5, start_y + 2 * s + 1)

    elif orientation == 'down':
        new_pos[aux_nodes[0]] = (start_x, start_y - s)
        new_pos[aux_nodes[1]] = (start_x, start_y - s - 1)
        new_pos[new_node]     = (start_x + 0.5, start_y - 2 * s - 1) if new_nodes_coords[new_node][0] >= new_nodes_coords[apex][0] else (start_x - 0.5, start_y - 2 * s - 1)
        new_pos[apex]         = (start_x + 0.5, start_y - 2 * s - 1) if new_nodes_coords[apex][0] > new_nodes_coords[new_node][0] else (start_x - 0.5, start_y - 2 * s - 1)

    else:
        raise ValueError("The nodes to be added don't match any expected configuration")
    
    new_nodes = set(new_pos.keys())
    current_idx += num_aux_nodes

    return new_pos, new_nodes, apex, current_idx

def orientation(apex, start_apex, coords):
    """
    Determine whether the relative placement of two apex nodes is
    predominantly vertical or horizontal.

    Parameters
    ----------
    apex : hashable
        Identifier of the first apex node.
    start_apex : hashable
        Identifier of the second apex node used as reference.
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z), ...}.

    Returns
    -------
    str
        "vertical" if the absolute y-displacement between the two nodes
        is greater than or equal to the absolute x-displacement;
        "horizontal" otherwise.

    Notes
    -----
    - The function uses the helper functions compute_abs_dy and
      compute_abs_dx for clarity and consistency.
    - The output describes the dominant geometric axis between the two nodes.
    """

    return ("vertical" if compute_abs_dy(apex, start_apex, coords) >= compute_abs_dx(apex, start_apex, coords) else "horizontal")

def reflect_positions(positions, pivot, axis = "vertical"):
    """
    Reflect a set of 2D coordinates across a vertical line, a horizontal line,
    or a point defined by a pivot node.

    Parameters
    ----------
    positions : dict
        Dictionary mapping node identifiers to 2D coordinates,
        e.g. {id: (x, y), ...}.
    pivot : hashable
        Node identifier whose coordinates define the reflection axis
        (vertical or horizontal) or the reflection point.
    axis : {'vertical', 'horizontal', 'point'}, optional
        Type of reflection:
        - "vertical": reflect across a vertical line through the pivot.
        - "horizontal": reflect across a horizontal line through the pivot.
        - "point": reflect through the pivot as a 180° rotation (point reflection).
        Default is "vertical".

    Returns
    -------
    dict
        A new dictionary containing the reflected coordinates for all nodes.

    Notes
    -----
    - The function does not modify the input dictionary; it returns a new one.
    - A KeyError is raised if the pivot is not present in positions.
    - Invalid values for axis raise a ValueError.
    """

    if pivot not in positions:
        raise KeyError(f"Pivot '{pivot}' is not a key in positions")

    px, py = positions[pivot]
    new_pos = {}

    for label, (x, y) in positions.items():
        if axis == "vertical":
            new_pos[label] = (2 * px - x, y)
        elif axis == "horizontal":
            new_pos[label] = (x, 2 * py - y)
        elif axis == "point":
            new_pos[label] = (2 * px - x, 2 * py - y)
        else:
            raise ValueError("axis must be 'vertical', 'horizontal', or 'point'")

    return new_pos

def apex_of(pyr, coords): 
    """
    Identify the apex node of a pyramid based on its 3D coordinates.

    Parameters
    ----------
    pyr : iterable
        Collection of node identifiers belonging to the pyramid.
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z), ...}.

    Returns
    -------
    hashable
        The node in pyr whose z-coordinate is nonzero, interpreted as
        the apex of the pyramid.

    Notes
    -----
    - The function assumes that exactly one node in pyr has a nonzero
      z-coordinate (the apex), while all base nodes have z = 0.
    - If no such node exists, the underlying next() call will raise a
      StopIteration error, indicating inconsistent data.
    """

    return next(n for n in pyr if coords[n][2] != 0)

def find_double_shared(target_pyr, pyramids, apex_list):
    """
    Find all pyramids that share exactly two base nodes with a target pyramid.

    Parameters
    ----------
    target_pyr : iterable
        Collection of node identifiers defining the target pyramid.
    pyramids : iterable
        Iterable of pyramid descriptors (each a collection of node identifiers).
    apex_list : iterable
        Collection of node identifiers that correspond to apex nodes.
        Used to distinguish apex nodes from base nodes.

    Returns
    -------
    list of tuple
        A list of pairs (pyramid, shared_nodes), where:
        - pyramid is one of the pyramids that shares exactly two base nodes
          with target_pyr.
        - shared_nodes is the set of the two shared base nodes.

    Notes
    -----
    - The function identifies base nodes of a pyramid using base_nodes,
      which removes nodes listed in apex_list.
    - target_pyr is excluded from the search.
    - A "double shared" pyramid is one whose base overlaps with the base of
      target_pyr in exactly two nodes.
    """    

    tgt_base = base_nodes(target_pyr, apex_list)
    out = []
    for p in pyramids:
        if p is target_pyr:
            continue
        shared = tgt_base & base_nodes(p, apex_list)
        if len(shared) == 2:
            out.append((p, shared))
    return out

# -------------------------------------------------
# ISOLATED PYRAMID FLATTENED TO A BOW-TIE STRUCTURE
# -------------------------------------------------
def single_bow_tie(idx_nodes, coords, start_idx = 0):
    """
    Construct the 2D bow-tie embedding for a pyramid whose four base nodes
    are known, together with its apex. This function places all necessary
    nodes (including auxiliary nodes) to form a single bow-tie unit.

    Parameters
    ----------
    idx_nodes : iterable
        Collection of node identifiers belonging to a single pyramid
        (typically 4 base nodes + 1 apex).
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z), ...}. The z-coordinate is used to
        detect the apex.
    start_idx : int, optional
        Index used to generate unique auxiliary node labels
        (prefix "W"). Two auxiliary nodes are created.

    Returns
    -------
    tuple
        - pos`` : dict  
          Dictionary mapping the newly positioned nodes (left tips, right tips,
          and auxiliary nodes) to their 2D coordinates defining the bow-tie.
        - all_nodes : list  
          List of all nodes involved in this bow-tie (in 2D).
        - apex : hashable  
          Identifier of the apex node (unchanged).
        - (rt, rb) : tuple  
          Identifiers of the right-top and right-bottom nodes.
        - (lt, lb) : tuple  
          Identifiers of the left-top and left-bottom nodes.
        - current_idx : int  
          Updated index for generating future auxiliary nodes.

    Notes
    -----
    - The apex is detected as the node with nonzero (or maximum) z-coordinate.
    - The two leftmost base nodes are placed at fixed positions, with vertical
      ordering determined by their 3D y-coordinates.
    - Two auxiliary nodes are placed at fixed offsets forming the central
      connector of the bow-tie.
    - The two rightmost base nodes are positioned symmetrically using fixed
      vertical offsets.
    - The function returns only the coordinates of the constructed bow-tie,
      not the entire graph.
    """

    import math
    s = math.sqrt(3)/2
    current_idx = start_idx

    ids = sorted(idx_nodes, key = lambda n: (coords[n][0], coords[n][1], n))

    apex = max(ids, key = lambda n: coords[n][2])
    left_nodes = sorted(ids, key=lambda n: coords[n][0])[:2]
    lt, lb = sorted(left_nodes, key=lambda n: - coords[n][1])

    d1   = 1 if coords[lb][1] > coords[lt][1] else - 1
    pos  = {lt : (0, 0), lb : (0, d1)}
    mid_y = d1 * 0.5

    aux_prefix = "W"  
    num_aux_nodes = 2
    aux_nodes = [f"{aux_prefix}{i}" for i in range(current_idx + 1, current_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (s, mid_y)
    pos[aux_nodes[1]] = (s + 1, mid_y)
    aux1 = aux_nodes[0]
    aux2 = aux_nodes[1]

    rest = [n for n in ids if n not in (lt, lb)]
    right_nodes = sorted(rest, key=lambda n: - coords[n][0])[:2] 
    rt, rb = sorted(right_nodes, key=lambda n: - coords[n][1])  

    pos[rt] = (2 * s + 1, mid_y + 0.5)
    pos[rb] = (2 * s + 1, mid_y - 0.5)

    all_nodes = [lt, lb, aux1, aux2, rt, rb]
    current_idx += num_aux_nodes

    return pos, all_nodes, apex, (rt, rb), (lt, lb), current_idx

# ----------------------------------
# HELPER FUNCTIONS FOR SINGLE SHARED 
# NODE BETWEEN PAIRS OF PYRAMIDS
# ----------------------------------

def top_right_horizontal_helper(start_node_id, start_node_coords, idx_nodes, coords, start_idx = 0):
    """
    Construct the 2D embedding of a bow-tie that attaches to an existing
    structure on its top-right side, assuming a horizontal orientation.

    Parameters
    ----------
    start_node_id : hashable
        Identifier of the shared node where the new bow-tie attaches.
    start_node_coords : tuple of float
        2D coordinates (x, y) of the shared node in the existing embedding.
    idx_nodes : iterable
        Collection of node identifiers belonging to the new pyramid
        (typically four base nodes + one apex).
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to detect the apex.
    start_idx : int, optional
        Starting index for generating unique auxiliary node labels
        (prefix "W"). Two auxiliary nodes are introduced.

    Returns
    -------
    tuple
        - pos : dict  
          Dictionary with the newly computed 2D positions for the apex,
          the two auxiliary nodes, and the two remaining base nodes.
        - apex : hashable  
          Identifier of the apex node of the new pyramid.
        - topB : hashable  
          Identifier of the top base node on the right side.
        - botB : hashable  
          Identifier of the bottom base node on the right side.
        - current_idx : int  
          Updated index after consuming two auxiliary node labels.

    Notes
    -----
    - The apex is selected as the node with the highest z-coordinate.
    - The two auxiliary nodes form the central diagonal part of the bow-tie.
    - The two remaining base nodes are assigned to top/bottom positions
      based on their 3D y-ordering.
    - The bow-tie is constructed to extend horizontally to the right of the
      shared starting node.
    """

    import math

    s = math.sqrt(3) / 2
    ids = idx_nodes.copy()

    # Start node coordinates (top-right node from the first pyramid)
    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    # The apex of the second pyramid is the highest node based on z-coordinate
    apex = max(ids, key = lambda n: coords[n][2])

    # Construct the second bow-tie starting from the shared node
    pos = {}
    pos[apex] = (start_x, start_y + 1)

    aux_prefix = "W"  
    num_aux_nodes = 2
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x + s, start_y + 0.5)
    pos[aux_nodes[1]] = (start_x + 1 + s, start_y + 0.5)

    remaining_nodes = [n for n in ids if n != apex and n != start_node_id]
    topB, botB = sorted(remaining_nodes, key = lambda n: coords[n][1], reverse = True)

    if coords[topB][1] < coords[botB][1]:
        topB, botB = botB, topB  # Swap topB and botB if topB has a lower y

    pos[topB] = (start_x + 2 * s + 1, start_y + 1)
    pos[botB] = (start_x + 2 * s + 1, start_y)
    current_idx = start_idx + num_aux_nodes

    return pos, apex, topB, botB, current_idx

def bottom_right_horizontal_helper(start_node_id, start_node_coords, idx_nodes, coords, start_idx = 0):
    """
    Construct the 2D embedding of a bow-tie that attaches to an existing
    structure on its bottom-right side, assuming a horizontal orientation.

    Parameters
    ----------
    start_node_id : hashable
        Identifier of the shared node where the new bow-tie attaches.
    start_node_coords : tuple of float
        2D coordinates (x, y) of the shared node in the existing embedding.
    idx_nodes : iterable
        Collection of node identifiers belonging to the new pyramid
        (typically four base nodes + one apex).
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to detect the apex.
    start_idx : int, optional
        Starting index for generating unique auxiliary node labels
        (prefix "W"). Two auxiliary nodes are introduced.

    Returns
    -------
    tuple
        - pos : dict  
          Dictionary with the computed 2D positions for the apex, the two
          auxiliary nodes, and the two remaining right-side base nodes.
        - apex : hashable  
          Identifier of the apex node of the new pyramid.
        - topB : hashable  
          Identifier of the upper base node on the right side.
        - botB : hashable  
          Identifier of the lower base node on the right side.
        - current_idx : int  
          Updated index after consuming two auxiliary node labels.

    Notes
    -----
    - The apex is selected as the node with the highest z-coordinate.
    - The two auxiliary nodes form the central diagonal segment of the bow-tie.
    - The two remaining base nodes are assigned to top/bottom based on their
      3D y-coordinate ordering.
    - The bow-tie is constructed to extend horizontally to the right of the
      shared starting node, but oriented downward relative to it.
    """

    import math

    s = math.sqrt(3) / 2
    ids = idx_nodes.copy()

    # Start node coordinates (top-right node from the first pyramid)
    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    # The apex of the second pyramid is the highest node based on z-coordinate
    apex = max(ids, key = lambda n: coords[n][2])

    # Construct the second bow-tie starting from the shared node
    pos = {}
    pos[apex] = (start_x, start_y - 1)

    aux_prefix = "W"  
    num_aux_nodes = 2
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x + s, start_y - 0.5)
    pos[aux_nodes[1]] = (start_x + s + 1, start_y - 0.5)

    remaining_nodes = [n for n in ids if n != apex and n != start_node_id]
    topB, botB = sorted(remaining_nodes, key = lambda n: coords[n][1], reverse = True)

    if coords[topB][1] < coords[botB][1]:
        topB, botB = botB, topB  # Swap topB and botB if topB has a lower y

    pos[topB] = (start_x + 2 * s + 1, start_y)
    pos[botB] = (start_x + 2 * s + 1, start_y - 1)
    current_idx = start_idx + num_aux_nodes

    return pos, apex, topB, botB, current_idx

def top_right_vertical_helper(start_node_id, start_node_coords, idx_nodes, coords, start_idx = 0):
    """
    Construct the 2D embedding of a bow-tie that attaches to an existing
    structure on its top-right side, assuming a vertical orientation.

    Parameters
    ----------
    start_node_id : hashable
        Identifier of the shared node where the new bow-tie attaches.
    start_node_coords : tuple of float
        2D coordinates (x, y) of the shared node in the existing embedding.
    idx_nodes : iterable
        Collection of node identifiers belonging to the new pyramid
        (typically four base nodes + one apex).
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to detect the apex.
    start_idx : int, optional
        Starting index for generating unique auxiliary node labels
        (prefix "W"). Two auxiliary nodes are introduced.

    Returns
    -------
    tuple
        - pos : dict  
          Dictionary with the newly computed 2D positions for the apex,
          auxiliary nodes, and the two remaining base nodes.
        - apex : hashable  
          Identifier of the apex node of the new pyramid.
        - topR : hashable  
          Identifier of the rightmost node on the upper horizontal segment.
        - topL : hashable  
          Identifier of the leftmost node on the upper horizontal segment.
        - current_idx : int  
          Updated index after consuming two auxiliary node labels.

    Notes
    -----
    - The apex is selected as the node with the highest z-coordinate.
    - Auxiliary nodes form the diagonal middle segment of the bow-tie,
      placed above the shared node.
    - The two remaining base nodes are placed horizontally at the top,
      ordered by their x-coordinates.
    - The final bow-tie extends vertically upward from the shared node.
    """

    import math

    s = math.sqrt(3) / 2
    ids = idx_nodes.copy()

    # Start node coordinates (top-right node from the first pyramid)
    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    # The apex of the second pyramid is the highest node based on z-coordinate
    apex = max(ids, key = lambda n: coords[n][2])

    # Construct the second bow-tie starting from the shared node
    pos = {}
    pos[apex] = (start_x + 1, start_y)

    aux_prefix = "W"  
    num_aux_nodes = 2
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x + 0.5, start_y + s)
    pos[aux_nodes[1]] = (start_x + 0.5, start_y + s + 1)

    remaining_nodes = [n for n in ids if n != apex and n != start_node_id]
    topR, topL = sorted(remaining_nodes, key = lambda n: coords[n][1], reverse = True)

    if coords[topR][0] < coords[topL][0]:
        topR, topL = topL, topR  # Swap topR and topL if topR has a lower x

    pos[topR] = (start_x + 1, start_y + 2 * s + 1)
    pos[topL] = (start_x, start_y + 2 * s + 1)
    current_idx = start_idx + num_aux_nodes

    return pos, apex, topR, topL, current_idx

def bottom_right_vertical_helper(start_node_id, start_node_coords, idx_nodes, coords, start_idx = 0):
    """
    Construct the 2D embedding of a bow-tie that attaches to an existing
    structure on its bottom-right side, assuming a vertical orientation.

    Parameters
    ----------
    start_node_id : hashable
        Identifier of the shared node where the new bow-tie attaches.
    start_node_coords : tuple of float
        2D coordinates (x, y) of the shared node in the existing embedding.
    idx_nodes : iterable
        Collection of node identifiers belonging to the new pyramid
        (typically four base nodes + one apex).
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to detect the apex.
    start_idx : int, optional
        Starting index for generating unique auxiliary node labels
        (prefix "W"). Two auxiliary nodes are introduced.

    Returns
    -------
    tuple
        - pos : dict  
          Dictionary with the newly computed 2D positions for the apex,
          auxiliary nodes, and the two remaining base nodes.
        - apex : hashable  
          Identifier of the apex node of the new pyramid.
        - botR : hashable  
          Identifier of the rightmost node on the lower horizontal segment.
        - botL : hashable  
          Identifier of the leftmost node on the lower horizontal segment.
        - current_idx : int  
          Updated index after consuming two auxiliary node labels.

    Notes
    -----
    - The apex is selected as the node with the highest z-coordinate.
    - Auxiliary nodes form the diagonal central segment of the bow-tie,
      placed below the shared node.
    - The two remaining base nodes are placed horizontally at the bottom,
      ordered by their x-coordinates.
    - The final bow-tie extends vertically downward from the shared node.
    """

    import math

    s = math.sqrt(3) / 2
    ids = idx_nodes.copy()

    # Start node coordinates (top-right node from the first pyramid)
    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    # The apex of the second pyramid is the highest node based on z-coordinate
    apex = max(ids, key = lambda n: coords[n][2])

    # Construct the second bow-tie starting from the shared node
    pos = {}
    pos[apex] = (start_x + 1, start_y)

    aux_prefix = "W"  
    num_aux_nodes = 2
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x + 0.5, start_y - s)
    pos[aux_nodes[1]] = (start_x + 0.5, start_y - s - 1)

    remaining_nodes = [n for n in ids if n != apex and n != start_node_id]
    botR, botL = sorted(remaining_nodes, key = lambda n: coords[n][1], reverse = True)

    if coords[botR][0] < coords[botL][0]:
        botR, botL = botL, botR  # Swap botR and botL if botR has a lower x

    pos[botR] = (start_x + 1, start_y - 2 * s - 1)
    pos[botL] = (start_x, start_y - 2 * s - 1)
    current_idx = start_idx + num_aux_nodes

    return pos, apex, botR, botL, current_idx

def top_left_horizontal_helper(start_node_id, start_node_coords, idx_nodes, coords, start_idx = 0):
    """
    Construct the 2D embedding of a bow-tie that attaches to an existing
    structure on its top-left side, assuming a horizontal orientation.

    Parameters
    ----------
    start_node_id : hashable
        Identifier of the shared node where the new bow-tie attaches.
    start_node_coords : tuple of float
        2D coordinates (x, y) of the shared node in the existing embedding.
    idx_nodes : iterable
        Collection of node identifiers belonging to the new pyramid
        (typically four base nodes + one apex).
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to detect the apex.
    start_idx : int, optional
        Starting index for generating unique auxiliary node labels
        (prefix "W"). Two auxiliary nodes are introduced.

    Returns
    -------
    tuple
        - pos : dict  
          Dictionary with the newly computed 2D positions for the apex,
          the two auxiliary nodes, and the two remaining left-side base nodes.
        - apex : hashable  
          Identifier of the apex node of the new pyramid.
        - topB : hashable  
          Identifier of the upper base node on the left side.
        - botB : hashable  
          Identifier of the lower base node on the left side.
        - current_idx : int  
          Updated index after consuming two auxiliary node labels.

    Notes
    -----
    - The apex is chosen as the node with the highest z-coordinate.
    - Auxiliary nodes form the central diagonal of the bow-tie, placed to
      the left of the shared node.
    - The two base nodes are ordered by their 3D y-coordinate (top/bottom).
    - The bow-tie extends horizontally to the left of the shared starting node.
    """    

    import math

    s = math.sqrt(3) / 2
    ids = idx_nodes.copy()

    # Start node coordinates (top-right node from the first pyramid)
    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    # The apex of the second pyramid is the highest node based on z-coordinate
    apex = max(ids, key = lambda n: coords[n][2])

    # Construct the second bow-tie starting from the shared node
    pos = {}
    pos[apex] = (start_x, start_y + 1)

    aux_prefix = "W"  
    num_aux_nodes = 2
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x - s, start_y + 0.5)
    pos[aux_nodes[1]] = (start_x - 1 - s, start_y + 0.5)

    remaining_nodes = [n for n in ids if n != apex and n != start_node_id]
    topB, botB = sorted(remaining_nodes, key = lambda n: coords[n][1], reverse = True)

    if coords[topB][1] < coords[botB][1]:
        topB, botB = botB, topB  # Swap topB and botB if topB has a lower y

    pos[topB] = (start_x - 1 - 2 * s, start_y + 1)
    pos[botB] = (start_x - 1 - 2 * s, start_y)
    current_idx = start_idx + num_aux_nodes

    return pos, apex, topB, botB, current_idx

def bottom_left_horizontal_helper(start_node_id, start_node_coords, idx_nodes, coords, start_idx = 0):
    """
    Construct the 2D embedding of a bow-tie that attaches to an existing
    structure on its bottom-left side, assuming a horizontal orientation.

    Parameters
    ----------
    start_node_id : hashable
        Identifier of the shared node where the new bow-tie attaches.
    start_node_coords : tuple of float
        2D coordinates (x, y) of the shared node in the existing embedding.
    idx_nodes : iterable
        Collection of node identifiers belonging to the new pyramid
        (typically four base nodes + one apex).
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to detect the apex.
    start_idx : int, optional
        Starting index for generating unique auxiliary node labels
        (prefix "W"). Two auxiliary nodes are introduced.

    Returns
    -------
    tuple
        - pos : dict  
          Dictionary containing the computed 2D positions for the apex,
          the auxiliary nodes, and the two remaining left-side base nodes.
        - apex : hashable  
          Identifier of the apex node of the new pyramid.
        - topB : hashable  
          Identifier of the upper base node on the left side.
        - botB : hashable  
          Identifier of the lower base node on the left side.
        - current_idx : int  
          Updated index after using two auxiliary node labels.

    Notes
    -----
    - The apex is selected as the node with the highest z-value.
    - Auxiliary nodes are positioned to the left of the shared node,
      forming the diagonal centre of the bow-tie.
    - The two remaining base nodes are ordered by their 3D y-coordinate.
    - The new bow-tie extends horizontally to the left of the shared node,
      but oriented downward relative to it.
    """

    import math

    s = math.sqrt(3) / 2
    ids = idx_nodes.copy()

    # Start node coordinates (top-right node from the first pyramid)
    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    # The apex of the second pyramid is the highest node based on z-coordinate
    apex = max(ids, key = lambda n: coords[n][2])

    # Construct the second bow-tie starting from the shared node
    pos = {}
    pos[apex] = (start_x, start_y - 1)

    aux_prefix = "W"  
    num_aux_nodes = 2
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x - s, start_y - 0.5)
    pos[aux_nodes[1]] = (start_x - s - 1, start_y - 0.5)

    remaining_nodes = [n for n in ids if n != apex and n != start_node_id]
    topB, botB = sorted(remaining_nodes, key = lambda n: coords[n][1], reverse = True)

    if coords[topB][1] < coords[botB][1]:
        topB, botB = botB, topB  # Swap topB and botB if topB has a lower y

    pos[topB] = (start_x - 2 * s - 1, start_y)
    pos[botB] = (start_x - 2 * s - 1, start_y - 1)
    current_idx = start_idx + num_aux_nodes

    return pos, apex, topB, botB, current_idx

def top_left_vertical_helper(start_node_id, start_node_coords, idx_nodes, coords, start_idx = 0):
    """
    Construct the 2D embedding of a bow-tie that attaches to an existing
    structure on its top-left side, assuming a vertical orientation.

    Parameters
    ----------
    start_node_id : hashable
        Identifier of the shared node where the new bow-tie attaches.
    start_node_coords : tuple of float
        2D coordinates (x, y) of the shared node in the existing embedding.
    idx_nodes : iterable
        Collection of node identifiers belonging to the new pyramid
        (typically four base nodes + one apex).
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-coordinate is used to detect the apex.
    start_idx : int, optional
        Starting index for generating unique auxiliary node labels
        (prefix "W"). Two auxiliary nodes are created.

    Returns
    -------
    tuple
        - pos : dict  
          Dictionary containing the new 2D positions of the apex,
          auxiliary nodes, and the upper base nodes.
        - apex : hashable  
          Identifier of the apex node of the new pyramid.
        - topR : hashable  
          Identifier of the rightmost top node in the new bow-tie.
        - topL : hashable  
          Identifier of the leftmost top node in the new bow-tie.
        - current_idx : int  
          Updated index after generating two auxiliary node labels.

    Notes
    -----
    - The apex is determined as the node with the greatest z-coordinate.
    - Auxiliary nodes are placed above the shared node to form the diagonal
      segment of the bow-tie.
    - The two non-apex, non-shared nodes are assigned based on their x-values
      so that topR is the one with larger x.
    - The final bow-tie extends vertically upward and leftward from
      the shared node.
    """

    import math

    s = math.sqrt(3) / 2
    ids = idx_nodes.copy()

    # Start node coordinates (top-right node from the first pyramid)
    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    # The apex of the second pyramid is the highest node based on z-coordinate
    apex = max(ids, key = lambda n: coords[n][2])

    # Construct the second bow-tie starting from the shared node
    pos = {}
    pos[apex] = (start_x - 1, start_y)

    aux_prefix = "W"  
    num_aux_nodes = 2
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x - 0.5, start_y + s)
    pos[aux_nodes[1]] = (start_x - 0.5, start_y + s + 1)

    remaining_nodes = [n for n in ids if n != apex and n != start_node_id]
    topR, topL = sorted(remaining_nodes, key = lambda n: coords[n][1], reverse = True)

    if coords[topR][0] < coords[topL][0]:
        topR, topL = topL, topR  # Swap topR and topL if topR has a lower x

    pos[topR] = (start_x, start_y + 2 * s + 1)
    pos[topL] = (start_x - 1, start_y + 2 * s + 1)
    current_idx = start_idx + num_aux_nodes

    return pos, apex, topR, topL, current_idx

def bottom_left_vertical_helper(start_node_id, start_node_coords, idx_nodes, coords, start_idx = 0):
    """
    Construct the 2D embedding of a bow-tie that attaches to an existing
    structure on its bottom-left side, assuming a vertical orientation.

    Parameters
    ----------
    start_node_id : hashable
        Identifier of the shared node where the new bow-tie attaches.
    start_node_coords : tuple of float
        2D coordinates (x, y) of the shared node in the existing embedding.
    idx_nodes : iterable
        Collection of node identifiers belonging to the new pyramid
        (typically four base nodes + one apex).
    coords : dict
        Dictionary mapping node identifiers to their 3D coordinates,
        e.g. {id: (x, y, z)}. The z-value is used to identify the apex.
    start_idx : int, optional
        Starting index for generating unique auxiliary node labels
        (prefix "W"). Two auxiliary nodes are created.

    Returns
    -------
    tuple
        - pos : dict  
          Mapping of the newly placed nodes (apex, auxiliary nodes,
          and two remaining base nodes) to their 2D coordinates.
        - apex : hashable  
          Identifier of the apex node of the new pyramid.
        - botR : hashable  
          Identifier of the rightmost bottom node in the new bow-tie.
        - botL : hashable  
          Identifier of the leftmost bottom node in the new bow-tie.
        - current_idx : int  
          Updated index after generating two auxiliary node labels.

    Notes
    -----
    - The apex is the node with the highest z-coordinate.
    - Auxiliary nodes are placed below the shared node to form the diagonal
      middle of the bow-tie.
    - The two remaining base nodes are ordered by their x-coordinates so
      that botR has the greater x-value.
    - The constructed bow-tie extends vertically downward and leftward
      from the shared node.
    """

    import math

    s = math.sqrt(3) / 2
    ids = idx_nodes.copy()

    # Start node coordinates (top-right node from the first pyramid)
    start_x = start_node_coords[0]
    start_y = start_node_coords[1]

    # The apex of the second pyramid is the highest node based on z-coordinate
    apex = max(ids, key = lambda n: coords[n][2])

    # Construct the second bow-tie starting from the shared node
    pos = {}
    pos[apex] = (start_x - 1, start_y)

    aux_prefix = "W"  
    num_aux_nodes = 2
    aux_nodes = [f"{aux_prefix}{i}" for i in range(start_idx + 1, start_idx + num_aux_nodes + 1)]
    pos[aux_nodes[0]] = (start_x - 0.5, start_y - s)
    pos[aux_nodes[1]] = (start_x - 0.5, start_y - s - 1)

    remaining_nodes = [n for n in ids if n != apex and n != start_node_id]
    botR, botL = sorted(remaining_nodes, key = lambda n: coords[n][1], reverse = True)

    if coords[botR][0] < coords[botL][0]:
        botR, botL = botL, botR  # Swap botR and botL if botR has a lower x

    pos[botR] = (start_x, start_y - 2 * s - 1)
    pos[botL] = (start_x - 1, start_y - 2 * s - 1)
    current_idx = start_idx + num_aux_nodes

    return pos, apex, botR, botL, current_idx
