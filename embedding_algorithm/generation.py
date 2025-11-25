"""This module provides functions to generate 2D point sets using
Sobol sequences, build unit disk graphs (UDGs), and create 3D UDGs 
with pyramid apex nodes based on triple overlaps. It also includes 
plotting functions for visualizing 2D and 3D UDGs."""

def generate_sobol_points(n_points, seed = None, border = None):
    """
    Generate 2D points inside a polygon using a Sobol low-discrepancy sequence.

    Points are drawn from a 2D Sobol sequence, rescaled to the bounding box
    of `border`, and kept only if they lie inside the polygon. If `border`
    is not provided, a default hexagonal polygon is used. Points are stored
    in a dictionary with string node IDs as keys.

    Parameters
    ----------
    n_points : int
        Number of points to generate inside the polygon.
    seed : int or None, optional
        Seed for the Sobol sampler (and optionally NumPy) to ensure
        reproducibility. Default is None.
    border : shapely.geometry.Polygon or None, optional
        Polygon that defines the allowed region. If None, a fixed hexagon
        is used.

    Returns
    -------
    dict
        Dictionary mapping string node IDs to 2D coordinates:
        {"1": (x1, y1), "2": (x2, y2), ...}, with all points inside `border`.
    """

    import numpy as np
    from scipy.stats import qmc
    from shapely.geometry import Point, Polygon
    
    if not seed:
        np.random.seed(seed)

    if not border:
        # Define an hexagon border
        vertices = [(1.3, 4.1), (5, 1), (9, 1.5), (10, 4), (8.5, 8.2), (4.3, 9)]
        border = Polygon(vertices)

    # Create the Sobol sequence sampler for 2D with an optional seed for reproducibility
    sampler = qmc.Sobol(d = 2, seed = seed)
    sobol_points_dict = {}
    node_id = 1
    
    # Generate points until we have the desired number inside the polygon
    while len(sobol_points_dict) < n_points:
        # Generate a new Sobol point
        new_points = sampler.random(1)  # Generate one point
        # Rescale to fit the polygon's bounding box
        minx, miny, maxx, maxy = border.bounds
        new_points[:, 0] = minx + new_points[:, 0] * (maxx - minx)
        new_points[:, 1] = miny + new_points[:, 1] * (maxy - miny)
        
        # Check if the point is inside the polygon
        p = Point(new_points[0, 0], new_points[0, 1])
        if border.contains(p):
            sobol_points_dict[str(node_id)] = (float(new_points[0, 0]), float(new_points[0, 1]))
            node_id += 1
    
    return sobol_points_dict

def build_udg(node_coords, radius = 1.0): 
    """
    Build a unit disk graph (UDG) from a set of 2D node coordinates.

    Nodes are connected by an edge if the Euclidean distance between their
    coordinates is less than or equal to `2 * radius`.

    Parameters
    ----------
    node_coords : dict
        Mapping from node identifiers to 2D coordinates, e.g. {id: (x, y), ...}.
    radius : float, optional
        Base radius used in the distance threshold (default 1.0).

    Returns
    -------
    tuple
        G : networkx.Graph
            The constructed geometric graph with node attribute "pos".
        edges : list of tuple
            List of edges (u, v) present in G.
    """

    import numpy as np 
    import networkx as nx 
    from itertools import combinations 
    
    # Initialise graph & positions 
    G = nx.Graph() 
    G.add_nodes_from(node_coords.keys()) 
    nx.set_node_attributes(G, node_coords, "pos") 

    # Add UDG edges 
    ids = list(node_coords.keys()) 
    for u, v in combinations(ids, 2): 
        if np.linalg.norm(np.array(node_coords[u]) - np.array(node_coords[v])) <= 2*radius: 
            G.add_edge(u, v) 

    # Return graph and edges 
    return G, list(G.edges())

def plot_udg(G, coords, edges, radius = 1.0, border = None):
    """
    Plot a unit disk graph and its coverage disks inside a polygonal border.

    The function optionally adds edges to `G`, draws the polygon `border`,
    plots the nodes and edges, and overlays a disk of radius `radius` around
    each node.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes and (optional) edges will be plotted.
    coords : dict
        Mapping node_id -> (x, y) with node coordinates. Keys are matched
        to graph nodes, with a best-effort cast to int if needed.
    edges : list of tuple or None
        Extra edges to add to `G` before plotting. If None, no edges are added.
    radius : float, optional
        Radius of the red disks drawn around each node (default 1.0).
    border : shapely.geometry.Polygon or None, optional
        Polygonal region to display as the outer boundary. If None, a
        default hexagon is used.
    """

    import matplotlib.pyplot as plt
    import networkx as nx
    from shapely.geometry import Polygon

    if not border:
        # Define an hexagon border
        vertices = [(1.3, 4.1), (5, 1), (9, 1.5), (10, 4), (8.5, 8.2), (4.3, 9)]
        border = Polygon(vertices)

    # Convert position keys to same type as graph nodes
    # (handles cases where keys are strings but G nodes are ints)
    pos = {}
    for k, v in coords.items():
        node = k
        if node not in G:
            try:
                node = int(k)
            except:
                pass
        pos[node] = v

    # Add edges if needed
    if edges is not None:
        G = G.copy()
        G.add_edges_from(edges)

    # Plot
    fig, ax = plt.subplots(figsize = (8, 8))

    # Plot the polygon border
    x_poly, y_poly = border.exterior.xy
    ax.plot(x_poly, y_poly, linewidth = 1, color = "black")
    # ax.margins(x = 0.15, y = 0.15) 

    # Add the points
    nx.draw(G, pos = pos, with_labels = False, node_size = 50, edge_color = "black", node_color = "black")

    # Draw a circle around each node
    for node, (x, y) in pos.items():
        circle = plt.Circle((x, y), radius, color = '#3e00ab', fill = 'red', alpha = 0.15)
        ax.add_artist(circle)

    plt.tight_layout()
    # plt.savefig("udg_plot.pdf", dpi = 400)
    plt.show()
    
def build_3d_udg(node_coords, radius = 1.0, height = 1.0, tol = 1e-9):
    """
    Build a 3D unit disk graph by adding pyramid apex nodes above triple
    overlaps of 2D disks.

    Starting from a 2D UDG on `node_coords`, the function finds all
    triangles in the base graph, computes the minimal enclosing circle of
    each triple, and adds an apex node A* at height `height` whenever that
    circle fits inside disks of radius `radius` (triple overlap). The apex
    is connected to the three base nodes, forming a pyramid.

    Parameters
    ----------
    node_coords : dict
        {node_id: (x, y)} 2D coordinates of base nodes.
    radius : float, optional
        Disk radius used for the base UDG and triple-overlap test (default 1.0).
    height : float, optional
        z-coordinate at which to place the apex nodes (default 2.0).
    tol : float, optional
        Numerical tolerance for deciding triple overlaps (default 1e-9).

    Returns
    -------
    G3d : networkx.Graph
        Graph with original base nodes (z = 0) and added apex nodes (z = height).
        Includes all 2D UDG edges plus edges from each apex to its three bases.
    coords_3d : dict
        {node_id: (x, y, z)} 3D coordinates of all nodes.
    edges_3d : list of tuple
        List of edges (u, v) present in G3d.
    apex_list : list
        List of apex node IDs added to G3d.
    """

    import numpy as np
    from itertools import combinations

    # ---- Helper: minimal enclosing circle of 3 points (2D) ----
    def minimal_enclosing_circle_three(p1, p2, p3):
        """
        Compute the minimal enclosing circle of three 2D points.

        For three points p1, p2, p3, this returns the center and radius of
        the smallest circle containing all three. If the triangle is obtuse,
        the circle is the one with diameter given by the longest side;
        otherwise it is the circumcircle. A nearly colinear case falls back
        to the bounding segment-based circle.

        Parameters
        ----------
        p1, p2, p3 : array-like
            2D points (x, y).

        Returns
        -------
        center : numpy.ndarray
            2D array giving the circle center.
        radius : float
            Radius of the minimal enclosing circle.
        """

        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        p3 = np.array(p3, dtype=float)

        def circle_from_two(a, b):
            """
            Construct the circle having segment AB as diameter.

            Parameters
            ----------
            a, b : array-like
                2D points (x, y).

            Returns
            -------
            c : numpy.ndarray
                Midpoint of a and b, i.e. the circle center.
            r : float
                Radius, equal to half the distance between a and b.
            """

            c = (a + b) / 2.0
            r = np.linalg.norm(a - b) / 2.0
            return c, r

        def circle_from_three(a, b, c):
            """
            Compute the circumcircle of triangle ABC (with colinear fallback).

            For three 2D points a, b, c, this returns the center and radius
            of the circumcircle. If the points are nearly colinear (singular
            system), it falls back to the minimal circle covering the three
            points based on their bounding segment.

            Parameters
            ----------
            a, b, c : array-like
                2D points (x, y).

            Returns
            -------
            center : numpy.ndarray
                Center of the resulting circle.
            r : float
                Circle radius.
            """

            # Circumcircle of triangle ABC
            A = np.array([[b[0] - a[0], b[1] - a[1]],
                          [c[0] - a[0], c[1] - a[1]]], dtype=float)
            B = np.array([
                ((b[0]**2 - a[0]**2) + (b[1]**2 - a[1]**2)) / 2.0,
                ((c[0]**2 - a[0]**2) + (c[1]**2 - a[1]**2)) / 2.0
            ], dtype=float)

            detA = np.linalg.det(A)
            if abs(detA) < 1e-14:
                # Points almost colinear – fall back to bounding circle
                xs = [a[0], b[0], c[0]]
                ys = [a[1], b[1], c[1]]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                center = np.array([(minx + maxx) / 2.0, (miny + maxy) / 2.0])
                r = max(np.linalg.norm(center - a),
                        np.linalg.norm(center - b),
                        np.linalg.norm(center - c))
                
                return center, r

            center = np.linalg.solve(A, B)
            r = max(np.linalg.norm(center - a),
                    np.linalg.norm(center - b),
                    np.linalg.norm(center - c))
            
            return center, r

        # Squared side lengths of triangle
        a2 = np.linalg.norm(p2 - p3)**2
        b2 = np.linalg.norm(p1 - p3)**2
        c2 = np.linalg.norm(p1 - p2)**2

        # If triangle has an obtuse angle, MEC is circle with diameter = longest side
        if a2 >= b2 + c2:
            return circle_from_two(p2, p3)
        if b2 >= a2 + c2:
            return circle_from_two(p1, p3)
        if c2 >= a2 + b2:
            return circle_from_two(p1, p2)

        # Otherwise it's the circumcircle
        return circle_from_three(p1, p2, p3)

    # Build the 2D UDG on the base layer
    G2d, _ = build_udg(node_coords, radius = radius)

    # Start the 3D graph, lifting original nodes to z = 0.0
    G3d = G2d.copy()
    coords_3d = {}

    for n, (x, y) in node_coords.items():
        coords_3d[n] = (float(x), float(y), 0.0)

    # Enumerate all node triples that form triangles in the 2D UDG
    triangles = set()
    for u in G2d.nodes():
        nbrs = list(G2d.neighbors(u))
        for v, w in combinations(nbrs, 2):
            if G2d.has_edge(v, w):
                tri = tuple(sorted((u, v, w), key=str))
                triangles.add(tri)

    # Prepare naming of apex nodes
    existing_ids = set(G3d.nodes())
    apex_counter = 1
    while f"A{apex_counter}" in existing_ids:
        apex_counter += 1

    pyramids = []

    # For each triangle, check triple overlap and add apex if needed
    for (n1, n2, n3) in triangles:
        p1 = node_coords[n1]
        p2 = node_coords[n2]
        p3 = node_coords[n3]

        center, r_mec = minimal_enclosing_circle_three(p1, p2, p3)

        # Triple overlap exists if minimal enclosing circle fits inside each disk
        if r_mec <= radius + tol:
            x_overlap, y_overlap = center

            apex_id = f"A{apex_counter}"
            apex_counter += 1

            # Add the apex node in 3D
            G3d.add_node(apex_id)
            coords_3d[apex_id] = (float(x_overlap), float(y_overlap), float(height))

            # Connect it manually to the three base nodes
            G3d.add_edge(apex_id, n1)
            G3d.add_edge(apex_id, n2)
            G3d.add_edge(apex_id, n3)

            pyramids.append((apex_id, (n1, n2, n3)))
            
    apex_list = [n for n in G3d.nodes() if str(n).startswith("A")]
    edges_3d = list(G3d.edges())

    return G3d, coords_3d, edges_3d, apex_list

def plot_3d_udg(G, pos, show_labels = False, renderer = "browser"):
    """
    Plot a 3D unit disk graph using Plotly.

    Nodes are drawn as 3D markers (apex nodes whose IDs start with "A"
    are coloured red, others black) and edges as gray line segments.
    Axes are hidden and the aspect ratio is fixed.

    Parameters
    ----------
    G : networkx.Graph
        Graph to plot. Only nodes present in `pos` are shown.
    pos : dict
        Mapping {node_id: (x, y, z)} with 3D coordinates for nodes in `G`.
    show_labels : bool, optional
        If True, display node IDs as text next to the markers (default False).
    renderer : str, optional
        Name of the Plotly renderer to use (e.g. "browser", "notebook").
        Default is "browser".
    """

    import plotly.graph_objects as go
    import plotly.io as pio

    # -----------------------
    # Gather edge coordinates 
    # -----------------------
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        # Skip edges for which we don't have coordinates
        if u not in pos or v not in pos:
            continue
        (x0, y0, z0) = pos[u]
        (x1, y1, z1) = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    # ---------------------------------
    # Gather node coordinates & colours
    # ---------------------------------
    node_ids = [n for n in G.nodes() if n in pos]

    node_x = [pos[n][0] for n in node_ids]
    node_y = [pos[n][1] for n in node_ids]
    node_z = [pos[n][2] for n in node_ids]

    # Apex nodes from build_3d_udg start with "A"
    node_color = ["red" if str(n).startswith("A") else "black" for n in node_ids]
    node_mode = "markers+text" if show_labels else "markers"

    # -------------------
    # Build Plotly figure
    # -------------------
    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter3d(x = edge_x, y = edge_y, z = edge_z, mode = "lines", line = dict(color = "gray", width = 4), hoverinfo = "skip"))

    # Nodes (and optional labels)
    fig.add_trace(go.Scatter3d(x = node_x, y = node_y, z = node_z, mode = node_mode, marker = dict(size = 10, color = node_color), text = [str(n) for n in node_ids] if show_labels else None, textposition = "top center", textfont = dict(color = "black", size = 10), hoverinfo = "text", hovertext = [str(n) for n in node_ids]))

    fig.update_layout(scene = dict(xaxis = dict(visible = False), yaxis = dict(visible = False), zaxis = dict(visible = False), bgcolor = "white", aspectmode = "manual", aspectratio = dict(x = 1, y = 1, z = 0.4),), showlegend = False, margin = dict(l = 0, r = 0, b = 0, t = 0),)

    # Open in the chosen renderer (by default: browser)
    pio.renderers.default = renderer
    fig.show()
