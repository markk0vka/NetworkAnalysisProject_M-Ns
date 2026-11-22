"""
Part D - Evacuation routes and bottlenecks   (src/bottlenecks.py)

What this script does
  1. every danger node -> shortest path (weight = travel_time) to its zone's assembly point
  2. load(e)             = number of evacuation routes that use edge e (1 route = 1 unit of people)
  3. bottleneck_score(e) = load(e) / capacity(e)            (capacity = number of lanes)
  4. compares load with ordinary (all-pairs) edge betweenness centrality
  5. saves two maps and the top-20 bottleneck table

Inputs (data/processed/)
  graph.graphml           <- part A   (node id = osmid, edges: length, travel_time, capacity, name, highway)
  danger_nodes.csv        <- part B   (osmid, danger)
  zones.csv               <- part C   (osmid, zone)
  assembly_points.csv     <- part C   (zone, osmid)
If a file does not exist yet, a simple placeholder is used instead, so the code runs today.
When the real file appears, nothing has to be changed - just re-run.

Usage
  python src/bottlenecks.py                  # real files where they exist, placeholders otherwise
  python src/bottlenecks.py --strict         # stop with an error if any real file is missing
  python src/bottlenecks.py --k 500          # sample 500 sources for ordinary betweenness (faster)
  python src/bottlenecks.py --exclude-ap-edges   # top-20 without the edges touching assembly points
"""

import argparse
import ast
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).resolve().parent.parent      # project root (the folder above src/)
PROC = ROOT / "data" / "processed"                 # shared files between the 4 parts
FIG = ROOT / "report" / "figures"                  # output maps
TOP_N = 20


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def edge_key(a, b):
    """Undirected edge id: identical for (a, b) and (b, a)."""
    return (a, b) if a <= b else (b, a)


def clean_name(raw, highway=None):
    """Street name for labels. Handles NaN, "['A', 'B']" strings, lists and unnamed roads."""
    if isinstance(raw, (list, tuple)):
        names = list(raw)
    elif isinstance(raw, str) and raw.strip().startswith("["):
        try:
            names = list(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            names = [raw]
    elif isinstance(raw, str) and raw.strip():
        names = [raw.strip()]
    else:
        names = []
    if names:
        return " / ".join(str(n) for n in names)
    return f"(unnamed {highway})" if isinstance(highway, str) else "(unnamed road)"


def safe_capacity(value):
    """Capacity must be a positive number, otherwise treat the edge as 1 lane."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    return value if np.isfinite(value) and value > 0 else 1.0


def find_file(name):
    """Look for a raw dataset file (node_list.csv / edge_list.csv) anywhere under data/."""
    hits = sorted((ROOT / "data").rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {ROOT / 'data'} - put the dataset there")
    return hits[0]


def pick_weight(G):
    """travel_time if every edge has it (output of part A), otherwise fall back to length."""
    if all("travel_time" in d for _, _, d in G.edges(data=True)):
        return "travel_time"
    print("[note] no travel_time on the graph -> using length as the routing weight")
    return "length"


# --------------------------------------------------------------------------------------
# Loading inputs from parts A, B, C  (and placeholders when they are not ready)
# --------------------------------------------------------------------------------------
def load_graph_real(path):
    """Part A: weighted graph saved with nx.write_graphml (node id = osmid)."""
    G = nx.read_graphml(path, node_type=int)
    if G.is_multigraph():                          # part A should already have merged parallel edges
        G = nx.Graph(G)
    for _, _, d in G.edges(data=True):             # graphml can return numbers as strings -> floats
        for attr in ("length", "speed", "travel_time", "capacity"):
            if attr in d:
                d[attr] = float(d[attr])
    # coordinates are needed for the maps; fill them from node_list.csv if the graphml lacks them
    if any("x" not in d or "y" not in d for _, d in G.nodes(data=True)):
        print("[note] graphml has no node coordinates -> taking x, y from node_list.csv")
        nodes = pd.read_csv(find_file("node_list.csv"))
        xy = nodes.set_index("osmid")[["x", "y"]].to_dict("index")
        for n, d in G.nodes(data=True):
            d["x"], d["y"] = float(xy[n]["x"]), float(xy[n]["y"])
    else:
        for _, d in G.nodes(data=True):
            d["x"], d["y"] = float(d["x"]), float(d["y"])
    return G


def build_placeholder_graph():
    """Stand-in for part A: undirected simple graph, largest component, weight = length, capacity = 1."""
    nodes = pd.read_csv(find_file("node_list.csv"))
    edges = pd.read_csv(find_file("edge_list.csv"))
    edges = edges[edges["u"] != edges["v"]].copy()                     # drop self-loops
    edges["a"] = edges[["u", "v"]].min(axis=1)                         # canonical direction ...
    edges["b"] = edges[["u", "v"]].max(axis=1)
    edges = edges.sort_values("length").drop_duplicates(["a", "b"])    # ... keep the shortest parallel edge
    G = nx.Graph()
    for r in edges.to_dict("records"):
        G.add_edge(int(r["a"]), int(r["b"]), length=float(r["length"]), capacity=1.0,
                   name=r.get("name"), highway=r.get("highway"))
    xy = nodes.set_index("osmid")[["x", "y"]].to_dict("index")         # node coordinates for the maps
    for n, d in G.nodes(data=True):
        d["x"], d["y"] = float(xy[n]["x"]), float(xy[n]["y"])
    biggest = max(nx.connected_components(G), key=len)                 # keep the largest component
    return G.subgraph(biggest).copy()


def load_danger_nodes(G):
    """Part B: danger_nodes.csv (osmid, danger) -> set of danger node ids that exist in G."""
    df = pd.read_csv(PROC / "danger_nodes.csv")
    flag = df["danger"].astype(str).str.lower().isin(["true", "1"])   # works for bool or "True"/"False"
    nodes = set(df.loc[flag, "osmid"].astype(int))
    missing = len(nodes - set(G))
    if missing:
        print(f"[note] {missing} danger nodes are not in the graph (outside the largest component) -> ignored")
    return nodes & set(G)


def placeholder_danger_nodes(G, share=0.20):
    """Stand-in for part B: the southern 20% of nodes by y (roughly the coastline)."""
    y = pd.Series({n: d["y"] for n, d in G.nodes(data=True)})
    return set(y[y <= y.quantile(share)].index)


def load_zones_and_assembly_points():
    """Part C: zones.csv (osmid, zone) and assembly_points.csv (zone, osmid) -> two dicts."""
    z = pd.read_csv(PROC / "zones.csv")
    a = pd.read_csv(PROC / "assembly_points.csv")
    zones = dict(zip(z["osmid"].astype(int), z["zone"]))               # node -> zone
    assembly = dict(zip(a["zone"], a["osmid"].astype(int)))            # zone -> assembly node
    return zones, assembly


def placeholder_assembly_points(G, danger_nodes, k=4):
    """Stand-in for part C: k safe nodes spread out (west->east) just north of the danger zone."""
    y = {n: d["y"] for n, d in G.nodes(data=True)}
    limit = np.quantile(list(y.values()), 0.35)
    band = sorted((n for n in G if n not in danger_nodes and y[n] <= limit), key=lambda n: G.nodes[n]["x"])
    idx = np.linspace(0, len(band) - 1, k + 2)[1:-1].astype(int)       # evenly spaced, skip the extremes
    return [band[i] for i in idx]


def load_inputs(strict=False):
    """Load every input; use a placeholder (or fail if strict) for whatever is missing."""
    def real_or_stub(files, label):
        ok = all((PROC / f).exists() for f in files)
        if not ok and strict:
            raise FileNotFoundError(f"--strict: missing {', '.join(files)} in {PROC}")
        print(f"[{'real' if ok else 'PLACEHOLDER'}] {label}")
        return ok

    G = load_graph_real(PROC / "graph.graphml") if real_or_stub(["graph.graphml"], "graph (part A)") \
        else build_placeholder_graph()

    danger = load_danger_nodes(G) if real_or_stub(["danger_nodes.csv"], "danger zone (part B)") \
        else placeholder_danger_nodes(G)

    if real_or_stub(["zones.csv", "assembly_points.csv"], "zones + assembly points (part C)"):
        zones, assembly = load_zones_and_assembly_points()
    else:
        zones, assembly = None, placeholder_assembly_points(G, danger)   # list -> "go to nearest" mode
    return G, danger, assembly, zones


# --------------------------------------------------------------------------------------
# Core analysis (all functions take graph + danger nodes + assembly points as arguments)
# --------------------------------------------------------------------------------------
def assign_targets(G, danger_nodes, assembly_points, zones=None, weight="travel_time"):
    """
    Decide where each danger node goes -> {danger node: assembly node}.
      - zones given and assembly_points is a dict {zone: node}: go to the assembly point of own zone
      - otherwise (placeholder mode): go to the nearest assembly point
    """
    zone_mode = zones is not None and isinstance(assembly_points, dict)
    ap_nodes = set(assembly_points.values()) if isinstance(assembly_points, dict) else set(assembly_points)
    # one multi-source Dijkstra gives the nearest assembly point of every node (also the fallback)
    _, nearest_paths = nx.multi_source_dijkstra(G, ap_nodes, weight=weight)

    targets, fallback = {}, 0
    for n in danger_nodes:
        ap = assembly_points.get(zones.get(n)) if zone_mode else None
        if ap is None:
            ap = nearest_paths[n][0]                    # path starts at the nearest assembly point
            fallback += zone_mode                       # counted only if a zone had no assembly point
        targets[n] = ap
    if fallback:
        print(f"[note] {fallback} danger nodes have no zone/assembly point -> sent to the nearest one")
    return targets


def find_routes(G, targets, weight="travel_time"):
    """Shortest path from every danger node to its assembly point -> {danger node: [node, ..., assembly]}."""
    by_ap = defaultdict(list)                           # group danger nodes by their assembly point
    for n, ap in targets.items():
        by_ap[ap].append(n)
    routes = {}
    for ap, sources in by_ap.items():
        # one Dijkstra per assembly point (graph is undirected, so the reversed path is also shortest)
        paths = nx.single_source_dijkstra_path(G, ap, weight=weight)
        for n in sources:
            routes[n] = paths[n][::-1]                  # reverse: danger node -> assembly point
    return routes


def compute_loads(routes, people=1.0):
    """
    Count how many routes use each edge / node.
      people: people per danger node (number, or dict {node: number}); 1 = uniform distribution.
    Node load counts intermediate nodes only (like betweenness: endpoints excluded).
    """
    edge_load, node_load = Counter(), Counter()
    for src, path in routes.items():
        w = people.get(src, 1.0) if isinstance(people, dict) else people
        for a, b in zip(path[:-1], path[1:]):
            edge_load[edge_key(a, b)] += w
        for n in path[1:-1]:
            node_load[n] += w
    return edge_load, node_load


def ordinary_edge_betweenness(G, weight="travel_time", k=None, seed=42):
    """Ordinary edge betweenness (all pairs). k = number of sampled sources, None = exact (slower)."""
    eb = nx.edge_betweenness_centrality(G, k=k, normalized=True, weight=weight, seed=seed)
    return {edge_key(u, v): val for (u, v), val in eb.items()}


def build_edge_table(G, edge_load, betweenness, ap_nodes):
    """One row per edge: attributes, load, bottleneck_score = load / capacity, ordinary betweenness."""
    rows = []
    for u, v, d in G.edges(data=True):
        a, b = edge_key(u, v)
        cap = safe_capacity(d.get("capacity"))
        load = float(edge_load.get((a, b), 0.0))
        rows.append(dict(
            u=a, v=b,
            name=clean_name(d.get("name"), d.get("highway")),
            highway=d.get("highway"),
            length_m=d.get("length"),
            travel_time_s=d.get("travel_time"),
            capacity=cap,
            load=load,
            bottleneck_score=load / cap,
            betweenness=betweenness.get((a, b), np.nan) if betweenness else np.nan,
            touches_assembly=(a in ap_nodes) or (b in ap_nodes),
        ))
    return pd.DataFrame(rows)


def build_node_table(G, node_load, danger_nodes, ap_nodes):
    """One row per node: coordinates, danger / assembly flags, evacuation load through the node."""
    return pd.DataFrame([dict(osmid=n, x=d["x"], y=d["y"], is_danger=n in danger_nodes,
                              is_assembly=n in ap_nodes, node_load=float(node_load.get(n, 0.0)))
                         for n, d in G.nodes(data=True)])


def run_bottleneck_analysis(G, danger_nodes, assembly_points, zones=None, weight=None,
                            people=1.0, k=None, with_betweenness=True, seed=42):
    """Whole part D pipeline. Returns (edge_table, node_table, routes)."""
    weight = weight or pick_weight(G)
    ap_nodes = set(assembly_points.values()) if isinstance(assembly_points, dict) else set(assembly_points)

    targets = assign_targets(G, danger_nodes, assembly_points, zones, weight)    # who goes where
    routes = find_routes(G, targets, weight)                                     # shortest paths
    edge_load, node_load = compute_loads(routes, people)                         # load per edge / node

    btw = None
    if with_betweenness:
        print(f"computing ordinary edge betweenness ({'exact' if k is None else f'k={k}'}) ...")
        btw = ordinary_edge_betweenness(G, weight, k, seed)

    edge_df = build_edge_table(G, edge_load, btw, ap_nodes)
    node_df = build_node_table(G, node_load, danger_nodes, ap_nodes)
    return edge_df, node_df, routes


def select_top(edge_df, n=TOP_N, exclude_ap_edges=False):
    """Top-n edges by bottleneck_score (ties broken by load)."""
    df = edge_df[~edge_df["touches_assembly"]] if exclude_ap_edges else edge_df
    top = df.sort_values(["bottleneck_score", "load"], ascending=False).head(n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    return top


def compare_load_vs_betweenness(edge_df, top_n=TOP_N):
    """What does the evacuation scenario add compared to plain betweenness? Returns summary numbers."""
    used = edge_df[edge_df["load"] > 0]
    spearman = lambda df: df["load"].rank().corr(df["betweenness"].rank())   # Spearman = Pearson on ranks
    top_load = set(edge_df.nlargest(top_n, "load").index)
    top_btw = set(edge_df.nlargest(top_n, "betweenness").index)
    return dict(share_edges_used=len(used) / len(edge_df),
                spearman_all_edges=spearman(edge_df),
                spearman_used_edges=spearman(used),
                top_overlap=len(top_load & top_btw), top_n=top_n)


# --------------------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------------------
def _segments(G, keys):
    """Line segments [(x1, y1), (x2, y2)] for a list of edge keys."""
    return [[(G.nodes[a]["x"], G.nodes[a]["y"]), (G.nodes[b]["x"], G.nodes[b]["y"])] for a, b in keys]


def _style_map(ax, G, title):
    """Same extent / aspect ratio / title for all maps."""
    xs = np.array([d["x"] for _, d in G.nodes(data=True)])
    ys = np.array([d["y"] for _, d in G.nodes(data=True)])
    mx, my = 0.02 * np.ptp(xs), 0.02 * np.ptp(ys)
    ax.set_xlim(xs.min() - mx, xs.max() + mx)
    ax.set_ylim(ys.min() - my, ys.max() + my)
    if np.abs(xs).max() <= 180 and np.abs(ys).max() <= 90:    # lon/lat -> correct the stretching
        ax.set_aspect(1 / np.cos(np.radians(ys.mean())))
    else:
        ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_axis_off()


def _draw_edge_values(fig, ax, G, values, cmap, label):
    """Grey network + edges with value > 0 coloured and thickened by value (drawn low -> high)."""
    all_keys = [edge_key(u, v) for u, v in G.edges]
    ax.add_collection(LineCollection(_segments(G, all_keys), colors="#d4d4d4", linewidths=0.3, zorder=1))
    keys = sorted((k for k in all_keys if values.get(k, 0) > 0), key=lambda k: values[k])
    if not keys:
        return
    vals = np.array([values[k] for k in keys])
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vals.max())      # gamma<1: separates the many low values
    scaled = np.asarray(norm(vals))
    ax.add_collection(LineCollection(_segments(G, keys), colors=cmap(scaled),
                                     linewidths=0.6 + 2.4 * scaled, zorder=2))
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.65, label=label)


def _overlay_points(ax, G, danger_nodes, ap_nodes):
    """Danger nodes (small red dots) and assembly points (stars)."""
    ax.scatter([G.nodes[n]["x"] for n in danger_nodes], [G.nodes[n]["y"] for n in danger_nodes],
               s=2, c="red", alpha=0.3, zorder=3, label="danger nodes")
    ax.scatter([G.nodes[n]["x"] for n in ap_nodes], [G.nodes[n]["y"] for n in ap_nodes],
               marker="*", s=160, c="gold", edgecolors="black", zorder=5, label="assembly points")
    ax.legend(loc="lower left", fontsize=7)


def plot_load_vs_betweenness(G, edge_df, danger_nodes, ap_nodes, path):
    """Map (a): evacuation load (left) next to ordinary betweenness (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    keys = list(zip(edge_df["u"], edge_df["v"]))
    load = dict(zip(keys, edge_df["load"]))
    _draw_edge_values(fig, axes[0], G, load, plt.cm.plasma_r, "evacuation load (routes per edge)")
    _overlay_points(axes[0], G, danger_nodes, ap_nodes)
    _style_map(axes[0], G, "Evacuation load: danger node -> assembly point")
    if edge_df["betweenness"].notna().any():
        btw = dict(zip(keys, edge_df["betweenness"].fillna(0)))
        _draw_edge_values(fig, axes[1], G, btw, plt.cm.viridis_r, "ordinary edge betweenness")
    _style_map(axes[1], G, "Ordinary edge betweenness (all pairs)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bottleneck_score(G, edge_df, top, danger_nodes, ap_nodes, path):
    """Map (b): bottleneck score = load / capacity, top bottlenecks numbered with street names."""
    fig, ax = plt.subplots(figsize=(11, 9))
    score = dict(zip(zip(edge_df["u"], edge_df["v"]), edge_df["bottleneck_score"]))
    _draw_edge_values(fig, ax, G, score, plt.cm.YlOrRd, "bottleneck score = load / capacity")
    _overlay_points(ax, G, danger_nodes, ap_nodes)
    for r in top.to_dict("records"):                        # ring + "rank. street" label at the edge midpoint
        mx = (G.nodes[r["u"]]["x"] + G.nodes[r["v"]]["x"]) / 2
        my = (G.nodes[r["u"]]["y"] + G.nodes[r["v"]]["y"]) / 2
        ax.scatter(mx, my, s=45, facecolors="none", edgecolors="black", linewidths=0.8, zorder=6)
        angle = np.radians(137.5 * r["rank"])               # golden-angle offsets spread neighbouring labels apart
        radius = 28 + 10 * (r["rank"] % 3)                  # label distance from the edge, in points
        ax.annotate(f"{r['rank']}. {r['name']}", xy=(mx, my), xytext=(radius * np.cos(angle), radius * np.sin(angle)),
                    textcoords="offset points", fontsize=6, ha="center", va="center", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6),
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="black"))
    _style_map(ax, G, f"Evacuation bottlenecks (top {len(top)} labelled)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Part D: evacuation routes and bottlenecks")
    parser.add_argument("--strict", action="store_true", help="fail if a real input file is missing")
    parser.add_argument("--k", type=int, default=None, help="sampled sources for ordinary betweenness (default: exact)")
    parser.add_argument("--exclude-ap-edges", action="store_true", help="skip edges touching assembly points in top-20")
    args = parser.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    G, danger, assembly, zones = load_inputs(args.strict)
    ap_nodes = set(assembly.values()) if isinstance(assembly, dict) else set(assembly)
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | "
          f"danger nodes: {len(danger)} | assembly points: {len(ap_nodes)}")

    edge_df, node_df, routes = run_bottleneck_analysis(G, danger, assembly, zones, k=args.k)
    top = select_top(edge_df, TOP_N, args.exclude_ap_edges)

    # tables
    edge_df.to_csv(PROC / "edge_loads.csv", index=False)
    node_df.to_csv(PROC / "node_loads.csv", index=False)
    top.to_csv(PROC / "bottleneck_top20.csv", index=False)

    # maps
    plot_load_vs_betweenness(G, edge_df, danger, ap_nodes, FIG / "D_load_vs_betweenness.png")
    plot_bottleneck_score(G, edge_df, top, danger, ap_nodes, FIG / "D_bottleneck_score.png")

    # short summary for the report / the meeting
    cmp = compare_load_vs_betweenness(edge_df)
    print(f"\nroutes: {len(routes)} | edges used by at least one route: {cmp['share_edges_used']:.1%}")
    print(f"max load: {edge_df['load'].max():.0f} | max bottleneck score: {edge_df['bottleneck_score'].max():.1f}")
    print(f"Spearman(load, betweenness): all edges {cmp['spearman_all_edges']:.2f}, "
          f"used edges only {cmp['spearman_used_edges']:.2f}")
    print(f"overlap of top-{cmp['top_n']} by load and top-{cmp['top_n']} by betweenness: {cmp['top_overlap']}")
    print(f"\nTop {len(top)} bottlenecks:")
    print(top[["rank", "name", "highway", "capacity", "load", "bottleneck_score"]].to_string(index=False))
    print(f"\nsaved tables to {PROC} and maps to {FIG}")


if __name__ == "__main__":
    main()
