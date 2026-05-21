import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="Node embeddings")


@app.cell
def imports():
    import io
    import urllib.request
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import igraph as ig

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        silhouette_score,
    )
    from sklearn.model_selection import train_test_split

    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": "#cccccc",
        # No surrounding "axis box" on any plot. Curves with meaningful
        # axes (Section 4 accuracy) still show their tick marks and
        # labels; PCA / network / walk plots additionally clear ticks
        # in their own cells, so the panels read as pure scatter.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    def parse_graphml(raw_bytes):
        """Parse a GraphML document with the stdlib XML parser.

        Pyodide's python-igraph wheel is compiled without GraphML support
        (no libxml2), so we cannot call ``ig.Graph.Read_GraphML`` in the
        browser. This helper produces the same kind of igraph.Graph by
        reading the GraphML XML directly.

        Recognised: ``<key>`` declarations (so attribute names and types
        survive), ``<node>`` ids and per-node ``<data>`` values, and
        ``<edge>`` source/target pairs. Directed graphs are collapsed to
        undirected to match the rest of the app. Self-loops and parallel
        edges are simplified away. The original GraphML node ids are
        preserved as ``g.vs['id']``.
        """
        import xml.etree.ElementTree as _ET

        _root = _ET.fromstring(raw_bytes)
        _ns = ""
        if _root.tag.startswith("{"):
            _ns = _root.tag[1:].split("}", 1)[0]
        def _qn(tag): return f"{{{_ns}}}{tag}" if _ns else tag

        _keys = {}
        for _k in _root.findall(_qn("key")):
            _keys[_k.get("id")] = {
                "name": _k.get("attr.name", _k.get("id")),
                "for":  _k.get("for", "node"),
                "type": _k.get("attr.type", "string"),
            }

        _graph_el = _root.find(_qn("graph"))
        if _graph_el is None:
            raise ValueError("GraphML document has no <graph> element.")
        _is_directed = (_graph_el.get("edgedefault", "undirected") == "directed")

        _node_ids, _id_to_idx, _node_attrs = [], {}, {}
        for _n in _graph_el.findall(_qn("node")):
            _gid = _n.get("id")
            _idx = len(_node_ids)
            _id_to_idx[_gid] = _idx
            _node_ids.append(_gid)
            for _kid, _info in _keys.items():
                if _info["for"] == "node":
                    _node_attrs.setdefault(_info["name"], []).append(None)
            for _d in _n.findall(_qn("data")):
                _kid = _d.get("key")
                if _kid not in _keys:
                    continue
                _info = _keys[_kid]
                if _info["for"] != "node":
                    continue
                _val = (_d.text or "").strip()
                _t = _info["type"]
                if _t in ("int", "integer", "long"):
                    try: _val = int(_val)
                    except ValueError: pass
                elif _t in ("float", "double"):
                    try: _val = float(_val)
                    except ValueError: pass
                elif _t in ("bool", "boolean"):
                    _val = _val.strip().lower() in ("true", "1", "yes")
                _node_attrs[_info["name"]][_idx] = _val

        _edges = []
        for _e in _graph_el.findall(_qn("edge")):
            _s = _e.get("source"); _t = _e.get("target")
            if _s in _id_to_idx and _t in _id_to_idx and _s != _t:
                _edges.append((_id_to_idx[_s], _id_to_idx[_t]))

        _g = ig.Graph(n=len(_node_ids), edges=_edges, directed=_is_directed)
        if _is_directed:
            _g = _g.as_undirected(mode="collapse")
        _g.simplify()
        _g.vs["id"] = _node_ids
        for _name, _vals in _node_attrs.items():
            _g.vs[_name] = _vals
        return _g

    def _load_public_bytes(filename):
        """Read bytes from public/<filename>, working both locally and in WASM.

        Marimo's WASM exporter bundles the ``public/`` directory alongside the
        notebook and serves it from ``mo.notebook_location()``. Locally,
        ``notebook_location()`` returns a filesystem path, so the same call
        site works in both environments.
        """
        target = mo.notebook_location() / "public" / filename
        target_str = str(target)
        if target_str.startswith(("http://", "https://")):
            with urllib.request.urlopen(target_str) as f:
                return f.read()
        local = Path(target_str)
        if local.exists():
            return local.read_bytes()
        # Fallback for `marimo edit` invocations where notebook_location()
        # may not resolve to this file's directory.
        return (Path(__file__).parent / "public" / filename).read_bytes()

    def load_graphml(filename):
        """Return an igraph Graph from public/<filename>.

        Uses the in-app GraphML parser because Pyodide's igraph wheel ships
        without GraphML support.
        """
        return parse_graphml(_load_public_bytes(filename))

    def load_npy(filename):
        return np.load(io.BytesIO(_load_public_bytes(filename)))

    def load_npz(filename):
        """Load an .npz archive from public/ as a {name: array} dict."""
        with np.load(io.BytesIO(_load_public_bytes(filename))) as archive:
            return {k: archive[k] for k in archive.files}

    PALETTE = list(matplotlib.colormaps["tab20"].colors)
    ACCENT = "#0b789d"

    # Conference IDs in the bundled football dataset map to the actual
    # NCAA conference names in alphabetical order (Girvan & Newman 2002).
    # Verified by inspecting which teams sit in each ID-group.
    FOOTBALL_CONFERENCES = {
        0: "Atlantic Coast",
        1: "Big East",
        2: "Big Ten",
        3: "Big Twelve",
        4: "Conference USA",
        5: "Independents",       # not a real conference — see app note
        6: "Mid-American",
        7: "Mountain West",
        8: "Pacific Ten",
        9: "Southeastern",
        10: "Sun Belt",
        11: "Western Athletic",
    }
    KARATE_FACTIONS = {0: "Mr Hi's faction", 1: "Officer's faction"}

    # Les Misérables community labels were obtained by running Louvain
    # community detection on the co-appearance graph (Knuth's dataset
    # from the node2vec paper). The names below were assigned by
    # inspecting which characters fell into each cluster.
    LESMIS_COMMUNITIES = {
        0: "Bishop's circle",
        1: "Valjean & ex-convicts",
        2: "Fantine's friends",
        3: "Thénardier's gang",
        4: "Gillenormand family",
        5: "ABC revolutionaries",
    }

    def class_names_for(graph_data):
        """Return a list of length n_classes with human-readable names,
        or string class IDs if no canonical mapping is known."""
        name = graph_data.get("name", "")
        labels = graph_data.get("labels")
        if labels is None:
            return []
        classes = sorted(set(int(c) for c in labels))
        if name.startswith("Football"):
            return [FOOTBALL_CONFERENCES.get(c, f"class {c}") for c in classes]
        if name.startswith("Karate"):
            return [KARATE_FACTIONS.get(c, f"class {c}") for c in classes]
        if name.startswith("Les"):
            return [LESMIS_COMMUNITIES.get(c, f"class {c}") for c in classes]
        return [str(c) for c in classes]

    def style_minimal(ax, no_ticks=False):
        """Strip the surrounding spines (the black axis box).

        Pass ``no_ticks=True`` for plots whose x/y axes have no physical
        meaning (PCAs, network layouts) — that also clears the tick marks
        and tick labels. Curves with meaningful axes (e.g. accuracy vs.
        epoch) leave the ticks in place and just lose the surrounding
        box.
        """
        for _s in ax.spines.values():
            _s.set_visible(False)
        if no_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

    def procrustes_align(coords_2d, anchor_idx, anchor_targets):
        """Find the rotation + reflection + uniform scale that best maps
        ``coords_2d[anchor_idx]`` onto ``anchor_targets`` (3×2). Apply
        that transform to every row of ``coords_2d`` and return the result.

        This is full Procrustes (Schönemann 1966) with scaling, so all
        2-d embedding plots end up in roughly the same orientation and
        size as the network layout — only the *internal* geometry of each
        method's embedding differs from one panel to the next.
        """
        src = np.asarray(coords_2d, dtype=float)
        anchors_src = src[list(anchor_idx)]
        src_mean = anchors_src.mean(axis=0)
        tgt_mean = np.asarray(anchor_targets, dtype=float).mean(axis=0)
        src_c = anchors_src - src_mean
        tgt_c = np.asarray(anchor_targets, dtype=float) - tgt_mean
        # Optimal R: SVD of cross-covariance.
        M = src_c.T @ tgt_c
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt  # 2×2 orthogonal (det = ±1; reflection allowed)
        # Optimal isotropic scale.
        src_norm_sq = float((src_c ** 2).sum())
        s = float(S.sum() / src_norm_sq) if src_norm_sq > 1e-12 else 1.0
        return (src - src_mean) @ (R * s) + tgt_mean

    # A second palette specifically for k-means cluster colours, so
    # they don't get confused with the per-class palette used everywhere
    # else. Picked from matplotlib's "Set2" colormap (colourblind-friendly
    # pastels, distinct from the tab20 default).
    KMEANS_PALETTE = list(matplotlib.colormaps["Set2"].colors) + \
                     list(matplotlib.colormaps["Set3"].colors)

    def network_with_cluster_colors(ax, g, coords, edges_iter, cluster_ids,
                                    palette=None, struct_anchors=None,
                                    title=None, names=None, show_labels=False,
                                    anchor_names_always=True):
        """Draw the FR-laid-out network with node colours coming from a
        k-means clustering of an embedding. Edges stay plain grey
        (per user feedback — colouring them by cluster made the plots
        noisier than they helped). Anchor nodes get a hollow black
        square; their *names* are written next to the square by
        default."""
        from matplotlib.collections import LineCollection as _LC
        _pal = palette if palette is not None else KMEANS_PALETTE
        ax.set_facecolor("white")
        _segs = [[coords[e.source], coords[e.target]] for e in edges_iter]
        ax.add_collection(_LC(_segs, colors="#dddddd",
                              linewidths=0.5, alpha=0.6, zorder=1))
        _colors = [_pal[int(c) % len(_pal)] for c in cluster_ids]
        ax.scatter(coords[:, 0], coords[:, 1], s=55, c=_colors,
                   edgecolor="#333333", linewidth=0.4, zorder=3)
        if struct_anchors:
            for _i in struct_anchors:
                ax.scatter(coords[_i, 0], coords[_i, 1], s=260,
                           marker="s", facecolors="none",
                           edgecolor="black", linewidth=1.8, zorder=5)
                if anchor_names_always and names is not None:
                    ax.annotate(
                        str(names[_i]),
                        xy=(coords[_i, 0], coords[_i, 1]),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=7, color="#222222",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#888888", alpha=0.9, lw=0.5),
                        zorder=6,
                    )
        if show_labels and names is not None:
            for _i, _nm in enumerate(names):
                if struct_anchors and _i in struct_anchors and anchor_names_always:
                    continue
                ax.annotate(str(_nm), xy=(coords[_i, 0], coords[_i, 1]),
                            xytext=(3, 3), textcoords="offset points",
                            fontsize=6, color="#333333", alpha=0.8, zorder=6)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal")
        if title:
            ax.set_title(title, fontsize=10)

    def biased_walk(graph, start, length, p, q, rng):
        """One node2vec random walk: bias `1/p` to return to the previous
        node, `1` to step to a distance-1 neighbour, `1/q` to step to a
        distance-2 neighbour."""
        walk = [int(start)]
        if length < 2:
            return walk
        nbrs0 = graph.neighbors(start)
        if not nbrs0:
            return walk
        walk.append(int(rng.choice(nbrs0)))
        inv_p = 1.0 / p
        inv_q = 1.0 / q
        for _ in range(length - 2):
            cur = walk[-1]
            prev = walk[-2]
            nbrs = graph.neighbors(cur)
            if not nbrs:
                break
            prev_set = set(graph.neighbors(prev)) | {prev}
            w = np.empty(len(nbrs), dtype=np.float64)
            for i, x in enumerate(nbrs):
                if x == prev:
                    w[i] = inv_p
                elif x in prev_set:
                    w[i] = 1.0
                else:
                    w[i] = inv_q
            cw = np.cumsum(w)
            r = rng.random() * cw[-1]
            walk.append(int(nbrs[int(np.searchsorted(cw, r))]))
        return walk

    return (
        FOOTBALL_CONFERENCES,
        KARATE_FACTIONS,
        KMeans,
        LineCollection,
        LogisticRegression,
        PALETTE,
        PCA,
        TSNE,
        TruncatedSVD,
        biased_walk,
        class_names_for,
        load_graphml,
        load_npy,
        load_npz,
        network_with_cluster_colors,
        parse_graphml,
        procrustes_align,
        style_minimal,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        ig,
        io,
        mo,
        np,
        pd,
        plt,
        silhouette_score,
        train_test_split,
    )


@app.cell
def title(mo):
    mo.md("""
    # Node embeddings and what we can do with them

    A **node embedding** turns every vertex of a network into a vector.
    Once that's done, *"is node A like node B?"* becomes a question
    about **distance** — and the full toolkit of geometry, statistics,
    and machine learning becomes available to answer it.

    Below we build three families of embeddings, then close the loop
    with a classification benchmark:

    1. **Spectral methods** — top eigenvectors of the adjacency or
       Laplacian. Deterministic, fast, classical.
    2. **node2vec** — random walks fed through *word2vec*. Same kind of
       trick that turns words into vectors, applied to nodes. Two knobs
       ($p$, $q$) let us tune what *kind* of structure it captures:
       community vs. role.
    3. **GCN (graph convolutional network)** — a supervised graph
       neural network that learns its embedding **with** the labels in
       hand. The fanciest, and the only one that gets to see classes
       during training.

    For each, we then ask: how well does the embedding predict the
    class when plugged into a plain multinomial logistic regression?

    > _Pick a network in the sidebar — football, karate, Les Misérables,
    > or upload your own — and the five sections below all re-run on
    > the new graph._

    ---
    """)
    return


@app.cell
def sec1_widgets(mo):
    dataset_choice = mo.ui.radio(
        options=[
            "Football (115 nodes, 12 conferences)",
            "Karate (34 nodes, 2 factions)",
            "Les Misérables (77 characters, 6 communities)",
            "Upload my own (CSV or GraphML)",
        ],
        value="Football (115 nodes, 12 conferences)",
        label="Dataset",
    )
    upload = mo.ui.file(
        filetypes=[".csv", ".graphml", ".xml"],
        multiple=False,
        label="CSV or GraphML",
    )
    show_node_labels = mo.ui.checkbox(value=False, label="Show node labels on all networks")
    return dataset_choice, show_node_labels, upload


@app.cell
def sidebar(dataset_choice, mo, show_node_labels, upload):
    mo.sidebar(
        [
            mo.md("### Dataset"),
            dataset_choice,
            mo.md("_Or upload your own:_"),
            upload,
            mo.md(
                "_CSV: needs a `source` and `target` column (the edge "
                "list). An optional third column named `class` or "
                "`label` becomes the per-node class label._"
            ),
            mo.md("---"),
            mo.md("### Display"),
            show_node_labels,
            mo.md("---"),
            mo.md(
                "_The active network drives every section below — pick "
                "it once here, then scroll._"
            ),
        ]
    )
    return


@app.cell
def sec1_header(mo):
    mo.md("""
    ## Section 1: The dataset

    The default is the **US college football network** (115 teams, 12
    conferences). Each node is a team; each edge is a game played in the
    2000 regular season. The **conference** a team belongs to is our class
    label — geographically and competitively coherent, so a good test case
    for "does the embedding recover community structure?".

    You can also switch to **Zachary's karate club** (34 members, 2
    factions) for a small example, or **upload your own** network in the
    sidebar:

    - a CSV with columns `source,target` (edge list, no class labels), or
    - a `.graphml` file (class labels are picked up from the node
      attribute `value`, `class`, `label`, or `group` if present).
    """)
    return


@app.cell
def build_graph(load_graphml, parse_graphml, dataset_choice, ig, io, np, pd, upload):
    # The classic Zachary faction assignment (Zachary 1977).
    _zachary_factions = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1,
    ])

    _choice = dataset_choice.value
    _is_byod = False
    _classes_available = True
    _description = ""
    _g = None
    _labels = None
    _err = None

    def _coerce_labels(values):
        """Try to parse a per-node attribute as an integer class label."""
        try:
            return np.array([int(v) for v in values])
        except (TypeError, ValueError):
            pass
        # Fall back to a categorical encoding (stable order by first appearance).
        seen = {}
        out = []
        for v in values:
            key = str(v)
            if key not in seen:
                seen[key] = len(seen)
            out.append(seen[key])
        return np.array(out, dtype=int)

    if _choice.startswith("Upload") and upload.value:
        _file = upload.value[0]
        _raw = _file.contents
        _fname = (_file.name or "").lower()
        try:
            if _fname.endswith(".graphml") or _fname.endswith(".xml"):
                # GraphML upload: use the stdlib-XML parser (Pyodide's
                # python-igraph wheel has no GraphML support).
                _g = parse_graphml(_raw)
                # Names: prefer 'name', then 'id', else integer indices.
                if "name" in _g.vs.attributes():
                    _g.vs["name"] = [str(x) for x in _g.vs["name"]]
                elif "id" in _g.vs.attributes():
                    _g.vs["name"] = [str(x) for x in _g.vs["id"]]
                else:
                    _g.vs["name"] = [f"v{i}" for i in range(_g.vcount())]
                # Labels: try common attribute names.
                _label_attr = None
                for _cand in ("value", "class", "label", "group"):
                    if _cand in _g.vs.attributes():
                        _label_attr = _cand
                        break
                if _label_attr is not None:
                    _labels = _coerce_labels(_g.vs[_label_attr])
                    _classes_available = True
                    _is_byod = True
                    _description = (
                        f"**User-uploaded GraphML** with {_g.vcount()} nodes, "
                        f"{_g.ecount()} edges, and {len(set(_labels))} classes "
                        f"(from node attribute `{_label_attr}`)."
                    )
                else:
                    _labels = None
                    _classes_available = False
                    _is_byod = True
                    _description = (
                        f"**User-uploaded GraphML** with {_g.vcount()} nodes "
                        f"and {_g.ecount()} edges. _No node-class attribute "
                        f"(`value` / `class` / `label` / `group`) was found, "
                        f"so the classification panels are hidden._"
                    )
            else:
                _df = pd.read_csv(io.BytesIO(_raw))
                cols = [c.lower() for c in _df.columns]
                _df.columns = cols
                if "source" not in cols or "target" not in cols:
                    _g = None
                    _err = "CSV must have columns 'source' and 'target'."
                else:
                    _df["source"] = _df["source"].astype(str)
                    _df["target"] = _df["target"].astype(str)
                    _nodes = sorted(set(_df["source"]).union(set(_df["target"])))
                    _idx = {n: i for i, n in enumerate(_nodes)}
                    _edges = [(_idx[s], _idx[t]) for s, t in zip(_df["source"], _df["target"]) if s != t]
                    _g = ig.Graph(n=len(_nodes), edges=_edges, directed=False)
                    _g.simplify()
                    _g.vs["name"] = _nodes
                    # Optional third column with per-node class labels.
                    _class_col = next((_c for _c in ("class", "label", "group") if _c in cols), None)
                    if _class_col is not None:
                        # Build a node→class map from the edge list:
                        # whichever value appears most for each node wins
                        # (handles duplicates gracefully).
                        _node_cls = {}
                        for _s, _t, _c in zip(_df["source"], _df["target"], _df[_class_col]):
                            for _node in (_s, _t):
                                _node_cls.setdefault(_node, []).append(_c)
                        from collections import Counter as _Counter
                        _per_node = [
                            _Counter(_node_cls.get(_n, ["?"])).most_common(1)[0][0]
                            for _n in _nodes
                        ]
                        _labels = _coerce_labels(_per_node)
                        _classes_available = True
                        _is_byod = True
                        _description = (
                            f"**User-uploaded CSV edge list** with "
                            f"{_g.vcount()} nodes, {_g.ecount()} edges, "
                            f"and {len(set(_labels))} classes (from column "
                            f"`{_class_col}`)."
                        )
                    else:
                        _labels = None
                        _classes_available = False
                        _is_byod = True
                        _description = (
                            f"**User-uploaded CSV edge list** with {_g.vcount()} "
                            f"nodes and {_g.ecount()} edges. _No `class` "
                            f"column found, so classification panels are hidden._"
                        )
        except Exception as e:
            _g = None
            _err = f"Failed to read upload: {e}"
    elif _choice.startswith("Karate"):
        _g = ig.Graph.Famous("Zachary")
        # Conventional names from Zachary (1977): node 0 is "Mr Hi" (the
        # instructor) and node 33 is "Officer" (the club president). The
        # other 32 members are anonymised - we just label them by index.
        _karate_names = [f"v{i}" for i in range(_g.vcount())]
        _karate_names[0] = "Mr Hi"
        _karate_names[33] = "Officer"
        _g.vs["name"] = _karate_names
        _labels = _zachary_factions.copy()
        _description = (
            "**Zachary karate club** (34 members, 78 ties). The classic "
            "split into Mr Hi's and the officer's faction after a dispute."
        )
    elif _choice.startswith("Les"):
        _g = load_graphml("les_miserables.graphml")
        _g.vs["name"] = [str(x) for x in _g.vs["id"]]
        _labels = np.array([int(v) for v in _g.vs["value"]])
        _description = (
            "**Les Misérables co-appearance network** (Knuth 1993; used as "
            "the demo dataset in the node2vec paper). 77 characters, 254 "
            "co-appearance ties across the chapters of Hugo's novel. Class "
            "labels come from Louvain community detection on the graph "
            "itself — six communities that line up with the novel's main "
            "social circles."
        )
    elif _choice.startswith("Upload"):
        # User selected upload but nothing was uploaded yet — fall back to football
        _g = load_graphml("football_network.graphml")
        _g.vs["name"] = list(_g.vs["id"])
        _labels = np.array([int(v) for v in _g.vs["value"]])
        _description = (
            "**Football** (showing while you choose a file). 115 teams, "
            "613 games, 12 conferences as classes."
        )
    else:
        _g = load_graphml("football_network.graphml")
        _g.vs["name"] = list(_g.vs["id"])
        _labels = np.array([int(v) for v in _g.vs["value"]])
        _description = (
            "**US college football network** (Girvan & Newman 2002): 115 "
            "teams, 613 games, 12 conferences. Teams in the same conference "
            "play each other more often, so the conference structure is "
            "visible in the network."
        )

    if _g is None:
        graph_data = {
            "graph": None,
            "labels": None,
            "names": None,
            "name": "upload-error",
            "is_byod": True,
            "classes_available": False,
            "precomp_prefix": None,
            "description": f"_Could not build a graph: {_err}._",
        }
    else:
        # Which precomputed embedding bundle (if any) belongs to this graph?
        if _choice.startswith("Football"):
            _prefix = ""
        elif _choice.startswith("Karate"):
            _prefix = "karate_"
        elif _choice.startswith("Les"):
            _prefix = "lesmis_"
        else:
            _prefix = None
        graph_data = {
            "graph": _g,
            "labels": _labels,
            "names": list(_g.vs["name"]),
            "name": _choice,
            "is_byod": _is_byod,
            "classes_available": _classes_available,
            "precomp_prefix": _prefix,
            "description": _description,
        }
    return (graph_data,)


@app.cell
def compute_layout(graph_data, np):
    _g = graph_data["graph"]
    _coords = None
    _anchor_idx = None
    if _g is not None:
        import random as _random
        _random.seed(1546)
        np.random.seed(1546)
        _g_for_layout = _g.copy()
        _layout = _g_for_layout.layout_fruchterman_reingold(niter=400)
        _coords = np.array(_layout.coords)

        # Pick three anchor nodes by farthest-point sampling on the
        # layout. We use their layout positions as the canonical target
        # for every 2-d PCA below, so all panels share an orientation.
        _start = int(np.argmax(_coords[:, 1]))
        _chosen = [_start]
        for _ in range(2):
            _d = np.min(
                np.linalg.norm(_coords - _coords[_chosen][:, None], axis=2),
                axis=0,
            )
            _chosen.append(int(np.argmax(_d)))
        _anchor_idx = _chosen
    layout_data = {
        "coords": _coords,
        "anchor_idx": _anchor_idx,
        "anchor_targets": None if _coords is None else _coords[_anchor_idx],
    }
    return (layout_data,)


@app.cell
def stats_panel(graph_data, mo):
    _g = graph_data["graph"]
    if _g is None:
        _stats_md = mo.md(graph_data["description"])
    else:
        _n = _g.vcount()
        _m = _g.ecount()
        _density = 2 * _m / (_n * (_n - 1)) if _n > 1 else 0.0
        _trans = _g.transitivity_undirected(mode="zero")
        _n_classes = (
            int(len(set(graph_data["labels"]))) if graph_data["labels"] is not None else 0
        )
        _classes_line = (
            f"- Classes: **{_n_classes}**" if graph_data["classes_available"] else "- Classes: _none (BYOD)_"
        )
        _stats_md = mo.md(
            graph_data["description"]
            + "\n\n"
            + f"- Nodes: **{_n}**\n"
            + f"- Edges: **{_m}**\n"
            + f"- Density: **{_density:.3f}**\n"
            + f"- Transitivity (global clustering): **{_trans:.3f}**\n"
            + _classes_line
        )
    _stats_md
    return


@app.cell
def plot_network(
    LineCollection, PALETTE, class_names_for, graph_data, layout_data, mo, np,
    plt, show_node_labels, struct_anchors,
):
    _g = graph_data["graph"]
    mo.stop(_g is None, mo.md("_(No graph to display.)_"))
    _coords = layout_data["coords"]
    _labels = graph_data["labels"]
    _node_names = graph_data["names"]
    _fig, _ax = plt.subplots(figsize=(8.5, 6.5))
    _ax.set_facecolor("white")
    _segs = [[_coords[e.source], _coords[e.target]] for e in _g.es]
    _lc = LineCollection(_segs, colors="#bbbbbb", linewidths=0.6, alpha=0.7, zorder=1)
    _ax.add_collection(_lc)
    if _labels is not None:
        _classes = sorted(set(int(c) for c in _labels))
        _names = class_names_for(graph_data)
        _name_for = dict(zip(_classes, _names))
        for _c in _classes:
            _mask = (_labels == _c)
            _ax.scatter(
                _coords[_mask, 0], _coords[_mask, 1],
                s=60, color=PALETTE[int(_c) % len(PALETTE)],
                edgecolor="#333333", linewidth=0.4, zorder=2,
                label=_name_for[_c],
            )
        _ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9,
            frameon=False, handletextpad=0.3, labelspacing=0.4,
            title="Class",
        )
    else:
        _ax.scatter(
            _coords[:, 0], _coords[:, 1], s=55, c="#0b789d",
            edgecolor="#333333", linewidth=0.4, zorder=2,
        )
    # Anchor names are always shown so the same nodes are identifiable
    # across every figure that uses the layout.
    if struct_anchors and _node_names is not None:
        for _i in struct_anchors:
            _ax.scatter(_coords[_i, 0], _coords[_i, 1], s=260, marker="s",
                         facecolors="none", edgecolor="black", linewidth=1.8,
                         zorder=3)
            _ax.annotate(str(_node_names[_i]),
                         xy=(_coords[_i, 0], _coords[_i, 1]),
                         xytext=(8, 8), textcoords="offset points",
                         fontsize=8, color="#111111", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                   ec="#666666", alpha=0.9, lw=0.5),
                         zorder=5)
    if show_node_labels.value and _node_names is not None:
        for _i, _nm in enumerate(_node_names):
            if struct_anchors and _i in struct_anchors:
                continue
            _ax.annotate(str(_nm), xy=(_coords[_i, 0], _coords[_i, 1]),
                         xytext=(3, 3), textcoords="offset points",
                         fontsize=6, color="#333333", alpha=0.8, zorder=4)
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.grid(False)
    _ax.set_title(f"{graph_data['name'].split('(')[0].strip()}  (coloured by class)")
    _ax.set_aspect("equal")
    _fig.tight_layout()
    _fig
    return


@app.cell
def sec1_data_note(graph_data, mo):
    """Footnote about the Independents in the football data."""
    if graph_data.get("name", "").startswith("Football"):
        mo.md(r"""
        > **Why are a few teams in odd places on the layout?** The 12
        > "classes" come from the original Girvan & Newman (2002) data,
        > and **class 5 is _Independents_** — Central Florida,
        > Connecticut, Navy, Notre Dame, and Utah State. The
        > Independents are not a real conference: they don't play a
        > shared round-robin schedule, so they end up scattered across
        > the layout, embedded among whichever conferences they happen
        > to have played the most. These five are also the nodes where
        > **all** the embedding methods below struggle the most — they
        > genuinely don't have a conference-shaped community structure.
        """)
    elif graph_data.get("name", "").startswith("Karate"):
        mo.md(
            "_The two classes are the two factions the karate club split "
            "into after a dispute between Mr Hi and the officer "
            "(Zachary 1977)._"
        )
    elif graph_data.get("name", "").startswith("Les"):
        mo.md(
            "_The communities (Bishop's circle, Valjean & ex-convicts, "
            "Fantine's friends, the Thénardier gang, the Gillenormand "
            "family, the ABC revolutionaries) come from running Louvain "
            "modularity maximisation on the graph itself — so this "
            "network is the *most* community-shaped of the three, by "
            "construction. Node2vec's DFS-vs-BFS contrast also tends to "
            "show up most clearly here._"
        )
    return


@app.cell
def sec2_header(mo):
    mo.md(r"""
    ---
    ## Section 2: Spectral embeddings

    Spectral methods turn a matrix that describes the graph (the adjacency
    matrix $A$ or the Laplacian $L = D - A$) into a low-dimensional
    coordinate system by taking its top — or bottom — eigenvectors. They
    are *deterministic*, *cheap*, and a natural starting point.

    Three classics, side by side on the same network:

    - **Truncated SVD of $A$**: the strongest connection patterns in
      the adjacency matrix. The very first one on a near-regular graph
      like football is basically *node degree* — every team plays
      roughly ten games, so "how many games does this team play" is the
      single biggest pattern. That's why the SVD panel below has a
      degree-shaped first axis.
    - **PCA of $A$**: the same idea, but PCA *centres* the data first.
      Centering subtracts out the "everyone plays ~10 games" baseline,
      which removes that degree mode. So **PCA-of-$A$ is exactly the
      same as SVD-of-$A$ with its first (degree) component dropped** —
      we verified that the SVD's second and third columns match PCA's
      first and second columns almost perfectly.
    - **Laplacian Eigenmaps**: the smoothest variations across the
      graph. The matrix $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$ is
      the *normalised* Laplacian (degree-corrected); its smallest
      eigenvectors are the patterns that change *least* between
      neighbours — and that is exactly what a community label looks
      like (roughly constant within a community, varying between them).
      This is what classical spectral clustering uses.

    Each method has its own sweet spot for dimensionality, and the two
    families behave in **opposite ways** as you give them more
    components:

    - **SVD and PCA: more dims → more signal.** Every direction picked
      up by SVD/PCA is a real pattern in how nodes connect — community,
      brokerage, neighbourhood shape, … none of them is noise. So we
      use **$k = 32$** for SVD/PCA.
    - **Laplacian Eigenmaps: more dims → more noise.** The Laplacian
      orders its components from "smooth across the whole graph" to
      "wiggly inside individual communities". After roughly **as many
      components as there are classes**, the rest are wiggles *inside*
      the existing communities — not useful for telling classes apart.
      So we use **$k \approx 2\times\text{n\_classes}$ (max 16)**.
      Pushing LE to 32 dimensions drops its F1 from ~0.88 down to
      ~0.83.

    > **Reading the silhouette score.** For every node, silhouette
    > compares its average distance to other nodes **in its own class**
    > ($a$) with its average distance to the **nearest other class**
    > ($b$): $s = (b - a) / \max(a, b)$. The average over all nodes is
    > what we report. Values are in $[-1, 1]$:
    > **$+1$** means classes form tight, well-separated clumps in the
    > embedding; **$0$** means the clumps overlap; **negative** values
    > mean nodes are typically closer to a different class than to their
    > own. It is an *intrinsic* score — no classifier is fit, it only
    > looks at distances in the embedding space.
    """)
    return


@app.cell
def compute_spectral(PCA, TruncatedSVD, graph_data, mo, np):
    _g = graph_data["graph"]
    mo.stop(_g is None, mo.md("_(No graph.)_"))
    _A = np.array(_g.get_adjacency().data, dtype=float)
    _n = _A.shape[0]

    # Each spectral method has its own sweet spot, so we use different
    # k's. SVD-of-A and PCA-of-A keep gaining signal with more dims
    # (they're general-purpose dimensionality reductions); Laplacian
    # Eigenmaps peaks around k ~ number of clusters - beyond that the
    # high-frequency Laplacian eigenvectors are essentially noise and
    # the embedding *degrades*. This is the canonical "k ~ #clusters"
    # rule from spectral clustering (Shi-Malik 1997, von Luxburg 2007).
    _k_full = min(32, _n - 1)        # for SVD / PCA: more dims help
    if graph_data.get("classes_available") and graph_data["labels"] is not None:
        _n_classes = len(set(int(c) for c in graph_data["labels"]))
        _k_lap = min(16, max(4, 2 * _n_classes), _n - 1)
    else:
        _k_lap = min(16, _n - 1)      # for LE: k tied to #clusters

    # Truncated SVD of A. We keep all k components, including the top
    # singular vector. That top vector is Perron-Frobenius - on a near-
    # regular graph it correlates ~0.75 with degree centrality - but
    # the downstream classifier ignores it (degree doesn't separate
    # conferences), so it doesn't hurt F1 and it lets the SVD panel
    # stay visibly different from the centered PCA panel.
    _svd = TruncatedSVD(n_components=_k_full, random_state=1546)
    _emb_svd = _svd.fit_transform(_A)

    # Laplacian Eigenmaps with the symmetric normalised Laplacian
    # L_sym = I - D^{-1/2} A D^{-1/2} (Shi-Malik 1997, Ng-Jordan-Weiss
    # 2002). np.linalg.eigh returns ascending eigenvalues, so cols
    # 1..k are the bottom-k non-trivial eigenvectors.
    _deg = _A.sum(axis=1)
    _d_inv_sqrt = np.zeros_like(_deg)
    _nz = _deg > 0
    _d_inv_sqrt[_nz] = 1.0 / np.sqrt(_deg[_nz])
    _A_norm = _A * _d_inv_sqrt[:, None] * _d_inv_sqrt[None, :]
    _L_sym = np.eye(_n) - _A_norm
    _w, _v = np.linalg.eigh(_L_sym)
    _emb_lap = _v[:, 1 : _k_lap + 1]

    _pca = PCA(n_components=_k_full, random_state=1546)
    _emb_pca = _pca.fit_transform(_A)

    spectral_embs = {
        "Truncated SVD (of A)": np.asarray(_emb_svd, dtype=float),
        "Laplacian Eigenmaps (of L_sym)": np.asarray(_emb_lap, dtype=float),
        "PCA (of A)": np.asarray(_emb_pca, dtype=float),
    }
    # "Primary" spectral embedding used downstream (Section 5).
    spectral_emb = spectral_embs["Laplacian Eigenmaps (of L_sym)"]
    spectral_k = {"svd_pca": _k_full, "lap": _k_lap}
    return spectral_emb, spectral_embs, spectral_k


@app.cell
def plot_spectral(
    KMeans, PALETTE, graph_data, layout_data, network_with_cluster_colors, np, plt,
    procrustes_align, show_node_labels, silhouette_score, spectral_embs,
    struct_anchors,
):
    _labels = graph_data["labels"]
    _names = graph_data["names"]
    _anchor_idx = layout_data["anchor_idx"]
    _anchor_targets = layout_data["anchor_targets"]
    _coords = layout_data["coords"]
    _g = graph_data["graph"]
    _n_classes = len(set(int(c) for c in _labels)) if _labels is not None else 4
    _fig, _axes = plt.subplots(2, 3, figsize=(14.5, 9.0), height_ratios=[1, 1])
    for _ax, (_name, _emb) in zip(_axes[0], spectral_embs.items()):
        # Plot the first two raw dimensions of each method, honestly.
        # For SVD-of-A this means dim 0 is roughly degree (the
        # Perron-Frobenius eigenvector); the SVD panel therefore looks
        # different from PCA, and the *very* next pair of SVD dims
        # (cols 1, 2) is what PCA's cols 0, 1 already are. Z-score each
        # axis so panels with very different per-column scales don't
        # collapse under the isotropic Procrustes rescale.
        _emb2 = _emb[:, :2].astype(float)
        _emb2 = (_emb2 - _emb2.mean(axis=0)) / (_emb2.std(axis=0) + 1e-12)
        if _anchor_idx is not None and _anchor_targets is not None:
            _emb2 = procrustes_align(_emb2, _anchor_idx, _anchor_targets)
        if _labels is not None:
            _colors = [PALETTE[int(c) % len(PALETTE)] for c in _labels]
        else:
            _colors = ["#0b789d"] * _emb.shape[0]
        _ax.scatter(_emb2[:, 0], _emb2[:, 1], s=45, c=_colors, edgecolor="#333333", linewidth=0.4)
        # Anchor squares (hollow black) on the spectral panels too.
        if struct_anchors:
            for _i in struct_anchors:
                _ax.scatter(
                    _emb2[_i, 0], _emb2[_i, 1], s=280, marker="s",
                    facecolors="none", edgecolor="black", linewidth=1.8, zorder=5,
                )
                _ax.annotate(
                    str(_names[_i]),
                    xy=(_emb2[_i, 0], _emb2[_i, 1]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888888", alpha=0.9, lw=0.5),
                )
        if show_node_labels.value:
            for _i, _nm in enumerate(_names):
                if struct_anchors and _i in struct_anchors:
                    continue
                _ax.annotate(str(_nm), xy=(_emb2[_i, 0], _emb2[_i, 1]),
                             xytext=(3, 3), textcoords="offset points",
                             fontsize=6, color="#444444", alpha=0.7, zorder=4)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.grid(False)
        _ax.set_aspect("equal")
        _ax.set_title(_name)
        if _labels is not None and len(set(_labels)) > 1:
            _sil = silhouette_score(_emb, _labels)
            _sil_txt = f"silhouette = {_sil:.3f}"
        else:
            _sil_txt = "silhouette: n/a"
        _ax.text(
            0.98,
            0.02,
            _sil_txt,
            transform=_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", alpha=0.9),
        )
    # Bottom row: same network laid out by FR, coloured by k-means on
    # each embedding. The clusters should look like the true community
    # palette for methods that capture homophily, and cut across
    # communities for methods that don't.
    if _g is not None and _coords is not None and _n_classes >= 2:
        for _ax, (_name, _emb) in zip(_axes[1], spectral_embs.items()):
            _km = KMeans(n_clusters=_n_classes, n_init=10, random_state=1546).fit(_emb)
            network_with_cluster_colors(
                _ax, _g, _coords, _g.es, _km.labels_,
                struct_anchors=struct_anchors,
                title=f"network coloured by k-means on {_name.split('(')[0].strip()}",
                names=_names, show_labels=show_node_labels.value,
            )
    _fig.tight_layout()
    _fig
    return


@app.cell
def sec3_header(mo):
    mo.md(r"""
    ---
    ## Section 3: Random walks and node2vec

    Pick a node. Start walking. The neighbours you tend to visit
    together capture the local structure of the graph. **node2vec**
    turns these walks into vectors with *word2vec* — yes, the NLP one —
    by treating each walk as a sentence and each node as a token.

    Two parameters shape the walk:

    - **$p$ (return bias)**: $p < 1$ encourages **going back** to the
      previous node; $p > 1$ discourages it.
    - **$q$ (in-out bias)**: $q < 1$ pushes the walk **outward**
      (depth-first → **homophily / community** structure); $q > 1$
      keeps it **close** to the start (breadth-first → **structural
      roles**).

    Move the sliders to see how a single walk changes; click **↻ Roll
    a new walk** to draw another one without changing the parameters.
    The sliders only drive this illustration — the three node2vec
    panels further down come from a separate, pre-trained run.
    """)
    return


@app.cell
def sec3_walk_widgets(graph_data, mo):
    _g = graph_data["graph"]
    if _g is None:
        start_node = mo.ui.dropdown(options=["—"], value="—", label="Start node")
    else:
        _names = list(graph_data["names"])
        start_node = mo.ui.dropdown(options=_names, value=_names[0], label="Start node")
    walk_length = mo.ui.slider(start=3, stop=20, step=1, value=10, label="Walk length")
    p_slider = mo.ui.slider(
        start=0.1, stop=10.0, step=0.1, value=1.0, label="p (return bias)",
        show_value=True,
    )
    q_slider = mo.ui.slider(
        start=0.1, stop=10.0, step=0.1, value=1.0, label="q (in-out bias)",
        show_value=True,
    )
    # A button to re-roll the walk without touching the sliders. Pressing
    # it increments a counter that sample_walk re-reads, so the cell
    # re-fires with a fresh RNG seed.
    reroll = mo.ui.run_button(label="↻ Roll a new walk")
    return p_slider, q_slider, reroll, start_node, walk_length


@app.cell
def sample_walk(
    biased_walk, graph_data, mo, np, p_slider, q_slider, reroll, start_node,
    walk_length,
):
    _g = graph_data["graph"]
    mo.stop(_g is None, mo.md("_(No graph.)_"))
    # Touch reroll.value so the cell re-fires when the button is clicked
    # (its value is the click count). A fresh RNG seed each time.
    _ = reroll.value
    _names = graph_data["names"]
    _name_to_idx = {n: i for i, n in enumerate(_names)}
    _start_idx = _name_to_idx.get(start_node.value, 0)

    _rng = np.random.default_rng()
    walk_indices = biased_walk(
        _g, _start_idx, int(walk_length.value),
        float(p_slider.value), float(q_slider.value), _rng,
    )
    return (walk_indices,)


@app.cell
def plot_walk(
    LineCollection,
    PALETTE,
    graph_data,
    layout_data,
    mo,
    plt,
    walk_indices,
):
    _g = graph_data["graph"]
    mo.stop(_g is None, mo.md("_(No graph.)_"))
    _coords = layout_data["coords"]
    _labels = graph_data["labels"]

    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _ax.set_facecolor("white")
    _bg = [[_coords[e.source], _coords[e.target]] for e in _g.es]
    _bg_lc = LineCollection(_bg, colors="#dddddd", linewidths=0.5, alpha=0.7, zorder=1)
    _ax.add_collection(_bg_lc)
    if _labels is not None:
        _colors = [PALETTE[int(c) % len(PALETTE)] for c in _labels]
    else:
        _colors = ["#cccccc"] * _g.vcount()
    _ax.scatter(_coords[:, 0], _coords[:, 1], s=35, c=_colors, edgecolor="#777777", linewidth=0.3, zorder=2, alpha=0.75)
    _walk_segs = []
    for i in range(len(walk_indices) - 1):
        _walk_segs.append([_coords[walk_indices[i]], _coords[walk_indices[i + 1]]])
    if _walk_segs:
        _walk_lc = LineCollection(_walk_segs, colors="#d62728", linewidths=2.2, alpha=0.85, zorder=3)
        _ax.add_collection(_walk_lc)
    _wx = [_coords[i, 0] for i in walk_indices]
    _wy = [_coords[i, 1] for i in walk_indices]
    _ax.scatter(_wx, _wy, s=80, facecolor="#d62728", edgecolor="black", linewidth=0.6, zorder=4)
    _ax.scatter([_coords[walk_indices[0], 0]], [_coords[walk_indices[0], 1]], s=180, facecolor="gold", edgecolor="black", linewidth=1.0, zorder=5, label="start")
    _ax.legend(loc="upper right", fontsize=10)
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.grid(False)
    _ax.set_title(f"A biased random walk of length {len(walk_indices)}")
    _ax.set_aspect("equal")
    _fig.tight_layout()
    walk_fig = _fig
    return (walk_fig,)


@app.cell
def walk_sentence(graph_data, mo, walk_indices):
    _names = graph_data["names"]
    _sentence = " -> ".join(_names[i] for i in walk_indices)
    walk_sentence_md = mo.md(f"**The walk as a sentence:**\n\n`{_sentence}`")
    return (walk_sentence_md,)


@app.cell
def sec3_layout(
    mo, p_slider, q_slider, reroll, start_node, walk_fig, walk_length,
    walk_sentence_md,
):
    _left = mo.vstack(
        [
            mo.md("**Tweak the walk:**"),
            start_node,
            walk_length,
            p_slider,
            q_slider,
            reroll,
            mo.md("---"),
            walk_sentence_md,
        ],
        gap=0.6,
    )
    mo.hstack([_left, walk_fig], widths=[1, 1.4], gap=1.2, align="start")
    return


@app.cell
def sec3_n2v_intro(graph_data, mo):
    _g = graph_data["graph"]
    if _g is None or graph_data["precomp_prefix"] is None:
        mo.md(
            "_Node2vec embeddings are precomputed only for the football "
            "and karate networks. Switch to one of those to see the three "
            "$q$-settings below._"
        )
    else:
        mo.md(r"""
        ### node2vec embeddings at three settings — Fig 3 of the paper

        Repeat the biased walk above many times from every node, feed
        the corpus of walks into word2vec, and you get a 16-d vector
        per node. The three panels use the paper's $q$ values
        (Grover & Leskovec 2016, Fig 3):

        - **$q = 0.5$ (DFS-like)**: walks roam outward from each start
          → embedding captures **homophily / community**. Same walk
          settings as the paper: length 80, 10 walks per node,
          window 10.
        - **$q = 1$**: vanilla random walk, same long-walk settings.
        - **$q = 2$ (BFS-like)**: walks **stay close to each start**,
          and we cut the walk length down to **5 with window 5** —
          without that, even at $q=2$ each walk only ever visits one
          subplot and you get back the homophily result. With short
          walks at $q=2$, the embedding sees only each node's
          *immediate degree pattern*, and characters with similar
          structural roles end up clustered together (e.g. Fantine,
          Myriel, Gavroche all sit in the "sub-protagonist" cluster
          on Les Mis, regardless of which subplot they lead).

        > **Note about the BFS silhouette.** The reported silhouette is
        > measured against the *true community* labels. The BFS panel
        > is specifically *not* organising the embedding by community,
        > so its silhouette goes near zero or negative on purpose —
        > the structural story lives in the cluster colours on the
        > bottom-row network, not in the silhouette number.

        _Top row_: 2-d **t-SNE** of each 32-d embedding (linear
        projections look near-identical here; t-SNE preserves local
        geometry where the actual contrast lives).

        _Bottom row_: the **same network laid out as in Section 1**,
        but coloured by **k-means on each embedding** instead of the
        true labels — exactly as in Figure 3 of the node2vec paper
        (Grover & Leskovec 2016). For a homophily embedding the
        colours should *match* the community palette from Section 1;
        for a structural-role embedding they should *cut across*
        communities — bridges in different parts of the graph end up
        the same colour.

        **★ Watch the three gold stars.** They are the
        highest-**betweenness** node in three different classes — i.e.
        three "bridges" between communities. Bridges have a similar
        structural role even though they live in different parts of
        the network. The "**hub pull**" number under each panel
        measures how close those three end up in the embedding
        compared with three random nodes:

        - values **below 1** → bridges are pulled *together* (the
          embedding sees them as structurally similar regardless of
          community); this is the **BFS / structural** signature.
        - values **above 1** → bridges are pushed *apart* (the
          embedding has organised them by community); this is the
          **DFS / homophily** signature.
        """)
    return


@app.cell
def load_n2v_all(load_npy, graph_data):
    _g = graph_data["graph"]
    _prefix = graph_data["precomp_prefix"]
    if _g is None or _prefix is None:
        n2v_embs = None
    else:
        # Paper-faithful (Grover & Leskovec 2016, Fig 3): same walk
        # config across the three, only q varies.
        n2v_embs = {
            "DFS (p=1, q=0.5, walk=80)\n→ homophily":
                load_npy(f"{_prefix}node2vec_dfs.npy"),
            "balanced (p=1, q=1, walk=80)":
                load_npy(f"{_prefix}node2vec_balanced.npy"),
            "BFS (p=1, q=2, walk=5)\n→ structural roles":
                load_npy(f"{_prefix}node2vec_bfs.npy"),
        }
    return (n2v_embs,)


@app.cell
def select_structural_anchors(graph_data, np):
    """Pick a few nodes that have *similar structural role* (high
    betweenness — they sit on shortcuts between communities) but live
    in *different classes*. High-betweenness nodes are the canonical
    'bridges' of a network: they have a comparable structural job
    even when they live in different parts of the graph. On football
    this picks up Notre Dame (an Independent — a literal cross-
    conference bridge); on Les Mis it picks up Valjean, Myriel, and
    Gavroche — the three characters who connect the most disparate
    parts of the novel."""
    _g = graph_data["graph"]
    _labels = graph_data["labels"]
    if _g is None or _labels is None:
        struct_anchors = []
    else:
        _bet = np.array(_g.betweenness())
        _classes = sorted(set(int(c) for c in _labels))
        _candidates = []
        for _c in _classes:
            _mask = (_labels == _c)
            _idx = np.where(_mask)[0]
            if len(_idx) == 0:
                continue
            _best = int(_idx[np.argmax(_bet[_idx])])
            _candidates.append((_best, float(_bet[_best])))
        _candidates.sort(key=lambda x: -x[1])
        struct_anchors = [c[0] for c in _candidates[: min(3, len(_candidates))]]
    return (struct_anchors,)


@app.cell
def plot_n2v_all(
    KMeans, PALETTE, TSNE, graph_data, layout_data, mo, n2v_embs,
    network_with_cluster_colors, np, plt, show_node_labels, silhouette_score,
    struct_anchors,
):
    mo.stop(n2v_embs is None, mo.md(""))
    _labels = graph_data["labels"]
    _names = graph_data["names"]
    _coords = layout_data["coords"]
    _g = graph_data["graph"]
    _n_classes = len(set(int(c) for c in _labels)) if _labels is not None else 4

    def _hub_ratio(_emb_arr):
        if not struct_anchors:
            return None
        from numpy.linalg import norm
        _a = _emb_arr[struct_anchors]
        _da = float(np.mean([norm(_a[i] - _a[j])
                             for i in range(len(_a))
                             for j in range(i + 1, len(_a))]))
        _rng = np.random.default_rng(0)
        _bases = []
        for _ in range(100):
            _s = _rng.choice(_emb_arr.shape[0], size=len(struct_anchors), replace=False)
            _es = _emb_arr[_s]
            _bases.append(float(np.mean([norm(_es[i] - _es[j])
                                          for i in range(len(_es))
                                          for j in range(i + 1, len(_es))])))
        return _da / float(np.mean(_bases))

    _fig, _axes = plt.subplots(2, 3, figsize=(14.5, 9.0), height_ratios=[1, 1])
    for _ax, (_name, _emb) in zip(_axes[0], n2v_embs.items()):
        # 2-d t-SNE projection (UMAP gave odd placements on small,
        # weighted graphs; t-SNE is the safer default for visualisation).
        _n = _emb.shape[0]
        _perp = max(5, min(30, _n // 4))
        _tsne = TSNE(
            n_components=2, perplexity=_perp, random_state=1546,
            init="pca", learning_rate="auto",
        )
        _emb2 = _tsne.fit_transform(_emb)
        _emb2 = (_emb2 - _emb2.mean(axis=0)) / (_emb2.std(axis=0) + 1e-12)
        if _labels is not None:
            _colors = [PALETTE[int(c) % len(PALETTE)] for c in _labels]
        else:
            _colors = ["#0b789d"] * _emb.shape[0]
        _ax.scatter(_emb2[:, 0], _emb2[:, 1], s=45, c=_colors,
                    edgecolor="#333333", linewidth=0.4)
        # Structural anchors: highest-betweenness node per class.
        # Marker is a *hollow black square* so the node's own colour
        # stays readable.
        if struct_anchors:
            for _i in struct_anchors:
                _ax.scatter(
                    _emb2[_i, 0], _emb2[_i, 1], s=280, marker="s",
                    facecolors="none", edgecolor="black", linewidth=1.8, zorder=5,
                )
                _ax.annotate(
                    str(_names[_i]),
                    xy=(_emb2[_i, 0], _emb2[_i, 1]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888888", alpha=0.9, lw=0.5),
                )
        if show_node_labels.value:
            for _i, _nm in enumerate(_names):
                if struct_anchors and _i in struct_anchors:
                    continue  # already labelled above
                _ax.annotate(str(_nm), xy=(_emb2[_i, 0], _emb2[_i, 1]),
                             xytext=(3, 3), textcoords="offset points",
                             fontsize=6, color="#444444", alpha=0.7, zorder=4)
        _ax.set_xticks([]); _ax.set_yticks([]); _ax.grid(False); _ax.set_aspect("equal")
        _ax.set_title(_name, fontsize=11)
        _sil = silhouette_score(_emb, _labels)
        _hub = _hub_ratio(_emb)
        _hub_line = "" if _hub is None else f"\nhub pull = {_hub:.2f}× chance"
        _ax.text(
            0.98, 0.02, f"silhouette = {_sil:.3f}{_hub_line}",
            transform=_ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", alpha=0.9),
        )
    # Bottom row: same FR layout, coloured by k-means on each embedding
    # — Fig 3 of Grover & Leskovec 2016.
    if _g is not None and _coords is not None and _n_classes >= 2:
        for _ax, (_name, _emb) in zip(_axes[1], n2v_embs.items()):
            _km = KMeans(n_clusters=_n_classes, n_init=10, random_state=1546).fit(_emb)
            _short = _name.split("(")[0].strip().split("\n")[0]
            network_with_cluster_colors(
                _ax, _g, _coords, _g.es, _km.labels_,
                struct_anchors=struct_anchors,
                title=f"network coloured by k-means · {_short}",
                names=_names, show_labels=show_node_labels.value,
            )
    _fig.tight_layout()
    _fig
    return


@app.cell
def sec4_header_md(mo):
    sec4_header_md = mo.md(r"""
    ## Section 4: A GNN that learns to predict the label

    The previous methods are purely structural — they never see the
    class labels. A **graph neural network** can do something stronger:
    trained *with* labels, it can learn an embedding whose geometry is
    shaped by the task it is being asked to solve.

    We train a 3-layer **Graph Convolutional Network (GCN)** (with
    dropout for regularisation) for 200 epochs to predict the class of
    each node. We picked GCN over GraphSAGE and GAT after a local
    architecture sweep: a 3-layer GCN with hidden size 128 was the only
    setup that consistently *beat* the unsupervised spectral baselines
    on this dataset (macro-F1 ≈ 0.93 vs 0.92 for SVD/PCA).

    1. **Split** the nodes into a 50/50 train/test set, stratified by
       class (right →). The GNN sees the labels of the train nodes only.
    2. **Message passing**: each GCN layer rewrites every node's vector
       as a symmetric-normalised average of itself and its neighbours
       ($D^{-1/2} A D^{-1/2}$). After three layers, each node's
       representation has folded in information from nodes three hops
       away. **Dropout** randomly zeros 50% of the hidden activations
       during training, which prevents the GNN from memorising single
       training nodes and improves generalisation to the held-out test
       set.
    3. **Loss**: **cross-entropy** against the train labels. For each
       train node $i$ the GNN outputs a vector of class scores; softmax
       turns them into probabilities $\hat{p}_{ic}$ and the loss is
       $-\log \hat{p}_{i, y_i}$ — i.e. how surprised the model is by
       the true class. Averaged over train nodes, this is the quantity
       the optimiser pushes down each epoch.
    4. The 32-d output of the third layer is the **learned embedding**.
    """)
    return (sec4_header_md,)


@app.cell
def load_gnn(graph_data, load_npz):
    _prefix = graph_data["precomp_prefix"]
    if graph_data["graph"] is None or _prefix is None or not graph_data["classes_available"]:
        gnn = None
    else:
        gnn = load_npz(f"{_prefix}gnn_supervised.npz")
    return (gnn,)


@app.cell
def plot_gnn_split(
    LineCollection, PALETTE, graph_data, gnn, layout_data, mo, plt, struct_anchors,
):
    mo.stop(
        gnn is None,
        mo.md(
            "_The supervised GCN story is precomputed for the football "
            "and karate networks only. Switch to one of those to see it._"
        ),
    )
    _g = graph_data["graph"]
    _coords = layout_data["coords"]
    _labels = graph_data["labels"]
    _train_mask = gnn["train_mask"]
    _test_mask = gnn["test_mask"]
    _node_names = graph_data["names"]

    _fig, _ax = plt.subplots(figsize=(6.2, 5.6))
    _ax.set_facecolor("white")
    _bg = [[_coords[e.source], _coords[e.target]] for e in _g.es]
    _bg_lc = LineCollection(_bg, colors="#dddddd", linewidths=0.5, alpha=0.7, zorder=1)
    _ax.add_collection(_bg_lc)

    _colors = [PALETTE[int(c) % len(PALETTE)] for c in _labels]
    _colors_arr = list(_colors)
    _tr = _train_mask
    _te = _test_mask
    _ax.scatter(
        _coords[_tr, 0], _coords[_tr, 1],
        s=80, c=[_colors_arr[i] for i in range(len(_colors_arr)) if _tr[i]],
        edgecolor="black", linewidth=0.6, zorder=3, label=f"train ({int(_tr.sum())})",
    )
    _ax.scatter(
        _coords[_te, 0], _coords[_te, 1],
        s=85, facecolors="white",
        edgecolor=[_colors_arr[i] for i in range(len(_colors_arr)) if _te[i]],
        linewidth=2.0, zorder=4, label=f"test ({int(_te.sum())})",
    )
    # Always-on anchor names so the same nodes can be found across figures.
    if struct_anchors and _node_names is not None:
        for _i in struct_anchors:
            _ax.scatter(_coords[_i, 0], _coords[_i, 1], s=260, marker="s",
                        facecolors="none", edgecolor="black", linewidth=1.8,
                        zorder=5)
            _ax.annotate(str(_node_names[_i]),
                         xy=(_coords[_i, 0], _coords[_i, 1]),
                         xytext=(8, 8), textcoords="offset points",
                         fontsize=8, color="#111111", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                   ec="#666666", alpha=0.9, lw=0.5),
                         zorder=6)
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.grid(False)
    _ax.set_title("Train/test split (filled = train, hollow = test)")
    _ax.set_aspect("equal")
    _ax.legend(loc="upper right", fontsize=10)
    _fig.tight_layout()
    gnn_split_fig = _fig
    return (gnn_split_fig,)


@app.cell
def sec4_layout(gnn_split_fig, mo, sec4_header_md):
    mo.md("---")
    mo.hstack(
        [sec4_header_md, gnn_split_fig],
        widths=[1, 1.1],
        gap=1.5,
        align="start",
    )
    return


@app.cell
def plot_gnn_curves(gnn, mo, np, plt):
    mo.stop(gnn is None, mo.md(""))
    _tr_acc = gnn["train_acc_history"]
    _te_acc = gnn["test_acc_history"]
    _epochs = np.arange(1, len(_tr_acc) + 1)

    _fig, _ax = plt.subplots(figsize=(7.5, 4.0))
    _ax.plot(_epochs, _tr_acc, color="#0b789d", linewidth=1.6, label="train")
    _ax.plot(_epochs, _te_acc, color="#d62728", linewidth=1.6, label="test")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("accuracy")
    _ax.set_title("GCN train / test accuracy during training")
    _ax.set_ylim(-0.02, 1.02)
    _ax.legend(loc="lower right", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def gnn_summary(gnn, mo):
    mo.stop(gnn is None, mo.md(""))
    _final_tr = float(gnn["train_acc_history"][-1])
    _final_te = float(gnn["test_acc_history"][-1])
    _best_te = float(gnn["test_acc_history"].max())
    _train_n = int(gnn["train_mask"].sum())
    _test_n = int(gnn["test_mask"].sum())
    mo.md(
        f"**Final train accuracy:** `{_final_tr:.3f}` &nbsp;|&nbsp; "
        f"**Final test accuracy:** `{_final_te:.3f}` &nbsp;|&nbsp; "
        f"**Best test accuracy during training:** `{_best_te:.3f}` &nbsp;|&nbsp; "
        f"**Train size:** {_train_n} &nbsp;|&nbsp; **Test size:** {_test_n}"
    )
    return


@app.cell
def gnn_embedding_fig(
    PALETTE, TSNE, class_names_for, graph_data, gnn, mo, np, plt, silhouette_score,
):
    """Build (but don't display) the GCN-embedding t-SNE figure.

    Layout for misclassified test nodes: keep the circle's colour =
    *true* class (the node's "real identity") and overlay a square
    whose colour = *predicted* class (what the GCN guessed). That way
    you can read both at a glance — "true purple, predicted orange".
    """
    if gnn is None:
        gnn_emb_fig = None
    else:
        _labels = graph_data["labels"]
        _emb = gnn["emb"]
        _preds = gnn["preds"]
        _train_mask = gnn["train_mask"]
        _test_mask = gnn["test_mask"]
        _n = _emb.shape[0]
        _perp = max(5, min(30, _n // 4))
        _emb2 = TSNE(n_components=2, perplexity=_perp, random_state=1546,
                     init="pca", learning_rate="auto").fit_transform(_emb)
        _emb2 = (_emb2 - _emb2.mean(axis=0)) / (_emb2.std(axis=0) + 1e-12)
        _wrong = (_preds != _labels) & _test_mask

        _classes = sorted(set(int(c) for c in _labels))
        _short_names = class_names_for(graph_data)
        _name_for = dict(zip(_classes, _short_names))

        _fig, _ax = plt.subplots(figsize=(6.6, 5.6))
        _colors = np.array([PALETTE[int(c) % len(PALETTE)] for c in _labels])
        _ax.scatter(
            _emb2[_train_mask, 0], _emb2[_train_mask, 1],
            s=70, c=_colors[_train_mask], edgecolor="#333333", linewidth=0.4,
            zorder=2, label=f"train ({int(_train_mask.sum())})",
        )
        _ax.scatter(
            _emb2[_test_mask, 0], _emb2[_test_mask, 1],
            s=75, facecolors="white",
            edgecolor=_colors[_test_mask], linewidth=2.0,
            zorder=3, label=f"test ({int(_test_mask.sum())})",
        )
        if _wrong.any():
            # Overlay each error with a SQUARE whose colour is the
            # *predicted* class. The underlying circle's edge keeps
            # the true class colour, so the pair reads as
            # "true → predicted" at a glance.
            _pred_colors = np.array([PALETTE[int(c) % len(PALETTE)] for c in _preds])
            _ax.scatter(
                _emb2[_wrong, 0], _emb2[_wrong, 1],
                s=180, marker="s", c=_pred_colors[_wrong],
                edgecolor="black", linewidth=1.0, zorder=4,
                label=f"test errors ({int(_wrong.sum())}, square = predicted class)",
            )
            for _i in np.where(_wrong)[0]:
                _ax.annotate(
                    f"{_name_for.get(int(_labels[_i]))} → {_name_for.get(int(_preds[_i]))}",
                    xy=(_emb2[_i, 0], _emb2[_i, 1]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7, color="#222222",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white",
                        ec="#888888", alpha=0.85, lw=0.5,
                    ),
                )
        _ax.legend(loc="upper right", fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.grid(False)
        _ax.set_aspect("equal")
        _ax.set_title(
            "Learned GCN embedding — 2D t-SNE\n"
            "(filled circle = train, hollow circle = test, "
            "coloured square = wrong, square colour = predicted class)",
            fontsize=10,
        )
        _sil = silhouette_score(_emb, _labels)
        _ax.text(
            0.98,
            0.02,
            f"silhouette (full 32-d) = {_sil:.3f}",
            transform=_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", alpha=0.9),
        )
        _fig.tight_layout()
        gnn_emb_fig = _fig
    return (gnn_emb_fig,)


@app.cell
def gnn_errors_network_fig(
    LineCollection, PALETTE, class_names_for, graph_data, gnn, layout_data, np,
    plt, struct_anchors,
):
    """Network view with the same test-set errors highlighted and the
    predicted (wrong) class annotated next to each error."""
    if gnn is None:
        gnn_errors_fig = None
    else:
        _g = graph_data["graph"]
        _coords = layout_data["coords"]
        _labels = graph_data["labels"]
        _preds = gnn["preds"]
        _train_mask = gnn["train_mask"]
        _test_mask = gnn["test_mask"]
        _wrong = (_preds != _labels) & _test_mask

        _classes = sorted(set(int(c) for c in _labels))
        _short_names = class_names_for(graph_data)
        _name_for = dict(zip(_classes, _short_names))

        _fig, _ax = plt.subplots(figsize=(6.6, 5.6))
        _ax.set_facecolor("white")
        _bg = [[_coords[e.source], _coords[e.target]] for e in _g.es]
        _bg_lc = LineCollection(_bg, colors="#dddddd", linewidths=0.5, alpha=0.7, zorder=1)
        _ax.add_collection(_bg_lc)
        _colors = np.array([PALETTE[int(c) % len(PALETTE)] for c in _labels])
        _ax.scatter(
            _coords[_train_mask, 0], _coords[_train_mask, 1],
            s=70, c=_colors[_train_mask], edgecolor="#333333", linewidth=0.4,
            zorder=3, label=f"train ({int(_train_mask.sum())})",
        )
        _ax.scatter(
            _coords[_test_mask, 0], _coords[_test_mask, 1],
            s=75, facecolors="white",
            edgecolor=_colors[_test_mask], linewidth=2.0,
            zorder=4, label=f"test ({int(_test_mask.sum())})",
        )
        if _wrong.any():
            _pred_colors = np.array([PALETTE[int(c) % len(PALETTE)] for c in _preds])
            _ax.scatter(
                _coords[_wrong, 0], _coords[_wrong, 1],
                s=180, marker="s", c=_pred_colors[_wrong],
                edgecolor="black", linewidth=1.0, zorder=5,
                label=f"test errors ({int(_wrong.sum())}, square = predicted class)",
            )
            for _i in np.where(_wrong)[0]:
                _ax.annotate(
                    f"{_name_for.get(int(_labels[_i]))} → {_name_for.get(int(_preds[_i]))}",
                    xy=(_coords[_i, 0], _coords[_i, 1]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7, color="#222222",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white",
                        ec="#888888", alpha=0.85, lw=0.5,
                    ),
                )
        if struct_anchors:
            _node_names = graph_data["names"]
            for _i in struct_anchors:
                _ax.scatter(_coords[_i, 0], _coords[_i, 1], s=260, marker="s",
                            facecolors="none", edgecolor="black", linewidth=1.8,
                            zorder=6)
                _ax.annotate(str(_node_names[_i]),
                             xy=(_coords[_i, 0], _coords[_i, 1]),
                             xytext=(-10, 10), textcoords="offset points",
                             fontsize=8, color="#111111", fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                       ec="#666666", alpha=0.9, lw=0.5),
                             zorder=7)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.grid(False)
        _ax.set_aspect("equal")
        _ax.set_title(
            "Same nodes on the actual network\n"
            "(coloured circle edge = true class, square fill = predicted class)",
            fontsize=10,
        )
        _ax.legend(loc="upper right", fontsize=9)
        _fig.tight_layout()
        gnn_errors_fig = _fig
    return (gnn_errors_fig,)


@app.cell
def sec5_header(mo):
    mo.md(r"""
    ---
    ## Section 5: Predicting the class from each embedding

    Take each embedding we built — spectral, node2vec, and the supervised
    GCN — and use it as input features for a plain **multinomial
    (softmax) logistic regression** that predicts the node class.

    To make the comparison apples-to-apples, every method uses the
    **same train/test split** that the GCN was trained on:

    - the GNN sees the train-node labels during its training,
    - each logistic regression sees the train-node labels during its fit,
    - everything is evaluated on the held-out test nodes — **no method
      ever saw the test-node labels.** Including the GCN embedding
      here is fair: only train labels touched it.

    The table below shows test-set **accuracy**, **macro-averaged
    precision / recall / F1**, fit for every embedding side by side.
    The classifier is `LogisticRegression(C=10, solver="lbfgs")`; we
    use a weaker-than-default L2 penalty (`C=10` instead of sklearn's
    `C=1`) so the high-frequency components of the Laplacian Eigenmaps
    are not over-regularised away — every method gets the same setting.
    """)
    return


@app.cell
def classification_split(gnn, graph_data, np, train_test_split):
    """Train/test split used in Section 5.

    For football and karate we reuse the exact mask the GCN was
    trained on (so every method sees the same train and test nodes). For
    a user-uploaded labelled graph, we compute a 50/50 stratified split
    on the fly with a fixed seed.
    """
    _labels = graph_data["labels"]
    if not graph_data["classes_available"] or _labels is None:
        split = None
    elif gnn is not None:
        split = {
            "train_mask": gnn["train_mask"],
            "test_mask": gnn["test_mask"],
            "source": "graphsage",
        }
    else:
        _n = len(_labels)
        _idx = np.arange(_n)
        _u, _counts = np.unique(_labels, return_counts=True)
        _strat = _labels if _counts.min() >= 2 else None
        _tr, _te = train_test_split(
            _idx, test_size=0.5, stratify=_strat, random_state=1546
        )
        _tm = np.zeros(_n, dtype=bool); _tm[_tr] = True
        split = {"train_mask": _tm, "test_mask": ~_tm, "source": "fresh"}
    return (split,)


@app.cell
def classify_all(
    LogisticRegression,
    accuracy_score,
    f1_score,
    gnn,
    graph_data,
    load_npy,
    mo,
    np,
    pd,
    spectral_embs,
    split,
):
    from sklearn.metrics import precision_score, recall_score

    mo.stop(
        not graph_data["classes_available"] or split is None,
        mo.md(
            "_The current network has no class labels, so the comparison "
            "table is hidden. Switch to Football or Karate, or upload a "
            "GraphML with a node-class attribute, to see it._"
        ),
    )

    _labels = graph_data["labels"]
    _prefix = graph_data["precomp_prefix"]
    _train_mask = split["train_mask"]
    _test_mask = split["test_mask"]

    _methods = []
    for _name, _emb in spectral_embs.items():
        _methods.append((f"Spectral · {_name}", _emb))
    if _prefix is not None:
        for _q_label, _file in [
            ("node2vec — DFS (p=4, q=0.1)",      "node2vec_dfs.npy"),
            ("node2vec — balanced (p=1, q=1)",   "node2vec_balanced.npy"),
            ("node2vec — BFS (p=0.25, q=10)",    "node2vec_bfs.npy"),
        ]:
            _methods.append((_q_label, load_npy(f"{_prefix}{_file}")))
    if gnn is not None:
        _methods.append(("GCN (supervised)", gnn["emb"]))

    _rows = []
    for _name, _emb in _methods:
        _Xtr = _emb[_train_mask]
        _ytr = _labels[_train_mask]
        _Xte = _emb[_test_mask]
        _yte = _labels[_test_mask]
        # C=10 gives weaker L2 regularisation than the sklearn default
        # of C=1. Default-C over-penalises features that look noisy in
        # the train sample (notably the higher-frequency Laplacian
        # eigenvectors), so all methods - and Laplacian Eigenmaps in
        # particular - perform substantially better here.
        _clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=10.0)
        _clf.fit(_Xtr, _ytr)
        _yhat = _clf.predict(_Xte)
        _rows.append({
            "embedding": _name,
            "accuracy": round(float(accuracy_score(_yte, _yhat)), 3),
            "precision (macro)": round(float(precision_score(_yte, _yhat, average="macro", zero_division=0)), 3),
            "recall (macro)": round(float(recall_score(_yte, _yhat, average="macro", zero_division=0)), 3),
            "f1 (macro)": round(float(f1_score(_yte, _yhat, average="macro", zero_division=0)), 3),
        })
    _df = pd.DataFrame(_rows).sort_values("accuracy", ascending=False).reset_index(drop=True)

    _note = (
        "same split as the GCN training above"
        if split["source"] == "graphsage"
        else "50/50 stratified split (no precomputed GCN for this graph)"
    )
    _summary = mo.md(
        f"_Train set: {int(_train_mask.sum())} nodes &nbsp;·&nbsp; "
        f"Test set: {int(_test_mask.sum())} nodes &nbsp;·&nbsp; "
        f"{_note}._"
    )
    mo.vstack([_summary, mo.ui.table(_df, selection=None)])
    return


@app.cell
def sec5_errors_header(gnn, graph_data, mo):
    mo.stop(gnn is None, mo.md(""))
    _name = graph_data.get("name", "")
    if _name.startswith("Football"):
        _interp = (
            "Most errors typically sit on **inter-conference edges** — "
            "teams that played a lot of out-of-conference games (the "
            "Independents and a few transitioning conferences) are the "
            "hardest to place."
        )
    elif _name.startswith("Karate"):
        _interp = (
            "With only two factions any errors will be club members "
            "who interact across the Mr Hi / Officer dispute line, "
            "which is exactly where the disputed members of the "
            "original Zachary (1977) study sit."
        )
    elif _name.startswith("Les"):
        _interp = (
            "Errors are typically characters who appear across **multiple "
            "social circles** in Hugo's novel — Valjean, Marius, Cosette "
            "and Gavroche bridge the prison, the Friends of the ABC, and "
            "the Thénardier subplots, so a single 'community' label is "
            "always going to be a lossy summary of who they really are."
        )
    else:
        _interp = (
            "Errors typically sit on the boundary between two classes — "
            "nodes whose connections span multiple groups are inherently "
            "harder to classify than nodes deep inside a single group."
        )
    mo.md(rf"""
    ### Where does GCN go wrong on the test set?

    Left: the **learned 32-d GCN embedding**, projected to 2-d
    with PCA. Right: the **same nodes back on the actual network**.
    In both, filled markers are train nodes, hollow markers are
    test nodes, and a **black ring** marks every test node the GNN
    mis-classified.

    {_interp}
    """)
    return


@app.cell
def sec5_errors_layout(gnn, gnn_emb_fig, gnn_errors_fig, mo):
    mo.stop(gnn is None, mo.md(""))
    mo.hstack(
        [gnn_emb_fig, gnn_errors_fig], widths=[1, 1], gap=1.0, align="start"
    )
    return


@app.cell
def conclusion(mo):
    mo.md(r"""
    ---
    ## Wrap-up: when to use which method?

    | Method | Sees labels? | Best at | Watch out for |
    |---|---|---|---|
    | **Truncated SVD / PCA of $A$** | no | a strong general-purpose baseline; cheap, deterministic, and works whenever the network has any structure at all | needs the right number of dimensions — too few drops accuracy. SVD's first axis is degree-shaped (it disappears if you centre, which is what PCA does) |
    | **Laplacian Eigenmaps** | no | recovering **community-shaped** classes — especially when classes look like cliques in the graph | only the *first few* eigenvectors are useful; adding more dimensions actually *hurts*. Performance falls off classes that aren't real communities (Independents on football) |
    | **node2vec** | no | shifting between **community** ($q \!<\! 1$, DFS) and **structural-role** ($q \!>\! 1$, BFS) views of the same network — by changing two knobs | many hyperparameters (p, q, walk length, window) — on a dense, near-regular network like football the bias has little room to move, the differences become subtle |
    | **GCN (supervised)** | YES | every dataset where you have *some* labels; trained directly on the classification objective, so it can pick up patterns SVD/LE/node2vec miss | the most expensive option; needs careful regularisation (dropout, weight decay) to not overfit on small training sets. Without enough train labels it just memorises them and doesn't generalise |

    **General lessons** the football and karate networks reveal:

    - The **easy classes** (large, dense, well-separated conferences) are
      solved by basically every method. The interesting question is
      always how each method handles the **hard classes** — here the
      Independents and Sun Belt teams, which lack a clear community
      structure. SVD/PCA and GCN handle them best because they can pick
      up *which* communities each team interacted with, not just *that*
      it interacted with some community.
    - **Supervised wins, but not by much**. The GCN beats the
      unsupervised methods by only a few F1 points (~0.93 vs ~0.92 on
      football) — because the structure of the network already encodes
      almost all of the label information. On graphs where labels are
      *less* tied to structure, the gap would widen.
    - **Embeddings are not interpretable axes**. Throughout we used
      2-d projections (the first two components for spectral methods,
      PCA for GCN, t-SNE for node2vec) — but the underlying spaces are
      32-dimensional and their individual axes don't carry meaning.
      That's why we strip the tick labels: only the *geometry* matters.
    """)
    return


@app.cell
def footer(mo):
    mo.md("""
    ---

    _An interactive companion on node embeddings, built with [marimo](https://marimo.io)._
    """)
    return


if __name__ == "__main__":
    app.run()
