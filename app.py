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

    from sklearn.decomposition import PCA, TruncatedSVD
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
        LineCollection,
        LogisticRegression,
        PALETTE,
        PCA,
        TruncatedSVD,
        biased_walk,
        class_names_for,
        load_graphml,
        load_npy,
        load_npz,
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

    A **node embedding** is a way of putting every vertex of a network into
    a vector space, so that geometry (distances, angles, clusters) becomes
    a tool we can use for prediction.

    We will build up three families of embeddings on the US college
    football network:

    1. **Spectral** — deterministic, from the eigenvectors of $A$ or $L$.
       Structure-only.
    2. **node2vec** — random walks fed through *word2vec*. Structure-only,
       but learned.
    3. **GraphSAGE** — a graph neural network that we train *with* the
       labels to predict each team's conference.

    Then we close the loop by asking: **how well do the two structure-only
    embeddings predict the labels** when we plug them into a plain
    logistic regression?

    ---
    """)
    return


@app.cell
def sec1_widgets(mo):
    dataset_choice = mo.ui.radio(
        options=[
            "Football (115 nodes, 12 conferences)",
            "Karate (34 nodes, 2 factions)",
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
    return dataset_choice, upload


@app.cell
def sidebar(dataset_choice, mo, upload):
    mo.sidebar(
        [
            mo.md("### Dataset"),
            dataset_choice,
            mo.md("_Or upload your own:_"),
            upload,
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
                    _labels = None
                    _classes_available = False
                    _is_byod = True
                    _description = (
                        f"**User-uploaded CSV edge list** with {_g.vcount()} "
                        f"nodes and {_g.ecount()} edges. _Classification "
                        f"panels are hidden (no class labels)._"
                    )
        except Exception as e:
            _g = None
            _err = f"Failed to read upload: {e}"
    elif _choice.startswith("Karate"):
        _g = ig.Graph.Famous("Zachary")
        _g.vs["name"] = [f"v{i}" for i in range(_g.vcount())]
        _labels = _zachary_factions.copy()
        _description = (
            "**Zachary karate club** (34 members, 78 ties). The classic "
            "split into Mr Hi's and the officer's faction after a dispute."
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
    LineCollection, PALETTE, class_names_for, graph_data, layout_data, mo, np, plt
):
    _g = graph_data["graph"]
    mo.stop(_g is None, mo.md("_(No graph to display.)_"))
    _coords = layout_data["coords"]
    _labels = graph_data["labels"]
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

    - **Truncated SVD of $A$**: the top singular vectors of the adjacency.
    - **Laplacian Eigenmaps**: the smallest non-trivial eigenvectors of
      $L = D - A$. This is what classical spectral clustering uses.
    - **PCA of $A$**: principal components of the adjacency rows.

    Each is computed in 32 dimensions for fair downstream comparison
    with node2vec and GraphSAGE; the scatter below shows the **first
    two** dimensions, rotated and scaled to share an orientation with
    the network layout above so the panels are easier to compare.

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
    _k = min(32, _n - 1)

    _svd = TruncatedSVD(n_components=_k, random_state=1546)
    _emb_svd = _svd.fit_transform(_A)

    # Laplacian Eigenmaps: use the *symmetric normalised* Laplacian
    # L_sym = I - D^{-1/2} A D^{-1/2}. This is the canonical choice for
    # community detection (Shi & Malik 1997, Ng-Jordan-Weiss 2002) and
    # gives much better small-class recall than the raw L = D - A.
    # np.linalg.eigh returns eigenvalues in ascending order, so columns
    # 1..k are the bottom-k non-trivial eigenvectors.
    _deg = _A.sum(axis=1)
    _d_inv_sqrt = np.zeros_like(_deg)
    _nz = _deg > 0
    _d_inv_sqrt[_nz] = 1.0 / np.sqrt(_deg[_nz])
    _A_norm = _A * _d_inv_sqrt[:, None] * _d_inv_sqrt[None, :]
    _L_sym = np.eye(_n) - _A_norm
    _w, _v = np.linalg.eigh(_L_sym)
    _emb_lap = _v[:, 1 : _k + 1]

    _pca = PCA(n_components=_k, random_state=1546)
    _emb_pca = _pca.fit_transform(_A)

    spectral_embs = {
        "Truncated SVD (of A)": np.asarray(_emb_svd, dtype=float),
        "Laplacian Eigenmaps (of L_sym)": np.asarray(_emb_lap, dtype=float),
        "PCA (of A)": np.asarray(_emb_pca, dtype=float),
    }
    # "Primary" spectral embedding used downstream (Section 5).
    spectral_emb = spectral_embs["Laplacian Eigenmaps (of L_sym)"]
    return spectral_emb, spectral_embs


@app.cell
def plot_spectral(
    PALETTE, graph_data, layout_data, plt, procrustes_align, silhouette_score, spectral_embs
):
    _labels = graph_data["labels"]
    _anchor_idx = layout_data["anchor_idx"]
    _anchor_targets = layout_data["anchor_targets"]
    _fig, _axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for _ax, (_name, _emb) in zip(_axes, spectral_embs.items()):
        # Spectral methods produce columns *already ordered by importance*
        # (singular values for SVD/PCA, ascending eigenvalues for the
        # Laplacian). PCA of these orthonormal columns washes that
        # ordering out and produces a meaningless rotation, so we plot
        # the first two raw dimensions directly. The Fiedler vector
        # (Laplacian column 0) is what carries the dominant community
        # cut, which is what we want to see in 2-d.
        _emb2 = _emb[:, :2]
        if _anchor_idx is not None and _anchor_targets is not None:
            _emb2 = procrustes_align(_emb2, _anchor_idx, _anchor_targets)
        if _labels is not None:
            _colors = [PALETTE[int(c) % len(PALETTE)] for c in _labels]
        else:
            _colors = ["#0b789d"] * _emb.shape[0]
        _ax.scatter(_emb2[:, 0], _emb2[:, 1], s=45, c=_colors, edgecolor="#333333", linewidth=0.4)
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
    _fig.tight_layout()
    _fig
    return


@app.cell
def sec3_header(mo):
    mo.md(r"""
    ---
    ## Section 3: Random walks and node2vec

    Pick a node. Start walking. The neighbours you tend to visit together
    capture the local structure of the graph. **Node2vec** turns these
    walks into vectors using *word2vec* — yes, the NLP one — by treating
    each walk as a sentence and each node as a token.

    The two bias parameters $p$ and $q$ shape the walks:

    - **$p$ (return bias)**: $p < 1$ encourages **going back** to the
      previous node, $p > 1$ discourages it.
    - **$q$ (in-out bias)**: $q < 1$ pushes the walk **outward**
      (depth-first → captures **homophily / local communities**), $q > 1$
      keeps it **close** to the start (breadth-first → captures
      **structural roles**).

    Move the four sliders below to see how the walk itself changes. The
    sliders only drive the walk illustration; the three node2vec panels
    further down were precomputed at three extreme settings so you can
    compare them at fixed quality.
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
    walk_length = mo.ui.slider(start=5, stop=50, step=1, value=20, label="Walk length")
    p_slider = mo.ui.slider(
        start=0.25, stop=4.0, step=0.25, value=1.0, label="p (return bias)",
        show_value=True,
    )
    q_slider = mo.ui.slider(
        start=0.25, stop=4.0, step=0.25, value=1.0, label="q (in-out bias)",
        show_value=True,
    )
    return p_slider, q_slider, start_node, walk_length


@app.cell
def sample_walk(
    biased_walk, graph_data, mo, np, p_slider, q_slider, start_node, walk_length
):
    _g = graph_data["graph"]
    mo.stop(_g is None, mo.md("_(No graph.)_"))
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
def sec3_layout(mo, p_slider, q_slider, start_node, walk_fig, walk_length, walk_sentence_md):
    _left = mo.vstack(
        [
            mo.md("**Tweak the walk:**"),
            start_node,
            walk_length,
            p_slider,
            q_slider,
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
        ### node2vec embeddings at three extreme settings

        Repeat the kind of biased walk above many times from every node,
        feed the corpus of walks into word2vec, and you get a 32-d
        vector per node. Below: precomputed node2vec embeddings on this
        graph at three corners of the $(p, q)$ plane. We push both
        parameters at once to make the contrast as visible as possible:

        - **$p = 4,\ q = 0.1$** — very depth-first: large $p$ discourages
          return, tiny $q$ pulls toward distance-2 neighbours, so walks
          stay inside the same community → embedding captures
          **homophily**.
        - **$p = 1,\ q = 1$** — neutral / vanilla random walk.
        - **$p = 0.25,\ q = 10$** — very breadth-first: small $p$
          encourages return, large $q$ keeps the walk close to where it
          started → embedding captures **structural roles**.
        """)
    return


@app.cell
def load_n2v_all(load_npy, graph_data):
    _g = graph_data["graph"]
    _prefix = graph_data["precomp_prefix"]
    if _g is None or _prefix is None:
        n2v_embs = None
    else:
        n2v_embs = {
            "p=4, q=0.1 — very DFS (homophily)":
                load_npy(f"{_prefix}node2vec_dfs.npy"),
            "p=1, q=1 — balanced":
                load_npy(f"{_prefix}node2vec_balanced.npy"),
            "p=0.25, q=10 — very BFS (structural)":
                load_npy(f"{_prefix}node2vec_bfs.npy"),
        }
    return (n2v_embs,)


@app.cell
def plot_n2v_all(
    PALETTE, PCA, graph_data, layout_data, mo, n2v_embs, plt, procrustes_align, silhouette_score
):
    mo.stop(n2v_embs is None, mo.md(""))
    _labels = graph_data["labels"]
    _anchor_idx = layout_data["anchor_idx"]
    _anchor_targets = layout_data["anchor_targets"]
    _fig, _axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for _ax, (_name, _emb) in zip(_axes, n2v_embs.items()):
        _emb2 = PCA(n_components=2, random_state=1546).fit_transform(_emb)
        if _anchor_idx is not None and _anchor_targets is not None:
            _emb2 = procrustes_align(_emb2, _anchor_idx, _anchor_targets)
        if _labels is not None:
            _colors = [PALETTE[int(c) % len(PALETTE)] for c in _labels]
        else:
            _colors = ["#0b789d"] * _emb.shape[0]
        _ax.scatter(_emb2[:, 0], _emb2[:, 1], s=45, c=_colors, edgecolor="#333333", linewidth=0.4)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.grid(False)
        _ax.set_aspect("equal")
        _ax.set_title(_name, fontsize=11)
        _sil = silhouette_score(_emb, _labels)
        _ax.text(
            0.98,
            0.02,
            f"silhouette = {_sil:.3f}",
            transform=_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", alpha=0.9),
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

    We train a 2-layer **GraphSAGE** (with dropout for regularisation)
    for 100 epochs to predict the class of each node:

    1. **Split** the nodes into a 50/50 train/test set, stratified by
       class (right →). The GNN sees the labels of the train nodes only.
    2. **Message passing**: each layer rewrites every node's vector by
       averaging its own vector with its neighbours' — so after two
       layers, each node's representation has folded in information
       from nodes two hops away. **Dropout** randomly zeros 50% of the
       hidden activations during training, which prevents the GNN from
       memorising single training nodes and improves generalisation to
       the held-out test set.
    3. **Loss**: **cross-entropy** against the train labels. For each
       train node $i$ the GNN outputs a vector of class scores; softmax
       turns them into probabilities $\hat{p}_{ic}$ and the loss is
       $-\log \hat{p}_{i, y_i}$ — i.e. how surprised the model is by
       the true class. Averaged over train nodes, this is the quantity
       the optimiser pushes down each epoch.
    4. The 32-d output of the second layer is the **learned embedding**.
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
def plot_gnn_split(LineCollection, PALETTE, graph_data, gnn, layout_data, mo, plt):
    mo.stop(
        gnn is None,
        mo.md(
            "_The supervised GraphSAGE story is precomputed for the football "
            "and karate networks only. Switch to one of those to see it._"
        ),
    )
    _g = graph_data["graph"]
    _coords = layout_data["coords"]
    _labels = graph_data["labels"]
    _train_mask = gnn["train_mask"]
    _test_mask = gnn["test_mask"]

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
    _ax.set_title("GraphSAGE train / test accuracy during training")
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
    PALETTE, PCA, class_names_for, graph_data, gnn, layout_data, mo, np, plt,
    procrustes_align, silhouette_score,
):
    """Build (but don't display) the GraphSAGE-embedding PCA figure.

    Section 5 picks it up and shows it side-by-side with the network view.
    """
    if gnn is None:
        gnn_emb_fig = None
    else:
        _labels = graph_data["labels"]
        _emb = gnn["emb"]
        _preds = gnn["preds"]
        _train_mask = gnn["train_mask"]
        _test_mask = gnn["test_mask"]
        _emb2 = PCA(n_components=2, random_state=1546).fit_transform(_emb)
        _anchor_idx = layout_data["anchor_idx"]
        _anchor_targets = layout_data["anchor_targets"]
        if _anchor_idx is not None and _anchor_targets is not None:
            _emb2 = procrustes_align(_emb2, _anchor_idx, _anchor_targets)
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
            _ax.scatter(
                _emb2[_wrong, 0], _emb2[_wrong, 1],
                s=240, facecolors="none", edgecolor="black", linewidth=1.6,
                zorder=4, label=f"test errors ({int(_wrong.sum())})",
            )
            for _i in np.where(_wrong)[0]:
                _ax.annotate(
                    f"→ {_name_for.get(int(_preds[_i]), str(_preds[_i]))}",
                    xy=(_emb2[_i, 0], _emb2[_i, 1]),
                    xytext=(6, 6), textcoords="offset points",
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
            "Learned GraphSAGE embedding — 2D PCA\n"
            "(filled = train, hollow = test, black ring + → label = wrong on test)",
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
    LineCollection, PALETTE, class_names_for, graph_data, gnn, layout_data, np, plt
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
            _ax.scatter(
                _coords[_wrong, 0], _coords[_wrong, 1],
                s=240, facecolors="none", edgecolor="black", linewidth=1.6,
                zorder=5, label=f"test errors ({int(_wrong.sum())})",
            )
            for _i in np.where(_wrong)[0]:
                _ax.annotate(
                    f"→ {_name_for.get(int(_preds[_i]), str(_preds[_i]))}",
                    xy=(_coords[_i, 0], _coords[_i, 1]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=7, color="#222222",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white",
                        ec="#888888", alpha=0.85, lw=0.5,
                    ),
                )
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.grid(False)
        _ax.set_aspect("equal")
        _ax.set_title(
            "Same nodes on the actual network\n"
            "(black ring + → label = misclassified test node, predicted class)",
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
    GraphSAGE — and use it as input features for a plain **multinomial
    (softmax) logistic regression** that predicts the node class.

    To make the comparison apples-to-apples, every method uses the
    **same train/test split** that the GraphSAGE was trained on:

    - the GNN sees the train-node labels during its training,
    - each logistic regression sees the train-node labels during its fit,
    - everything is evaluated on the held-out test nodes — **no method
      ever saw the test-node labels.** Including the GraphSAGE embedding
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

    For football and karate we reuse the exact mask the GraphSAGE was
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
        _methods.append(("GraphSAGE (supervised)", gnn["emb"]))

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
        "same split as the GraphSAGE training above"
        if split["source"] == "graphsage"
        else "50/50 stratified split (no precomputed GraphSAGE for this graph)"
    )
    _summary = mo.md(
        f"_Train set: {int(_train_mask.sum())} nodes &nbsp;·&nbsp; "
        f"Test set: {int(_test_mask.sum())} nodes &nbsp;·&nbsp; "
        f"{_note}._"
    )
    mo.vstack([_summary, mo.ui.table(_df, selection=None)])
    return


@app.cell
def sec5_errors_header(gnn, mo):
    mo.stop(gnn is None, mo.md(""))
    mo.md(r"""
    ### Where does GraphSAGE go wrong on the test set?

    Left: the **learned 32-d GraphSAGE embedding**, projected to 2-d
    with PCA. Right: the **same nodes back on the actual network**.
    In both, filled markers are train nodes, hollow markers are
    test nodes, and a **black ring** marks every test node the GNN
    mis-classified. Most errors cluster on the boundary between two
    classes in the embedding, but on the network you can see they
    typically sit on **inter-conference edges** — teams that played a
    lot of out-of-conference games are the hardest to place.
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
def footer(mo):
    mo.md("""
    ---

    _An interactive companion on node embeddings, built with [marimo](https://marimo.io)._
    """)
    return


if __name__ == "__main__":
    app.run()
