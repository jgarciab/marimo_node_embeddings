"""Systematic node2vec parameter sweep on Les Misérables, with the goal
of finding a configuration where Fantine / Myriel / Gavroche cluster
together (as the user expects from Fig 3 of the node2vec paper).

Score:
  triplet_match: how many of {Fantine, Myriel, Gavroche} share the same
                 k-means cluster (max 3 = all together)
  peri_concentration: fraction of degree-1 peripheral nodes that end up
                      in their *most common* cluster (max 1 = all in one
                      cluster)
  total: triplet_match + peri_concentration  (max 4, ideally ~ 4)
"""
import itertools
import numpy as np
import igraph as ig
import networkx as nx
from node2vec import Node2Vec
from sklearn.cluster import KMeans

SEED = 1546
g = ig.Graph.Read_GraphML("public/les_miserables.graphml")
names = list(g.vs["id"])
labels_true = np.array([int(v) for v in g.vs["value"]])
n = g.vcount()
deg = np.array(g.degree())
periph_idx = [i for i in range(n) if deg[i] == 1]
target = [names.index(t) for t in ("Fantine", "Myriel", "Gavroche")]

# Build weighted NX
G = nx.Graph()
G.add_nodes_from(range(n))
for e in g.es:
    G.add_edge(e.source, e.target,
               weight=float(e["weight"]) if "weight" in g.es.attributes() else 1.0)


def run_one(p, q, wl, nw, w, dim=16, k=6, use_weights=True):
    n2v = Node2Vec(G, dimensions=dim, walk_length=wl, num_walks=nw,
                   p=p, q=q, workers=1, seed=SEED, quiet=True,
                   weight_key="weight" if use_weights else None)
    m = n2v.fit(window=w, min_count=1, batch_words=4, seed=SEED, workers=1)
    emb = np.zeros((n, dim))
    for i in range(n):
        emb[i] = m.wv[str(i)]
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(emb)
    cl = km.labels_
    # triplet_match: are F/M/G in same cluster?
    tcs = [cl[t] for t in target]
    from collections import Counter
    most_common, count = Counter(tcs).most_common(1)[0]
    triplet_match = count  # 1, 2, or 3
    # peri_concentration
    pcs = [cl[i] for i in periph_idx]
    _, peri_count = Counter(pcs).most_common(1)[0]
    peri_conc = peri_count / max(1, len(periph_idx))
    return triplet_match, peri_conc, cl


# Smart grid
configs = []
for wl in (3, 5, 10, 20, 40, 80, 160):
    for nw in (10, 50):
        for w in (1, 2, 5, 10):
            for q in (0.25, 0.5, 1, 2, 4, 10):
                configs.append(dict(p=1, q=q, wl=wl, nw=nw, w=w))

print(f"Running {len(configs)} configs ...")
results = []
for i, cfg in enumerate(configs):
    try:
        tm, pc, _ = run_one(**cfg)
        results.append((tm + pc, tm, pc, cfg))
    except Exception as e:
        print(f"  skip {cfg}: {e}")
    if (i + 1) % 30 == 0:
        print(f"  done {i+1}/{len(configs)}")

# Sort by combined score
results.sort(key=lambda r: -r[0])
print("\n=== TOP 20 by triplet_match + peri_concentration ===")
print(f'{"score":<6} {"triplet":<8} {"peri":<5} {"params"}')
for total, tm, pc, cfg in results[:20]:
    print(f"{total:<6.2f} {tm:<8} {pc:<5.2f} p={cfg['p']} q={cfg['q']} wl={cfg['wl']} nw={cfg['nw']} w={cfg['w']}")

# Also try varying p
print("\n=== Best with varying p (fixed wl, nw, w from top result) ===")
best = results[0][3]
for p in (0.1, 0.25, 0.5, 1, 2, 4):
    for q in (0.5, 1, 2):
        tm, pc, _ = run_one(p=p, q=q, wl=best["wl"], nw=best["nw"], w=best["w"])
        print(f"  p={p:<5} q={q:<3} → triplet={tm}/3 peri={pc:.2f}")
