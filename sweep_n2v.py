"""Node2vec parameter sweep on football.

The current three settings (very DFS / balanced / very BFS) at
walk_length=40 still produce embeddings whose principal angles between
top-5 subspaces are ~10 degrees - i.e., basically the same. Find
settings that produce *visibly* different embeddings.

The structural-roles story needs the BFS embedding to actually pull the
three highest-degree nodes (from different conferences) close together.
Measure that explicitly with a 'hub-pull distance ratio'.
"""
import numpy as np
import igraph as ig
import networkx as nx
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

SEED = 1546
np.random.seed(SEED)

g = ig.Graph.Read_GraphML("data/football_network.graphml")
labels = np.array([int(v) for v in g.vs["value"]])
n = g.vcount()
deg = np.array(g.degree())
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from([(e.source, e.target) for e in g.es])

# Pick three structural anchors: highest-degree node from three different classes
classes = sorted(set(labels))
cand = []
for c in classes:
    idx = np.where(labels == c)[0]
    cand.append((idx[np.argmax(deg[idx])], deg[idx[np.argmax(deg[idx])]], c))
cand.sort(key=lambda x: -x[1])
anchors = [c[0] for c in cand[:3]]
print(f"Anchors: idx={anchors}, classes={[labels[a] for a in anchors]}, degrees={[deg[a] for a in anchors]}")


def hub_pull_ratio(emb):
    """Mean pairwise distance among anchor nodes, normalised by mean
    pairwise distance among random samples of same size. <1 means
    anchors are CLOSER than chance (structural-role embedding); >1 means
    they're FAR APART (community embedding)."""
    from numpy.linalg import norm
    a = emb[anchors]
    da = np.mean([norm(a[i] - a[j]) for i in range(len(a)) for j in range(i + 1, len(a))])
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(200):
        s = rng.choice(n, size=len(anchors), replace=False)
        es = emb[s]
        ratios.append(np.mean([norm(es[i] - es[j]) for i in range(len(es)) for j in range(i + 1, len(es))]))
    base = np.mean(ratios)
    return da / base


def run(p, q, walk_length=40, num_walks=20, window=5, dim=32):
    n2v = Node2Vec(G, dimensions=dim, walk_length=walk_length, num_walks=num_walks,
                   p=p, q=q, workers=1, seed=SEED, quiet=True)
    model = n2v.fit(window=window, min_count=1, batch_words=4, seed=SEED, workers=1)
    emb = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        emb[i] = model.wv[str(i)]
    sil = silhouette_score(emb, labels)
    hpr = hub_pull_ratio(emb)
    return emb, sil, hpr


print(f"\n{'config':<55} {'sil':<6} {'hub_ratio':<10}")
configs = [
    # baseline
    ("p=1   q=1   wl=40 nw=20 w=5",   (1.0, 1.0, 40, 20, 5)),
    # extreme q variations
    ("p=4   q=0.1 wl=40 nw=20 w=5",   (4.0, 0.1, 40, 20, 5)),
    ("p=0.25 q=10 wl=40 nw=20 w=5",   (0.25, 10.0, 40, 20, 5)),
    # ultra-extreme p, q
    ("p=10  q=0.05 wl=40 nw=20 w=5",  (10.0, 0.05, 40, 20, 5)),
    ("p=0.05 q=10 wl=40 nw=20 w=5",   (0.05, 10.0, 40, 20, 5)),
    # smaller window: more BFS-like for the same (p,q)
    ("p=0.25 q=10 wl=40 nw=20 w=2",   (0.25, 10.0, 40, 20, 2)),
    ("p=4   q=0.1 wl=40 nw=20 w=10",  (4.0, 0.1, 40, 20, 10)),
    # short vs long walks
    ("p=4   q=0.1 wl=80 nw=10 w=5",   (4.0, 0.1, 80, 10, 5)),
    ("p=0.25 q=10 wl=10 nw=40 w=3",   (0.25, 10.0, 10, 40, 3)),
    ("p=0.25 q=10 wl=5  nw=80 w=2",   (0.25, 10.0, 5, 80, 2)),
    ("p=4   q=0.1 wl=160 nw=10 w=5",  (4.0, 0.1, 160, 10, 5)),
    # extreme BFS with short walk + small window — should bring hubs together
    ("p=0.1 q=20  wl=8  nw=80 w=2",   (0.1, 20.0, 8, 80, 2)),
    ("p=0.1 q=50  wl=6  nw=100 w=2",  (0.1, 50.0, 6, 100, 2)),
    # extreme DFS with long walks
    ("p=10  q=0.02 wl=80 nw=20 w=10", (10.0, 0.02, 80, 20, 10)),
]

results = {}
for name, params in configs:
    emb, sil, hpr = run(*params)
    print(f"{name:<55} {sil:<6.3f} {hpr:<10.3f}")
    results[name] = (emb, sil, hpr, params)

print("\n=== Best DFS (anchors FAR APART; hub_ratio > 1) ===")
for name, (_, sil, hpr, _) in sorted(results.items(), key=lambda x: -x[1][2])[:5]:
    print(f"  {name:<55} hub_ratio={hpr:.3f} sil={sil:.3f}")

print("\n=== Best BFS (anchors CLOSE TOGETHER; hub_ratio < 1) ===")
for name, (_, sil, hpr, _) in sorted(results.items(), key=lambda x: x[1][2])[:5]:
    print(f"  {name:<55} hub_ratio={hpr:.3f} sil={sil:.3f}")

# Save the best DFS / balanced / BFS as candidates
print("\nWriting candidate embeddings to data/n2v_explore/*.npy ...")
import os
os.makedirs("data/n2v_explore", exist_ok=True)
np.save("data/n2v_explore/balanced.npy", results["p=1   q=1   wl=40 nw=20 w=5"][0])
best_dfs = sorted(results.items(), key=lambda x: -x[1][2])[0]
best_bfs = sorted(results.items(), key=lambda x: x[1][2])[0]
print(f"  picked DFS: {best_dfs[0]}")
print(f"  picked BFS: {best_bfs[0]}")
np.save("data/n2v_explore/dfs.npy", best_dfs[1][0])
np.save("data/n2v_explore/bfs.npy", best_bfs[1][0])
