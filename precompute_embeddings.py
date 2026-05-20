"""Precompute node2vec and GraphSAGE embeddings for the football network.

Run with the networks conda env (has torch, torch_geometric, node2vec, gensim):

    /Users/garci061/miniforge3/envs/networks/bin/python precompute_embeddings.py
    /Users/garci061/miniforge3/envs/networks/bin/python precompute_embeddings.py --only supervised

Produces in data/:
    node2vec_p1_q1.npy
    node2vec_p1_q0.5.npy
    node2vec_p1_q2.npy
    gnn_graphsage.npy                 (self-supervised, random-walk objective)
    gnn_graphsage_supervised.npz      (supervised, predicts the conference label;
                                       bundles emb, train/test masks, preds,
                                       and full loss/accuracy histories)
    node_names.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
from node2vec import Node2Vec
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    "--only",
    choices=["all", "node2vec", "unsupervised", "supervised", "karate"],
    default="all",
    help="Restrict to a subset of the precompute steps (handy for re-runs).",
)
args = parser.parse_args()
RUN = args.only

SEED = 1546
np.random.seed(SEED)
torch.manual_seed(SEED)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# ---------------------------------------------------------------------------
# Load football network
# ---------------------------------------------------------------------------
g = ig.Graph.Read_GraphML(str(DATA / "football_network.graphml"))
n = g.vcount()
names = list(g.vs["id"])
labels = np.array([int(v) for v in g.vs["value"]])
print(f"loaded football: n={n}, m={g.ecount()}, classes={len(set(labels))}")

# Edge list
edges = [(e.source, e.target) for e in g.es]

# Save names
pd.DataFrame({"name": names}).to_csv(DATA / "node_names.csv", index=False)
print(f"wrote node_names.csv ({len(names)} rows)")

# ---------------------------------------------------------------------------
# Build networkx graph for node2vec
# ---------------------------------------------------------------------------
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from(edges)


def run_node2vec(p: float, q: float, out_path: Path) -> np.ndarray:
    print(f"  node2vec p={p}, q={q} ...")
    n2v = Node2Vec(
        G,
        dimensions=32,
        walk_length=20,
        num_walks=10,
        p=p,
        q=q,
        workers=1,
        seed=SEED,
        quiet=True,
    )
    model = n2v.fit(window=5, min_count=1, batch_words=4, seed=SEED, workers=1)
    emb = np.zeros((n, 32), dtype=np.float32)
    for i in range(n):
        emb[i] = model.wv[str(i)]
    np.save(out_path, emb)
    sil = silhouette_score(emb, labels)
    print(f"    saved {out_path.name}  shape={emb.shape}  silhouette={sil:.3f}")
    return emb


if RUN in ("all", "node2vec"):
    run_node2vec(1.0, 0.25, DATA / "node2vec_p1_q0.25.npy")
    run_node2vec(1.0, 1.0, DATA / "node2vec_p1_q1.npy")
    run_node2vec(1.0, 4.0, DATA / "node2vec_p1_q4.npy")
else:
    print(f"skipping node2vec (only={RUN})")

# ---------------------------------------------------------------------------
# GraphSAGE (unsupervised, random-walk objective, short training)
# ---------------------------------------------------------------------------
print("graphsage (self-supervised) ...")

# PyG-style edge_index (undirected => both directions)
src = []
dst = []
for u, v in edges:
    src.extend([u, v])
    dst.extend([v, u])
edge_index = torch.tensor([src, dst], dtype=torch.long)

# Identity features so the GNN has something to propagate
x = torch.eye(n, dtype=torch.float32)
data = Data(x=x, edge_index=edge_index)


class SAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, out_dim)

    def forward(self, x_in, ei):
        h = F.relu(self.conv1(x_in, ei))
        h = self.conv2(h, ei)
        return h


model = SAGE(in_dim=n, hid=64, out_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Pre-sample random walks (length 10) for the positive pairs
rng = np.random.default_rng(SEED)
neighbors = [list(G.neighbors(i)) for i in range(n)]


def sample_walks(num_walks_per_node: int = 5, length: int = 10):
    walks = []
    for _ in range(num_walks_per_node):
        for start in range(n):
            walk = [start]
            cur = start
            for _ in range(length - 1):
                nbrs = neighbors[cur]
                if not nbrs:
                    break
                cur = nbrs[rng.integers(len(nbrs))]
                walk.append(cur)
            walks.append(walk)
    return walks


def pos_neg_pairs(walks, window: int = 3, num_neg: int = 5):
    pos_a, pos_b = [], []
    for w in walks:
        L = len(w)
        for i in range(L):
            for j in range(max(0, i - window), min(L, i + window + 1)):
                if i == j:
                    continue
                pos_a.append(w[i])
                pos_b.append(w[j])
    pos_a_t = torch.tensor(pos_a, dtype=torch.long)
    pos_b_t = torch.tensor(pos_b, dtype=torch.long)
    neg_b_t = torch.tensor(rng.integers(0, n, size=len(pos_a) * num_neg), dtype=torch.long)
    neg_a_t = pos_a_t.repeat_interleave(num_neg)
    return pos_a_t, pos_b_t, neg_a_t, neg_b_t


if RUN in ("all", "unsupervised"):
    model.train()
    EPOCHS = 80
    for epoch in range(EPOCHS):
        walks = sample_walks(num_walks_per_node=3, length=10)
        pa, pb, na, nb = pos_neg_pairs(walks, window=3, num_neg=3)
        optimizer.zero_grad()
        h = model(data.x, data.edge_index)
        pos_score = (h[pa] * h[pb]).sum(dim=1)
        neg_score = (h[na] * h[nb]).sum(dim=1)
        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1:>3}/{EPOCHS}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        emb_sage = model(data.x, data.edge_index).detach().cpu().numpy().astype(np.float32)

    np.save(DATA / "gnn_graphsage.npy", emb_sage)
    sil_sage = silhouette_score(emb_sage, labels)
    print(f"  saved gnn_graphsage.npy  shape={emb_sage.shape}  silhouette={sil_sage:.3f}")
else:
    print(f"skipping self-supervised graphsage (only={RUN})")

# ---------------------------------------------------------------------------
# GraphSAGE (supervised: predict the conference label from a 50/50 split)
# ---------------------------------------------------------------------------
class SupervisedSAGE(torch.nn.Module):
    """2-layer GraphSAGE with dropout for the supervised setting.

    Dropout is applied:
      - on the input features (small p, to regularise the identity feats),
      - on the hidden activations between the two SAGE conv layers (p=0.5).
    Inference (model.eval()) disables dropout automatically.
    """

    def __init__(self, in_dim: int, hid: int, emb_dim: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, emb_dim)
        self.head = torch.nn.Linear(emb_dim, n_classes)
        self.dropout_p = dropout

    def encode(self, x_in, ei):
        h = F.dropout(x_in, p=min(self.dropout_p, 0.2), training=self.training)
        h = F.relu(self.conv1(h, ei))
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        return self.conv2(h, ei)

    def forward(self, x_in, ei):
        z = self.encode(x_in, ei)
        return z, self.head(z)


if RUN in ("all", "supervised"):
    print("graphsage (supervised) ...")

    num_classes = int(labels.max() + 1)
    y = torch.tensor(labels, dtype=torch.long)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.5, stratify=labels, random_state=SEED
    )
    train_mask_np = np.zeros(n, dtype=bool)
    train_mask_np[train_idx] = True
    test_mask_np = ~train_mask_np
    train_mask_t = torch.tensor(train_mask_np)
    test_mask_t = torch.tensor(test_mask_np)

    torch.manual_seed(SEED)
    sup = SupervisedSAGE(in_dim=n, hid=64, emb_dim=32, n_classes=num_classes)
    opt = torch.optim.Adam(sup.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_hist, tr_acc_hist, te_acc_hist = [], [], []
    EPOCHS_SUP = 100
    for epoch in range(EPOCHS_SUP):
        sup.train()
        opt.zero_grad()
        _, logits = sup(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask_t], y[train_mask_t])
        loss.backward()
        opt.step()
        sup.eval()
        with torch.no_grad():
            _, logits_eval = sup(data.x, data.edge_index)
            pred = logits_eval.argmax(dim=1)
            tr_acc = (pred[train_mask_t] == y[train_mask_t]).float().mean().item()
            te_acc = (pred[test_mask_t] == y[test_mask_t]).float().mean().item()
        loss_hist.append(loss.item())
        tr_acc_hist.append(tr_acc)
        te_acc_hist.append(te_acc)
        if (epoch + 1) % 25 == 0:
            print(
                f"  epoch {epoch+1:>3}/{EPOCHS_SUP}  loss={loss.item():.4f}  "
                f"train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}"
            )

    sup.eval()
    with torch.no_grad():
        z_final = sup.encode(data.x, data.edge_index).cpu().numpy().astype(np.float32)
        _, logits_final = sup(data.x, data.edge_index)
        preds_final = logits_final.argmax(dim=1).cpu().numpy().astype(np.int32)

    out_path = DATA / "gnn_supervised.npz"
    np.savez(
        out_path,
        emb=z_final,
        train_mask=train_mask_np,
        test_mask=test_mask_np,
        preds=preds_final,
        true=labels.astype(np.int32),
        loss_history=np.array(loss_hist, dtype=np.float32),
        train_acc_history=np.array(tr_acc_hist, dtype=np.float32),
        test_acc_history=np.array(te_acc_hist, dtype=np.float32),
    )
    final_te = te_acc_hist[-1]
    final_tr = tr_acc_hist[-1]
    sil_sup = silhouette_score(z_final, labels)
    print(
        f"  saved {out_path.name}  shape={z_final.shape}  "
        f"train_acc={final_tr:.3f}  test_acc={final_te:.3f}  silhouette={sil_sup:.3f}"
    )
else:
    print(f"skipping supervised graphsage (only={RUN})")

# ===========================================================================
# Karate club — same suite (3× node2vec + supervised GraphSAGE)
# ===========================================================================
if RUN in ("all", "karate"):
    print("\nkarate club precompute ...")
    g_k = ig.Graph.Famous("Zachary")
    n_k = g_k.vcount()
    labels_k = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1,
    ])
    edges_k = [(e.source, e.target) for e in g_k.es]
    G_k = nx.Graph()
    G_k.add_nodes_from(range(n_k))
    G_k.add_edges_from(edges_k)

    def run_node2vec_k(p: float, q: float, out_path: Path) -> None:
        print(f"  karate node2vec p={p}, q={q} ...")
        n2v = Node2Vec(
            G_k, dimensions=32, walk_length=10, num_walks=20,
            p=p, q=q, workers=1, seed=SEED, quiet=True,
        )
        model = n2v.fit(window=5, min_count=1, batch_words=4, seed=SEED, workers=1)
        emb = np.zeros((n_k, 32), dtype=np.float32)
        for i in range(n_k):
            emb[i] = model.wv[str(i)]
        np.save(out_path, emb)
        sil = silhouette_score(emb, labels_k)
        print(f"    saved {out_path.name}  shape={emb.shape}  silhouette={sil:.3f}")

    run_node2vec_k(1.0, 0.25, DATA / "karate_node2vec_p1_q0.25.npy")
    run_node2vec_k(1.0, 1.0, DATA / "karate_node2vec_p1_q1.npy")
    run_node2vec_k(1.0, 4.0, DATA / "karate_node2vec_p1_q4.npy")

    # Supervised GraphSAGE for karate
    print("  karate graphsage (supervised) ...")
    src_k = []
    dst_k = []
    for u, v in edges_k:
        src_k.extend([u, v])
        dst_k.extend([v, u])
    edge_index_k = torch.tensor([src_k, dst_k], dtype=torch.long)
    x_k = torch.eye(n_k, dtype=torch.float32)
    data_k = Data(x=x_k, edge_index=edge_index_k)

    indices_k = np.arange(n_k)
    train_idx_k, test_idx_k = train_test_split(
        indices_k, test_size=0.5, stratify=labels_k, random_state=SEED
    )
    train_mask_k = np.zeros(n_k, dtype=bool); train_mask_k[train_idx_k] = True
    test_mask_k = ~train_mask_k
    train_mask_kt = torch.tensor(train_mask_k)
    test_mask_kt = torch.tensor(test_mask_k)
    y_k = torch.tensor(labels_k, dtype=torch.long)

    torch.manual_seed(SEED)
    sup_k = SupervisedSAGE(in_dim=n_k, hid=32, emb_dim=32, n_classes=2)
    opt_k = torch.optim.Adam(sup_k.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_hist_k, tr_acc_hist_k, te_acc_hist_k = [], [], []
    EPOCHS_K = 100
    for epoch in range(EPOCHS_K):
        sup_k.train()
        opt_k.zero_grad()
        _, logits = sup_k(data_k.x, data_k.edge_index)
        loss = F.cross_entropy(logits[train_mask_kt], y_k[train_mask_kt])
        loss.backward()
        opt_k.step()
        sup_k.eval()
        with torch.no_grad():
            _, logits_eval = sup_k(data_k.x, data_k.edge_index)
            pred = logits_eval.argmax(dim=1)
            tr = (pred[train_mask_kt] == y_k[train_mask_kt]).float().mean().item()
            te = (pred[test_mask_kt] == y_k[test_mask_kt]).float().mean().item()
        loss_hist_k.append(loss.item())
        tr_acc_hist_k.append(tr)
        te_acc_hist_k.append(te)
        if (epoch + 1) % 25 == 0:
            print(
                f"    epoch {epoch+1:>3}/{EPOCHS_K}  loss={loss.item():.4f}  "
                f"train_acc={tr:.3f}  test_acc={te:.3f}"
            )

    sup_k.eval()
    with torch.no_grad():
        z_final_k = sup_k.encode(data_k.x, data_k.edge_index).cpu().numpy().astype(np.float32)
        _, logits_final_k = sup_k(data_k.x, data_k.edge_index)
        preds_final_k = logits_final_k.argmax(dim=1).cpu().numpy().astype(np.int32)

    out_k = DATA / "karate_gnn_supervised.npz"
    np.savez(
        out_k,
        emb=z_final_k,
        train_mask=train_mask_k,
        test_mask=test_mask_k,
        preds=preds_final_k,
        true=labels_k.astype(np.int32),
        loss_history=np.array(loss_hist_k, dtype=np.float32),
        train_acc_history=np.array(tr_acc_hist_k, dtype=np.float32),
        test_acc_history=np.array(te_acc_hist_k, dtype=np.float32),
    )
    print(
        f"    saved {out_k.name}  shape={z_final_k.shape}  "
        f"train_acc={tr_acc_hist_k[-1]:.3f}  test_acc={te_acc_hist_k[-1]:.3f}"
    )
else:
    print(f"skipping karate (only={RUN})")

print("done.")
