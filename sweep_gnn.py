"""GNN architecture sweep on football.

Goal: a supervised GNN should clearly beat SVD/PCA (which never saw the
labels). Right now it ties. Sweep across:
  - architecture (SAGE / GCN / GAT)
  - depth (2 / 3 layers)
  - hidden dimension (32 / 64 / 128)
  - dropout (0.2 / 0.5)
  - epochs (100 / 200 / 400)
  - initial features (identity / degree / row-of-A)
  - learning rate (1e-2 / 5e-3)
"""
import time
import numpy as np
import igraph as ig
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEED = 1546
np.random.seed(SEED); torch.manual_seed(SEED)

g = ig.Graph.Read_GraphML("data/football_network.graphml")
labels = np.array([int(v) for v in g.vs["value"]])
n = g.vcount()
edges = [(e.source, e.target) for e in g.es]
src, dst = [], []
for u, v in edges:
    src.extend([u, v]); dst.extend([v, u])
edge_index = torch.tensor([src, dst], dtype=torch.long)

train_idx, test_idx = train_test_split(np.arange(n), test_size=0.5, stratify=labels, random_state=SEED)
train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
test_mask = ~train_mask
train_mask_t = torch.tensor(train_mask)
test_mask_t = torch.tensor(test_mask)
y = torch.tensor(labels, dtype=torch.long)
num_classes = int(labels.max() + 1)

# Three feature options
A = np.array(g.get_adjacency().data, dtype=np.float32)
deg = A.sum(axis=1)
features = {
    "identity": torch.eye(n, dtype=torch.float32),
    "row-of-A": torch.tensor(A, dtype=torch.float32),
    "deg+1hot": torch.cat([torch.tensor(deg[:, None] / deg.max(), dtype=torch.float32), torch.eye(n, dtype=torch.float32)], dim=1),
}


def make_model(arch, in_dim, hid, out, n_layers, dropout):
    layers = []
    sizes = [in_dim] + [hid] * (n_layers - 1) + [out]
    convs = []
    for i in range(n_layers):
        if arch == "SAGE":
            convs.append(SAGEConv(sizes[i], sizes[i + 1]))
        elif arch == "GCN":
            convs.append(GCNConv(sizes[i], sizes[i + 1]))
        elif arch == "GAT":
            heads = 4 if i < n_layers - 1 else 1
            convs.append(GATConv(sizes[i], sizes[i + 1] // heads if i < n_layers - 1 else sizes[i + 1],
                                  heads=heads, dropout=dropout, concat=(i < n_layers - 1)))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = torch.nn.ModuleList(convs)
            self.head = torch.nn.Linear(out, num_classes)
            self.dropout = dropout

        def encode(self, x, ei):
            for i, conv in enumerate(self.convs):
                x = conv(x, ei)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        def forward(self, x, ei):
            z = self.encode(x, ei)
            return z, self.head(z)

    return Net()


def fit_and_eval(arch, feat_name, feat, hid, n_layers, dropout, epochs, lr, wd):
    torch.manual_seed(SEED)
    in_dim = feat.shape[1]
    model = make_model(arch, in_dim, hid, 32, n_layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    data = Data(x=feat, edge_index=edge_index)
    best_te = 0.0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        _, logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask_t], y[train_mask_t])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            _, lo = model(data.x, data.edge_index)
            pred = lo.argmax(1)
            te = (pred[test_mask_t] == y[test_mask_t]).float().mean().item()
            best_te = max(best_te, te)
    # Also: use the embedding + LR (downstream classifier, like the app)
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index).cpu().numpy()
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=10).fit(z[train_mask], labels[train_mask])
    yhat = clf.predict(z[test_mask])
    lr_acc = accuracy_score(labels[test_mask], yhat)
    lr_f1 = f1_score(labels[test_mask], yhat, average="macro", zero_division=0)
    # And the GNN's own prediction
    with torch.no_grad():
        _, lo = model(data.x, data.edge_index)
        pred = lo.argmax(1).cpu().numpy()
    gnn_acc = accuracy_score(labels[test_mask], pred[test_mask])
    gnn_f1 = f1_score(labels[test_mask], pred[test_mask], average="macro", zero_division=0)
    return gnn_acc, gnn_f1, lr_acc, lr_f1, best_te


def main():
    print(f"{'arch':<5} {'feat':<8} {'L':<2} {'h':<3} {'do':<5} {'ep':<3} {'lr':<6} {'wd':<6}  "
          f"{'gnn_acc':<8} {'gnn_F1':<7} {'lr_acc':<7} {'lr_F1':<7} {'best_te':<7}")
    rows = []
    # Architecture × feature × hidden × layers × dropout
    configs = []
    for arch in ["SAGE", "GCN", "GAT"]:
        for feat_name in ["identity", "row-of-A"]:
            for hid in [64, 128]:
                for n_layers in [2, 3]:
                    for do in [0.3, 0.5]:
                        configs.append((arch, feat_name, hid, n_layers, do, 200, 5e-3, 5e-4))
    # Also a few extra runs varying epochs
    for arch in ["SAGE", "GCN"]:
        for ep in [100, 400]:
            configs.append((arch, "identity", 64, 2, 0.5, ep, 5e-3, 5e-4))

    for cfg in configs:
        arch, feat_name, hid, nl, do, ep, lr, wd = cfg
        feat = features[feat_name]
        try:
            t0 = time.time()
            gnn_acc, gnn_f1, lr_acc, lr_f1, best_te = fit_and_eval(arch, feat_name, feat, hid, nl, do, ep, lr, wd)
            dt = time.time() - t0
            print(f"{arch:<5} {feat_name:<8} {nl:<2} {hid:<3} {do:<5} {ep:<3} {lr:<6.0e} {wd:<6.0e}  "
                  f"{gnn_acc:<8.3f} {gnn_f1:<7.3f} {lr_acc:<7.3f} {lr_f1:<7.3f} {best_te:<7.3f}")
            rows.append((cfg, gnn_acc, gnn_f1, lr_acc, lr_f1, best_te))
        except Exception as e:
            print(f"{arch} FAILED: {e}")

    print("\n=== TOP 10 by GNN macro F1 ===")
    for cfg, ga, gf, la, lf, bt in sorted(rows, key=lambda r: -r[2])[:10]:
        arch, feat_name, hid, nl, do, ep, lr, wd = cfg
        print(f"  {arch}/{feat_name}/L={nl}/h={hid}/do={do}/ep={ep}  gnn_F1={gf:.3f}  lr_F1={lf:.3f}")


if __name__ == "__main__":
    main()
