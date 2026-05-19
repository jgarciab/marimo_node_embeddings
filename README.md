# marimo_node_embeddings

An interactive [marimo](https://marimo.io) companion app for node embeddings
and node classification, built for the **Network Science Summer School**
(Utrecht University, Day 3b).

**Live app (runs entirely in the browser, no server needed):**
👉 https://jgarciab.github.io/marimo_node_embeddings/

## What's in it

Five sections, built up on the US college football network (115 teams, 12
conferences), Zachary's karate club (34 members, 2 factions), or a network
you upload yourself (CSV edge list or GraphML):

1. **The dataset** — pick a network in the sidebar; see stats, layout, and
   a colour-by-class plot.
2. **Spectral embeddings** — Truncated SVD, Laplacian Eigenmaps, and PCA of
   the adjacency, side by side. Includes a primer on the silhouette score.
3. **Random walks → node2vec** — visualise a single random walk on the
   network and read it as a sentence; then see three node2vec embeddings
   trained with $q \in \{0.25, 1, 4\}$ to show how the bias parameter
   reshapes the embedding.
4. **Supervised GraphSAGE** — a 2-layer GNN trained to predict the node
   class on a 50/50 train/test split. Shows the split on the network, the
   train/test accuracy curve, and the learned 32-d embedding (with test
   errors highlighted).
5. **Comparison: predicting the class from each embedding** — every
   embedding plugged into a multinomial logistic regression on the *same*
   train/test mask, reported as one comparison table (accuracy / macro
   precision, recall, F1).

## Running locally

```bash
./run.sh --setup   # first time: create the uv venv and install deps
./run.sh           # launch `marimo edit app.py`
```

`run.sh` points `uv` at an external venv (`~/.uv_envs/day3b_embeddings`)
because the project lives on pCloud Drive, which breaks symlinks. If you
clone this somewhere with proper symlink support, you can also just run:

```bash
uv sync
uv run marimo edit app.py
```

## Building the WASM bundle locally

```bash
bash export_wasm.sh
cd build && python -m http.server 8000
```

GitHub Pages is configured to do this automatically on every push to
`main` via `.github/workflows/deploy.yml`.

## Re-computing the bundled embeddings

The node2vec and GraphSAGE embeddings are precomputed (the browser cannot
run torch). To regenerate them:

```bash
# Needs torch, torch_geometric, node2vec, gensim. The script's docstring
# points at one conda env that has them installed.
python precompute_embeddings.py            # everything
python precompute_embeddings.py --only supervised   # just one block
```

Outputs land in `data/`; copy the ones the app needs (`*node2vec*.npy`,
`*gnn_supervised.npz`, `football_network.graphml`) into `public/` —
marimo's WASM exporter bundles `public/` next to the notebook.

## Credits

US college football network: Girvan & Newman, *PNAS* 2002.
Karate club: Zachary, *J. Anthropol. Res.* 1977.
