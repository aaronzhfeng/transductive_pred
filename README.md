## Graph2Transforms: Reaction → Autoregressive Edits Corpus

This toolkit converts tokenized reaction SMILES (reactants → products) into a canonical, autoregressive sequence of chemically valid edit actions. It verifies the sequence by rolling edits forward, and emits per-step graph snapshots, targets, masks, and vocabularies for training graph-to-edits models.

### Inputs
- Six files total in a directory, named `src-{split}` and `tgt-{split}` for `{train,val,test}`.
- Lines contain tokenized SMILES. Components are separated by a `.` token. Tokens are space-separated. To reconstruct standard SMILES, remove spaces.

### Outputs
- Sharded JSONL files under `out/{split}/shard-*.jsonl` with fields:
  - `reaction_id`, `step`, `graph`, `target_action`, `args`, `masks`, `maps`, `sanity_flags`.
- `vocabs/actions.json` and optional `vocabs/leaving_groups.json`.
- `manifest.json` with shard paths and counts.
- `run.json` capturing versions, thresholds, and provenance.

### Quick start
```bash
# 1) Create and activate a Python 3.10+ environment
# 2) Install
pip install -r requirements.txt

# 3) Build corpus
python -m graph2transforms.cli.build_corpus \
  --data-dir ../uspto \
  --out-dir uspto_edits \
  --num-workers 12 \
  --map-threshold 0.85 \
  --shard-size 20000 \
  --compress-lg --topk-lg 512
```

The CLI expects the six files to be present in `--data-dir`:
- `src-train.txt`, `tgt-train.txt`, `src-val.txt`, `tgt-val.txt`, `src-test.txt`, `tgt-test.txt`.

### Processing overview
1. Standardize reactant/product SMILES with RDKit (sanitize, optional kekulization, salt/solvent removal, canonical SMILES).
2. Create/verify atom mapping with RXNMapper, requiring confidence ≥ threshold.
3. Build graph indices by atom map IDs for deterministic diffs.
4. Diff bonds, hydrogens, and selected atom properties to produce ordered gold edits.
5. Optionally detect and compress leaving-groups into macro actions.
6. Validate by rolling edits forward with sanitization after each step; discard failures.
7. Emit per-step training examples with features and masks.

### Feature schema (encoder)
- Atoms: element (Z), degree, total valence, formal charge, hybridization, aromatic flag, implicit/explicit H, chiral tag.
- Bonds: order (single/double/triple/aromatic), conjugation, ring flag, E/Z stereo.

### Action schema (decoder)
- `DEL_BOND(i,j,order,stereo)`
- `CHANGE_BOND(i,j,old→new,stereo_old→stereo_new)`
- `ADD_BOND(i,j,order,stereo)`
- `H_LOSS(i)` / `H_GAIN(i)`
- Optional `CHANGE_ATOM(i,prop,old→new)`
- Optional `REMOVE_LG(fragment_id)` or `DELETE_SUBGRAPH(node_set_id)`
- `STOP`

Arguments use the current intermediate's atom indices. Masks are precomputed from the live graph to prune illegal actions.

### Model/training expectations (for later work)
- Encoder: graph transformer/GAT with multi-scale atom environment embeddings and optional graph positional encodings.
- Decoder: autoregressive edit policy with action-type head and typed arguments, re-encoding the updated graph at each step.
- Loss: masked cross-entropy over legal actions and arguments.

### Performance and caching
- Multiprocessing is used for dataset creation. Shard outputs for streaming during training.
- Cache masks per step if memory allows; otherwise recompute on load.

### Optional stereo validation
- If installed, `rdchiral` can be used to spot-check stereo consistency on a sample.

### Tests
Run `pytest` to execute minimal unit tests covering common transformations and rollout integrity.

### Provenance and reproducibility
- The pipeline records package versions and thresholds in `run.json`. Fix random seeds to regenerate the same corpus.

### Troubleshooting
- NumPy/RDKit ABI error (`_ARRAY_API not found`): RDKit wheels compiled against NumPy 1.x crash under NumPy 2.x.
  - Quick fix (pip): `pip install "numpy<2" --upgrade`
  - Quick fix (conda): `conda install "numpy<2" -y`
  - Alternatively upgrade RDKit to wheels compatible with NumPy 2.x or rebuild from source.
  - The CLI will fail fast if NumPy 2.x is detected to avoid cryptic crashes in workers.

### License
MIT
