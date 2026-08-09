# CQU-LSH experiments

This repository contains the modified FALCONN implementation and the grid
search used to compare standard LSH, QU-LSH (MTQ), and CQU-LSH. The published
JSON results and plotting notebooks are in `experiments/`.

Build the reproducible environment:

```sh
docker build -t cqu-lsh .
```

Place datasets under one root using the paths declared at the top of
`experiments/run_experiment.py`, then run one dataset at a time:

```sh
docker run --rm \
  -v /absolute/path/to/datasets:/datasets:ro \
  -v "$PWD/experiments:/results" \
  cqu-lsh glove
```

Valid dataset names are shown by `docker run --rm cqu-lsh --help`. Results are
appended to `<dataset>/{vanilla_lsh,mtq_lsh,cqu_lsh}.json`. The full grid over
all datasets takes several days; run only the datasets needed and use the
matching `graph_*.ipynb` notebook to reproduce plots.
