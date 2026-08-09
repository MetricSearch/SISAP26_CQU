docker build -t cqu-lsh . &&
for dataset in glove gooaq mf_dino2 pubmed; do
  docker run --rm \
    -v /Volumes/Data:/datasets:ro \
    -v "$PWD/experiments:/results" \
    cqu-lsh "$dataset"
done