QUERY="../result/cnn_mlp_add_cellstate/single_label_v1/kernels/pwm_selected/meme_selected.txt"
TARGET=".ref/jaspar/JASPAR2024_CORE_vertebrates_nr.meme"
OUTDIR="../result/cnn_mlp_add_cellstate/single_label_v1/kernels/tomtom_jaspar_v2024"

mkdir -p "${OUTDIR}"

tomtom \
  -oc "${OUTDIR}" \
  -no-ssc \
  -dist pearson \
  -min-overlap 5 \
  -evalue \
  -thresh 0.05 \
  "${QUERY}" \
  "${TARGET}"
