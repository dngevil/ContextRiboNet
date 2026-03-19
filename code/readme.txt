##Download data
wget ftp://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz
wget ftp://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz
wget ftp://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz

gunzip *.gz


python pipeline_prepare_and_train.py \
  --rna_path ../data/GSE197265_h_GVtohESC_28500_RNA_merge_average_fpkm.txt \
  --ribo_path ../data/GSE197265_h_GVtohESC_28500_Ribo_merge_average_fpkm.txt \
  --cdna_fa  ../data/Homo_sapiens.GRCh38.cdna.all.fa \
  --cds_fa   ../data/Homo_sapiens.GRCh38.cds.all.fa \
  --outdir   ../preprocess_data/pipeline_out \
  --target   ribo \
  --log1p \
  --model    ridge \
  --plot_example_stage 2C \
  --save_sequences


python build_model_inputs.py \
  --seq_features ../preprocess_data/pipeline_out/seq_features.csv \
  --ribo_path   ../data/GSE197265_h_GVtohESC_28500_Ribo_merge_average_fpkm.txt \
  --rna_path    ../data/GSE197265_h_GVtohESC_28500_RNA_merge_average_fpkm.txt \
  --out_npz     ../preprocess_data/pipeline_out/model_inputs_cds_lenle5000_withRNA.npz \
  --seq_mode cds \
  --max_seq_len 5000 \
  --dedup_ribo --dedup_rna \
  --log1p --log1p_rna



##The relationship between RNA seq and ribo
python RNAseq_vs_Riboseq.py \
  --rna_path  ../data/GSE197265_h_GVtohESC_28500_RNA_merge_average_fpkm.txt \
  --ribo_path ../data/GSE197265_h_GVtohESC_28500_Ribo_merge_average_fpkm.txt \
  --outdir    ../preprocess_data/analysis_r2 \
  --dedup \
  --log1p

##Based on the analysis of the characteristic preferences of cell lines at different life cycles

python analyze_lasso_contrib_by_stage.py \
  --npz ../preprocess_data/pipeline_out/model_inputs_cds_lenle5000_withRNA.npz \
  --kernel_npz ../preprocess_data/cnn_mlp_runs/kernels/kernel_scores_topk.npz \
  --coeff_csv ../preprocess_data/lasso_on_kernels/lasso_coefficients.csv \
  --outdir ./preprocess_data/lasso_on_kernels/lasso_contrib_by_stage \
  --include_other --drop_rna_col \
  --top_k_plot 40 --relative --pdf_all


##Increase cell line state characterization
python train_cnn_mlp_add_cellstate.py \
  --npz    ../preprocess_data/pipeline_out/model_inputs_cds_lenle5000_withRNA.npz \
  --outdir ../result/cnn_mlp_add_cellstate/single_label_v1 \
  --epochs 200 \
  --batch_size 32 \
  --lr 2e-4 --wd 1e-2 \
  --scheduler plateau --lr_factor 0.5 --lr_patience 10 --min_lr 1e-6 \
  --early_patience 30 \
  --test_ratio_per_stage 0.2 --val_ratio_in_train 0.2 \
  --seed 42

##For increasing the characterization of cell line states
python extract_kernels_and_scores_add_cellstate.py   --npz ../preprocess_data/pipeline_out/model_inputs_cds_lenle5000_withRNA.npz   --ckpt ../result/cnn_mlp_add_cellstate/single_label_v1/best.pt   --outdir ../result/cnn_mlp_add_cellstate/single_label_v1/kernels   --split_mode per_gene   --test_ratio_per_stage 0.2   --val_ratio_in_train 0.2   --seed 42   --kernel_sizes 6 10 12 16 20   --cnn_channels 64   --mlp_hidden 128 --mlp_blocks 3 --dropout 0.1   --topk_per_kernel 500 --min_hits 100 --save_meme

## Supplementary Analysis of Figure 3
python export_pwms_for_figure3.py \
  --npz ../preprocess_data/pipeline_out/model_inputs_cds_lenle5000_withRNA.npz \
  --ckpt ../result/cnn_mlp_add_cellstate/single_label_v1/best.pt \
  --outdir ../result/cnn_mlp_add_cellstate/single_label_v1/kernels \
  --kernels kernel_k20_f8 kernel_k12_f17 kernel_k12_f29 kernel_k16_f11


python make_figure3.py \
    --activity-csv ../result/cnn_mlp_add_cellstate/single_label_v1/kernels/kernel_stage_activity.csv \
    --pwm-dir ../result/cnn_mlp_add_cellstate/single_label_v1/kernels/pwm_selected \
    --outdir figure3_out \
    --global-kernel kernel_k20_f8 \
    --stage-kernel 4C:kernel_k12_f17 \
    --stage-kernel 8C:kernel_k12_f29 \
    --stage-kernel hESC:kernel_k16_f11 \
    --delta-csv ../result/cnn_mlp_add_cellstate/single_label_v1/kernels/cellstate_effect_stage_means_topK.csv



##Motifs are filtered based on high-scoring kernels, and then the scan score of each motif on each sequence is obtained.
python scan_motif_hits_figure3.py \
  --npz  ./preprocess_data/pipeline_out/model_inputs_cds_lenle5000_withRNA.npz \
  --ckpt ./result/cnn_mlp_add_cellstate/single_label_v1/best.pt \
  --kernels kernel_k20_f8 kernel_k12_f17 kernel_k12_f29 kernel_k16_f11 \
  --out_csv figure3_out/motif_scan_hits.csv \
  --batch_size 256 \
  --top_hits_per_sample 3 \
  --global_quantile 0.995


