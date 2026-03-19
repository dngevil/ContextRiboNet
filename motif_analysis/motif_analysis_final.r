setwd("E:/projects/合作课题/亢雨笺/ContextRiboNet/")
library(data.table)
library(ggplot2)
library(dplyr)

df <- fread("motif_scan_hits.csv")

#get motif position
motif_position_histogram <- function(hits, motif){
  tmp <- hits[hits$kernel==motif,]
  tmp <- tmp %>%
    mutate(
      motif_len = pos_end_raw0 - pos_start_raw0 + 1,
      mid = (pos_start_raw0 + pos_end_raw0) / 2,
      rel_mid = mid / seq_len_raw,
      rel_start = pos_start_raw0 / seq_len_raw,
      rel_start1 = ifelse(rel_start < 0, 0, rel_start),
      rel_end   = pos_end_raw0 / seq_len_raw
    )

  p1 <- ggplot(tmp, aes(x = rel_mid)) +
    geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.6) +
    geom_density(linewidth = 1) +
    theme_bw() +
    theme(
      axis.text = element_text(size=14),
      axis.title = element_text(size = 16),
      plot.title = element_text(hjust = 0.5)
    ) +
    labs(
      x = "Relative position of motif midpoint",
      y = "Density",
      title = motif
    ) 
  return(p1)
}

k20_f8 <- motif_position_histogram(df, "kernel_k20_f8")
k20_f8
k12_f17 <- motif_position_histogram(df, "kernel_k12_f17")
k12_f17
k12_f29 <- motif_position_histogram(df, "kernel_k12_f29")
k12_f29
k16_f11 <- motif_position_histogram(df, "kernel_k16_f11")
k16_f11

###=====================function enrichment=====================================
dotplot <- function(df, n=10, cat, title){
  df.tmp <- df[df$Category==cat, ]
  df.tmp <- df.tmp %>% top_n(n = n, wt = -P.Value)
  df.dotplot <- ggplot(df.tmp,aes(x=Fold.Enrichment,y=reorder(Term, Fold.Enrichment)))+
    geom_point(aes(size=Count,color=P.Value))+
    theme_bw()+
    ggtitle(title) +
    theme(
      axis.text = element_text(size=14),
      panel.grid=element_blank(),
      axis.title = element_text(size = 16),
      plot.title = element_text(hjust = 0.5)
    ) + 
    ylab("") +
    xlab("Fold Enrichment") +
    scale_colour_gradient(low = "red",high = "blue")
  df.dotplot  
  return(df.dotplot)
}

barplot <- function(df, n=5, cat, title){
  df.tmp <- df[df$Category==cat, ]
  df.tmp <- df.tmp %>% top_n(n = n, wt = -P.Value)
  df.dotplot <- ggplot(df.tmp,aes(x=-log10(P.Value),y=reorder(Term, -P.Value)))+
    geom_bar(stat="identity", fill="#84CAC0")+
    theme_bw()+
    ggtitle(title) +
    theme(
      axis.text = element_text(size=14),
      panel.grid=element_blank(),
      axis.title = element_text(size = 16),
      plot.title = element_text(hjust = 0.5)
    ) + 
    ylab("") +
    xlab("-log10(pvalue)")
  df.dotplot  
  return(df.dotplot)
}

k20_f8_func <- read.csv("function_enrichment/DAVIDChartReport_k20_f8.csv")
k20_f8_func.fig <- dotplot(k20_f8_func, n=5, cat = "GOTERM_BP_DIRECT", title = "kernel_k20_f8")
k20_f8_func.fig <- barplot(k20_f8_func, n=5, cat = "GOTERM_BP_DIRECT", title = "kernel_k20_f8")
k20_f8_func.fig

k12_f17_func <- read.csv("function_enrichment/DAVIDChartReport_k12_f17.csv")
k12_f17_func.fig <- dotplot(k12_f17_func, n=5, cat = "GOTERM_BP_DIRECT", title = "kernel_k12_f17")
k12_f17_func.fig <- barplot(k12_f17_func, n=5, cat = "GOTERM_BP_DIRECT", title = "kernel_k12_f17")
k12_f17_func.fig

k12_f29_func <- read.csv("function_enrichment/DAVIDChartReport_k12_f29.csv")
k12_f29_func.fig <- dotplot(k12_f29_func, n=5, cat = "GOTERM_BP_DIRECT", title = "kernel_k12_f29")
k12_f29_func.fig <- barplot(k12_f29_func, n=5, cat = "GOTERM_BP_DIRECT", title = "kernel_k12_f29")
k12_f29_func.fig

k16_f11_func <- read.csv("function_enrichment/DAVIDChartReport_k16_f11.csv")
k16_f11_func.fig <- dotplot(k16_f11_func, n=5, cat = "GOTERM_BP_DIRECT", title = "kernel_k16_f11")
k16_f11_func.fig <- barplot(k16_f11_func, n=5, cat = "GOTERM_BP_DIRECT", title = "kernel_k16_f11")
k16_f11_func.fig
