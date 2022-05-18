# CCST：Cell clustering for spatial transcriptomics data with graph neural network 

Taking advantages of two recent technical development, spatial transcriptomics and graph neural network, we thus introduce CCST, Cell Clustering for Spatial Transcriptomics data with graph neural network, an unsupervised cell clustering method based on graph convolutional network to improve ab initio cell clustering and discovering of novel sub cell types based on curated cell category annotation. CCST is a general framework for dealing with various kinds of spatially resolved transcriptomics.

Framework

![image](https://github.com/xiaoyeye/CCST/blob/main/figure/figure1.png)


The code is licensed under the MIT license. 

# 1. Requirements 

## 1.1 Operating systems:

The code in python has been tested on both Linux (Ubuntu 16.04.6 LTS) and windows 10 system.

## 1.2 Required packages in python: 

numpy==1.19.2

pandas==1.2.3

sklearn==0.24.1

matplotlib==3.3.4

scipy==1.7.0

pytorch== 1.7.1

torch_geometric==1.6.3

torch_sparse==0.6.8

torch_scatter==2.0.5

seaborn==0.11.1

scanorama==1.7.1

openpyxl==3.0.7

umap-learn==0.5.1

stlearn==0.3.2


# 2. Instructions: Demo on MERFISH.

## 2.1 Raw data 

Raw data should be placed in the folder ***dataset***.

we put the MERFISH dataset, which is downloaded from https://www.pnas.org/content/116/39/19490/tab-figures-data, in ***dataset/MERFISH***. Need to be extracted firstly.

For 10x Spatial Transcripts (ST) datasets, taking V1_Breast_Cancer_Block_A_Section_1 for instance. Files should be put in the following structure, which is the same with that provided by 10x website.

*--V1_Breast_Cancer_Block_A_Section_1 
 *--spatial/  # The folder where files for spatial information can be found 
 *--metadata.tsv # mainly for annotation
 *--V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5 # gene expression data


## 2.2 Data Preprocessing and Graph Construction

Run ***data_generation_merfish.py*** to preprocess the raw data. 

`python data_generation_merfish.py`

We provide the pocessed data in folder ***generated_data***. You can also directly use it without runing ***data_generation_merfish.py***. Specifically, the following five files will be used in CCST.

(1) ***features_array_after_removal_low_var.npy*** saves the preprocessed gene expression. 

(2) ***Adjacent_200*** saves the constructed adjacency matrix.

(3) ***cell_batch_info.npy*** saves the cell batch information of each cell.

(4) ***gene_names_after_removal_low_var.txt*** saves names of selected genes. 

(5) ***all_genes.txt*** saves names of all genes.


## 2.3 Run CCST 

Run ***CCST_merfish.py*** for node clustering and differential expressed gene extracting. The meaning of each argument is listed below.

**--lambda_I**: the value of hyperparameter lambda, which should be within [0,1].

**--DGI**: whether to run the DGI (set to 1) or not (set to 0). 

**--load**：whether to load the pretrained DGI model (set to 1) or not (set to 0). 

**--num_epoch**: the number of epochs in training DGI. 

**--hidden**: the dimension of each hidden layer. 

**--cluster**: whether to perform cluster (set to 1) or not (set to 0).

**--PCA**: whether to perform PCA on the embedding (set to 1) or not (set to 0).

**--n_clusters**: the number of desired clusters.

**--merge**: whether to merge clustered groups with less than three cells into the closest group (set to 1) or not (set to 0).

**--draw_map**: whether to draw the spatial distribution of cells (set to 1) or not (set to 0).

**--diff_gene**: whether to take differential expressed gene analysis (set to 1) or not (set to 0).

**--calculate_score**: whether to calculate the Silhouette score of clustering (set to 1) or not (set to 0).

**--model_path**: the path for saving model.

**--embedding_data_path**: the path for saving embedding data.

**--result_path**: the path for saving results.


## 2.4 Usage and Results Analysis

The trained model, embedding data and analysis results will be saved in folder ***model***, ***embedding_data*** and ***results_CCST*** by default.

We provide the output of DGI in the folder ***embedding_data***. If you want to directly use it, run 

 `python CCST_merfish --DGI 0.  `

We provide the trained model of DGI in the folder ***model***. If you want to directly use it, run

 `python CCST_merfish --DGI 1 --load 1.  `

All results are saved in the results folder. We provide our results in the folder ***results_CCST*** for taking further analysis. 

(1) The cell clustering label are saved in ***types.txt***, where three columns refer to cell index, batch information and cell cluster label, respectively. 

(2) The barplot of the neighborhood ratio is shown in fig ***barplot_subx.png***. 

(3) The spatial distribution of cells within each batch are illustrated in ***cluster_Batchx.png***. 

(4) The top-200 highly expressed genes of each cluster are listed in ***clusterx_gene_cur.txt***. They are sorted in the decent of significance.




# 3. Download all datasets used in CCST:

3.1 MERFISH

Data is available at https://www.pnas.org/content/116/39/19490/tab-figures-data 

3.2 DLPFC

Data is available at https://research.libd.org/spatialLIBD/

3.3 SeqFISH+

Data is available at https://github.com/CaiGroup/seqFISH-PLUS. 

3.4 10x Visium spatial transcriptomics data of human breast cancer

Data: https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1 

Annotation:  https://github.com/JinmiaoChenLab/SEDR_analyses/tree/master/data/BRCA1/metadata.tsv
