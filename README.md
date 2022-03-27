# CCST：Cell clustering for spatial transcriptomics data with graph neural network 

Taking advantages of two recent technical development, spatial transcriptomics and graph neural network, we  thus introduce CCST, Cell Clustering for Spatial Transcriptomics data with graph neural network, an unsupervised cell clustering method based on graph convolutional network to improve ab initio cell clustering and discovering of novel sub cell types based on curated cell category annotation. CCST is a general framework for dealing with various kinds of spatially resolved transcriptomics.

Framework

![image](https://github.com/xiaoyeye/CCST/blob/main/figure/figure1.png)


The code is licensed under the MIT license. 

# 1. Requirements 

1.1 Operating systems:

The code in python has been tested on both linux (Ubuntu 16.04.6 LTS) and windows 10 system.

1.2 Required packages in python: 

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

1.3 GPU is required for training.



# 2. Intructions of usage: demo code is tested on MERFISH.

2.1 Raw data is put in the folder "merfish". Need to be extracted firstly.

2.2 Run 'data_generation_merfish.py' to preprocess the raw data. The pocessed data will be save in folder "generated_data". After perpeocessing, there are 1368 cells with 1892 selected genes. specificlly, the following five files will be used in CCST.
(1) features_array_after_removal_low_var.npy saves the preprocessed gene expression. Shape=(1368,1892)
(2) Adjacent_200 saves the constructed adjacency matrix. Shape=(1368,1368)
(3) cell_batch_info.npy saves the cell batch information of each cell. Length=1368 
(4) gene_names_after_removal_low_var.txt saves names of selected genes.  Length=1892 
(5) all_genes.txt saves names of all genes. length=12903


2.3 Run 'CCST_merfish.py' for node clustering and differential expressed gene extracting. 

2.4 The trained model, embedding data and analysis results will be saved in folder "model", "embedding_data" and "results_CCST" by defult.

2.5 Run Time to use for training and clustering: 284 seconds on GPU RTX 3090. GPU Memory usage: 1635MiB 


# 3. Download all datasets used in CCST:

3.1 MERFISH

Data is avalieble at https://www.pnas.org/content/116/39/19490/tab-figures-data 

3.2 DLPFC

Data is avalieble at https://research.libd.org/spatialLIBD/

3.3 SeqFISH+

Data is avalieble at https://github.com/CaiGroup/seqFISH-PLUS. 

3.4 10x Visium spatial transcriptomics data of human breast cancer

Data: https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1 

Annotation:  https://github.com/JinmiaoChenLab/SEDR_analyses/tree/master/data/BRCA1/metadata.tsv
