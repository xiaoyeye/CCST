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

For 10x Spatial Transcripts (ST) datasets, files should be put in the same structure with that provided by 10x website. Taking V1_Breast_Cancer_Block_A_Section_1 for instance:

> dataset/V1_Breast_Cancer_Block_A_Section_1/ 
  >> spatial/  # The folder where files for spatial information can be found 
  
  >> metadata.tsv # mainly for annotation
  
  >> V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5 # gene expression data


## 2.2 Data Preprocessing and Graph Construction

Run ***data_generation_merfish.py*** to preprocess the raw MERFISH data:

`python data_generation_merfish.py`

Augments:

**--gene_expression_path**: the path to gene expression file.

**--spatial_location_path**: the path to cell location file.

For dealing other single cell datasets (e.g. seqFISH+), please modify the Arguments to change the path of input files, and the codes for dataset loading if required. 


---------------------------------------------------------------------------

Run ***data_generation_ST.py*** to preprocess the raw 10x ST data:

`python data_generation_ST.py`

Augments:

**--data_name**: the name of 10x ST dataset. Files need to be put in the required structure.


For dealing other 10x ST datasets, please modify the Arguments to change the data name. 

---------------------------------------------------------------------------

We provide the pocessed data in folder ***generated_data*** for both MERFISH and V1_Breast_Cancer_Block_A_Section_1 dataset. You can also directly use it without runing data preprocessing. Specifically, the following five files will be used in CCST.

(1) ***features.npy*** saves the preprocessed gene expression. 

(2) ***Adjacent*** saves the constructed adjacency matrix.



## 2.3 Run CCST 

The CCST model is implemented in ***CCST.py***. We give examples on both MERFISH and 10x ST datasets. When running CCST, the data type need to be firstly specified, so that the dataloader and image plotting functions can be load accordingly. We show demos on MERFISH and V1_Breast_Cancer_Block_A_Section_1. For apllying CCST on other datasets, if the corresponding data preprocessing has been done, please modify the  **--data_type** and **--data_name** here.

The meaning of each argument in ***run_CCST.py*** is listed below.

**--data_type**: 'sc' or 'nsc'. To specify whether the dataset is in single cell resolution (e.g. MERFISH) or non single cell resolution(e.g. ST)

**--data_name**: the name of dataset, which is utilized for locating the preprocessed data.

**--lambda_I**: the value of hyperparameter lambda for intracellular (gene) and extracellular (spatial) information balance, which should be within [0,1]. 

**--DGI**: whether to run the Deep Graph Infomax (DGI) model (set to 1) or directly load node embedding without utilizing DGI (set to 0).

**--load**：whether to load a pretrained DGI model (set to 1) or train a new model from the begining (set to 0). 

**--num_epoch**: the number of epochs in training DGI. 

**--hidden**: the dimension of each hidden layer. 

**--cluster**: whether to perform cluster (set to 1) or not (set to 0). If set to 0, the model will only conduct node embedding without further clustering.

**--PCA**: whether to perform PCA on the embedding (set to 1) or not (set to 0).

**--n_clusters**: the number of desired clusters.

**--draw_map**: whether to draw the spatial distribution of cells (set to 1) or not (set to 0).

**--diff_gene**: whether to take differential expressed gene analysis (set to 1) or not (set to 0).

**--model_path**: the path for saving model.

**--embedding_data_path**: the path for saving embedding data.

**--result_path**: the path for saving results.


## 2.4 Usage

The trained model, embedding data and analysis results will be saved in folder ***model***, ***embedding_data*** and ***results*** by default.

We provide the learned cell embedding of the model in the folder ***embedding_data***. If you want to directly use it, run 

 `python run_CCST --data_type sc --data_name MERFISH --lambda_I 0.8 --DGI 0.  ` on MERFISH and
 
 `python run_CCST --data_type nsc --data_name V1_Breast_Cancer_Block_A_Section_1 --lambda_I 0.3 --DGI 0.  ` on V1_Breast_Cancer_Block_A_Section_1.

We provide the trained model in the folder ***model***. If you want to directly use it, run

 `python run_CCST --data_type sc --data_name MERFISH --lambda_I 0.8 --DGI 1 --load 1.  ` on MERFISH and
 
 `python run_CCST --data_type nsc --data_name V1_Breast_Cancer_Block_A_Section_1 --lambda_I 0.3 --DGI 1 --load 1.  ` on V1_Breast_Cancer_Block_A_Section_1.
 
For training your own model, run

 `python run_CCST --data_type sc --data_name MERFISH --lambda_I 0.8 --DGI 1 --load 0.  ` on MERFISH and
 
 `python run_CCST --data_type nsc --data_name V1_Breast_Cancer_Block_A_Section_1 --lambda_I 0.3 --DGI 1 --load 0.  ` on V1_Breast_Cancer_Block_A_Section_1.
 

All results are saved in the results folder. We provide our results in the folder ***results*** for taking further analysis. 

(1) The cell clustering labels are saved in ***types.txt***, where the first column refers to cell index, and the last column refers to cell cluster label. 

(3) The spatial distribution of cells within each batch are illustrated in ***.png*** files. 

(4) On MERFISH dataset, the top-200 highly expressed genes of each cluster are listed in ***clusterx_gene_cur.txt***. They are sorted in the decent of statistical significance.





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
