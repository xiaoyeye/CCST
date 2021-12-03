# CCST
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

seaborn==0.11.1

pickle==4.0

scanorama==1.7.1

1.3 GPU is required for training.



# 2. Intructions of usage: demo code is runing on MERFISH.

2.1 Raw data is put in the folder "merfish". Need to be extracted firstly.

2.2 Run 'data_generation_merfish.py' to preprocess the raw data. The pocessed data will be save in folder "generated_data".

2.3 Run 'CCST_merfish.py' for node clustering and differential expressed gene extracting. 

2.4 The trained model, embedding data and analysis results will be saved in folder "model", "embedding_data" and "results_CCST" by defult.

2.5 Run Time to use for training and clustering: 284 seconds on GPU RTX 3090. GPU Memory usage: 1635MiB 


# 3. Download all datasets used in CCST:

3.1 MERFISH

Data are avalieble at https://www.pnas.org/content/116/39/19490/tab-figures-data 

3.2 DLPFC

Data are avalieble at https://research.libd.org/spatialLIBD/

3.3 SeqFISH+

Data are avalieble at https://github.com/CaiGroup/seqFISH-PLUS. 

3.4 10x Visium spatial transcriptomics data of human breast cancer

Data: https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1 

Annotation:  https://github.com/JinmiaoChenLab/SEDR_analyses/tree/master/data/BRCA1/metadata.tsv
