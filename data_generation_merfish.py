#import pandas as pd
import numpy as np
#import seaborn as sns
#from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import os
import sys
import scipy
from scipy import sparse
import pickle
import scipy.linalg
####################  get the whole training dataset


rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath+'/CCST')

generated_data_path = 'generated_data/'
if not os.path.exists(generated_data_path):
    os.makedirs(generated_data_path) 
    
def get_adj(data_path):
    import openpyxl
    if 1:
        workbook = openpyxl.load_workbook(data_path)
        distance_matrix_list = [] 
        all_distance_list = []
        num_cells = 0

        for sheet_name in ['Batch 1', 'Batch 2', 'Batch 3']:
            worksheet = workbook[sheet_name]

            sheet_X = [item.value for item in list(worksheet.columns)[1]]
            sheet_X = sheet_X[1:]
            sheet_X[sheet_X == ''] = 0.0
            sheet_Y = [item.value for item in list(worksheet.columns)[2]]
            sheet_Y = sheet_Y[1:]
            sheet_Y[sheet_Y == ''] = 0.0
            sheet_X = np.array(sheet_X).astype(np.float)
            sheet_Y = np.array(sheet_Y).astype(np.float)
            #all_X = np.array([n for a in all_X for n in a]).astype(np.float)
            #all_Y = np.array([n for a in all_Y for n in a]).astype(np.float)

            sheet_num_cells = len(sheet_X)
            num_cells += sheet_num_cells

            #print('num_cells:' , num_cells)        
            distance_list = []
            sheet_distance_matrix = np.zeros((sheet_num_cells, sheet_num_cells))
            for i in range(sheet_num_cells):
                for j in range(sheet_num_cells):
                    if i!=j:
                        dist = np.linalg.norm(np.array([sheet_X[i], sheet_Y[i]])-np.array([sheet_X[j], sheet_Y[j]]))
                        distance_list.append(dist)
                        sheet_distance_matrix[i,j] = dist
            all_distance_list += distance_list
            distance_matrix_list.append(sheet_distance_matrix)
        
        all_distance_array = np.array(all_distance_list) # shape: num_cell * (num_cell-1)

        all_distance_matrix = scipy.linalg.block_diag(distance_matrix_list[0],
                                                        distance_matrix_list[1],
                                                        distance_matrix_list[2])

        np.save( generated_data_path + 'all_distance_array.npy', all_distance_array)
        np.save( generated_data_path + 'all_distance_matrix.npy', all_distance_matrix)

    else:
        all_distance_array = np.load( generated_data_path + 'all_distance_array.npy')
        all_distance_matrix = np.load( generated_data_path + 'all_distance_matrix.npy')
        num_cells = len(all_distance_matrix)

    ###try different distance threshold, so that on average, each cell has x neighbor cells, see Tab. S1 for results
    print('num_cells:', num_cells)
    for threshold in [200]:
        num_big = np.where(all_distance_array<threshold)[0].shape[0]
        print (threshold,num_big,str(num_big/(num_cells)))
        distance_matrix_threshold_I_list = []
        distance_matrix_threshold_W_list = []
        from sklearn.metrics.pairwise import euclidean_distances
        
        distance_matrix_threshold_I = np.zeros(all_distance_matrix.shape)
        distance_matrix_threshold_W = np.zeros(all_distance_matrix.shape)
        for i in range(distance_matrix_threshold_I.shape[0]):
            for j in range(distance_matrix_threshold_I.shape[1]):
                if all_distance_matrix[i,j] <= threshold and all_distance_matrix[i,j] > 0:
                    distance_matrix_threshold_I[i,j] = 1
                    distance_matrix_threshold_W[i,j] = all_distance_matrix[i,j]
        distance_matrix_threshold_I_list.append(distance_matrix_threshold_I)
        distance_matrix_threshold_W_list.append(distance_matrix_threshold_W)

        whole_distance_matrix_threshold_I = scipy.linalg.block_diag(distance_matrix_threshold_I_list[0])
                                                            
        
        ############### get normalized sparse adjacent matrix
        distance_matrix_threshold_I_N = np.float32(whole_distance_matrix_threshold_I) ## do not normalize adjcent matrix
        distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
        with open( generated_data_path + 'Adjacent_'+str(threshold), 'wb') as fp:
            pickle.dump(distance_matrix_threshold_I_N_crs, fp)
        


def get_attribute(data_path):
    # attribute 
    all_data = np.loadtxt(data_path, str, delimiter = ",")
    all_features = all_data[1:, 1: ]
    all_features[all_features == ''] = 0.0
    all_features = all_features.astype(np.float)
    all_features = np.swapaxes(all_features, 0, 1)
    print('feature shape: ', all_features.shape)
    np.save( generated_data_path + 'features_array.npy', all_features)

    all_features_batch_1, all_features_batch_2, all_features_batch_3 = [], [], []

    all_cell_names = all_data[0, 1:]
    all_batch_info = []
    for i,cell_name in enumerate(all_cell_names):
        if 'B1' in cell_name:
            all_batch_info.append(1)
            all_features_batch_1.append(all_features[i])
        elif 'B2' in cell_name:
            all_batch_info.append(2)
            all_features_batch_2.append(all_features[i])
        elif 'B3' in cell_name:
            all_batch_info.append(3)
            all_features_batch_3.append(all_features[i])
    all_features_all_batch = [np.array(all_features_batch_1), np.array(all_features_batch_2), np.array(all_features_batch_3)] 
    np.save( generated_data_path + 'cell_batch_info.npy', all_batch_info)
    return all_features, all_features_all_batch


def get_all_gene(p):
    import csv
    with open(p, encoding = 'utf-8') as f:
        all_gene_names = np.loadtxt(f, str, delimiter = ",")[1:, 0]
    all_genes_file = generated_data_path+'all_genes.txt'
    file_handle=open(all_genes_file, mode='w')
    for gene_name in all_gene_names:
        file_handle.write(gene_name+'\n')
    file_handle.close()
    return all_gene_names


def remove_low_gene(all_features_all_batch, all_gene_names):
    all_features = np.load( generated_data_path + 'features_array.npy')

    each_feature_mean = all_features.mean(0)
    tmp_mean = np.arange(len(each_feature_mean))
    indeces_after_removal = tmp_mean[each_feature_mean>=1]
    features_after_removal = all_features[:,indeces_after_removal]
    gene_names_after_removal = all_gene_names[indeces_after_removal]
    print(features_after_removal.shape)


    genes_file = generated_data_path+'gene_names_after_removal_mean.txt'
    file_handle=open(genes_file, mode='w')
    for gene_name in gene_names_after_removal:
        file_handle.write(gene_name+'\n')
    file_handle.close()

    np.save( generated_data_path + 'features_array_after_removal_low_mean.npy', features_after_removal)

    features_all_batch_after_removal = []
    for all_features_each_batch in all_features_all_batch:
        features_all_batch_after_removal.append(all_features_each_batch[:,indeces_after_removal])
    return features_all_batch_after_removal, gene_names_after_removal


def batch_correction(datasets, genes_list):
    # List of datasets (matrices of cells-by-genes):
    #datasets = [ list of scipy.sparse.csr_matrix or numpy.ndarray ]
    # List of gene lists:
    genes_lists = [ genes_list, genes_list, genes_list ]

    import scanorama

    # Batch correction.
    corrected, _ = scanorama.correct(datasets, genes_lists)
    np.save( generated_data_path + 'corrected_batches.npy', corrected)

    features_corrected = []
    for i, corrected_each_batch in enumerate(corrected):
        #features_corrected.append(np.array(corrected_each_batch.A))
        if i == 0:
            features_corrected = corrected_each_batch.A
        else:
            features_corrected = np.vstack((features_corrected, corrected_each_batch.A))
    features_corrected = np.array(features_corrected)
    np.save( generated_data_path + 'features_array_corrected.npy', features_corrected)
    print('corrected size: ', features_corrected.shape)
    return features_corrected

    
def normalization(features):
    node_num = features.shape[0]
    feature_num = features.shape[1]
    #max_in_each_node = features.max(1)
    sum_in_each_node = features.sum(1)
    sum_in_each_node = sum_in_each_node.reshape(-1, 1)
    sum_matrix = (np.repeat(sum_in_each_node, axis=1, repeats=feature_num))
    feature_normed = features/sum_matrix * 10000
    np.save( generated_data_path + 'features_array_normed.npy', feature_normed)
    return feature_normed



def remove_low_var_gene(features_input, gene_names, thres=0.4): # thres=1
    each_feature_var = features_input.var(0)
    tmp_var = np.arange(len(each_feature_var))
    indeces_after_removal = tmp_var[each_feature_var>=thres] 
    features_after_removal = features_input[:,indeces_after_removal]
    gene_names_after_removal = gene_names[indeces_after_removal]
    print(features_after_removal.shape)

    genes_file = generated_data_path+'gene_names_after_removal_low_var.txt'
    file_handle=open(genes_file, mode='w')
    for gene_name in gene_names_after_removal:
        file_handle.write(gene_name+'\n')
    file_handle.close()

    np.save( generated_data_path + 'features_array_after_removal_low_var.npy', features_after_removal)

    return gene_names_after_removal


def main():
    # B1：645, B2:400, B3:323
    
    all_features, all_features_all_batch = get_attribute('merfish/pnas.1912459116.sd12.csv')
    get_adj('merfish/pnas.1912459116.sd15.xlsx')
    all_gene_names = get_all_gene('merfish/pnas.1912459116.sd12.csv')

    features_all_batch_after_removal, gene_names_after_removal = remove_low_gene(all_features_all_batch, all_gene_names)
    features_corrected = batch_correction(features_all_batch_after_removal, gene_names_after_removal)
    normalized_features = normalization(features_corrected)
    remove_low_var_gene(normalized_features, gene_names_after_removal)

if __name__ == "__main__":
    main()
