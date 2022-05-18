import pandas as pd
import scanpy as sc
import numpy as np
import stlearn as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from h5py import Dataset, Group
####################  get the whole training dataset


rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath+'/CCST')


def read_h5(f, i=0):
    for k in f.keys():
        if isinstance(f[k], Group):
            print('Group', f[k])
            print('-'*(10-5*i))
            read_h5(f[k], i=i+1)
            print('-'*(10-5*i))
        elif isinstance(f[k], Dataset):
            print('Dataset', f[k])
            print(f[k][()])
        else:
            print('Name', f[k].name)

def adata_preprocess(i_adata, min_cells=3, pca_n_comps=300):
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X



def get_adj(generated_data_fold):
    coordinates = np.load(generated_data_fold + 'coordinates.npy')
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold) 
    ############# get batch adjacent matrix
    cell_num = len(coordinates)

    ############ the distribution of distance 
    if 1:#not os.path.exists(generated_data_fold + 'distance_array.npy'):
        distance_list = []
        print ('calculating distance matrix, it takes a while')
        
        distance_list = []
        for j in range(cell_num):
            for i in range (cell_num):
                if i!=j:
                    distance_list.append(np.linalg.norm(coordinates[j]-coordinates[i]))

        distance_array = np.array(distance_list)
        #np.save(generated_data_fold + 'distance_array.npy', distance_array)
    else:
        distance_array = np.load(generated_data_fold + 'distance_array.npy')

    ###try different distance threshold, so that on average, each cell has x neighbor cells, see Tab. S1 for results
    from scipy import sparse
    import pickle
    import scipy.linalg

    for threshold in [300]:#range (210,211):#(100,400,40):
        num_big = np.where(distance_array<threshold)[0].shape[0]
        print (threshold,num_big,str(num_big/(cell_num*2))) #300 22064 2.9046866771985256
        from sklearn.metrics.pairwise import euclidean_distances

        distance_matrix = euclidean_distances(coordinates, coordinates)
        distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
        distance_matrix_threshold_W = np.zeros(distance_matrix.shape)
        for i in range(distance_matrix_threshold_I.shape[0]):
            for j in range(distance_matrix_threshold_I.shape[1]):
                if distance_matrix[i,j] <= threshold and distance_matrix[i,j] > 0:
                    distance_matrix_threshold_I[i,j] = 1
                    distance_matrix_threshold_W[i,j] = distance_matrix[i,j]
            
        
        ############### get normalized sparse adjacent matrix
        distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I) ## do not normalize adjcent matrix
        distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
        with open(generated_data_fold + 'Adjacent', 'wb') as fp:
            pickle.dump(distance_matrix_threshold_I_N_crs, fp)


def get_type(args, cell_types, generated_data_fold):
    types_dic = []
    types_idx = []
    for t in cell_types:
        if not t in types_dic:
            types_dic.append(t) 
        id = types_dic.index(t)
        types_idx.append(id)

    n_types = max(types_idx) + 1 # start from 0
    # For human breast cancer dataset, sort the cells for better visualization
    if args.data_name == 'V1_Breast_Cancer_Block_A_Section_1':
        types_dic_sorted = ['Healthy_1', 'Healthy_2', 'Tumor_edge_1', 'Tumor_edge_2', 'Tumor_edge_3', 'Tumor_edge_4', 'Tumor_edge_5', 'Tumor_edge_6',
            'DCIS/LCIS_1', 'DCIS/LCIS_2', 'DCIS/LCIS_3', 'DCIS/LCIS_4', 'DCIS/LCIS_5', 'IDC_1', 'IDC_2', 'IDC_3', 'IDC_4', 'IDC_5', 'IDC_6', 'IDC_7']
        relabel_map = {}
        cell_types_relabel=[]
        for i in range(n_types):
            relabel_map[i]= types_dic_sorted.index(types_dic[i])
        for old_index in types_idx:
            cell_types_relabel.append(relabel_map[old_index])
        
    np.save(generated_data_fold+'cell_types.npy', np.array(cell_types_relabel))
    np.savetxt(generated_data_fold+'types_dic.txt', np.array(types_dic_sorted), fmt='%s', delimiter='\t')


def draw_map(generated_data_fold):
    coordinates = np.load(generated_data_fold + 'coordinates.npy')
    cell_types = np.load(generated_data_fold+'cell_types.npy')
    n_cells = len(cell_types)
    n_types = max(cell_types) + 1 # start from 0

    types_dic = np.loadtxt(generated_data_fold+'types_dic.txt', dtype='|S15',   delimiter='\t').tolist()
    for i,tmp in enumerate(types_dic):
        types_dic[i] = tmp.decode()
    print(types_dic)

    sc_cluster = plt.scatter(x=coordinates[:,0], y=-coordinates[:,1], s=5, c=cell_types, cmap='rainbow')  
    plt.legend(handles = sc_cluster.legend_elements(num=n_types)[0],labels=types_dic, bbox_to_anchor=(1,0.5), loc='center left', prop={'size': 9}) 
    
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    plt.title('Annotation')
    plt.savefig(generated_data_fold+'/spacial.png', dpi=400, bbox_inches='tight') 
    plt.clf()




def main(args):
    data_fold = args.data_path+args.data_name+'/'
    generated_data_fold = args.generated_data_path + args.data_name+'/'
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold)
    adata_h5 = st.Read10X(path=data_fold, count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
    print(adata_h5)
    #count = adata_h5.X 
    features = adata_preprocess(adata_h5, min_cells=args.min_cells, pca_n_comps=args.Dim_PCA)
    gene_ids = adata_h5.var['gene_ids']
    coordinates = adata_h5.obsm['spatial']

    np.save(generated_data_fold + 'features.npy', features)
    np.save(generated_data_fold + 'coordinates.npy', np.array(coordinates))


    df_meta = pd.read_csv(data_fold +'metadata.tsv', sep='\t')
    ## The cell_type are put in df_meta['fine_annot_type'] in V1_Breast_Cancer_Block_A_Section_1 dataset. This is labeled by SEDR
    #df_meta = df_meta[~pd.isnull(df_meta['layer_guess'])] 
    cell_types = df_meta['fine_annot_type'] 

    get_adj(generated_data_fold)
    get_type(args, cell_types, generated_data_fold)
    draw_map(generated_data_fold)


    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--min_cells', type=float, default=1, help='Lowly expressed genes which appear in fewer than this number of cells will be filtered out')
    parser.add_argument( '--Dim_PCA', type=int, default=200, help='The output dimention of PCA')
    parser.add_argument( '--data_path', type=str, default='dataset/', help='The path to dataset')
    parser.add_argument( '--data_name', type=str, default='V1_Breast_Cancer_Block_A_Section_1', help='The name of dataset')
    parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
    args = parser.parse_args() 

    main(args)

