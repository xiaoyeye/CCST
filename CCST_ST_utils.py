##exocrine GCNG with normalized graph matrix 
import os
import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn import metrics
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score

import numpy as np
from scipy import sparse
import pickle
import pandas as pd
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader

from CCST import get_graph, train_DGI, train_DGI, PCA_process, Kmeans_cluster

rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath+'/CCST')

def get_data(args):
    data_file = args.data_path + args.data_name +'/'
    with open(data_file + 'Adjacent', 'rb') as fp:
        adj_0 = pickle.load(fp)
    X_data = np.load(data_file + 'features.npy')

    num_points = X_data.shape[0]
    adj_I = np.eye(num_points)
    adj_I = sparse.csr_matrix(adj_I)
    adj = (1-args.lambda_I)*adj_0 + args.lambda_I*adj_I

    cell_type_indeces = np.load(data_file + 'cell_types.npy')
    
    return adj_0, adj, X_data, cell_type_indeces



def clean_labels(gt_labels, cluster_labels, NAN_idx):
    cleaned_gt_labels, cleaned_cluster_labels = [], []
    for i,tmp in enumerate(gt_labels):
        if tmp != NAN_idx:
            cleaned_gt_labels.append(tmp)
            cleaned_cluster_labels.append(cluster_labels[i])
    print('cleaned length', len(cleaned_gt_labels), len(cleaned_cluster_labels))
    return np.array(cleaned_gt_labels), np.array(cleaned_cluster_labels)



def compare_labels(save_path, gt_labels, cluster_labels): 
    # re-order cluster labels for constructing diagonal-like matrix
    if max(gt_labels)==max(cluster_labels):
        matrix = np.zeros([max(gt_labels)+1, max(cluster_labels)+1], dtype=int)
        n_samples = len(cluster_labels)
        for i in range(n_samples):
            matrix[gt_labels[i], cluster_labels[i]] += 1
        matrix_size = max(gt_labels)+1
        order_seq = np.arange(matrix_size)
        matrix = np.array(matrix)
        #print(matrix)
        norm_matrix = matrix/matrix.sum(1).reshape(-1,1)
        #print(norm_matrix)
        norm_matrix_2_arr = norm_matrix.flatten()
        sort_index = np.argsort(-norm_matrix_2_arr)
        #print(sort_index)
        sort_row, sort_col = [], []
        for tmp in sort_index:
            sort_row.append(int(tmp/matrix_size))
            sort_col.append(int(tmp%matrix_size))
        sort_row = np.array(sort_row)
        sort_col = np.array(sort_col)
        #print(sort_row)
        #print(sort_col)
        done_list = []
        for j in range(len(sort_index)):
            if len(done_list) == matrix_size:
                break
            if (sort_row[j] in done_list) or (sort_col[j] in done_list):
                continue
            done_list.append(sort_row[j])
            tmp = sort_col[j]
            sort_col[sort_col == tmp] = -1
            sort_col[sort_col == sort_row[j]] = tmp
            sort_col[sort_col == -1] = sort_row[j]
            order_seq[sort_row[j]], order_seq[tmp] = order_seq[tmp], order_seq[sort_row[j]]

        reorder_cluster_labels = []
        for k in cluster_labels:
            reorder_cluster_labels.append(order_seq.tolist().index(k))
        matrix = matrix[:, order_seq]
        norm_matrix = norm_matrix[:, order_seq]
        plt.imshow(norm_matrix)
        plt.savefig(save_path + '/compare_labels_Matrix.png')
        plt.close()
        np.savetxt(save_path+ '/compare_labels_Matrix.txt', matrix, fmt='%3d', delimiter='\t')
        reorder_cluster_labels = np.array(reorder_cluster_labels, dtype=int)

    else:
        print('not square matrix!!')
        reorder_cluster_labels = cluster_labels
    return reorder_cluster_labels



def draw_map(args, adj_0, barplot=False):
    data_folder = args.data_path + args.data_name+'/'
    save_path = args.result_path
    f = open(save_path+'/types.txt')            
    line = f.readline() # drop the first line  
    cell_cluster_type_list = []

    while line: 
        tmp = line.split('\t')
        cell_id = int(tmp[0]) # index start is start from 0 here
        #cell_type_index = int(tmp[1])
        cell_cluster_type = int(tmp[1].replace('\n', ''))
        cell_cluster_type_list.append(cell_cluster_type)
        line = f.readline() 
    f.close() 
    n_clusters = max(cell_cluster_type_list) + 1 # start from 0
    print('n clusters in drwaing:', n_clusters)
    coordinates = np.load(data_folder+'coordinates.npy')

    sc_cluster = plt.scatter(x=coordinates[:,0], y=-coordinates[:,1], s=5, c=cell_cluster_type_list, cmap='rainbow')  
    plt.legend(handles = sc_cluster.legend_elements(num=n_clusters)[0],labels=np.arange(n_clusters).tolist(), bbox_to_anchor=(1,0.5), loc='center left', prop={'size': 9}) 
    #cb_cluster = plt.colorbar(sc_cluster, boundaries=np.arange(n_types+1)-0.5).set_ticks(np.arange(n_types))    
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    plt.title('CCST')
    plt.savefig(save_path+'/spacial.png', dpi=400, bbox_inches='tight') 
    plt.clf()


    # draw barplot
    if barplot:
        total_cell_num = len(cell_cluster_type_list)
        barplot = np.zeros([n_clusters, n_clusters], dtype=int)
        source_cluster_type_count = np.zeros(n_clusters, dtype=int)
        p1, p2 = adj_0.nonzero()
        def get_all_index(lst=None, item=''):
            return [i for i in range(len(lst)) if lst[i] == item]

        for i in range(total_cell_num):
            source_cluster_type_index = cell_cluster_type_list[i]
            edge_indeces = get_all_index(p1, item=i)
            paired_vertices = p2[edge_indeces]
            for j in paired_vertices:
                neighbor_type_index = cell_cluster_type_list[j]
                barplot[source_cluster_type_index, neighbor_type_index] += 1
                source_cluster_type_count[source_cluster_type_index] += 1

        np.savetxt(save_path + '/cluster_' + str(n_clusters) + '_barplot.txt', barplot, fmt='%3d', delimiter='\t')
        norm_barplot = barplot/(source_cluster_type_count.reshape(-1, 1))
        np.savetxt(save_path + '/cluster_' + str(n_clusters) + '_barplot_normalize.txt', norm_barplot, fmt='%3f', delimiter='\t')

        for clusters_i in range(n_clusters):
            plt.bar(range(n_clusters), norm_barplot[clusters_i], label='graph '+str(clusters_i))
            plt.xlabel('cell type index')
            plt.ylabel('value')
            plt.title('barplot_'+str(clusters_i))
            plt.savefig(save_path + '/barplot_sub' + str(clusters_i)+ '.jpg')
            plt.clf()

    return 



def res_search_fixed_clus(cluster_type, adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[int]
    '''
    if cluster_type == 'leiden':
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            if count_unique_leiden == fixed_clus_count:
                break
    elif cluster_type == 'louvain':
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            if count_unique_louvain == fixed_clus_count:
                break
    return res



def CCST_on_ST(args):
    lambda_I = args.lambda_I
    # Parameters
    batch_size = 1  # Batch size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adj_0, adj, X_data, cell_type_indeces = get_data(args)


    num_cell = X_data.shape[0]
    num_feature = X_data.shape[1]
    print('Adj:', adj.shape, 'Edges:', len(adj.data))
    print('X:', X_data.shape)

    n_clusters = max(cell_type_indeces)+1 #num_cell_types, start from 0

    print('n clusters:', n_clusters)

    if args.DGI and (lambda_I>=0):
        print("-----------Deep Graph Infomax-------------")
        data_list = get_graph(adj, X_data)
        data_loader = DataLoader(data_list, batch_size=batch_size)
        DGI_model = train_DGI(args, data_loader=data_loader, in_channels=num_feature)

        for data in data_loader:
            data.to(device)
            X_embedding, _, _ = DGI_model(data)
            X_embedding = X_embedding.cpu().detach().numpy()
            X_embedding_filename =  args.embedding_data_path+'lambdaI' + str(lambda_I) + '_epoch' + str(args.num_epoch) + '_Embed_X.npy'
            np.save(X_embedding_filename, X_embedding)

    if args.cluster:
        cluster_type = 'kmeans' # 'louvain' leiden kmeans

        print("-----------Clustering-------------")
    
        X_embedding_filename =  args.embedding_data_path+'lambdaI' + str(lambda_I) + '_epoch' + str(args.num_epoch) + '_Embed_X.npy'
        X_embedding = np.load(X_embedding_filename)

        X_embedding = PCA_process(X_embedding, nps=30)

        #X_data_PCA = PCA_process(X_data, nps=X_embedding.shape[1])

        # concate
        #X_embedding = np.concatenate((X_embedding, X_data), axis=1)
        
        print('Shape of data to cluster:', X_embedding.shape)

        if cluster_type == 'kmeans':
            cluster_labels, score = Kmeans_cluster(X_embedding, n_clusters) 
        else:
            results_file = args.result_path + '/adata.h5ad'
            adata = ad.AnnData(X_embedding)
            sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
            sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50) # 20
            eval_resolution = res_search_fixed_clus(cluster_type, adata, n_clusters)
            if cluster_type == 'leiden':
                sc.tl.leiden(adata, key_added="CCST_leiden", resolution=eval_resolution)
                cluster_labels = np.array(adata.obs['leiden'])
            if cluster_type == 'louvain':
                sc.tl.louvain(adata, key_added="CCST_louvain", resolution=eval_resolution)
                cluster_labels = np.array(adata.obs['louvain'])
            #sc.tl.umap(adata)
            #sc.pl.umap(adata, color=['leiden'], save='_lambdaI_' + str(lambda_I) + '.png')
            adata.write(results_file)
            cluster_labels = [ int(x) for x in cluster_labels ]
            score = False

        all_data = [] 
        for index in range(num_cell):
            #all_data.append([index, cell_type_indeces[index], cluster_labels[index]])  # txt: cell_id, gt_labels, cluster type 
            all_data.append([index,  cluster_labels[index]])   #txt: cell_id, cluster type 
        np.savetxt(args.result_path+'/types.txt', np.array(all_data), fmt='%3d', delimiter='\t')
        

    if args.draw_map:
        print("-----------Drawing map-------------")
        draw_map(args, adj_0)

