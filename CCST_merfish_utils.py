##exocrine GCNG with normalized graph matrix 
import os
import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import pylab as pl
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn import metrics
from scipy import sparse
#from sklearn.metrics import roc_curve, auc, roc_auc_score

import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader
from datetime import datetime 

from CCST import get_graph, train_DGI, train_DGI, PCA_process, Kmeans_cluster


rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath+'/CCST')

def get_data(args):
    data_folder = args.data_path + args.data_name +'/'
    with open(data_folder + 'Adjacent', 'rb') as fp:
        adj_0 = pickle.load(fp)
    X_data = np.load(data_folder + 'features.npy') 

    num_points = X_data.shape[0]
    adj_I = np.eye(num_points)
    adj_I = sparse.csr_matrix(adj_I)
    adj = (1-args.lambda_I)*adj_0 + args.lambda_I*adj_I

    cell_batch_info = np.load(data_folder + 'cell_batch_info.npy')

    return adj_0, adj, X_data, cell_batch_info


def draw_map(args, lambda_I, cell_batch_info, adj_0, draw_nerighbor=1):
    f = open(args.result_path+'/types.txt')            
    line = f.readline() # drop the first line  
    cell_cluster_type_list = []

    while line: 
        tmp = line.split('\t')
        cell_id = int(tmp[0]) # index start is start from 0 here
        cell_type_index = int(tmp[1])
        cell_cluster_type = int(tmp[2].replace('\n', ''))
        cell_cluster_type_list.append(cell_cluster_type)
        line = f.readline() 
    f.close() 
    n_clusters = max(cell_cluster_type_list) + 1 # start from 0
    num_cell_batch = max(cell_batch_info) # start from 1

    import openpyxl
    workbook = openpyxl.load_workbook('dataset/MERFISH/pnas.1912459116.sd15.xlsx')
    start_cell_index = 0
    if n_clusters <=5:
        from matplotlib.colors import ListedColormap
        all_colors_map = {0:'DarkBlue', 1:'orange', 2:'Crimson', 3:'green', 4:'yellow'}
        #sc_all = plt.scatter(x=np.arange(n_clusters), y=np.arange(n_clusters), c=np.arange(n_clusters), cmap=ListedColormap(colors_list))    
    else:
        import matplotlib.cm as cm
        colors = cm.viridis(np.linspace(0, 1, n_clusters))
        all_cluster = plt.scatter(x=np.arange(n_clusters), y=np.arange(n_clusters), s=100, c=np.arange(n_clusters))  
    plt.clf()

    for sheet_name in ['Batch 1', 'Batch 2', 'Batch 3']:
        worksheet = workbook[sheet_name]
        sheet_X = [item.value for item in list(worksheet.columns)[1]]
        sheet_X = sheet_X[1:]
        sheet_X = np.array(sheet_X)
        sheet_Y = [item.value for item in list(worksheet.columns)[2]]
        sheet_Y = sheet_Y[1:]
        sheet_Y = np.array(sheet_Y)
        end_cell_index = start_cell_index + len(sheet_X)
        cell_cluster_batch_list = cell_cluster_type_list[start_cell_index: end_cell_index]

        plt.title(sheet_name)
        if n_clusters <=5:
            colors_list = []
            clusters_name = []
            for tmp in range(n_clusters):
                if tmp in cell_cluster_batch_list:
                    colors_list.append(all_colors_map[tmp])
                    clusters_name.append('C'+str(tmp))
            sc_cluster = plt.scatter(x=sheet_X, y=sheet_Y, s=30, c=cell_cluster_batch_list, cmap=ListedColormap(colors_list))  
            plt.legend(handles = sc_cluster.legend_elements()[0],labels=clusters_name, bbox_to_anchor=(1,0.8), loc='center left')  
        else:
            sc_cluster = plt.scatter(x=sheet_X, y=sheet_Y, s=10, c=colors[cell_cluster_batch_list])  
            cb_cluster = plt.colorbar(all_cluster, boundaries=np.arange(n_clusters+1)-0.5).set_ticks(np.arange(n_clusters))    
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(args.result_path+'/cluster_'+ sheet_name.replace(' ','')+'.png', dpi=400, bbox_inches='tight') 
        plt.clf()

        start_cell_index = end_cell_index

     
    # draw barplot
    if draw_nerighbor:
        total_cell_num = end_cell_index
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

        np.savetxt(args.result_path + '/cluster_' + str(n_clusters) + '_barplot.txt', barplot, fmt='%3d', delimiter='\t')
        norm_barplot = barplot/(source_cluster_type_count.reshape(-1, 1))
        np.savetxt(args.result_path+ '/cluster_' + str(n_clusters) + '_barplot_normalize.txt', norm_barplot, fmt='%3f', delimiter='\t')

        for clusters_i in range(n_clusters):
            plt.bar(range(n_clusters), norm_barplot[clusters_i], label='graph '+str(clusters_i))
            plt.xlabel('cell type index')
            plt.ylabel('value')
            plt.title('barplot_'+str(clusters_i))
            plt.savefig(args.result_path+ '/barplot_sub' + str(clusters_i)+ '.jpg')
            plt.clf()

    return 


def get_gene(args):
    genes_file = args.data_path+args.data_name+'/gene_names.txt'  
    genes = []
    f = open(genes_file)               
    line = f.readline()              
    while line: 
        genes.append(line.replace('\n', '')) 
        line = f.readline() 
    f.close()
    return np.array(genes)

def test(data1, data2):
    from scipy.stats import ttest_ind
    from scipy import stats
    # mannwhitneyu
    stat, p = stats.mannwhitneyu(data1, data2)
    # t test
    # stat, p = ttest_ind(data1, data2, equal_var=False)
    return p



def CCST_on_MERFISH(args):
    lambda_I = args.lambda_I
    # Parameters
    batch_size = 1  # Batch size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = GCNNet(num_features=2, num_classes=1).to(device)
    #criterion = nn.BCELoss().to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    adj_0, adj, X_data, cell_batch_info = get_data(args)

    num_cell = X_data.shape[0]
    num_feature = X_data.shape[1]
    print('Adj:', adj.shape, 'Edges:', len(adj.data))
    print('X:', X_data.shape)

    n_clusters = args.n_clusters #num_cell_types

    if args.DGI:
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
        print("-----------Clustering-------------")
        X_embedding_filename =  args.embedding_data_path+'lambdaI' + str(lambda_I) + '_epoch' + str(args.num_epoch) + '_Embed_X.npy'
        X_embedding = np.load(X_embedding_filename)

        if args.PCA:
            X_embedding = PCA_process(X_embedding, nps=30)

        print('Shape of data to cluster:', X_embedding.shape)
        cluster_labels, score = Kmeans_cluster(X_embedding, n_clusters) 
        n_clusters = cluster_labels.max()+1

        #Umap(args, X_embedding, cluster_labels, n_clusters, score)
        all_data = [] # txt: cell_id, cell batch, cluster type 
        for index in range(num_cell):
            all_data.append([index, cell_batch_info[index], cluster_labels[index]])

        np.savetxt(args.result_path+'/types.txt', np.array(all_data), fmt='%3d', delimiter='\t')


    if args.draw_map:
        print("-----------Drawing map-------------")
        draw_map(args, lambda_I, cell_batch_info, adj_0)

    if args.diff_gene:
        print("-----------Differential gene expression analysis-------------")
        sys.setrecursionlimit(1000000)
        gene = get_gene(args)
        all_cate_info = np.loadtxt(args.result_path+'/types.txt', dtype=int)
        cell_index_all_cluster=[[] for i in range(n_clusters)]
        for cate_info in all_cate_info:
            cell_index_all_cluster[cate_info[2]].append(cate_info[0])

        for i in range(n_clusters): #num_cell_types
            cell_index_cur_cluster = cell_index_all_cluster[i]
            cur_cate_num = len(cell_index_cur_cluster)
            cells_cur_type = []
            cells_features_cur_type = []
            cells_other_type = []
            cells_features_other_type = []
            for cell_index in range(num_cell):
                if cell_index in cell_index_cur_cluster:
                    cells_cur_type.append(cell_index)
                    cells_features_cur_type.append(X_data[cell_index].tolist())
                else:
                    cells_other_type.append(cell_index)
                    cells_features_other_type.append(X_data[cell_index].tolist())

            cells_features_cur_type = np.array(cells_features_cur_type)
            cells_features_other_type = np.array(cells_features_other_type)
            pvalues_features = np.zeros(num_feature)
            for k in range(num_feature):
                cells_curfeature_cur_type = cells_features_cur_type[:,k]
                cells_curfeature_other_type = cells_features_other_type[:,k]
                if np.all(cells_curfeature_cur_type == 0) and np.all(cells_curfeature_other_type == 0):
                    pvalues_features[k] = 1
                else:
                    pvalues_features[k] = test(cells_curfeature_cur_type, cells_curfeature_other_type)

            num_diff_gene = 200
            num_gene = int(len(gene))
            print('Cluster', i, '. Cell num:', cur_cate_num)
            min_p_index = pvalues_features.argsort()
            min_p = pvalues_features[min_p_index]
            min_p_gene = gene[min_p_index] # all candidate gene
            gene_cur_type = []
            gene_other_type = []
            diff_gene_sub = []
            for ind, diff_gene_index in enumerate(min_p_index):
                flag_0 = (len(gene_cur_type) >= num_diff_gene)
                flag_1 = (len(gene_other_type) >= num_diff_gene)
                if (flag_0 and flag_1) or (min_p[ind] > 0.05):
                    break
                if cells_features_cur_type[:,diff_gene_index].mean() > cells_features_other_type[:, diff_gene_index].mean():
                    if not flag_0:
                        gene_cur_type.append(gene[diff_gene_index])  # diff gene that higher in subpopulation 0
                        diff_gene_sub.append('cur '+str(len(gene_cur_type)-1))
                    else:
                        diff_gene_sub.append('None')
                if cells_features_cur_type[:,diff_gene_index].mean() < cells_features_other_type[:, diff_gene_index].mean():
                    if not flag_1:
                        gene_other_type.append(gene[diff_gene_index])  # diff gene that higher in subpopulation 1
                        diff_gene_sub.append('other '+str(len(gene_other_type)-1))
                    else:
                        diff_gene_sub.append('None')
            print('diff gene for cur type:', len(gene_cur_type))
            print('diff gene for other type:', len(gene_other_type))

            file_handle=open(args.result_path+'/cluster' + str(i) + '_diff_gene.txt', mode='w')
            for m in range(len(diff_gene_sub)):
                file_handle.write(min_p_gene[m] + '\t' + str(min_p[m]) + '\t' + diff_gene_sub[m] + '\n')
            file_handle.close()
            file_handle_0 = open(args.result_path+'/cluster' + str(i) + '_gene_cur.txt', mode='w')
            for m in range(len(gene_cur_type)):
                file_handle_0.write(gene_cur_type[m]+'\n')
            file_handle_0.close()
            file_handle_1 = open(args.result_path+'/cluster' + str(i) + '_gene_other.txt', mode='w')
            for m in range(len(gene_other_type)):
                file_handle_1.write(gene_other_type[m]+'\n')
            file_handle_1.close()

    return 

    
