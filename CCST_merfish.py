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

rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath+'/CCST')

def get_data(lambda_I=0, Adjacent_threshold=200):
    data_path = 'generated_data/'

    with open(data_path + 'Adjacent_' + str(Adjacent_threshold), 'rb') as fp:
        adj_0 = pickle.load(fp)
    X_data = np.load(data_path + 'features_array_after_removal_low_var.npy') 
    #features_array_normed features_array_after_removal_low_var

    num_points = X_data.shape[0]
    adj_I = np.eye(num_points)
    adj_I = sparse.csr_matrix(adj_I)
    adj = (1-lambda_I)*adj_0 + lambda_I*adj_I

    cell_batch_info = np.load(data_path + 'cell_batch_info.npy')

    return adj_0, adj, X_data, cell_batch_info



def get_graph(adj, X):
    # create sparse matrix
    row_col = []
    edge_weight = []
    rows, cols = adj.nonzero()
    edge_nums = adj.getnnz() 
    for i in range(edge_nums):
        row_col.append([rows[i], cols[i]])
        edge_weight.append(adj.data[i])
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    graph_bags = []
    graph = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)  
    graph_bags.append(graph)
    return graph_bags


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_4 = GCNConv(hidden_channels, hidden_channels)
        
        self.prelu = nn.PReLU(hidden_channels)
        
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.conv_2(x, edge_index, edge_weight=edge_weight)
        x = self.conv_3(x, edge_index, edge_weight=edge_weight)
        x = self.conv_4(x, edge_index, edge_weight=edge_weight)
        x = self.prelu(x)

        return x

class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

def corruption(data):
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)



def train_DGI(args, data_loader, in_channels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=Encoder(in_channels=in_channels, hidden_channels=args.hidden),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=1e-6)
    if args.load:
        DGI_filename = args.model_path+'DGI_lambdaI_' + str(args.lambda_I) + '_epoch' + str(args.num_epoch) + '.pth.tar'
        DGI_model.load_state_dict(torch.load(DGI_filename))
    else:
        import datetime
        start_time = datetime.datetime.now()
        for epoch in range(args.num_epoch):
            DGI_model.train()
            DGI_optimizer.zero_grad()
        
            DGI_all_loss = []
            
            for data in data_loader:
                data = data.to(device)
                pos_z, neg_z, summary = DGI_model(data=data)

                DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
                DGI_loss.backward()
                DGI_all_loss.append(DGI_loss.item())
                DGI_optimizer.step()

            if ((epoch+1)%100) == 0:
                print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch+1, np.mean(DGI_all_loss)))

        end_time = datetime.datetime.now()
        DGI_filename =  args.model_path+'DGI_lambdaI_' + str(args.lambda_I) + '_epoch' + str(args.num_epoch) + '.pth.tar'
        torch.save(DGI_model.state_dict(), DGI_filename)
        print('Training time in seconds: ', (end_time-start_time).seconds)
    return DGI_model



def Umap(args, X, label, n_clusters, score, lambda_I):
    import umap
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=20)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(n_clusters+1)-0.5).set_ticks(np.arange(n_clusters))
    plt.title('UMAP projection')
    plt.text(0.0, 0.0, score, fontdict={'size':'16','color':'black'},  transform = plt.gca().transAxes)
    plt.savefig(args.result_path+'lambdaI' + str(lambda_I) + '/Umap.jpg')
    #plt.show()
    plt.close()


def draw_map(args, lambda_I, cell_batch_info, adj_0):
    f = open(args.result_path+'lambdaI' + str(lambda_I)+'/types.txt')            
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
    workbook = openpyxl.load_workbook('merfish/pnas.1912459116.sd15.xlsx')
    start_cell_index = 0
    if n_clusters <=5:
        all_colors_map = {0:'DarkBlue', 1:'orange', 2:'Crimson', 3:'green', 4:'yellow'}
        colors_map={}
        for tmp in range(n_clusters):
            colors_map[tmp] = all_colors_map[tmp]
        colors_list = list(colors_map.values()) #['blue', 'orange', 'yellow', 'green', 'Crimson']
        clusters_name = []
        for tmp in list(colors_map.keys()):
            clusters_name.append('C'+str(tmp))
    else:
        import matplotlib.cm as cm
        colors = cm.viridis(np.linspace(0, 1, n_clusters))
        all_cluster = plt.scatter(x=np.arange(n_clusters), y=np.arange(n_clusters), s=100, c=np.arange(n_clusters))  
    plt.clf()
    for sheet_name in ['Batch 1', 'Batch 2', 'Batch 3']:
        worksheet = workbook[sheet_name]
        sheet_X = [item.value for item in list(worksheet.columns)[1]]
        sheet_X = sheet_X[1:]
        sheet_X[sheet_X == ''] = 0.0
        sheet_X = np.array(sheet_X)
        sheet_Y = [item.value for item in list(worksheet.columns)[2]]
        sheet_Y = sheet_Y[1:]
        sheet_Y[sheet_Y == ''] = 0.0
        sheet_Y = np.array(sheet_Y)
        end_cell_index = start_cell_index + len(sheet_X)
        cell_cluster_batch_list = cell_cluster_type_list[start_cell_index: end_cell_index]

        fig_name = 'lambdaI'+str(lambda_I) + '/cluster_'+ sheet_name.replace(' ','')
        plt.title(sheet_name)
        if n_clusters <=5:
            from matplotlib.colors import ListedColormap
            sc_all = plt.scatter(x=np.arange(n_clusters), y=np.arange(n_clusters), c=np.arange(n_clusters), cmap=ListedColormap(colors_list))    
            sc_cluster = plt.scatter(x=sheet_X, y=sheet_Y, s=30, c=cell_cluster_batch_list, cmap=ListedColormap(colors_list))  
            plt.legend(handles = sc_all.legend_elements()[0],labels=clusters_name, bbox_to_anchor=(1,0.8), loc='center left')  
        else:
            sc_cluster = plt.scatter(x=sheet_X, y=sheet_Y, s=10, c=colors[cell_cluster_batch_list])  
            cb_cluster = plt.colorbar(all_cluster, boundaries=np.arange(n_clusters+1)-0.5).set_ticks(np.arange(n_clusters))    
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(args.result_path+''+fig_name+'.png', dpi=400, bbox_inches='tight') 
        plt.clf()

        start_cell_index = end_cell_index

     
    # draw barplot
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

    np.savetxt(args.result_path+'lambdaI' + str(lambda_I) + '/cluster_' + str(n_clusters) + '_barplot.txt', barplot, fmt='%3d', delimiter='\t')
    norm_barplot = barplot/(source_cluster_type_count.reshape(-1, 1))
    np.savetxt(args.result_path+'lambdaI' + str(lambda_I) + '/cluster_' + str(n_clusters) + '_barplot_normalize.txt', norm_barplot, fmt='%3f', delimiter='\t')

    for clusters_i in range(n_clusters):
        plt.bar(range(n_clusters), norm_barplot[clusters_i], label='graph '+str(clusters_i))
        plt.xlabel('cell type index')
        plt.ylabel('value')
        plt.title('barplot_'+str(clusters_i))
        plt.savefig(args.result_path+'lambdaI' + str(lambda_I) + '/barplot_sub' + str(clusters_i)+ '.jpg')
        plt.clf()

    return 


def get_gene():
    genes_file = 'generated_data/gene_names_after_removal_low_var.txt'  
    # gene_names_after_removal_low_var.txt  gene_names_after_removal_mean.txt
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

def merge_cluser(X_embedding, cluster_labels):
    count_dict, out_count_dict = {}, {}
    for cluster in cluster_labels:
        count_dict[cluster] = count_dict.get(cluster, 0) + 1
    clusters = count_dict.keys()
    n_clusters = len(clusters)
    for cluster in clusters: 
        out_count_dict[cluster] = count_dict[cluster] 
    for cluster in clusters: 
        cur_n = count_dict[cluster]
        if cur_n <=3:
            min_dis = 1000
            merge_to = cluster
            center_cluster = X_embedding[cluster_labels==cluster].mean(0)
            for cluster_2 in clusters:
                if cluster_2 == cluster:
                    continue
                center_cluster_2 = X_embedding[cluster_labels==cluster_2].mean(0)
                dist = np.linalg.norm(center_cluster - center_cluster_2)
                if dist < min_dis:
                    min_dis = dist
                    merge_to = cluster_2

            cluster_labels[cluster_labels==cluster] = merge_to
            print('Merge group', cluster, 'to group', merge_to, 'with', cur_n, 'samples')
            out_count_dict[cluster] = 0
            out_count_dict[merge_to] += cur_n
            if cluster < n_clusters-1:
                cluster_labels[cluster_labels==n_clusters-1] = cluster
                print('Group', n_clusters-1, 'is renamed to group', cluster)
                out_count_dict[cluster] = out_count_dict[n_clusters-1]
                del out_count_dict[n_clusters-1]
            print(out_count_dict)

    return cluster_labels


from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
def Kmeans_cluster(X_embedding, n_clusters, merge):
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(X_embedding)

    # merge clusters with less than 3 cells
    if merge:
        cluster_labels = merge_cluser(X_embedding, cluster_labels)

    score = metrics.silhouette_score(X_embedding, cluster_labels, metric='euclidean')
    
    return cluster_labels, score



def main(args):
    lambda_I = args.lambda_I
    # Parameters
    batch_size = 1  # Batch size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = GCNNet(num_features=2, num_classes=1).to(device)
    #criterion = nn.BCELoss().to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    adj_0, adj, X_data, cell_batch_info = get_data(lambda_I)

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
            from sklearn.decomposition import PCA
            print('------PCA--------')
            print('Shape of data to PCA:', X_embedding.shape)

            pca = PCA(n_components=30)
            X_embedding = pca.fit_transform(X_embedding)     #等价于pca.fit(X) pca.transform(X)
            #inv_X = pca.inverse_transform(X_embedding) 
            print('PCA recover:', pca.explained_variance_ratio_.sum())

        print('Shape of data to cluster:', X_embedding.shape)

        cluster_labels, score = Kmeans_cluster(X_embedding, n_clusters, args.merge) 
        n_clusters = cluster_labels.max()+1

        #Umap(args, X_embedding, cluster_labels, n_clusters, score, lambda_I)
        all_data = [] # txt: cell_id, cell batch, cluster type 
        for index in range(num_cell):
            all_data.append([index, cell_batch_info[index], cluster_labels[index]])

        np.savetxt(args.result_path+'lambdaI' +str(lambda_I)+'/types.txt', np.array(all_data), fmt='%3d', delimiter='\t')


    if args.merge:
        X_embedding_filename =  args.embedding_data_path+'lambdaI' + str(lambda_I) + '_epoch' + str(args.num_epoch) + '_Embed_X.npy'
        X_embedding = np.load(X_embedding_filename)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=30)
        X_embedding = pca.fit_transform(X_embedding)     #等价于pca.fit(X) pca.transform(X)

        all_cate_info = np.loadtxt(args.result_path+'lambdaI'+str(lambda_I)+'/types.txt', dtype=int)
        cluster_labels = []
        for cate_info in all_cate_info:
            cluster_labels.append(cate_info[2])
        cluster_labels = np.array(cluster_labels)
        cluster_labels = merge_cluser(X_embedding, cluster_labels)
        n_clusters = max(cluster_labels)+1
        
        all_data = [] # txt: cell_id, cell batch, cluster type 
        for index in range(num_cell):
            all_data.append([index, cell_batch_info[index], cluster_labels[index]])

        np.savetxt(args.result_path+'lambdaI' +str(lambda_I)+'/types.txt', np.array(all_data), fmt='%3d', delimiter='\t')


    if args.draw_map:
        print("-----------Drawing map-------------")
        draw_map(args, lambda_I, cell_batch_info, adj_0)

    if args.diff_gene:
        print("-----------Differential gene expression analysis-------------")
        sys.setrecursionlimit(1000000)
        gene = get_gene()
        all_cate_info = np.loadtxt(args.result_path+'lambdaI'+str(lambda_I)+'/types.txt', dtype=int)
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

            file_handle=open(args.result_path+'lambdaI'+str(lambda_I)+'/cluster' + str(i) + '_diff_gene.txt', mode='w')
            for m in range(len(diff_gene_sub)):
                file_handle.write(min_p_gene[m] + '\t' + str(min_p[m]) + '\t' + diff_gene_sub[m] + '\n')
            file_handle.close()
            file_handle_0 = open(args.result_path+'lambdaI'+str(lambda_I)+'/cluster' + str(i) + '_gene_cur.txt', mode='w')
            for m in range(len(gene_cur_type)):
                file_handle_0.write(gene_cur_type[m]+'\n')
            file_handle_0.close()
            file_handle_1 = open(args.result_path+'lambdaI'+str(lambda_I)+'/cluster' + str(i) + '_gene_other.txt', mode='w')
            for m in range(len(gene_other_type)):
                file_handle_1.write(gene_other_type[m]+'\n')
            file_handle_1.close()
    

    if args.calculate_score:
        f = open(args.result_path+'lambdaI' + str(lambda_I)+'/types.txt')            
        line = f.readline() # drop the first line  
        cell_cluster_type_list = []

        while line: 
            tmp = line.split('\t')
            cell_id = int(tmp[0]) # index start is start from 0 here
            #cell_type_index = int(tmp[1])
            cell_cluster_type = int(tmp[2].replace('\n', ''))
            cell_cluster_type_list.append(cell_cluster_type)
            line = f.readline() 
        f.close() 
        n_clusters = max(cell_cluster_type_list) + 1 # start from 0
        cluster_labels=np.array(cell_cluster_type_list)

        X_embedding_filename =  args.embedding_data_path+'lambdaI' + str(lambda_I) + '_epoch' + str(args.num_epoch) + '_Embed_X.npy'
        X_embedding = np.load(X_embedding_filename)
        if args.PCA:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=30)
            X_embedding = pca.fit_transform(X_embedding)  

        score = metrics.silhouette_score(X_embedding, cluster_labels, metric='euclidean')
        print('Silhouette score:', score)

    return 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lambda_I', default=0.8) #0.8
    parser.add_argument( '--DGI', type=int, default=1, help='Run DGI')
    parser.add_argument( '--load', type=int, default=1, help='Load pretrained DGI model')
    parser.add_argument( '--num_epoch', type=int, default=5000, help='numebr of epoch in training DGI')
    parser.add_argument( '--hidden', type=int, default=256, help='hidden channels in DGI')   
    parser.add_argument( '--cluster', type=int, default=1, help='run cluster or not')
    parser.add_argument( '--PCA', type=int, default=1, help='run PCA or not') 
    parser.add_argument( '--n_clusters', type=int, default=5, help='run DGI and cluster') #5
    parser.add_argument( '--merge', type=int, default=1, help='Merge cluster with less than 3 cells to the closest group') #5
    parser.add_argument( '--draw_map', type=int, default=1, help='run drawing map')
    parser.add_argument( '--diff_gene', type=int, default=1, help='run differential gene expression')
    parser.add_argument( '--calculate_score', type=int, default=1, help='calculate silhouette score')
    parser.add_argument( '--model_path', type=str, default='model/') 
    parser.add_argument( '--embedding_data_path', type=str, default='Embedding_data/') 
    parser.add_argument( '--result_path', type=str, default='results_CCST/') 
    parser.add_argument( '--save_name', type=str, default='') 
    args = parser.parse_args() 

    args.embedding_data_path = args.embedding_data_path.replace('/', args.save_name) +'/'
    args.model_path = args.model_path.replace('/', args.save_name)+'/'
    args.result_path = args.result_path.replace('/', args.save_name)+'/'
    if not os.path.exists(args.embedding_data_path):
        os.makedirs(args.embedding_data_path) 
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path) 

    
    path = args.result_path+'lambdaI'+str(args.lambda_I)
    if not os.path.exists(path):
        os.makedirs(path) 
    print ('------------------------Model and Training Details--------------------------')
    print(args) 
    main(args)
    
