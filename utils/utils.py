from plotly.subplots import make_subplots

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_samples, silhouette_score

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from hdbscan import HDBSCAN
from umap import UMAP
from tqdm import tqdm

import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly
import random
import torch
import pymp

import scipy.stats as stats

import os

colors = ["#FF00FF", "#3FFF00", "#00FFFF", "#FFF700", "#FF0000", "#0000FF", "#006600",
          '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', 'black',"gray"]
# colors = px.colors.sequential.Rainbow

edge_embeddings_name = ["AverageEmbedder", "HadamardEmbedder", "WeightedL1Embedder", "WeightedL2Embedder"]
name_reduction = ["PCA", "TSNE", "UMAP"]

cpu_count = os.cpu_count()

def sort_df_edges(df_edges):
    s = []
    t = []
    for row in df_edges.itertuples():
        if row[1] > row[2]:
            s.append(row[2])
            t.append(row[1])
        else:
            s.append(row[1])
            t.append(row[2])
    df_edges["source"] = s
    df_edges["target"] = t
    
def create_graph_data_other(exp, groups_id, subgroups_id, option):
    for group in tqdm(groups_id):
        list_graphs = []
        for subgroup in tqdm(subgroups_id[group]):
            df_weighted_edges = pd.read_csv("output/{}/preprocessing/edges/edges_{}_{}.csv".format(exp, group, subgroup), dtype={"source": "string", "target": "string"})
            G = nx.from_pandas_edgelist(df_weighted_edges, "source", "target", edge_attr="weight")
            list_graphs.append(G)

        rename = [chr(k + 65) for k in range(len(list_graphs))]
        list_edges = []

        for k in range(len(list_graphs) - 1):
            nodes = list(list_graphs[k].nodes())
            for node in nodes:
                if list_graphs[k + 1].has_node(node):
                    list_edges.append((rename[k] + str(node), rename[k + 1] + str(node), 0))
                    if option == "str":
                        break

        U = nx.union_all(list_graphs, rename=rename)
        # append edges
        U.add_weighted_edges_from(list_edges)

        mapping = dict(zip(list(U.nodes()), range(U.number_of_nodes())))
        U = nx.relabel_nodes(U, mapping)
        degree = dict(U.degree())

        df_nodes = pd.DataFrame(degree.items(), columns=["idx", "degree"])
        df_nodes["id"] = list(mapping.keys())
        df_nodes.to_csv("output/{}/preprocessing/graphs_data/nodes_data_{}_{}.csv".format(exp, group, option), index=False)

        edges = list(U.edges())
        df_edges = pd.DataFrame(edges, columns=["source", "target"])
        df_edges["weight"] = [U.get_edge_data(edge[0], edge[1])["weight"] for edge in edges]
        df_edges.to_csv("output/{}/preprocessing/graphs_data/edges_data_{}_{}.csv".format(exp, group, option), index=False)

def get_label(weights, th=0.8):
    w1 = weights.get("weight1")
    w2 = weights.get("weight2")

    l1 = "?"
    l2 = "?"

    if w1:
        if w1 > 0:
            if w1 >= th:
                l1 = "P"
            else:
                l1 = "p"
        else:
            if w1 <= -th:
                l1 = "N"
            else:
                l1 = "n"
    if w2:
        if w2 > 0:
            if w2 >= th:
                l2 = "P"
            else:
                l2 = "p"
        else:
            if w2 <= -th:
                l2 = "N"
            else:
                l2 = "n"
    label = l1 + l2
    return label

def std_global(dict_df_edges_filter_weight, exp, method, groups_id, option, th=0.3, plot=True, save=False):
    dict_df_common_edges_std = {}

    for group in tqdm(groups_id):
        df_edges_filter_weight = dict_df_edges_filter_weight[group].copy()

        # calculate std
        df_edges_filter_weight["std"] = np.std(df_edges_filter_weight.iloc[:, 2:], axis=1)

        # filter std < 0.3
        df_edges_filter_weight_std = df_edges_filter_weight[df_edges_filter_weight["std"] < th]

        # average weight
        df_edges_filter_weight_std_avg = df_edges_filter_weight_std.iloc[:, :-1]
        df_edges_filter_weight_std_avg["weight"] = df_edges_filter_weight_std_avg.iloc[:, 2:].mean(axis=1)
        df_edges_filter_weight_std_avg = df_edges_filter_weight_std_avg.iloc[:, [0, 1, -1]]
        df_edges_filter_weight_std_avg.reset_index(drop=True, inplace=True)

        # save
        if save:
            df_edges_filter_weight_std_avg.to_csv("output/{}/common_edges/common_edges_{}_{}_{}.csv".format(exp, method, group, option), index=False)
            
            G = nx.from_pandas_edgelist(df_edges_filter_weight_std_avg, "source", "target", edge_attr=["weight"])
            nx.write_gexf(G, "output/{}/common_edges/common_edges_{}_{}_{}.gexf".format(exp, method, group, option))

        # plot
        if plot:
            x = df_edges_filter_weight["std"]
            plt.hist(x, bins=100)
            plt.axvline(x=th, color="red", lw=1)
            l = len(df_edges_filter_weight) - len(df_edges_filter_weight_std)
            t = len(df_edges_filter_weight)
            plt.title("Loss: {} of {} ({}%)".format(l, t, round(l*100/t)))
            plt.savefig("output/{}/plots/common_edges_std_{}_{}_{}.png".format(exp, method, group, option))
            # plt.show()
            plt.clf()

        dict_df_common_edges_std[group] = df_edges_filter_weight_std_avg
    return dict_df_common_edges_std

def get_weight_global(dict_df_edges_filter, exp, groups_id, subgroups_id):
    dict_df_edges_filter_weight = {}

    for group in tqdm(groups_id):
        df_edges_filter_weight = dict_df_edges_filter[group].copy()

        s = []
        t = []
        for row in df_edges_filter_weight.itertuples():
            if row[1] > row[2]:
                s.append(row[2])
                t.append(row[1])
            else:
                s.append(row[1])
                t.append(row[2])
        df_edges_filter_weight["source"] = s
        df_edges_filter_weight["target"] = t

        df_edges_filter_weight.sort_values(["source", "target"], ascending=True, inplace=True)
        df_edges_filter_weight["idx"] = df_edges_filter_weight["source"].astype(str) + "-" + df_edges_filter_weight["target"].astype(str)
        list_aux = df_edges_filter_weight.iloc[:, -1].values

        for subgroup in tqdm(subgroups_id[group]):
            df_edges = pd.read_csv("output/{}/preprocessing/edges/edges_{}_{}.csv".format(exp, group, subgroup),
                                   dtype={"source": "string", "target": "string"})
            s = []
            t = []
            for row in df_edges.itertuples():
                if row[1] > row[2]:
                    s.append(row[2])
                    t.append(row[1])
                else:
                    s.append(row[1])
                    t.append(row[2])
            df_edges["source"] = s
            df_edges["target"] = t

            df_edges.sort_values(["source", "target"], ascending=True, inplace=True)
            df_edges["idx"] = df_edges["source"].astype(str) + "-" + df_edges["target"].astype(str)
            
            filter = df_edges["idx"].isin(list_aux)
            temp = df_edges[filter]
            list_temp = temp.iloc[:, -2].values
            df_edges_filter_weight["subgroup{}".format(subgroup)] = list_temp
            
        df_edges_filter_weight.drop(["idx"], inplace=True, axis=1)
        
        dict_df_edges_filter_weight[group] = df_edges_filter_weight
    return dict_df_edges_filter_weight

def std_global_(dict_df_edges_filter_weight, exp, method, dimension, groups_id, th=0.3, plot=True, save=False):
    dict_df_common_edges_std = {}

    for group in tqdm(groups_id):
        df_edges_filter_weight = dict_df_edges_filter_weight[group].copy()

        # calculate std
        df_edges_filter_weight["std"] = np.std(df_edges_filter_weight.iloc[:, 2:], axis=1)

        # filter std < 0.3
        df_edges_filter_weight_std = df_edges_filter_weight[df_edges_filter_weight["std"] < th]

        # average weight
        df_edges_filter_weight_std_avg = df_edges_filter_weight_std.iloc[:, :-1]
        df_edges_filter_weight_std_avg["weight"] = df_edges_filter_weight_std_avg.iloc[:, 2:].mean(axis=1)
        df_edges_filter_weight_std_avg = df_edges_filter_weight_std_avg.iloc[:, [0, 1, -1]]
        df_edges_filter_weight_std_avg.reset_index(drop=True, inplace=True)

        # save
        if save:
            df_edges_filter_weight_std_avg.to_csv("output/{}/baseline/common_edges/common_edges_std_{}_{}_{}_{}.csv".format(exp, group, method, dimension, "L2"), index=False)
            
            G = nx.from_pandas_edgelist(df_edges_filter_weight_std_avg, "source", "target", edge_attr=["weight"])
            nx.write_gexf(G, "output/{}/baseline/common_edges/common_edges_std_{}_{}_{}_{}.gexf".format(exp, group, method, dimension, "L2"))

        # plot
        if plot:
            x = df_edges_filter_weight["std"]
            plt.hist(x, bins=100)
            plt.axvline(x=th, color="red", lw=1)
            l = len(df_edges_filter_weight) - len(df_edges_filter_weight_std)
            t = len(df_edges_filter_weight)
            plt.title("Loss: {} of {} ({}%)".format(l, t, round(l*100/t)))
            plt.savefig("output/{}/baseline/plots/common_edges_std_{}_{}_{}_{}.png".format(exp, group, method, dimension, "L2"))
            # plt.show()
            plt.clf()

        dict_df_common_edges_std[group] = df_edges_filter_weight_std_avg

    return dict_df_common_edges_std

def get_subgraphs_global(dict_graphs, groups_id):
    dict_df_edges_filter = {}

    for group in tqdm(groups_id):
        SG = get_subgraphs(dict_graphs[group])
        graph_detail(SG)
        df_edges_filter = nx.to_pandas_edgelist(SG)
        
        dict_df_edges_filter[group] = df_edges_filter
    return dict_df_edges_filter

def edge_embeddings_global(exp, method, groups_id, subgroups_id):
    for group in tqdm(groups_id):
        for subgroup in tqdm(subgroups_id[group]):
            # read dataset
            df_node_embeddings = pd.read_csv("output/{}/node_embeddings/node-embeddings_{}_{}_{}.csv".format(exp, method, group, subgroup), index_col=0)
            df_edges = pd.read_csv("output/{}/preprocessing/graphs_data/edges_data_{}_{}.csv".format(exp, group, subgroup))
            
            # get edges embeddings
            df_edge_embeddings = edge2vec_l2(df_edges, df_node_embeddings)
            df_edge_embeddings.to_csv("output/{}/edge_embeddings/edge-embeddings_{}_{}_{}.csv".format(exp, method, group, subgroup), index=True)

def create_graph_data(exp, groups_id, subgroups_id):
    for group in tqdm(groups_id):
        for subgroup in tqdm(subgroups_id[group]):
            df_weighted_edges = pd.read_csv("output/{}/preprocessing/edges/edges_{}_{}.csv".format(exp, group, subgroup))
            G = nx.from_pandas_edgelist(df_weighted_edges, "source", "target", edge_attr="weight")
            mapping = dict(zip(list(G.nodes()), range(G.number_of_nodes())))
            G = nx.relabel_nodes(G, mapping)
            degree = dict(G.degree())

            # graph_detail(G)

            df_nodes = pd.DataFrame(degree.items(), columns=["idx", "degree"])
            df_nodes["id"] = list(mapping.keys())
            df_nodes.to_csv("output/{}/preprocessing/graphs_data/nodes_data_{}_{}.csv".format(exp, group, subgroup), index=False)

            edges = list(G.edges())
            df_edges = pd.DataFrame(edges, columns=["source", "target"])
            df_edges["weight"] = [G.get_edge_data(edge[0], edge[1])["weight"] for edge in edges]
            df_edges.to_csv("output/{}/preprocessing/graphs_data/edges_data_{}_{}.csv".format(exp, group, subgroup), index=False)

def build_graph_weight_global(exp, list_groups_subgroups_t_corr, groups_id, subgroups_id, threshold=0.5):
    list_groups_subgroups_t_corr_g = []

    for i in tqdm(range(len(list_groups_subgroups_t_corr))):
        list_aux = []
        for j in tqdm(range(len(list_groups_subgroups_t_corr[i]))):
            weighted_edges = p_build_graph_weight(list_groups_subgroups_t_corr[i][j], threshold)
            df_weighted_edges = pd.DataFrame(weighted_edges, columns=["source", "target", "weight"])
            df_weighted_edges = df_weighted_edges[df_weighted_edges["weight"] != 0]

            df_weighted_edges.to_csv("output/{}/preprocessing/edges/edges_{}_{}.csv".format(exp, groups_id[i], subgroups_id[groups_id[i]][j]), index=False)
            G = nx.from_pandas_edgelist(df_weighted_edges, "source", "target", edge_attr=["weight"])
            nx.write_gexf(G, "output/{}/preprocessing/graphs/graphs_{}_{}.gexf".format(exp, groups_id[i], subgroups_id[groups_id[i]][j]))

            list_aux.append(df_weighted_edges)
        list_groups_subgroups_t_corr_g.append(list_aux)
    return list_groups_subgroups_t_corr_g

def correlation_global(list_groups_subgroups_t, method="pearson"):
    list_groups_subgroups_t_corr = []
    for list_groups in tqdm(list_groups_subgroups_t):
        list_aux = []
        for subgroup in tqdm(list_groups):
            matrix = subgroup.corr(method=method) # pearson, kendall, spearman
            list_aux.append(matrix)
        list_groups_subgroups_t_corr.append(list_aux)
    return list_groups_subgroups_t_corr

def split_groups_subgroups(df_join_raw_log, groups_id, subgroups_id):
    list_df_groups_subgroups = []
    for group in groups_id:
        df_aux = df_join_raw_log.filter(like=group)
        list_aux = []
        
        for subgroup in subgroups_id[group]:
            list_aux.append(df_aux.filter(like="{}{}.".format(group, subgroup)))
        list_df_groups_subgroups.append(list_aux)
    return list_df_groups_subgroups

def get_subgroups_id(df_join_raw, groups):
    dict_groups_id = {}
    for group in groups:
        # get group
        columns = list(df_join_raw.filter(like=group).columns)

        subgroups = [item.split("{}".format(group))[1].split(".")[0] for item in columns]
        subgroups = np.unique(subgroups)
        dict_groups_id[group] = subgroups.tolist()
    return dict_groups_id

def transpose_global(list_groups_subgroups):
    list_groups_subgroups_t = []
    for list_subgroups in list_groups_subgroups:
        list_aux = []
        for subgroup in list_subgroups:
            aux_t = transpose(subgroup)
            list_aux.append(aux_t)
        list_groups_subgroups_t.append(list_aux)
    return list_groups_subgroups_t

def log10_global(df_join_raw):
    df_join_raw_log = df_join_raw.copy()
    for column in df_join_raw.columns:
        df_join_raw_log[column] = np.log10(df_join_raw[column], where=df_join_raw[column]>0)
    return df_join_raw_log

def get_edges_std(G, dir, group, subgroups, ddof):
    # ddof = 0, poblacional
    # ddof = 1, muestrals
    df_edge_embeddings_join_filter_count = pd.DataFrame(G.edges())
    df_edge_embeddings_join_filter_count.columns = ["source", "target"]
    df_edge_embeddings_join_filter_count

    # Get weight
    df_edge_embeddings_join_filter_count_weight = df_edge_embeddings_join_filter_count.copy()
    s = []
    t = []
    for row in df_edge_embeddings_join_filter_count_weight.itertuples():
        if row[1] > row[2]:
            s.append(row[2])
            t.append(row[1])
        else:
            s.append(row[1])
            t.append(row[2])
    df_edge_embeddings_join_filter_count_weight["source"] = s
    df_edge_embeddings_join_filter_count_weight["target"] = t

    # df_edge_embeddings_join_filter_count_weight = df_edge_embeddings_join_filter_count.copy()
    df_edge_embeddings_join_filter_count_weight.sort_values(["source", "target"], ascending=True, inplace=True)
    df_edge_embeddings_join_filter_count_weight["idx"] = df_edge_embeddings_join_filter_count_weight["source"].astype(str) + "-" + df_edge_embeddings_join_filter_count_weight["target"].astype(str)
    list_aux = df_edge_embeddings_join_filter_count_weight.iloc[:, -1].values

    for i in tqdm(subgroups):
        df_edges = pd.read_csv("{}/output_preprocessing/edges/{}_edges_{}.csv".format(dir, group[0], i))
        df_edges.sort_values(["source", "target"], ascending=True, inplace=True)
        df_edges["idx"] = df_edges["source"].astype(str) + "-" + df_edges["target"].astype(str)
        
        filter = df_edges["idx"].isin(list_aux)
        temp = df_edges[filter]
        list_temp = temp.iloc[:, -2].values
        df_edge_embeddings_join_filter_count_weight["subgroup{}".format(i)] = list_temp
    df_edge_embeddings_join_filter_count_weight

    # Dispersion (std)
    df_edge_embeddings_join_filter_count_weight_std = df_edge_embeddings_join_filter_count_weight.copy()
    df_edge_embeddings_join_filter_count_weight_std["std"] = np.std(df_edge_embeddings_join_filter_count_weight_std.iloc[:, -len(subgroups):], axis=1, ddof=ddof)
    df_edge_embeddings_join_filter_count_weight_std

    # Average weight
    df_edge_embeddings_join_filter_count_weight_std_avg_all = df_edge_embeddings_join_filter_count_weight_std.copy()
    df_edge_embeddings_join_filter_count_weight_std_avg_all["weight"] = df_edge_embeddings_join_filter_count_weight_std_avg_all.iloc[:, -(len(subgroups) + 1):-1].mean(axis=1)
    # df_edge_embeddings_join_filter_count_weight_std_avg_all.to_csv("{}/output_greedy/edges_filter_weight_std_avg_all/greedy_{}_edge-filter-weight-std-avg-all.csv".format(dir, group[0]), index=False)
    df_edge_embeddings_join_filter_count_weight_std_avg_all

    df_edges_all = df_edge_embeddings_join_filter_count_weight_std_avg_all.iloc[:, [0, 1, -1, -2]]
    return df_edges_all

def get_nodes_anova(G, dir, group):
    # Load dataset Groups
    df1 = pd.read_csv("{}/input/Edwin_proyecto2/{}.csv".format(dir, "int1"), delimiter="|")
    df2 = pd.read_csv("{}/input/Edwin_proyecto2/{}.csv".format(dir, "int2"), delimiter="|")
    df3 = pd.read_csv("{}/input/Edwin_proyecto2/{}.csv".format(dir, "int3"), delimiter="|")
    df4 = pd.read_csv("{}/input/Edwin_proyecto2/{}.csv".format(dir, "int4"), delimiter="|")
    # df5_ = pd.read_csv("{}/inputs/Edwin_proyecto2/{}.csv".format(dir, "int5"), delimiter="|")

    # concat
    # df_join_raw = pd.concat([df1.iloc[:,1:], df2.iloc[:, 2:], df3.iloc[:, 2:], df4.iloc[:, 2:], df5.iloc[:, 2:]], axis=1)
    df_join_raw = pd.concat([df1.iloc[:, 1:], df2.iloc[:, 2:], df3.iloc[:, 2:], df4.iloc[:, 2:]], axis=1)
    df_join_raw.set_index(["ionMz"], inplace=True)
    # print(df_join_raw.shape)
    df_join_raw

    df_raw_group = df_join_raw.filter(like=group[0], axis=1)
    df_raw_group

    # logarithm
    df_raw_log = df_raw_group.copy()
    for column in df_raw_group.columns:
        df_raw_log[column] = np.log10(df_raw_group[column], where=df_raw_group[column]>0)
    # df_raw_log[column] = np.log10(df_raw_group[column], out=np.zeros_like(df_raw_group[column]), where=df_raw_group[column]>0)
    df_raw_log

    subgroups = [item.split("{} ".format(group[0]))[1].split(".")[0] for item in list(df_raw_log.columns)]
    subgroups = np.unique(subgroups)
    subgroups

    # split graph
    list_raw = []

    for item in subgroups:
        list_raw.append(df_raw_log.filter(like="{} {}.".format(group[0], item)))

    # print(len(list_raw))
    list_raw[0]

    list_raw_copy = list_raw[:]

    for k, item in enumerate(list_raw_copy):
        item.columns = [chr(65 + k)]*len(item.columns)

    # filter by graph and concat 
    nodes = list(G.nodes())
    df_raw_filter = list_raw_copy[0].loc[nodes, :]

    for k in range(1, len(subgroups)):
        df_temp = list_raw_copy[k].loc[nodes, :]
        # df_raw_filter = df_raw_filter.join(df_temp)
        df_raw_filter = pd.concat([df_raw_filter, df_temp], axis=1)

    # df_raw_filter.to_csv("{}/output_greedy/matrix/greedy_{}_matrix_copy.csv".format(dir, group[0]), index=True)
    df_raw_filter

    # ANOVA
    df_raw_filter_anova = df_raw_filter.copy()
    p_values = anova(df_raw_filter_anova)
    df_raw_filter_anova["p-value"] = p_values
    
    df_raw_filter_anova = df_raw_filter_anova.iloc[:, [-1]]

    return df_raw_filter_anova

def anova(df_raw_filter):
    columns = np.unique(list(df_raw_filter.columns))
    p_values = []

    for i in range(len(df_raw_filter)):
        row = df_raw_filter.iloc[i,:]
        list_global = []
        for column in columns:
            list_global.append(row[column].values)
        fvalue, pvalue = stats.f_oneway(*list_global)
        p_values.append(pvalue)
    return p_values

def get_subgraphs(graphs):
    # get common nodes
    common_nodes = set(list(graphs[0].nodes()))
    # print(nodes1)
    for graph in tqdm(graphs[1:]):
        nodes = set(list(graph.nodes()))
        # print(nodes2)
        common_nodes = common_nodes & nodes
    # print("Num. of common nodes:", len(common_nodes))

    # get subgraphs
    G = graphs[0].subgraph(common_nodes)
    common_edges = set(sort_edges(G.edges()))
    for graph in tqdm(graphs[1:]):
        G = graph.subgraph(common_nodes)
        edges = set(sort_edges(G.edges()))
        common_edges = common_edges & edges
    # print("Num. of common edges:", len(common_edges))

    H = nx.Graph()
    H.add_edges_from(list(common_edges))

    return H

def get_common_nodes_edges(G1, G2, group1, group2):
    # get common nodes
    nodes1 = set(list(G1.nodes()))
    nodes2 = set(list(G2.nodes()))

    print("Nodes:")
    n1_inte_n2 = nodes1 & nodes2
    print("{} & {}:".format(group1, group2), len(n1_inte_n2))

    n1_diff_n2 = nodes1 - nodes2
    print("{} - {}:".format(group1, group2), len(n1_diff_n2))

    n2_diff_n1 = nodes2 - nodes1
    print("{} - {}:".format(group2, group1), len(n2_diff_n1))

    # get common edges
    edges1 = set(sort_edges(G1.edges()))
    edges2 = set(sort_edges(G2.edges()))

    print("Edges:")
    e1_inte_e2 = edges1 & edges2
    print("{} & {}:".format(group1, group2), len(e1_inte_e2))

    e1_diff_e2 = edges1 - edges2
    print("{} - {}:".format(group1, group2), len(e1_diff_e2))

    e2_diff_e1 = edges2 - edges1
    print("{} - {}:".format(group2, group1), len(e2_diff_e1))

    return [[n1_inte_n2, e1_inte_e2], [n1_diff_n2, e1_diff_e2], [n2_diff_n1, e2_diff_e1]]

def get_change_subgraphs(G1, G2, group1, group2):
    # get common nodes
    nodes1 = set(list(G1.nodes()))
    # print(nodes1)

    nodes2 = set(list(G2.nodes()))
    # print(nodes2)

    common_nodes = list(nodes1 & nodes2)
    print("Num. of common nodes:", len(common_nodes))

    # get subgraphs
    H1 = G1.subgraph(common_nodes)
    H2 = G2.subgraph(common_nodes)

    # get common edges
    edges1 = set(sort_edges(H1.edges()))
    edges2 = set(sort_edges(H2.edges()))

    e1_inte_e2 = edges1 & edges2
    print("{} & {}:".format(group1, group2), len(e1_inte_e2))
    # print(list(e1_inte_e2))

    e1_diff_e2 = edges1 - edges2
    print("{} - {}:".format(group1, group2), len(e1_diff_e2))
    # print(list(e1_diff_e2))

    e2_diff_e1 = edges2 - edges1
    print("{} - {}:".format(group2, group1), len(e2_diff_e1))
    # print(list(e2_diff_e1))

    return e1_inte_e2, e1_diff_e2, e2_diff_e1

def correlation_labels(df_subgraphs, threshold=0.8):
    conditions = [
        ((df_subgraphs["weight1"] < 0) & (df_subgraphs["weight2"].isnull()) & (df_subgraphs["weight1"] >= threshold)),
        ((df_subgraphs["weight1"] < 0) & (df_subgraphs["weight2"].isnull()) & (df_subgraphs["weight1"] < threshold)),
        ((df_subgraphs["weight1"] > 0) & (df_subgraphs["weight2"].isnull()) & (df_subgraphs["weight1"] >= threshold)),
        ((df_subgraphs["weight1"] > 0) & (df_subgraphs["weight2"].isnull()) & (df_subgraphs["weight1"] < threshold)),

        ((df_subgraphs["weight2"] < 0) & (df_subgraphs["weight1"].isnull()) & (df_subgraphs["weight2"] >= threshold)),
        ((df_subgraphs["weight2"] < 0) & (df_subgraphs["weight1"].isnull()) & (df_subgraphs["weight2"] < threshold)),
        ((df_subgraphs["weight2"] > 0) & (df_subgraphs["weight1"].isnull()) & (df_subgraphs["weight2"] >= threshold)),
        ((df_subgraphs["weight2"] > 0) & (df_subgraphs["weight1"].isnull()) & (df_subgraphs["weight2"] < threshold)),

        ((df_subgraphs["weight1"] < 0) & (df_subgraphs["weight2"] > 0) & (df_subgraphs["weight1"] > -threshold) & (df_subgraphs["weight2"] < threshold)),
        ((df_subgraphs["weight1"] < 0) & (df_subgraphs["weight2"] > 0) & (df_subgraphs["weight1"] > -threshold) & (df_subgraphs["weight2"] >= threshold)),
        ((df_subgraphs["weight1"] < 0) & (df_subgraphs["weight2"] > 0) & (df_subgraphs["weight1"] <= -threshold) & (df_subgraphs["weight2"] < threshold)),
        ((df_subgraphs["weight1"] < 0) & (df_subgraphs["weight2"] > 0) & (df_subgraphs["weight1"] <= -threshold) & (df_subgraphs["weight2"] >= threshold)),

        ((df_subgraphs["weight1"] > 0) & (df_subgraphs["weight2"] < 0) & (df_subgraphs["weight1"] < threshold) & (df_subgraphs["weight2"] > -threshold)),
        ((df_subgraphs["weight1"] > 0) & (df_subgraphs["weight2"] < 0) & (df_subgraphs["weight1"] >= threshold) & (df_subgraphs["weight2"] > -threshold)),
        ((df_subgraphs["weight1"] > 0) & (df_subgraphs["weight2"] < 0) & (df_subgraphs["weight1"] < threshold) & (df_subgraphs["weight2"] <= -threshold)),
        ((df_subgraphs["weight1"] > 0) & (df_subgraphs["weight2"] < 0) & (df_subgraphs["weight1"] >= threshold) & (df_subgraphs["weight2"] <= -threshold)),

        ((df_subgraphs["weight1"] > 0) & (df_subgraphs["weight2"] > 0) & (df_subgraphs["weight1"] >= threshold) & (df_subgraphs["weight2"] >= threshold)),
        ((df_subgraphs["weight1"] > 0) & (df_subgraphs["weight2"] > 0) & (df_subgraphs["weight1"] < threshold) & (df_subgraphs["weight2"] < threshold)),
        ((df_subgraphs["weight1"] > 0) & (df_subgraphs["weight2"] > 0) & (df_subgraphs["weight1"] >= threshold) & (df_subgraphs["weight2"] < threshold)),
        ((df_subgraphs["weight1"] > 0) & (df_subgraphs["weight2"] > 0) & (df_subgraphs["weight1"] < threshold) & (df_subgraphs["weight2"] >= threshold)),

        ((df_subgraphs["weight1"] < 0) & (df_subgraphs["weight2"] < 0) & (df_subgraphs["weight1"] <= -threshold) & (df_subgraphs["weight2"] <= -threshold)),
        ((df_subgraphs["weight1"] < 0) & (df_subgraphs["weight2"] < 0) & (df_subgraphs["weight1"] > -threshold) & (df_subgraphs["weight2"] > -threshold)),
        ((df_subgraphs["weight1"] < 0) & (df_subgraphs["weight2"] < 0) & (df_subgraphs["weight1"] <= -threshold) & (df_subgraphs["weight2"] > -threshold)),
        ((df_subgraphs["weight1"] < 0) & (df_subgraphs["weight2"] < 0) & (df_subgraphs["weight1"] > -threshold) & (df_subgraphs["weight2"] <= -threshold)),
    ]
    # print(conditions)

    values = ["N?", "n?", "P?", "p?", "?N", "?n", "?P", "?p", "np", "nP", "Np", "NP", "pn", "Pn", "pN", "PN", "PP", "pp", "Pp", "pP", "NN", "nn", "Nn", "nN"]
    changes_labels = np.select(conditions, values)
    return changes_labels

def sort_edges(edges):
    edges = list(edges)
    for k in range(len(edges)):
        if edges[k][0] > edges[k][1]:
            edges[k] = (edges[k][1], edges[k][0])
    return edges

def degree_distibution(graps, labels=[]):
    list_hist = []
    for graph in graps:
        list_hist.append(list(dict(graph.degree()).values()))
    
    plt.hist(list_hist, label=labels)
    plt.legend(loc='upper center')

    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()
    
def edge2vecx(list_df_node_embeddings, list_df_edges, list_node_embeddings_legend):    
    list_df_edge_embeddings = []
    list_edge_embeddings_legend = []

    for i, df_node_embedding in enumerate(list_df_node_embeddings):
        for j, embedder in enumerate([e2v_l2x]):
            df_edge_embeddings = embedder(list_df_edges[i], df_node_embedding)

            legends = ["{}{}".format(list_node_embeddings_legend[i], str("L2"))]

            list_df_edge_embeddings += [df_edge_embeddings]
            list_edge_embeddings_legend += legends
    return list_df_edge_embeddings, list_edge_embeddings_legend

def p_edge2vec_l2(df_edges, df_node_embeddings):
    dimension = df_node_embeddings.shape[1]
    index = pymp.shared.array((len(df_edges), 2), dtype="int")
    data = pymp.shared.array((len(df_edges), dimension), dtype="float")

    with pymp.Parallel(24) as p:
        for k in p.range(len(df_edges)):
            i = df_edges.iloc[k, 0]
            j = df_edges.iloc[k, 1]

            u = df_node_embeddings.loc[i].values
            v = df_node_embeddings.loc[j].values
            r = (u - v) ** 2
            
            data[k] = r
            index[k] = (i, j)

    index =list(map(tuple, index))
    index = pd.MultiIndex.from_tuples(index)
    df_edge_embeddings = pd.DataFrame(data, index=index)
    return df_edge_embeddings

def edge2vec_l2(df_edges, df_node_embeddings):
    index = []
    data = []
    for row in df_edges.itertuples():
        i = row[1]
        j = row[2]

        u = df_node_embeddings.loc[i].values
        v = df_node_embeddings.loc[j].values
        r = (u - v) ** 2
        
        index.append((i, j))
        data.append(r)

    index = pd.MultiIndex.from_tuples(index)
    df_edge_embeddings = pd.DataFrame(data, index=index)
    return df_edge_embeddings

def edge2vec_l2_v2(df_edges, df_node_embeddings):
    index = []
    data = []
    for k in range(len(df_edges)):
        i = df_edges.iloc[k, 0]
        j = df_edges.iloc[k, 1]

        u = df_node_embeddings.loc[i].values
        v = df_node_embeddings.loc[j].values
        r = (u - v) ** 2
        
        data.append(r)
        index.append((i, j))

    index = pd.MultiIndex.from_tuples(index)
    df_edge_embeddings = pd.DataFrame(data, index=index)
    return df_edge_embeddings

def edge2vec_l2_v1(df_edges, df_node_embeddings):
    edge2vec = {}
    for k in range(len(df_edges)):
        i = df_edges.iloc[k, 0]
        j = df_edges.iloc[k, 1]

        u = df_node_embeddings.loc[i].values
        v = df_node_embeddings.loc[j].values
        # edge2vec[str(tuple(sorted((i, j))))] = (u - v) ** 2
        # edge2vec[tuple(sorted((i, j)))] = (u - v) ** 2
        edge2vec[tuple((i, j))] = (u - v) ** 2
        # edge2vec["source"] = i
        # edge2vec["target"] = j
        # print(edge2vec)
    return pd.DataFrame.from_dict(edge2vec, orient="index")

def graph_detail(G):
    print("Num. nodes: {}".format(G.number_of_nodes()))
    print("Num. edges: {}".format(G.number_of_edges()))
    print()

def clustering_analysis(list_df_embeddings, list_embeddings_legend):
    list_df_embeddings_clusters = []
    for k, df_embedding in enumerate(list_df_embeddings):
        X_train = df_embedding
        # print(X_train)

        clustering = HDBSCAN(min_cluster_size=5, min_samples=None, core_dist_n_jobs=-1, allow_single_cluster=False)
        clustering.fit(X_train)
        X_train["labels"] = clustering.labels_
        list_df_embeddings_clusters.append(X_train)
    return list_df_embeddings_clusters, list_embeddings_legend

def filter_edges(list_df_embeddings_clusters, list_embeddings_clusters_legend):
    list_edges = []
    for df_embeddings_clusters in list_df_embeddings_clusters:
        a = df_embeddings_clusters
        unique_labels = np.unique(df_embeddings_clusters["labels"].values)
        if -1 in unique_labels:
            unique_labels = np.delete(unique_labels, 0)
        # print("labels:", unique_labels)
        list_edges_temp = []
        for label in unique_labels:
            # print(label)
            b = a[a["labels"] == label].index
            c = pd.Index([item.replace("A", "").replace("B", "").replace("(", "").replace(")", "").replace("'","").split(", ") for item in b])
            # print(c)
            d = c.value_counts()
            e = d[d == 2].index
            # print(list(e))
            list_edges_temp += list(e)
        list_edges.append(list_edges_temp)
    # return list_edges, list_embeddings_clusters_legend
    
    df_list_edges = pd.DataFrame()
    df_list_edges["legend"] = list_embeddings_clusters_legend
    df_list_edges["edges"] = list_edges
    df_list_edges["length"] = [len(edges) for edges in df_list_edges["edges"]]
    df_list_edges.sort_values(by=["length"],  ascending=False, inplace=True)
    return df_list_edges

def set_labels(index):
    labels = []
    for item in index:
        try:
            if "A" in item:
                labels.append(0)
            else:
                labels.append(1)
        except:
            labels.append(-1)
    return labels

def join_embeddings(df_embeddings_1, df_embeddings_2):
    df_embeddings = df_embeddings_1.copy()
    df_embeddings = pd.concat([df_embeddings, df_embeddings_2])
    # print(df_node_embeddings_.shape)
    return df_embeddings

def reduction_embeddings(list_df_embeddings, list_embeddings_legend, n_component=2):
    list_reduction_embeddings = []
    list_reduction_embeddings_legend = []

    for k, item in enumerate(list_df_embeddings):
        transform = PCA(n_components=n_component, random_state=42)
        embeddings_2d_transform = transform.fit_transform(item.values)
        df_embeddings_2d_1 = pd.DataFrame(embeddings_2d_transform, item.index)

        list_reduction_embeddings.append(df_embeddings_2d_1)
        list_reduction_embeddings_legend.append("{}-PCA".format(list_embeddings_legend[k]))
        
        # transform = TSNE(n_components=n_component, learning_rate="auto", metric="euclidean", init="random", perplexity=10, random_state=42, n_jobs=-1)
        transform = TSNE(n_components=n_component, metric="euclidean", init="random", perplexity=10, random_state=42, n_jobs=-1)
        embeddings_2d_transform = transform.fit_transform(item.values)
        df_embeddings_2d_2 = pd.DataFrame(embeddings_2d_transform, item.index)

        list_reduction_embeddings.append(df_embeddings_2d_2)
        list_reduction_embeddings_legend.append("{}-TSNE".format(list_embeddings_legend[k]))

        transform = UMAP(n_components=n_component, init="random", random_state=42)
        embeddings_2d_transform = transform.fit_transform(item.values)
        df_embeddings_2d_3 = pd.DataFrame(embeddings_2d_transform, item.index)

        list_reduction_embeddings.append(df_embeddings_2d_3)
        list_reduction_embeddings_legend.append("{}-UMAP".format(list_embeddings_legend[k]))

    return list_reduction_embeddings, list_reduction_embeddings_legend

def edge2vec(list_df_node_embeddings, list_graphs, list_node_embeddings_legend):
    # input: [df_node_embeddings_1, df_node_embeddings_2, df_node_embeddings_3, df_node_embeddings_4,...], [G1, G2, G3, G4,...]
    
    list_edge_embeddings = []
    list_edge_embeddings_legend = []

    for i, node_embedding in enumerate(list_df_node_embeddings):
        for j, embedder in enumerate([e2v_average, e2v_hadamar, e2v_l1, e2v_l2]):
            df_edge_embeddings = embedder(list_graphs[0], node_embedding)

            legends = ["{}-{}".format(list_node_embeddings_legend[i], edge_embeddings_name[j])]

            list_edge_embeddings += [df_edge_embeddings]
            list_edge_embeddings_legend += legends
    return list_edge_embeddings, list_edge_embeddings_legend

def edge2vec_(list_df_node_embeddings, list_graphs, list_node_embeddings_legend):
    # input: [df_node_embeddings_1, df_node_embeddings_2, df_node_embeddings_3, df_node_embeddings_4], [G1, G2, G3, G4]
    
    list_edge_embeddings = []
    list_edge_embeddings_legend = []

    list_df_node_embeddings[0].set_index(pd.Index([item.replace("A", "") for item in list_df_node_embeddings[0].index]), inplace=True)
    list_df_node_embeddings[1].set_index(pd.Index([item.replace("B", "") for item in list_df_node_embeddings[1].index]), inplace=True)

    for k, embedder in enumerate([e2v_average, e2v_hadamar, e2v_l1, e2v_l2]):
        df_edge_embeddings1 = embedder(list_graphs[0], list_df_node_embeddings[0])
        df_edge_embeddings2 = embedder(list_graphs[1], list_df_node_embeddings[1])
        df_edge_embeddings3 = embedder(list_graphs[2], list_df_node_embeddings[2])
        df_edge_embeddings4 = embedder(list_graphs[3], list_df_node_embeddings[3])

        df_edge_embeddings1.set_index(pd.Index(["A" + str(item) for item in df_edge_embeddings1.index]), inplace=True)
        df_edge_embeddings2.set_index(pd.Index(["B" + str(item) for item in df_edge_embeddings2.index]), inplace=True)

        df_edge_embeddings5 = join_embeddings(df_edge_embeddings1, df_edge_embeddings2)
        
        legends = ["G1 G2-{}".format(edge_embeddings_name[k]), "G3-{}".format(edge_embeddings_name[k]), "G4-{}".format(edge_embeddings_name[k])]

        list_edge_embeddings += [df_edge_embeddings5, df_edge_embeddings3, df_edge_embeddings4]
        list_edge_embeddings_legend += legends

    list_df_node_embeddings[0].set_index(pd.Index(["A" + str(item) for item in list_df_node_embeddings[0].index]), inplace=True)
    list_df_node_embeddings[1].set_index(pd.Index(["B" + str(item) for item in list_df_node_embeddings[1].index]), inplace=True)
    return list_edge_embeddings, list_edge_embeddings_legend

def e2v_average(G, node_embeddings):
    edge2vec = {}
    for edge in list(G.edges()):
        u = node_embeddings.loc[str(edge[0])].values
        v = node_embeddings.loc[str(edge[1])].values
        edge2vec[str(tuple(sorted(edge)))] = (u + v) / 2
    return pd.DataFrame.from_dict(edge2vec, orient='index')

def e2v_hadamar(G, node_embeddings):
    edge2vec = {}
    for edge in list(G.edges()):
        u = node_embeddings.loc[str(edge[0])].values
        v = node_embeddings.loc[str(edge[1])].values
        edge2vec[str(tuple(sorted(edge)))] = u * v
    return pd.DataFrame.from_dict(edge2vec, orient='index')

def e2v_l1(G, node_embeddings):
    edge2vec = {}
    for edge in list(G.edges()):
        u = node_embeddings.loc[str(edge[0])].values
        v = node_embeddings.loc[str(edge[1])].values
        edge2vec[str(tuple(sorted(edge)))] = np.abs(u - v)
    return pd.DataFrame.from_dict(edge2vec, orient='index')

def e2v_l2(G, node_embeddings):
    edge2vec = {}
    for edge in list(G.edges()):
        u = node_embeddings.loc[str(edge[0])].values
        v = node_embeddings.loc[str(edge[1])].values
        edge2vec[str(tuple(sorted(edge)))] = (u - v) ** 2
    return pd.DataFrame.from_dict(edge2vec, orient='index')

def get_daltons(formula):
    d = {"C": 12, 
        "H": 1.00782503207, 
        "N": 14.0030740048,
        "O": 15.99491461956,
        "P": 30.97376163,
        "S": 31.972071,
        "Si": 27.9769265325,
        "F": 18.99840322,
        "Cl": 34.96885268,
        "Br": 78.9183371}

    f = ""
    n = ""
    daltons = 0
    try:
        for k in range(len(formula)):
            if formula[k].isnumeric():
                n += formula[k]
            else:
                if formula[k].isupper() and f != "":
                    # print(f, n)
                    if n == "":
                        n = "1"
                    daltons += d[f] * int(n)
                    f = formula[k]
                    n = ""
                else:
                    f += formula[k]
            if n == "":
                n = "1"
            # print(f, n)
            daltons += d[f] * int(n)
    except:
        daltons = 0
    return daltons

def get_node_class(node_embeddings_2d1, node_embeddings_2d2):
    nodes = []
    classes1 = []
    classes2 = []

    for k, node_id in enumerate(node_embeddings_2d2.index):
        nodes.append(node_id)
        classes1.append(node_embeddings_2d2["labels"][k])
        if node_id in node_embeddings_2d1.index:
            index = list(node_embeddings_2d1.index).index(node_id)
            classes2.append(node_embeddings_2d1["labels"][index])
        else:
            classes2.append("X")
    data = {"Node id": nodes, "Class G1": classes1, "Class G2": classes2}
    df_match1 = pd.DataFrame(data=data)
    return df_match1.sort_values(by="Class G1", ascending=True)

def matching(node_embeddings_2d1, node_embeddings_2d2, node_embeddings_2d):
    n_classes = np.unique(node_embeddings_2d1["labels"].values)
    if -1 in n_classes:
        len1 = len(n_classes) - 1
    else:
        len1 = len(n_classes)

    labels1_ = node_embeddings_2d1["labels"].values.copy()
    size1_ = [8] * len(node_embeddings_2d1.index)
    opacity1_ = [0.3] * len(node_embeddings_2d1.index)
    for k, node_id in enumerate(node_embeddings_2d2.index):
        if node_id in node_embeddings_2d1.index and node_embeddings_2d2["labels"][k] != -1:
            index = list(node_embeddings_2d1.index).index(node_id)
            labels1_[index] = node_embeddings_2d2["labels"][k] + len1
            # size1_[index] = 10
            opacity1_[index] = 0.9
    node_embeddings_2d1["labels_"] = labels1_
  
    # Plot
    fig = make_subplots(rows=2, cols=2,
                      subplot_titles=("Group 1", "Group 2", "Group 1 - Group 2", "Group 1 + Group 2"),
                      horizontal_spacing=0.05, vertical_spacing=0.05)

    fig.add_trace(
        go.Scatter(
            x=node_embeddings_2d1.iloc[:, 0].values,
            y=node_embeddings_2d1.iloc[:, 1].values,
            mode="markers",
            name="markers",
            text=node_embeddings_2d1.index,
            hovertemplate="Node id: " + node_embeddings_2d1.index + "<br>Class: " + node_embeddings_2d1["labels"].astype(str),
            textposition="bottom center",
            showlegend=True,
            marker=dict(
                size=8,
                color=node_embeddings_2d1["labels"].values,
                opacity=0.9,
                colorscale=list(np.array(colors)[np.unique(node_embeddings_2d1["labels"])]), # "Rainbow",
                line_width=1
            ),),
          row=1, col=1)

    colorscale_ = np.unique(node_embeddings_2d2["labels"])
    for k in range(len(colorscale_)):
        if colorscale_[k] != -1:
            colorscale_[k] +=  len1

    fig.add_trace(
        go.Scatter(
            x=node_embeddings_2d2.iloc[:, 0].values,
            y=node_embeddings_2d2.iloc[:, 1].values,
            mode="markers",
            name="markers",
            text=node_embeddings_2d2.index,
            hovertemplate="Node id: " + node_embeddings_2d2.index + "<br>Class: " + node_embeddings_2d2["labels"].astype(str),
            textposition="bottom center",
            showlegend=True,
            marker=dict(
                size=8,
                color=node_embeddings_2d2["labels"].values,
                opacity=0.9,
                colorscale=list(np.array(colors)[colorscale_]), # "Rainbow",
                line_width=1
            ),
            textfont=dict(
                family="sans serif",
                size=10,
                color="black"
            ),
        ),
        row=1, col=2
    )

    fig.add_trace(
    go.Scatter(
        x=node_embeddings_2d1.iloc[:, 0].values,
        y=node_embeddings_2d1.iloc[:, 1].values,
        mode="markers",
        name="markers",
        text=node_embeddings_2d1.index,
        hovertemplate="Node id: " + node_embeddings_2d1.index + "<br>Class: " + node_embeddings_2d1["labels_"].astype(str),
        textposition="bottom center",
        showlegend=True,
        marker=dict(
            size=8,
            color=labels1_,
            opacity=opacity1_,
            colorscale=list(np.array(colors)[np.unique(labels1_)]), # "Rainbow",
            line_width=1
        ),
    ),
    row=2, col=1
    ),

    fig.add_trace(
        go.Scatter(
            x=node_embeddings_2d.iloc[:, 0].values,
            y=node_embeddings_2d.iloc[:, 1].values,
            mode="markers",
            name="markers",
            text="Node id: " + node_embeddings_2d.index + "<br>Class: " + node_embeddings_2d["labels"].astype(str),
            hovertemplate="Node id: " + node_embeddings_2d.index + "<br>Class: " + node_embeddings_2d["labels"].astype(str),
            textposition="bottom center",
            textfont_size=10,
            showlegend=True,
            marker=dict(
                size=8,
                color=node_embeddings_2d["labels"].values,
                opacity=0.9,
                colorscale=list(np.array(colors)[np.unique(node_embeddings_2d["labels"])]), # "Rainbow",
                line_width=1
            ),
        ),
        row=2, col=2
    )
    fig.update_layout(height=1000, width=1000, title_text="Clustering Embeddings") # ,legend_tracegroupgap=50, showlegend=False)
    fig.show()

def visualization_cluster_embeddings(list_embeddings_2d, titles=None, title_text="Embeddings", cols=2):
    # cols = 2
    rows = math.ceil(len(list_embeddings_2d) / cols)

    if not titles:
        titles = []
        for k in range(len(list_embeddings_2d)):
            titles.append("Group {}".format(k + 1))
  
    fig = plotly.subplots.make_subplots(rows=rows, cols=cols,
                                        subplot_titles=titles,
                                        horizontal_spacing=0.05, vertical_spacing=0.05)
    pos = ["top", "bottom"]
    for i, node_embeddings_2d in enumerate(list_embeddings_2d):
        try:
            fig.add_trace(
                go.Scatter(
                    x=node_embeddings_2d.iloc[:, 0].values,
                    y=node_embeddings_2d.iloc[:, 1].values,
                    mode="markers+text",
                    name="markers",
                    hovertemplate="Node id: " + node_embeddings_2d.index + "<br>Class: " + node_embeddings_2d["labels"].astype(str),
                    text=[item.replace("A", "").replace("B", "") for item in list(node_embeddings_2d.index)],
                    textposition="{} center".format(pos[i % len(pos)]),
                        marker=dict(
                        size=8,
                        color=node_embeddings_2d["labels"].values,
                        opacity=0.9,
                        colorscale=list(np.array(colors)[np.unique(node_embeddings_2d["labels"])]), # "Rainbow",
                        line_width=1
                    ),
                ),
                row=math.ceil((i + 1) / cols), col=(i % cols) + 1
            )
        except:
            pass
    fig.update_layout(height=500*rows, width=500*cols, title_text=title_text, showlegend=False, title_x=0.5)
    fig.show()

def visualization_pseudo_cluster_embeddings(list_embeddings_2d, titles=None, title_text="Embeddings", cols=2):
    # cols = 2
    rows = math.ceil(len(list_embeddings_2d) / cols)
    print(rows, cols)
    if not titles:
        titles = []
        for k in range(len(list_embeddings_2d)):
            titles.append("Group {}".format(k + 1))
  
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=titles,
                        horizontal_spacing=0.05, vertical_spacing=0.05)
    pos = ["top", "bottom"]
    for i, node_embeddings_2d in enumerate(list_embeddings_2d):
        node_embeddings_2d["labels"] = set_labels(node_embeddings_2d.index)
        # print(math.ceil((i + 1) / cols), (i % cols) + 1)
        fig.add_trace(
            go.Scatter(
                x=node_embeddings_2d.iloc[:, 0].values,
                y=node_embeddings_2d.iloc[:, 1].values,
                # z=node_embeddings_2d.iloc[:, 1].values,
                mode="markers+text",
                # name="markers",
                ##hovertemplate="Node id: " + str(node_embeddings_2d.index) + "<br>Class: " + node_embeddings_2d["labels"].astype(str),
                ##text=[str(item).replace("A", "").replace("B", "") for item in list(node_embeddings_2d.index)],
                textposition="bottom center",
                marker=dict(
                    size=8,
                    color=node_embeddings_2d["labels"].values,
                    opacity=0.9,
                    ###colorscale=list(np.array(colors)[np.unique(node_embeddings_2d["labels"].values)]), # "Rainbow",
                    line_width=1
                ),
            ),
            row=math.ceil((i + 1) / cols), col=(i % cols) + 1
        )
    print("height:", 500*rows, "width", 500*cols)
    fig.update_layout(height=500*rows, width=500*cols, title_text=title_text, showlegend=False, title_x=0.5)
    fig.show()

def visualization_embeddings(list_embeddings_2d):
    cols = 2
    rows = math.ceil(len(list_embeddings_2d) / cols)

    titles = []
    for k in range(len(list_embeddings_2d)):
        titles.append("Group {}".format(k + 1))

    fig = make_subplots(rows=rows, cols=cols,
                         subplot_titles=titles,
                        horizontal_spacing=0.05, vertical_spacing=0.05)

    for i, node_embedding_2d in enumerate(list_embeddings_2d):
        fig.add_trace(
            go.Scatter(
                x=node_embedding_2d.iloc[:, 0].values,
                y=node_embedding_2d.iloc[:, 1].values,
                mode="markers",
                name="markers",
                text=list(node_embedding_2d.index),
                textposition="bottom center",
                marker=dict(
                    size=6,
                    color=colors[0],
                    opacity=0.9,
                    # colorscale="Rainbow",
                    line_width=1
                ),
            ),
            row=math.ceil((i + 1) / cols), col=(i % cols) + 1
        )
    fig.update_layout(height=500*rows, width=1000,
                           title_text="Embeddings", showlegend=True)
    fig.show()

def get_random_walk(graph, node, n_steps=4):
    random.seed(1)
    # Given a graph and a node, return a random walk starting from the node   
    local_path = [str(node),]
    target_node = node  
    for _ in range(n_steps):
        neighbors = list(nx.all_neighbors(graph, target_node))
        target_node = random.choice(neighbors)
        local_path.append(str(target_node))
    return local_path

def info_graph(graph):
    print(f"N° nodes:\t{graph.number_of_nodes()}")
    print(f"N° edges:\t{graph.number_of_edges()}")
    print(f"Radius:\t\t{nx.radius(graph)}")
    print(f"Diameter:\t{nx.diameter(graph)}")
    print(f"Density:\t{nx.density(graph)}")
    # print(f"Eccentricity: {nx.eccentricity(graph)}")
    # print(f"Center: {nx.center(graph)}")
    # print(f"Periphery: {nx.periphery(graph)}")
    # print(f"Length: {len(graph)}")
    # print(f"Nodes: {sorted(graph.nodes())}")
    # print(f"Edges: {graph.edges()}")
    
def join_sub(df, blocks):
    # blocks= [(start1, end1), (start2, end2), ...]
    sdf = df.iloc[:, blocks[0][0]:blocks[0][1]]
    for block in blocks[1:]:
        sdf = sdf.join(df.iloc[:, block[0]:block[1]])
    return sdf

def transpose(df):
    df = df.T
    df.reset_index(drop=True, inplace=True)
    return df

def build_graph(matrix, threshold=0.5):
    edges = []
    for i in matrix.index:
        for j in matrix.columns:
            if i != j:
                if not math.isnan(matrix[i][j]) and abs(matrix[i][j]) >= threshold:
                    edges.append([i, j])
    return edges

def build_graph_weight_(matrix, threshold=0.5):
    edges = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if not math.isnan(matrix.iloc[i, j]) and abs(matrix.iloc[i, j]) >= threshold:
                edges.append([matrix.index[i], matrix.columns[j], matrix.iloc[i, j]])
    return edges

def build_graph_weight(matrix, threshold=0.5):
    edges = []
    for k, i in enumerate(matrix.index):
        for j in matrix.columns[k + 1:]:
            if not math.isnan(matrix[i][j]) and abs(matrix[i][j]) >= threshold:
                edges.append([i, j, matrix[i][j]])
    return edges

def p_build_graph_weight(matrix, threshold=0.5):
    c = len(matrix)
    edges = pymp.shared.array((c**2, 3), dtype="float")
    index = list(matrix.index)
    columns = list(matrix.columns)

    with pymp.Parallel(24) as p:
        for i in p.range(len(index)):
            for j in range(i + 1, c):
                i_ = index[i]
                j_ = index[j]
                if not math.isnan(matrix[i_][j_]) and abs(matrix[i_][j_]) >= threshold:
                    edges[i * c + j] = [i_, j_, matrix[i_][j_]]
    return edges

def deepwalk(G, num_walk, num_step):
    walk_paths = []
    for node in G.nodes():
        for _ in range(num_walk):
            walk_paths.append(get_random_walk(G, node, n_steps=num_step))
    return walk_paths

def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index2entity)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')

def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(10, 10))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

def detail(model):
    words = list(model.wv.vocab)
    print("Words\t", words)
    words_id = list(model.wv.index2word)
    print("Words id", words_id)
    words_id = list(model.wv.index2entity)
    print("Words id", words_id)
    print("Nodes\t", G.nodes())
    # print("Vector", model.wv["22"])
    print("Vector", model.wv.get_vector("1"))

def silhouette(X, k):
    # Generating the sample data from make_blobs
    # This particular setting has one distinct cluster and 3 clusters placed close
    # together.

    range_n_clusters = range(2, k)

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()

def complete_nodes(df, length):
    indexes = df.index
    for index in range(length):
        if index not in indexes:
            df.loc[index] = [-1]

    df = df.sort_index()  # sorting by index
    return df

def matching():
    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=("Raw data: 111-125", "Raw data: 411-425", 
                                        "Process data: 111-125", "Process data: 411-425"),
                        horizontal_spacing=0.05, vertical_spacing=0.05)

    fig.add_trace(
        go.Scatter(
            x=node_embeddings_2d1[:,0],
            y=node_embeddings_2d1[:,1],
            mode="markers",
            text=labels1,
            textposition="bottom center",
            marker=dict(
                size=8,
                color=labels1,
                opacity=0.9,
                colorscale="Rainbow",
                line_width=0.5
            ),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=node_embeddings_2d2[:,0],
            y=node_embeddings_2d2[:,1],
            mode="markers",
            text=labels2,
            textposition="bottom center",
            marker=dict(
                size=8,
                color=labels2,
                opacity=0.9,
                colorscale="Rainbow",
                line_width=0.5
            ),
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=node_embeddings_2d3[:,0],
            y=node_embeddings_2d3[:,1],
            mode="markers+text",
            text=node_ids3,
            textposition="bottom center",
            marker=dict(
                size=8,
                color=labels3,
                opacity=0.9,
                colorscale="Rainbow", # ["red", "blue"],
                line_width=0.5
            ),
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=node_embeddings_2d4[:,0],
            y=node_embeddings_2d4[:,1],
            mode="markers+text",
            text=node_ids4,
            textposition="bottom center",
            marker=dict(
                size=8,
                color=labels4,
                opacity=0.9,
                colorscale="Rainbow", # ["red", "blue"],
                line_width=0.5
            ),
        ),
        row=2, col=2
    )

    labels1_ = [2] * len(node_ids1)
    size1_ = [8] * len(node_ids1)
    opacity1_ = [0.1] * len(node_ids1)
    for k, node_id in enumerate(node_ids3):
        if node_id in node_ids1:
            index = node_ids1.index(node_id)
            labels1_[index] = labels3[k]
            # size1_[index] = 10
            opacity1_[index] = 0.9

    fig.add_trace(
        go.Scatter(
            x=node_embeddings_2d1[:,0],
            y=node_embeddings_2d1[:,1],
            mode="markers",
            text=node_ids1,
            textposition="bottom center",
            marker=dict(
                size=8, # size1_,
                color=labels1_,
                opacity=opacity1_,
                colorscale="Rainbow", # ["red", "blue", "gray"],
                line_width=0.5
            ),
        ),
        row=3, col=1
    )

    labels2_ = [2] * len(node_ids2)
    size2_ = [8] * len(node_ids2)
    opacity2_ = [0.1] * len(node_ids2)
    for k, node_id in enumerate(node_ids4):
        if node_id in node_ids2:
            index = node_ids2.index(node_id)
            labels2_[index] = labels4[k]
            # size2_[index] = 10
            opacity2_[index] = 0.9

    fig.add_trace(
        go.Scatter(
            x=node_embeddings_2d2[:,0],
            y=node_embeddings_2d2[:,1],
            mode="markers",
            text=node_ids2,
            textposition="bottom center",
            marker=dict(
                size=8, # size2_,
                color=labels2_,
                opacity=opacity2_,
                colorscale="Rainbow", # ["red", "blue", "gray"],
                line_width=0.5
            ),
        ),
        row=3, col=2
    )

    fig.update_layout(height=1000, width=1000, title_text="Clustering Embeddings",
                    showlegend=False)
    fig.show()

def similarity(df_embedding, w1, w2):
    u = df_embedding.loc[w1].to_numpy()
    v = df_embedding.loc[w2].to_numpy()

    similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return similarity

    # similarity(node_embeddings1, '1', '173')
    # model1.wv["798"]

def most_similar(df_embedding, w, topn=10):
    # u = embedding.loc[w].to_numpy() # model1.wv["1"]
    similarities = []
    for index in df_embedding.index:
        similar = similarity(df_embedding, w, index)
        similarities.append((index, similar))
    similarities = np.array(similarities)

    sorted_array = similarities[np.argsort(similarities[:, 1])]
    sorted_array = sorted_array[::-1][1:topn + 1]
    return sorted_array
    # most_similar(node_embeddings1, "1", 10)

# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
def similarity_cos(u, v):
    similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return similarity

def euclidean_distance(u, v):
    dist = np.linalg.norm(u - v)
    return dist

def most_similar2(embedding1, embedding2, topn=10, metric="euclidean"):
    # u = embedding.loc[w].to_numpy() # model1.wv["1"]
    data = []
    for w1 in embedding1.index:
        u = embedding1.loc[w1].to_numpy()
        similarities = []
        for w2 in embedding2.index:
            v = embedding2.loc[w2].to_numpy()
            # similar = euclidean_distance(u, v)
            # similar = similarity_cos(u, v)
            similar = distance.cdist([u], [v], metric=metric)
            similarities.append((int(w1), int(w2), similar[0][0]))

        similarities = np.array(similarities)

        sorted_array = similarities[np.argsort(similarities[:, 2])]
        if metric == "cosine":
            sorted_array = sorted_array[:][:topn]
        else:
            sorted_array = sorted_array[:topn]

        for item in sorted_array:
            data.append(item)
    df = pd.DataFrame(data, columns=["u", "u'", "{}".format(metric)])
    df["u"] = df["u"].astype("int")
    df["u'"] = df["u'"].astype("int")
    return df

def get_epsilon(X_train):
    neigh = NearestNeighbors(n_neighbors=2 * X_train.shape[1])
    nbrs = neigh.fit(X_train)
    distances, indices = nbrs.kneighbors(X_train)

    # Plotting K-distance Graph
    distances = np.sort(distances, axis=0)
    distances_ = distances[:,1]

    i = np.arange(len(distances))
    knee = KneeLocator(i, distances_, S=1, curve='convex', direction='increasing', interp_method='polynomial')

    plt.figure(figsize=(12, 6))
    knee.plot_knee()
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.grid()

    print(distances[knee.knee])