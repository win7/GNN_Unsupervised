from matplotlib.patches import ConnectionPatch # for plotting matching result
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt # for plotting
import matplotlib.colors as mcolors
import networkx as nx # for plotting graphs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
from torch_geometric.utils import to_networkx

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

class CustomDataset(Dataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]

        # Extract nodes and edges
        x = torch.tensor(data['nodes'], dtype=torch.float)
        edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()

        # Extract labels
        y = torch.tensor(data['labels'], dtype=torch.long)

        # Create PyTorch Geometric Data object
        return Data(x=x, edge_index=edge_index, y=y)

def plot_graphs(A1, A2):
    print("222222")
    plt.figure(figsize=(16, 8))
    print("222222333")
    G1 = nx.from_numpy_array(np.array(A1))
    G2 = nx.from_numpy_array(np.array(A2))

    print("123")
    G1 = nx.relabel_nodes(G1, mapping1)
    G2 = nx.relabel_nodes(G2, mapping2)
    
    plt.subplot(1, 2, 1)
    plt.title('Graph 1')
    nx.draw_networkx(G1, font_color="w")
    plt.subplot(1, 2, 2)
    plt.title('Graph 2')
    nx.draw_networkx(G2, font_color="w")
    return G1, G2

def similarity_matrix(X, G1, G2):
    df_matrix = pd.DataFrame(data=X, index=list(G1.nodes()), columns=list(G2.nodes()))
    # df_matrix = df_matrix.style.apply(highlight_max, color='red')
    return df_matrix

def plot_similarity_matrix(df_matrix):
    """ fig = make_subplots(rows=1, cols=1, subplot_titles=["Similarity Matrix"])

    fig.add_trace(
        go.Heatmap(x=df_matrix.columns, y=df_matrix.index, 
        z=df_matrix.values, colorscale="Viridis_r"), row=1, col=1
    )

    fig.update_layout(height=700, width=700, title_text="Similarity Matrix")
    fig.show() """

    fig = px.imshow(df_matrix,)
    fig.update_xaxes(side="top")
    fig.update_layout(height=700, width=700, title_text="Similarity Matrix")
    fig.show()

def plot_match(G1, G2, df_matrix):
    colors = list(mcolors.TABLEAU_COLORS)

    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(1, 2, 1)
    plt.title('Graph 1')
    nx.draw_networkx(G1, pos=initialpos, font_color="w")
    ax2 = plt.subplot(1, 2, 2)
    plt.title('Graph 2')
    nx.draw_networkx(G2, pos=initialpos, font_color="w")

    node_match = df_matrix.idxmax(axis="columns")
    node_match = list(zip(node_match.index, node_match.values))

    for k, item in enumerate(node_match):
        # print(item)
        con = ConnectionPatch(xyA=initialpos[item[0]], xyB=initialpos[item[1]], coordsA="data", coordsB="data",
                            axesA=ax1, axesB=ax2, color=colors[k % len(colors)])
        plt.gca().add_artist(con)
        if item[0] == item[1]:
            print("Match\t\t", item[0], item[1])
        else:
            print("Unmatch\t", item[0], item[1])
FFF = 6
N1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
N1_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
A1 = [
    #a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  r
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # a
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # b
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # c
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # d
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # e
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # f
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # g
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0], # h
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # i
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], # j
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], # k
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # l
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], # m
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], # n
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], # o
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # p
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # q
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # r
]
# A1 = torch.tensor(A1)
E1 = [
    ('a', 'b'),
    ('a', 'i'),
    ('b', 'c'),
    ('b', 'e'),
    ('b', 'j'),
    ('c', 'd'),
    ('c', 'f'),
    ('c', 'h'),
    ('d', 'f'),
    ('d', 'g'),
    ('e', 'g'),
    ('f', 'g'),
    ('f', 'h'),
    ('f', 'p'),
    ('g', 'r'),
    ('h', 'm'),
    ('h', 'p'),
    ('i', 'm'),
    ('i', 'n'),
    ('j', 'k'),
    ('j', 'l'),
    ('k', 'l'),
    ('m', 'n'),
    ('n', 'o'),
    ('o', 'p'),
    ('q', 'r')
]
E1_ = [
    (0, 1), 
    (0, 8), 
    (1, 2), 
    (1, 4), 
    (1, 9), 
    (2, 3), 
    (2, 5), 
    (2, 7),
    (3, 5), 
    (3, 6), 
    (4, 6), 
    (5, 6), 
    (5, 7), 
    (5, 15), 
    (6, 17), 
    (7, 12), 
    (7, 15), 
    (8, 12), 
    (8, 13), 
    (9, 10), 
    (9, 11), 
    (10, 11),
    (12, 13), 
    (13, 14), 
    (14, 15), 
    (16, 17)
]


N2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't']
N2_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
A2 = [
    #a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  s  t
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # a
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # b
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # c
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # d
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # e
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # f
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # g
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # h
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # j
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], # k
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # l
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # m
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # n
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], # o
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # p
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # q
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # s
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # t
]
# A2 = torch.tensor(A2)
E2 = [
    ('a', 'b'),
    ('a', 'i'),
    ('a', 's'),
    ('b', 'c'),
    ('b', 'e'),
    ('c', 'd'),
    ('c', 'f'),
    ('c', 'h'),
    ('c', 'i'),
    ('c', 'm'),
    ('d', 'f'),
    ('d', 'g'),
    ('e', 'g'),
    ('e', 'l'),
    ('f', 'g'),
    ('g', 'q'),
    ('h', 'o'),
    ('j', 'k'),
    ('j', 'l'),
    ('k', 'l'),
    ('m', 'n'),
    ('n', 'o'),
    ('o', 'p'),
    ('s', 't')
]
E2_ = [
    (0, 1),
    (0, 8), 
    (0, 17), 
    (1, 2), 
    (1, 4), 
    (2, 3), 
    (2, 5), 
    (2, 7), 
    (2, 8), 
    (2, 12), 
    (3, 5), 
    (3, 6), 
    (4, 6), 
    (4, 11), 
    (5, 6), 
    (6, 16), 
    (7, 14), 
    (9, 10), 
    (9, 11), 
    (10, 11), 
    (12, 13), 
    (13, 14), 
    (14, 15), 
    (17, 18)
]


nodelist1 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r"]
mapping1 = dict(zip(range(len(nodelist1)), nodelist1))

nodelist2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "s", "t"]
mapping2 = dict(zip(range(len(nodelist2)), nodelist2))

initialpos = {
    "a": (0, 0),
    "b": (-1, -1),
    "c": (1, -1),
    "d": (0, -2),
    "e": (-1, -3),
    "f": (1, -3),
    "g": (0, -4),
    "h": (2, -2),
    "i": (2, 0),
    "j": (-2, -1),
    "k": (-3, -2),
    "l": (-2, -3),
    "m": (2, -1),
    "n": (3, -1),
    "o": (3, -2),
    "p": (2, -3),
    "q": (-1, -5),
    "r": (1, -5),
    "s": (-1, 0),
    "t": (-2, 0)
}

initialpos_ = {
    0: (0, 0),
    1: (-1, -1),
    2: (1, -1),
    3: (0, -2),
    4: (-1, -3),
    5: (1, -3),
    6: (0, -4),
    7: (2, -2),
    8: (2, 0),
    9: (-2, -1),
    10: (-3, -2),
    11: (-2, -3),
    12: (2, -1),
    13: (3, -1),
    14: (3, -2),
    15: (2, -3),
    16: (-1, -5),
    17: (1, -5),
    18: (-1, 0),
    19: (-2, 0)
}

"""
Instructions

# Step 1
#---
import sys
sys.path.append('../')
import test_match as tm
#---

# Step 2
#---
A1 = torch.tensor(tm.A1)
A2 = torch.tensor(tm.A2)
#---

# Step 3
#---
G1, G2 = tm.plot_graphs(A1, A2)
#---

# Step 4
# Appy algorithm

# Step 5
#---
df_matrix = tm.similarity_matrix(X, G1, G2)
df_matrix
#---

# Step 6
#---
tm.plot_similarity_matrix(df_matrix)
#---

# Step 7
#---
tm.plot_match(G1, G2, df_matrix)
#---
"""

