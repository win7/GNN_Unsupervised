# GNN Unsupervised

### Input format dataset
A file in .csv with follow struture:

| Id  | X1.1 | ... | X1.n | X2.1 | ... | X2.m | X3.1 | ... | X3.p | Y1.1 | ... | Y1.q | Y2.1 | ... | Y2.r | Y3.1 | ... | Y3.s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m2 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m5 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m12 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m17 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m22 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m23 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| ... | ... | ... | ... |... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Files

1. format_input: format input dataset according to above table 
2. preprocessing:and create graph data (nodes + edges)
3. node-embeddings: generate node-embeddings with DGI or VGAE
4. processing: generate edge-embeddings and outlier detection

## To-Do
- Dynamic edge embeddings operator
- Parallel edge2vec operator

df_edges.to_csv("output/{}/baseline/common_edges/common_edges_{}_{}.csv".format(exp, group, method), index=False)
nx.write_gexf(SG, "output/{}/baseline/common_edges/common_edges_{}_{}.gexf".format(exp, group, method))

df_edges = pd.read_csv("output/{}/baseline/common_edges/common_edges_{}_{}.csv".format(exp, group, method))
df_edges.head()