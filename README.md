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

1. preprocessing: format input dataset and create graph data (nodes + edges)
2. node-embeddings: generate node-embeddings with DGI or VGAE
3. processing: generate edge-embeddings and outlier detection



## To-Do
- Dynamic edge embeddings operator