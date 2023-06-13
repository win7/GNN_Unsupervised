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


1. format_input: format input dataset according to above table (ok)
2. preprocessing: create graph data (nodes, edges) (ok)
3. node-embeddings: generate node-embeddings with DGI or VGAE (ok)
4. processing: generate edge-embeddings and outlier detection (ok)
5. baseline: Greedy algorithm for get maximun common subgraph (ok)
6. comparation: compare baseline (Greedy) with GNN (DGI, VGAE) on data variation
7. change_detection
8. 

Aditional Files
1. synthetic_graphs:
3. 
## To-Do
- Dynamic edge embeddings operator
- Parallel edge2vec operator
- improve mapping idx with id

## Notes
- exp1 MS GNN Greedy (experimet 1) (this for manuscript)
- exp2 MS GNN Greedy (experimet 2)
- exp3 Syntectic GNN Greedy (test)
- exp4 Syntectic GNN Greedy (n=1000, p=0.5)
- exp5 Syntectic GNN Greedy (n=1000, p=0.3) (this for manuscript)