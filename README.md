# SILVA
This is a PyTorch implementation of the paper: A Selective Multi-View Representation
Augmentation Framework for Crystal Property Estimation.
`@author:Haomin Yu, Jilin Hu, Chenjuan Guo, Bin Yang`

## Requirements
- arrow==0.15.4
- ase==3.22.1
- bresenham==0.2.1
- dgl==0.6.1
- dgl_cu110==0.6.1
- dill==0.3.6
- geopy==1.20.0
- ipdb==0.13.9
- matbench==0.6
- matplotlib==3.1.1
- MetPy==1.3.1
- networkx==2.7.1
- numpy==1.22.4
- p_tqdm==1.4.0
- pandas==1.4.2
- pydantic==1.10.4
- pymatgen==2022.11.7
- PyYAML==6.0
- scikit_learn==1.0.1
- scipy==1.7.3
- sympy==1.10.1
- timm==0.4.12
- torch==1.13.1
- torch_cluster==1.6.0
- torch_geometric==2.2.0
- torch_scatter==2.0.9
- torch_sparse==0.6.16
- torchvision==0.14.1
- tqdm==4.64.0


## Model Training
The models have been trained on the CIF files from the eight databases.
To execute the pre-trained model, use the following command:
```python
 python silva_train.py
```
The parameters for the  model can be modified in the config.yaml


## References 

In this project, we have referred to and utilized the following resources:

1. https://github.com/txie-93/cdvae - Parts of the code or ideas in this project were based on this.
2. https://github.com/RishikeshMagar/Crystal-Twins - Parts of the code or ideas in this project were based on this.
3. https://github.com/topics/graph-isomorphism-network - Parts of the code or ideas in this project were based on this.
