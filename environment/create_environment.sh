#!/bin/bash

pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

#pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
#pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
#pip install torch-sparse==0.6.13 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
#pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html

pip install torch-cluster==1.6.0 torch-scatter==2.0.9 torch-sparse==0.6.13 torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html

pip install torch-geometric
