# CoupledGNN
This repository is an implementation of our proposed CoupledGNN model in the following paper:

```
Qi Cao, Huawei Shen, Jinhua Gao, Bingzheng Wei, Xueqi Cheng. 2020. Popularity Prediction on Social Platforms 
with Coupled Graph Neural Networks. In WSDM'20, February 3-7, 2020, Houston, TX, USA, 9 pages.
```

The CoupledGNN model solves the *network-aware popularity prediction* problem, capturing the cascading effect explicitly by two coupled graph neural networks.

For more details, you can download this paper [Here](https://arxiv.org/abs/1906.09032)

## Requirements

Python 2.7.5

Tensorflow 1.14.0

## Usage
***Example Usage***

`python -u train.py --dataset=artificial1 --learning_rate=5e-4 --graph_learning_rate=5e-5 --n_layers=3 `

For detailed description of all parameters, you can run

`python -u train.py --help`

## Cite
Please cite our paper if you use this code in your own work:
```
@inproceedings{cao2020coupledgnn,
  title={Popularity Prediction on Social Platforms with Coupled Graph Neural Networks},
  author={Cao, Qi and Shen, Huawei and Gao, Jinhua and Wei, Bingzheng and Cheng, Xueqi},
  booktitle={Proceedings of the 13th ACM International Conference on Web Search and Data Mining},
  series={WSDM'20},
  year={2020},
  location={Houston, TX, USA},
  numpages={9}
}
```










