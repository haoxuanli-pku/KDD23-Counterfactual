# This is the code of KDD-23 paper "Who should be Given Incentives? Counterfactual Optimal Treatment Regimes Learning for Recommendation".

## Datasets
We provide our training data for Yelp and ML-1M in the corresponding folder, named "yelp.rating" and "ratings.dat" respectively. However, the scale of the KuaiRec dataset is too large, so we provide the dataset link here: https://github.com/chongminggao/KuaiRec.

## To run the code
The main code is in the matrix_factorization file. Our PyTorch version is 1.9.1 + cu111. To reconstruct our experiment results, please first run the "construct_data.ipynb" file, then run the "evaluate_policy.ipynb" file.

## Citation
lf you use our code, please kindly cite:
~~~
@inproceedings{li2023counterfactual,
 Author = {Haoxuan Li and Chunyuan Zheng and Peng Wu and Kun Kuang and Yue Liu and Peng Cui},
 Booktitle = {KDD},
 Title = {Who should be Given Incentives? Counterfactual Optimal Treatment Regimes Learning for Recommendation},
 Year = {2023}}
~~~
