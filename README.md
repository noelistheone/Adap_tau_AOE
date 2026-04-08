## AOE: Adversarial Optimized bias Elimination for embedding-based Recommender System

## Dependencies
- pytorch==1.11.0
- numpy==1.21.5
- scipy==1.7.3
- torch-scatter==2.0.9

## Training model:
- mkdir log
- cd bash
### MF
#### amazon-beauty
```
# Adap_tau_0
bash Adap_tau_novel.sh amazon-beauty 1e-3 1e-7 3 1024 2048 nopdrop 1.0 1.0 uniform_gpu 1 100 cosine mf weight_v0 1 0.1 Adap_tau_Loss
# Adap_tau
bash Adap_tau_novel.sh amazon-beauty 1e-3 1e-7 3 1024 2048 nopdrop 1.0 1.5 uniform_gpu 1 100 cosine mf weight_mean 1 0.05 Adap_tau_Loss
```
#### iFashion
```
# Adap_tau_0
bash Adap_tau_novel.sh iFashion 1e-3 1e-5 3 1024 2048 drop 0.8 1.0 uniform_gpu 0 100 cosine mf weight_v0 1 0.1 Adap_tau_Loss
# Adap_tau
bash Adap_tau_novel.sh iFashion 1e-3 1e-7 3 1024 2048 drop 0.8 0.1 uniform_gpu 0 100 cosine mf weight_mean 1 0.5 Adap_tau_Loss
```


### LightGCN
#### amazon-beauty
```
# Adap_tau_0
bash Adap_tau_novel.sh amazon-beauty 1e-3 1e-1 3 1024 2048 drop 1.0 1.0 no_sample 1 100 nocosine lgn weight_v0 1 0.5 Adap_tau_Loss
# Adap_tau
bash Adap_tau_novel.sh amazon-beauty 1e-3 1e-1 3 1024 2048 nopdrop 1.0 1.5 no_sample 0 100 nocosine lgn weight_mean 1 0.5 Adap_tau_Loss
```
#### iFashion
```
# Adap_tau_0
bash Adap_tau_novel.sh iFashion 1e-3 1e-1 3 1024 2048 nopdrop 0.8 1.0 no_sample 0 100 nocosine lgn weight_v0 1 1.0 Adap_tau_Loss
# Adap_tau
bash Adap_tau_novel.sh iFashion 1e-3 1e-1 3 1024 2048 nopdrop 0.8 1.5 no_sample 1 100 nocosine lgn weight_mean 1 1.0 Adap_tau_Loss
```
