# RecSys-SAS-team
HSE RecSys course seminar

#### Article: [Neural Collaborative Filtering vs. Matrix Factorization Revisited](https://arxiv.org/pdf/2005.09683.pdf)

We explored the above article, revisited the experiments, and proposed some modifications to speed up the training process.

#### Technical issues:
Requires the code and datasets from https://github.com/hexiangnan/neural_collaborative_filtering (this code assumes a Python 2 runtime)

Example how to run experiment 1:
```
python Experiment1_train.py \
  --mf_n_components 16 \
  --gmf_n_components 8 \
  --loss_reduction mean \
  --optimizer Adam
 ```
  
The 2-nd experiment could be run directly in the notebook.
