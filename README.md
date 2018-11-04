# neural_recommender
Movie recommender system based on Deep Factorization Machine DeepFM [1] . It combines matrix factorization and deep neural network to build a hybrid recommender system. The neural network captures the high-order features and the factorization method captures the low-order features. 

Let the input be x , output of the factorization method be y_MF(x) and the output of the neural network be y_NN(x). Then the final prediction is given by,

y_hat(x) = \sigma(y_MF(x) + y_NN(x)), 

where \sigma is the sigmoid activation function. For more details, refer to [1].


[1]: Guo, Huifeng, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. "Deepfm: a factorization-machine based neural network for ctr prediction." arXiv preprint arXiv:1703.04247 (2017).
