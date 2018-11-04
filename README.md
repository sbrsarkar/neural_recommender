# neural_recommender
Movie recommender system based on Deep Factorization Machine, DeepFM [1] . It combines matrix factorization and deep neural network to build a hybrid recommendation system. The neural network captures the high-order features and the factorization method captures the low-order features. 



Let the input be x , output of the factorization method be y<sub>MF</sub>(x) and the output of the neural network be y<sub>NN</sub>(x). Then the final prediction is given by,

<img src="http://latex.codecogs.com/gif.latex?\widehat{y}(x)&space;=&space;\sigma(y_{MF}(x)&space;&plus;&space;y_{NN}(x))" title="\widehat{y}(x) = \sigma(y_{MF}(x) + y_{NN}(x))" />

where <img src="http://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /> is the sigmoid activation function. For more details, refer to [1].


## training
To train the network, run the following command in terminal
```
python train.py 
```
It will train the network using default settings, show the training loss figure and save the trained model as 'model.pt'.

<img src="https://github.com/sbrsarkar/neural_recommender/blob/master/loss.png" alt="training loss" width="450" height="350">

training mse: 0.01361, testing mse: 0.05956

## model architecture
<img src="https://github.com/sbrsarkar/neural_recommender/blob/master/model.png" alt="training loss" width="450" height="350">








[1]: Guo, Huifeng, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. "Deepfm: a factorization-machine based neural network for ctr prediction." arXiv preprint arXiv:1703.04247 (2017).
