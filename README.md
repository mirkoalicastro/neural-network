# Neural Network
Neural network is a MATLAB project which implements a feed-forward full-connected multilayered network.
### What is implemented?
  - Principal Component Analysis
  - Resilient Batch learning 
  - Gradient Batch learning
  - Stopping criterion ( [Early Stopping --- but when?](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf))
  - Plot of errors on training set and on validation set, during learning epoches
  - Confusion matrix on test set
### Example
The file *src/main.m* contains an usage example of the neural network. The problem the network wants to solve is the digit recognition, using the [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/). The network is able to overcame a level of **96%** of accuracy on test set.
#### Resilient Batch learning
In particular, with the parameters (setted as in main.m), it has reached about 96.2% of accuracy with the resilient batch learning.

![alt text](https://raw.githubusercontent.com/mirkoalicastro/neural-network/master/demo/resilientbatch.jpg)
#### Gradient Batch learning
Moreover, removing the stopping criterion, using sigmoid instead of relu6 and using the gradient batch learning the network has reached about 95.3% of accuracy.

![alt text](https://raw.githubusercontent.com/mirkoalicastro/neural-network/master/demo/gradientbatch.jpg)
