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
The file *src/main.m* contains an usage example of the neural network. The problem the network wants to solve is the digit recognition, using the [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/). The network is able to reach a level of **95%** of accuracy on test set.