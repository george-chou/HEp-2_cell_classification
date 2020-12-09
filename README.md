# HEp-2_cell_classification

Classifying Cell Images Using Deep Learning Models

[![license](https://img.shields.io/github/license/george-chou/HEp-2_cell_classification.svg)](https://github.com/george-chou/HEp-2_cell_classification/blob/master/LICENSE)
[![Python application](https://github.com/george-chou/HEp-2_cell_classification/workflows/Python%20application/badge.svg)](https://github.com/george-chou/HEp-2_cell_classification/actions)
[![Github All Releases](https://img.shields.io/github/downloads-pre/george-chou/HEp-2_cell_classification/v1.1/total)](https://github.com/george-chou/HEp-2_cell_classification/releases)

## Aims and Background

### Aims

This project aims to classify cell images using deep CNN models.
 
Our task consists of five steps: 
1)	Construct our cell image classification model using AlexNet framework; 
2)	Initialise the weights in the pre-trained AlexNet for our model; 
3)	Modify the number of neurons in the output layer in our model into six classes; 
4)	Train our model for a few epochs; 
5)	Classify the test cell images using the fine-tuned model.
 
The training set contains 8,701 images, the validation set includes 2,175 cell images, and the test set contains 2,720 cell images. Besides, there is a label file containing ID and category of the overall 13,596 cell images. These are all our experiment materials from the international competition on cell image classification hosted by the International Conference on Pattern Recognition in 2014 (ICPR2014).

<div align=center>
<b>Table 1: The usage of three sorts of dataset</b><br>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/t1.PNG"/>
</div>

### Background

The human epithelial 2 (HEp-2) cells are epithelial cells of carcinoma of the larynx human. They are used for laboratory diagnostics for the detection of autoimmune antibodies and antinuclear antibodies. This specific type of cells represents the substrate of choice for the search for autoimmune antibodies with the indirect immunofluorescence technique due to their tumour nature.

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f0.png"/><br>
<b>Figure 0: HEp-2 example cell images of ICPR2014 dataset</b>
</div>

Efficient image classification of HEp-2 cells can facilitate the diagnosis of many autoimmune diseases. That is why we make a classification for HEp-2 cells. However, image classification is a heavy workload and requires high accuracy. Manual search is unrealistic; traditional image segmentation algorithms are not enough to meet the demand. Therefore, the use of a deep CNN-based image segmentation algorithm is more appropriate.

## Methodology

The main technical methods used in the whole experiment are listed below:

### Python

Python is a cross-platform programming language, as well as a high-level scripting language that combines interpretation, compilation, interactivity and object-oriented, which was initially designed by shell coders. With the continuous update of the version and the addition of new language features, the more it is used for the development of independent and large-scale projects.

Python’s package management, which is a collection of modules, can provide us with functions written by others. For example, the API of the CSV module implements the reading and writing of form data in CSV format; its mean syntax is as follow:

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f1.PNG"/><br>
<b>Figure 1: Syntax of csv module in Python</b>
</div>

### Deep Learning

Deep learning is a sort of AI function that mimics the way the human brain worked when processing data used to detect objects, recognise speech, translate language and make decisions. Deep learning AI can learn unsupervised from unstructured and unlabelled data. The process of deep learning is equivalent to the operation of regression. A neural network is a model that contains a large number of undetermined parameters. The learning process is to fill in these uncertain parameters. This process needs to refer to the backpropagation algorithm.
 
The backpropagation algorithm works by calculating the gradient of the loss function for each weight through a chain rule, one slope at a time, iterating backwards from the last level to avoid redundant calculation of intermediate terms in the chain rule. The steps to perform a deep learning model is as follow: 
 
1)	Design layer structures; 
2)	Set up deep learning layers; 
3)	Define loss function and optimiser; 4) Train the model with training data; 5) Predict the test data.

### CNN

The CNN stands for the convolutional neural network, which is a kind of feedforward neural network that includes convolution computing and has a fully connected system. As one of the usual methods in deep learning, CNN has representative learning capabilities and can do the shift-invariant classification of input information due to its hierarchical structure, so they are also named shift-invariant artificial neural networks.

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f2.png"/><br>
<b>Figure 2: The architecture of original CNN</b>
</div>

The training steps of CNN model are as below: 
 
1)	Input the training data into the model; 
2)	Obtain the result from the current model; 
3)	Calculate the loss with loss functions; 
4)	Gain gradients by backpropagating the loss; 
5)	Update model weights with gradient vectors and pre-defined learning rate.

### Pytorch

PyTorch is an open-source library for Python machine learning, based on Torch, C++ implements the bottom layer, and it plays an essential role in AI fields, such as natural language processing. It was initially developed by Facebook’s AI research team and used in Uber’s probabilistic programming software Pyro. PyTorch has two main features: Tensor calculation like NumPy, which can be accelerated by GPU;
 
To install it via pip, we can use the command line generator on Pytorch official index (https://pytorch.org/get-started/locally/) to automatically create the installation command line and copy it into the terminal of our system. Directly using ‘pip install pytorch’ or ‘pip3 install --upgrade alexnet_pytorch’ may cause pip to return errors.

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f3.png"/><br>
<b>Figure 3: The command line generator on Pytorch official website</b>
</div>

### AlexNet

AlexNet took part in the ImageNet Large-Scale Visual Recognition Challenge on 30 September 2012. The network's top 5 errors were 15.3%, 10.8 percentage points lower than the second place. The main result of the original paper was that the depth of the model was critical to its high performance and that the model was computationally expensive but feasible due to the use of a GPU in the training process.

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f4.png"/><br>
<b>Figure 4: The structure of the AlexNet</b>
</div>

The steps to finetune the pre-trained AlexNet on Pytorch are as follow: 
 
1)	Construct the cell image classification model; 
2)	Initialise the weights of the model from the pre-trained model; 
3)	Modify class number to six; 
4)	Train the model with preferred epochs; 
5)	Justify the accuracy of the model by test sets.

### Data Augmentation

Data augmentation is like an image pre-processing in the training phase. When our dataset is not enough for training, and we cannot find more images to supplement our dataset, data augmentation is a good way to expand our dataset. This process is like the practice questions for students are not enough, change the existing questions and continue to train the students. There are many data augmentation methods, such as:

1)  Crop, rotation, flip, hue, saturation, exposure, aspect; 
2)  Mix-up; 
3)	Cut Mix; 
4)	Mosaic; 
5)  Blur;

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f5.png"/><br>
<b>Figure 5: All kinds of data augmentation methods</b>
</div>

But we need to pay attention to maintaining the correctness of the answer when performing data augmentation, or our model would learn the wrong knowledge.

## Results

### Dataset loading and pre-processing

In this section, we finish the dataset loading and pre-processing task.
 
First, we want to use the “ImageFolder” function to finish the load the dataset and finish the transform process. Therefore, due to the limitation of “ImageFolder” function, we must put all train, validation and test images to different folders that use their label name to name the file name. Therefore, we can use the package from “csv”, “shutil” and “os” to move images to the folder which correspond to their labels.

We use the “csv.reader” function to read the csv file which has the information of the images’ file names and labels. Then we use the “os.path.exist” function to search the filename and follow the first row in the csv file. Finally, we use the “shutil.move” function to move the files to their corresponding files. Overall, we finish the pre-process of using “ImageFolder” function.

Then we use pytorch function to load and move dataset to dataloader of model.

We use the “transforms.Compose” function to conclude several augmentation functions that we use in this task. For example, using the “Resize” to change the image size. This is because the initial image size is small which cannot provide enough information for the image classification process. In addition, we use the “CenterCrop” to cut image to square input due to the demand of the AlexNet model. The “Normalize” function is to finish the normalization process and can decrease the contrast issue of initial data. Furthermore, the “RandomAffine” function is used to increase the amount and randomicity of input data.

Then we use the “DataLoader” function to make the dataset to several load groups. The batch size is used to control the number of datasets in one loader group. According to the experiment experience, too small batch size will lead to the fluctuating training process, which is hard to train. In addition, too large batch size will increase the memory of GPU and lead to the decline of model generalization ability. Therefore, we apply batch size to 4 in this task.

### Deep learning model

Then we develop the AlexNet model and the whole code is shown below:

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f6.png"/><br>
<b>Figure 6: Code of the AlexNet</b>
</div>

First we use the “torch.hub.load” function to load the pretrained AlexNet. Then due to the classification target of this task being 6, we must change the fully connected layers. We must use the “require_grad=False” function to set the returned grad to none. This is because the pretrained model has their own grad which we do not know. In addition, the change of model architecture will also lead to the change of grad. Therefore, in order to change some parts of the pretrained net, we want to freeze some parts of the model and cannot return the grad again. So, we have to set the returned grad to “False”. Then we rewrite the classifier layer of the AlexNet and add one dropout layer to avoid the overfitting, one activation function (“Relu”) to increase non-linear and one linear layer to change the final output channels to finish the classify task. We can use the “model.eval()” to evaluate the whole model, which is shown as follow:

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f7.png"/><br>
<b>Figure 7: The model structure of the AlexNet</b>
</div>

We can see that the final output channels are changed from 1,000 to 6.

### Training and evaluation process

Then we set some basic functions of the training process such as loss function, optimizer function, learning rate schedule and GPU CUDA setting. The whole code is shown below:

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f8.png"/><br>
<b>Figure 8: Code of some functions in training process</b>
</div>

We use the cross-entropy loss in this task. This is because this cross-entropy loss has been approved to perform well in image classification tasks. Then we use the “SGD” optimizer to optimize the model. It is important that we only change the parameters of the classifier layer, this is because the pretrained model already trained well for the initial convolutional layer. Then we use the “ReduceLROnPlateau” function to decrease the learning rate to one tenth if the loss hasn't changed too much for 5 epochs. This function requires us to input the loss to finish. Finally, we put model, data and criterion to GPU to increase the training speed.

Then we develop the training and evaluation function of this task, which is shown as follow:

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f9.png"/><br>
<b>Figure 9: Code of training function</b>
</div>

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f10.png"/><br>
<b>Figure 10: Code of evaluation function</b>
</div>

For the training process, we first get the input data and move them to GPU. Then we use this model to train the input image and receive output results. After this process, we use loss function to calculate the loss by predicting output and ground truth label and move this loss backwards for the optimizer function. After several iterations, we collect the average loss in order to draw the loss curve. In addition, we run evaluation functions to evaluate the training results by accuracy. In the final of one epoch training, we use the schedule function to change the learning rate for the next epoch.

For the evaluation process, we input the image and use the model to finish the classification process and receive one classification result. Then we compare this predicted result with the ground truth label to calculate the accuracy. We need to evaluate the training dataset accuracy and validation dataset accuracy in the training process. The main purpose of collecting validation accuracy is to avoid overfitting.

Then we use the “matplot” function to plot the loss curve and accuracy curve. The code is shown below:

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f11.png"/><br>
<b>Figure 11: Code of plot image</b>
</div>

Then we can receive the loss curve and show it in the following picture:

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f12.png"/><br>
<b>Figure 12: Loss curve</b>
</div>

Although the loss is fluctuating during the whole process, the main trend of this curve is in decreasing.

Then we plot the accuracy curve of training and validation data loader, which is shown as follow:

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f13.png"/><br>
<b>Figure 13: Training and validation accuracy</b>
</div>

We can find that the training and validation accuracy are increasing at beginning and levelling off after about 20 epochs. In addition, the accuracy curve of training and validation dataset are separated at about 20 epochs, this is because of the influence of overfitting.

Then we test the accuracy on the test loader and get 92% accuracy, which is shown below:

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/f14.png"/><br>
<b>Figure 14: Accuracy of test</b>
</div>

## Discussion and Conclusion

### Discussion

Transfer learning: we can use the pretrained model and change some part of it to improve the training performance. This is because this pretrained model has trained in large dataset such as ImageNet. Therefore, this model has a stronger learning ability. In addition, we need to save the model parameters within the training process in order to evaluate it and retrain this task.
 
Overfitting: due to the small dataset of our input images, the overfitting problem will become the most effective issue. The best way to decrease the influence of overfitting is using the “Dropout” function during the fully connected layer.
 
Learning rate: according to our experiment, the large learning will study fast in the beginning. However, the large learning rate will easily lead training accuracy to not change too much after several epochs. In addition, too small a learning rate still cannot receive better performance. Therefore, we can use schedule function to change learning from large to small to receive better performance.

### Conclusion

In conclusion, we finish the dataset establishing process, model establishing process and model training and evaluating process. We finally received a test accuracy of 92% using the following parameters:

<div align=center>
<b>Table 2: Experiment parameters</b><br>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/t2.PNG"/>
</div>

### Future work

We need to fine-tune the topology structure of AlexNet network to find more accurate parameters for the application scenario of HEp-2 cell image classification but finding the network structure most adaptive for this application scenario requires a lot of experiments. It is a task requiring a considerable workload. To improve search efficiency, using auto-machine learning to explore more optimised network structure is an efficient option. That is to use a small part of the data set to test the network performance under a specific topological structure, convolution kernel size and other parameters. When we obtain the locally optimal solution after multiple trials, then we can use the extensive data set for training.
 
We can also try other CNN frameworks as the base network to fine-tune structure and look for a more accurate method for HEp-2 cell image classification. No matter from the aspect of accuracy or time cost, any small optimisation is an improvement. The brief orientations for labour focus on the classifier, activation functions, loss functions, topology structure of convolutional layers, etc. For example, we can change the FC layer into SVM as the classifier and compare the performance of the two kinds of the classifier. We can also change Cross-entropy loss function into smooth L1 loss and make a comparison. Besides, we can change the current activation function into Mish function and make a comparison.
