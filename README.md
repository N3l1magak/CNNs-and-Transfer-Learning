# Project2: CNNs and Transfer Learning
Project for CS_6364 Machine Learning

## Description

In this project, a classifer for detecting different breeds of dogs will be built. The dataset will be using has a number of interesting properties:

* There are only about ~150 images per class (breed of dog)
* Some dog breeds look very similar to one another, and therefore are difficult to distinguish
* The images are of dogs in different poses and natural backgrounds, making classification potentially more difficult

## Getting Started
### Part 1: Dataset download and extraction
This dataset comes from the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), where you will find a description of the dataset. 

As you can see from the accuracy graph on that webpage, building a classifier from so few images per class, especially when classes can look similar to one another, is very difficult. To make our lives easier, and to reduce training time, we're going to focus on just three classes from their datase:

* n02109047-Great_Dane
* n02094114-Norfolk_terrier
* n02094258-Norwich_terrier

(A zip file of the dataset has been provided in the repository.)

**Check if using CUDA**

![image](https://user-images.githubusercontent.com/105015948/169910733-5e336b69-9b67-4409-b68a-3357a9272e42.png)


**Create train and holdout folders, where each folder has the three classes as subfolders.**

![image](https://user-images.githubusercontent.com/105015948/169911069-e24f41bf-ba22-47b0-b22b-bf84ce3a2a04.png)

The train, validation, and holdout folders are at the same directory level, as can be seen in the path above. I have chosen 34 images from each breeds to be the holdout set (approximately 20% of total dataset), 17 images from each breeds in train set to be the validation set (approximately 10% of total dataset). All images are selected randomly.

**Define and implement a list of image transformations to be used during training**

![image](https://user-images.githubusercontent.com/105015948/169911306-3dab2c53-310d-4719-8d3f-b2ccae544d8a.png)

For train_transform, a random roation within 45 degree can simulate dog turning head; a colorjitter to change brightness, contrast and saturation of the image for augmentation; resize to 224x224 to satisfy ImgNet; random horizontal flip also for augmentation; and convert to tensor file and normalize at last.

For validation and holdout set, only resize to 224x224 is applied, as they are used for testing the model created.

**Set up DataLoaders for the three folders (train, validation, holdout)**

![image](https://user-images.githubusercontent.com/105015948/169911514-5d6596d5-4391-4ded-9660-52f69fd1367e.png)

### Part 2: Data model setup
#

**Instantiate ResNet152 pre-trained ImageNet model**

![image](https://user-images.githubusercontent.com/105015948/169911652-d4634553-8075-4b57-9bba-3884420c902c.png)


**Freeze/unfreeze the pretrained model layers**

![image](https://user-images.githubusercontent.com/105015948/169912055-4e36bce8-b037-45da-954f-fc931a1c8775.png)


**Replace the head of the model with sequential layer(s) to predict three classes**

![image](https://user-images.githubusercontent.com/105015948/169912118-0c46a66f-b094-48ec-9f28-99149cc1d19c.png)


* I'm using ReLU for the activation function of fc layer. First, it is trivial to implement ReLU comparing to activation functions like sigmoid, which makes it faster; secondly, for CNN, ReLU will be useful for regularization, i.e., it is capable of outputting zero value for negative inputs.
* Dropout is used, because it is an effective technique for regularization and preventing the co-adaptation of neurons, as during training, it randomly zeroes some of the elements of the input tensor with a probability (default 0.5 in this case).
* I'm not using batch normalization, at least not for the step above. If it is asking about the whole model then normalization is needed. Because batch normalization is used to help coordinate the update of multiple layers in the model. Whereas in the step above I am only modifying the last layer (the output classification layer). In fact, as shown above, each layer does have batch normalization implemented (bn1, bn2, bn3, for instance).


**Instantiate an optimizer**

![image](https://user-images.githubusercontent.com/105015948/169912727-ac1568ba-3b95-445e-9ee0-9bf25f5910ea.png)


**instantiate a loss function**

![image](https://user-images.githubusercontent.com/105015948/169912777-58f0b487-1fd6-4098-b706-250cce922e2e.png)


**Places the model on the GPU, if it exists, otherwise using the CPU**

![image](https://user-images.githubusercontent.com/105015948/169912877-01df0f62-aa44-470e-bcaa-335483363369.png)

### Part 3: Training and testing the model
#

![image](https://user-images.githubusercontent.com/105015948/169913003-c0c28c76-c3cd-4e48-becb-bc4e3ab10461.png)

* Set up model to train over 20+ epochs (if GPU) or 2+ epochs (if CPU)
* Set up your model to use batches for training
* Make predictions with the model
* call loss function and back-propagate its results
* Use the optimizer to update weights/gradients
* Record training losses for each epoch
* Set up validation at each epoch
* Record validation losses for each epoch
* Record training and validation accuracies for each epoch

### Part 4: Model evaluation
#

**Graph training versus validation loss**

![image](https://user-images.githubusercontent.com/105015948/169913707-165c1054-22fc-4feb-adb9-9bc5734ea5cf.png)

From the result above, the model is overfitting, as the training loss keep decreasing, but validation loss is fluctuating.

**Possible reasons for under-perform**

* Limited dataset, the number of image is only around 500
* Too many epochs, the model may converge early, and thus the rest of epochs only leads to overfitting
* only "layer4" and "fc" layers are unfrozen, other layers may also need to update

**Possible ways to improve model performance

* using scheduler to control the learning rate of the model after each epoch, depends on the loss calculated
* implement an early stop when the loss won't decrease for a number of epochs
* unfreeze more layers for back propagation

**Graph training versus validation accuracy**

![image](https://user-images.githubusercontent.com/105015948/169913931-0587c0a2-56b3-4cd6-82d0-770f93ceacd0.png)

Even though the dataset is limited, and overfitting during training, the accuracy is actually good (aprox. 90%). So it may be a good model and the result will generalize.

### Before executing ipynb

the following libraries are needed:
* pandas
* Seaborn
* numpy
* matplotlib.pyplot
* sklearn
* torch

## Author

Reynolds_Z @ 2022

