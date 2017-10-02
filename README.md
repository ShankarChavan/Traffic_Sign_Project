# Traffic Sign Recognition

[//]: # (Image References)

[image3]: ./Plots/test.png "test hist"
[image1]: ./Plots/training.png "train hist"
[image2]: ./Plots/validation.png "valid hist"
[image4]: ./Plots/softmax_prob_img.png "softmax prob"
[image5]: ./web_images/No_entry.png "No Entry"
[image6]: ./web_images/keep_left.png "Keep Left"
[image7]: ./web_images/keep_right.png "Keep Right"
[image8]: ./web_images/turn_right_ahead.svg.png "Turn Right"
[image9]: ./web_images/turn_left_ahead.png "Turn Left"


The task was to buid a traffic sign recognition convolution nueral network.

Following were the tools used to build the project.
* python 3
* CV2
* pandas
* tensorflow
* numpy
* sklearn

The goal / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
*****
Link for the [project code](https://github.com/ShankarChavan/Traffic_Sign_Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Dataset Summary and Exploration

##### 1.Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python and numpy to calculate some the summary statistics of the traffic
sign data set

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in training,test and validation set.

![alt text][image1]
![alt text][image2]
![alt text][image3]




### Design and Test a Model Architecture
#### 1. Pre-Processing

**Step 1. Grayscale**
 As a first step, I decided to convert the images to grayscale because as per the paper *Sermanet and LeCun* too have tried it on their images and achieved great results.I have also tried with RGB but the error was too high.
 
 Here is an example of a traffic sign image before and after grayscaling.
 
 **Step 2. Normalize**
 In the second step I normalized data, before normalizing mean of the training data was around 82 but after normalizing mean got shifted to -0.35.
Normalizing bought the data between scale of -1 and 1 and thus it helped the algorithm learn faster because the features will be given equal weights and distribution of data will not be wider.

 **Step 2. Shuffle**
 In the 3rd step shuffling was done in order to avoid highly correlated batches and biasness of the network.
 
#### 2. Model Architecture 
 
 My final model consisted of following layers 
 
 | Layer          |Description|
 |----------------|:----------| 
 |Input           | 32x32x1 Grayscale Image  |
 |Convolution 5x5 |1x1 stride, Valid padding, outputs 28x28x6|
 |RELU            |           |
 |Max pooling  2x2|2x2 stride, outputs 14x14x6|
 |Convolution 5x5 |1x1 stride, Valid padding, outputs 10x10x16|
 |RELU            |           |
 |Max pooling  2x2|2x2 stride, outputs 5x5x16|
 |Flatten	      |outputs 400|
 |Fully Connected Layer|outputs 120|
 |RELU            |           |	
 |Fully Connected Layer|outputs 84|
 |RELU           |	         |
 |Dropout	     |50% Keep probability|
 |Fully Connected Layer|outputs 43   |
 
#### 3. Hyperparamters used to train model

To train the model, I used the following hyperparameters:

* preprocessed images
* Adam optimizer
* Learning reate 0.0009
* Batch Size 128
* Epochs 15
* Dropout Keep Probability 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Answer:
1. Preprocessed the data by using grayscale and normalizing
2. Trained withe Lenet architecture as problem seems to be similar to image classfification which they described in their paper.Initially the acuracy was very low under 90% so I added the dropout layer,tried with different learning rate but **0.0009** gave good result.Also added dropout in the layer 1 and layer 2 but they didn't helped much.
3. Keep _probability of 50% was giving good results compare to 70% and 60%.

### Test a Model on New Images
#### 1.Here are the 5 german traffic signs I found on internet

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

#### 2. Here are the results of the prediction:

| Image | Prediction |
| ----- | :---------:|
| Turn Right Ahead | Turn Right Ahead |
| Keep Left | Keep Left |
| Turn Left Ahead | Speed limit (30 km/h) |
| Keep Right | Keep Right |
| No entry | No entry |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
![alt text][image4]
