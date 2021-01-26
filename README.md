# Image-Recognition
Google Landmark, Kaggle, 2019. In this competition, it's required to build models that recognize the correct landmark (if any). A 53.3% accuracy rate is achived with a combination Mini Batch k-means with 500 clusters and 3-layer neural network with the number of epochs equal to 5.
	
The image recognition in this project consists of three major parts: feature extraction, representation simplification, and classification. The detailed descriptions of related algorithms are as follows. 
## Feature extraction

The scale-invariant feature transform (SIFT) is a feature algorithm that extracts keypoints
and computes descriptors from images. The algorithm has four steps: scale-space extrema detection, keypoint localization, orientation assignment, and keypoint descriptor. For each image input, SIFT will return 128-dimensional vectors as a descriptor.

## Representation simplification

Bag of words model is a common natural language processing techniques. Here, we consider each image as a document of SIFT “words”. We can now extend the bag-of-words model to classify images instead of text documents. The Mini Batch K-means is adopted as the encoding method.  A detailed algorithm is presented below:

## Classification

### Neural network with GPU, Logistic regression, SVM, Random forest is used to classify.

Part of result is show as below:

![image](../master/Neural_Network_result.png)
![image](../master/accuracy_of_different_active_function.jpg)
![image](../master/confusion_matrix.png)
 

