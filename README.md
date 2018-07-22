# ML-GenderClassifier

## Testing out Data Science concepts for Machine Learning
The code uses the scikit-learn machine learning library to train a decision tree on a small dataset of body metrics (height, width, and shoe size) labeled male or female. 
The script test out the different classifers and determine its accuracy. Then the prediction of the gender of someone is made given the set of body metrics.

## Dependencies
* scikit-learn (pip install -U scikit-learn)
* numpy (pip install numpy)
* scipy (pip install scipy)

## Usage
Run the script in the terminal
```
python gender_class.py
```

Change the data set inside the results of each classifer (result_dtc, result_svc, result_pt, result_knn) to test predictions
## Decision Tree Classification
* Used to separate dataset into different classes, based on response variable.
* Used when response variable is categorical in nature.

## Sector Vector Machine Learning Algo
* SVM is a supervised ML algorithm for classification and regression problems.
* The dataset teaches SVM about the classes then it can classify new data.
* When classifying the data into different classes it is finding a line (hyperplane) which separates training data set itno classes.
* SVM offers best classification performance (accuracy) on the training data.
* SVM renders more efficiency for correct classification of the future data.
* The best thing about SVM is that it does not make any strong assumptions on data.
* It does not over-fit the data.
* SVM is commonly used for stock market forecasting by various financial institutions.

## Perceptrons
* Single layer of neural network, usually used to classify data into two parts.
* It is used in supervised learning.
* Consists of 4 layers - Input values or One input layer, Weights and Bias, Net sum, Activation Function
* All inputs are multiplied by their weights, added up to a weighted sum, then applied to the correct Activation Function.
* Activation Function are used to map the input between required values. (a node that you add to the ouput end of the neural network)

## K Nearest Neighbors
* Used for both classification and regression predictive problems.
* 3 important aspects - Ease to interpret output, calculation time, predictive power
* The way it works is if you have two sets of data and you try to figure out what the underlying value will fall in which set.
* The underlying value that is closest to one of two sets will be considered to belong in that set, nearest neighbor.
