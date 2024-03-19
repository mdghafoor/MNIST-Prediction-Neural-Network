Neural Network learning project using MNIST handwritten numbers dataset and course guidelines from Andrew Ng. 
https://www.coursera.org/specializations/machine-learning-introduction

mnist_id.py script does the following:

    1. Reads a sample MNIST dataset curated at: https://www.kaggle.com/datasets/hojjatk/mnist-dataset 
    and splits into a training set and test set. 
    2. Sets up a 3 layer neural network for handwritten number prediction.
    3. Compiles and fits the model.
    4. Tests the model using the test set and compares predictions with actual value to calculate 
    model accuracy.
    5. Displays a random set of 64 total number images, its actual value, the model's prediction, 
    and the model's accuracy in predicting the actual value. The first 57 images are examples of 
    True readings while the final 8 images are examples of False readings. 


Visual Test Result Sample at a glance: 

![mnist_model_accuracy](https://github.com/mdghafoor/MLProject/assets/158994486/b9a2b870-e42b-4b06-9809-bcb83c7baa24)
 
For more details see:
    
    1. Training set: mnist_test.csv.gz
    2. Test set: mnist_train.csv.gz
    3. Keras model: mnist_model.keras 
    4. Test Results: mnist_model_acuracy.csv and mnist_model_accuracy.csv
