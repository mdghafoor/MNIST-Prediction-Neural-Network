# -*- coding: utf-8 -*-
__author__ = 'Muhammad Ghafoor'
__version__ = '1.0.1'
__email__ = "mdhgafoor@outlook.com"

"""
File Name: mnist_id.py
Description: Neural Network learning project using MNIST dataset and 
             course guidelines from Andrew Ng. 
             https://www.coursera.org/specializations/machine-learning-introduction
             
             This script does the following:
              1. Reads a sample MNIST dataset curated at: https://www.kaggle.com/datasets/hojjatk/mnist-dataset 
                 and splits into a training set and test set. 
              2. Sets up a 3 layer neural network for handwritten number prediction.
              3. Compiles and fits the model.
              4. Tests the model using the test set and compares predictions with actual value to calculate 
                 model accuracy.
              5. Displays a random set of 64 total number images, its actual value, the model's prediction, 
                 and the model's accuracy in predicting the actual value. The first 57 images are examples of 
                 True readings while the final 8 images are examples of False readings. 
"""

import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

class MNIST:

    def load_data_set(self):
        """
        Reads and stores the training set and test set into Dataframes. 
        Splits x and y values where x is the pixel data for the handwritten number
        while y is the true value.
        """
        self.train_df = pd.read_csv('mnist/mnist_train.csv.gz', compression='gzip')
        self.test_df = pd.read_csv('mnist/mnist_test.csv.gz', compression='gzip')
        self.y_train = self.train_df.iloc[:,0].to_numpy()
        self.y_test = self.test_df.iloc[:,0].to_numpy()
        self.x_train = self.train_df.iloc[:,1:].to_numpy()
        self.x_test = self.test_df.iloc[:,1:].to_numpy()


    def sequential_model_setup(self):
        """
        Sets up a Keras Sequential Model and Dense layer. The Dense layers have 2 relu activations 
        and 1 linear activation to simluate a Softmax activation, improving numerical stability.

        """
        tf.random.set_seed(81294)
        self.model = Sequential(
            [
                tf.keras.Input(shape=self.x_train[0].shape),
                Dense(25, activation = 'relu', name = 'L1'),
                Dense(15, activation = 'relu', name = 'L2'),
                Dense(10, activation = 'linear', name = 'L3'),
            ],
            name = "mnist_model"
        )


    def compile_model(self):
        """
        Compiles the Keras Sequential Model with Softmax Activation set up. 
        """
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        )
        self.model.fit(self.x_train, self.y_train, epochs=50)


    def load_model(self, model_filename):
        """
        Load a previously saved model.
        Args:
            model_filename: [str] Filename and extension to save the compiled model as. 
            Ex: "mnist_model.keras" 
        """
        self.model = tf.keras.models.load_model(model_filename)


    def save_model(self, model_filename):
        """
        Stores the compiled model.
        Args:
            model_filename: [str] Filename and extension to save the compiled model as. 
            Ex: "mnist_model.keras"
        """
        self.model_name = model_filename.split('.')[0]
        self.model.save(model_filename)


    def test_model(self, save_data=False, filepath=None):
        """
        Tests model by using the test data set's pixel data for handwritten numbers 
        stored in x and comparing to its true value stored in y. The data will be stored in
        a Dataframe where the first column is the prediction, the second column is the 
        true value, the third column is if the prediction matches the true value, and the final
        column stores the overal model accuracy by determining the % of correct predictions.

        Optionally, the user may store this data in a csv file.

        Args:
            save_data: [bool] Flag to determine if test_model data should be stored.
            filepath: [str] Filepath and Filename to save the compiled model as. 
        """
        predictions_headers = ['Prediction', 'True Value', 'Accuracy', 'Model Accuracy']
        self.predictions = pd.DataFrame(columns=predictions_headers)
        self.predictions['Prediction'] = [np.argmax(i) for i in self.model.predict(self.x_test)]
        self.predictions['True Value'] = self.y_test
        self.predictions['Accuracy'] = self.predictions['Prediction']==self.predictions['True Value']
        self.predictions.loc[0, 'Model Accuracy'] = round((self.predictions['Accuracy']==True).mean()*100,2)
        if save_data and filepath:
            self.predictions.to_csv(filepath,header=True,index=False)
        elif (save_data and not filepath) or (filepath and not save_data):
            print("Arguments to properly save data not given.")
            filepath = input("If you would like to save this data, please provide the complete filepath or press enter to continue: ")
            if filepath != '':
                self.predictions.to_csv(filepath,header=True,index=False)


    def visualize_sample_test(self, save_filepath=None, data_filepath=None):
        """
        Create a plot of 64 randomized handwritten number images. The first 57 will display those with True predictions.
        The final 8 will display those with False predictions. Optionally this image can be saved. This function can be ran
        using any test result by providing the data_filepath. 

        Args:
            save_filepath: [str] filepath for where to save image.
            data_filepath: [str] filepath for where test result data is stored.
        """
        if not hasattr(self, 'predictions'):
            self.predictions = pd.read_csv(data_filepath)

        m, n = self.x_test.shape

        fig, axes = plt.subplots(8,8, figsize=(6,6))
        fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.suptitle(f"Prediction Results [Actual, Prediction] | Overall Accuracy: {self.predictions.loc[0, "Model Accuracy"]}")
        
        for i,ax in enumerate(axes.flat):
            random_index = np.random.randint(m)
            if i < 56:
                title_flag = 'T'
                set_color = 'green'
                while self.predictions.loc[random_index, 'Accuracy'] != True:
                    random_index = np.random.randint(m)
                
            else:
                title_flag = 'F'
                set_color = 'red'
                while self.predictions.loc[random_index, 'Accuracy'] != False:
                    random_index = np.random.randint(m)

            X_random_reshaped = self.x_test[random_index].reshape((28,28))
            ax.imshow(X_random_reshaped, cmap='gray')
            ax.set_title(f'{self.y_test[random_index]},{self.predictions.loc[random_index, "Prediction"]},{title_flag}', color = set_color)
            ax.set_axis_off()
        
        if save_filepath:
            fig.savefig(save_filepath)

    
if __name__ == "__main__":
    """
    Example of how to run program above.
    """
    
    #Instantiate class and load data set. 
    mnist = MNIST()
    mnist.load_data_set()

    #Set up, run, compile, fit, and save model. 
    mnist.sequential_model_setup()
    mnist.compile_model()
    mnist.save_model("mnist/mnist_model.keras")
    mnist.test_model(save_data=True, filepath="mnist/mnist_model_accuracy.csv")

    #For use when loading model instead of new run.
    # mnist.load_model("mnist/mnist_model.keras") 

    #Test model, save data, and create visualization.
    mnist.test_model(save_data=True, filepath="mnist/mnist_model_accuracy.csv")
    mnist.visualize_sample_test(save_filepath='mnist/mnist_model_accuracy.png', data_filepath='mnist/mnist_model_accuracy.csv') 

