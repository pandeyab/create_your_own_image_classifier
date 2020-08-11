# Udacity DS ND program, Deep learning - Image Classifier Project

Project code.

In this project, 1st step to develop code for an image classifier built with PyTorch, then convert it into a python application which will run from command line.

This app will 1st train the deep learning model on data set of images, then use the trained model to classify images.

Install
This project requires Python 3.x and the following Python libraries installed:

NumPy
pytorch
Pandas
matplotlib
scikit-learn
You will also need to have software installed to run and execute an iPython Notebook

We recommend to install Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

Code

This project contains three files:

*image_classifier.ipynb: This is the main file where all works description is given.
*train.py: a python code to train the model
*predict.py: a python code which will be used in prediction



Executions:
*notebook image_classifier.ipynb -->
In a terminal or command window, navigate to the top-level project directory image_classifier/ (that contains this README) and run one of the following commands:

ipython notebook image_classifier.ipynb
or

jupyter notebook image_classifier.ipynb

*train.py -->
$  python train.py flowers --gpu True --epochs 2 --learning_rate 0.001 --arch 'vgg16' --hidden_units '4200, 3000, 1000' --output_units 202

*predict.py -->
$ python predict.py flowers/test/17/image_03864.jpg checkpoint.pth --category_names cat_to_name.json --top_k 5 --gpu True


Data

*cat_to_name.json: a json file containing names 
*checkpoint.pth :saved checkpoint location

