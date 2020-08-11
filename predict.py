#!/usr/bin/env python3
#                                                                        
# author: Abhishek Pandey
# date: 09-08-2020     
# description: Use a trained network to predict the class for an input image.Prints the most likely classes.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py </path/to/image> <checkpoint>
#                      --top_k <return top K most likely classes> 
#                      --category_names <path to a JSON file that maps the class values to other category names>
#                      --gpu 
#   Example command:
#    python predict.py flowers/test/17/image_03864.jpg checkpoint.pth --category_names cat_to_name.json --top_k 5 --gpu True
##
#main imports
import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from time import time, sleep
import time
from collections import OrderedDict


#import torch
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from PIL import Image



# Main program function defined below
def main():
    # start time
    startTime = time.time()
    
    # Creates & retrieves Command Line Arugments
    args = getArguments()
     
    # Set device to cuda if gpu flag is set
    if args.gpu==True:
        device = 'cuda'
    else:
        device = 'cpu'
    
    # If given, read the mapping of categories to class names
    cat_to_name = {}
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    
    # Load checkpoint and get the model
    model = load_checkpoint(args.checkpoint, args.gpu)
    print(model)
    
    # setting actual class labels convrter on probabilities
    model.idx_to_class = dict([[v,k] for k, v in model.class_to_idx.items()])
  
    # Predict probabilities and  classes
    probs, clas = predict(args.img_path, model, args.top_k, args.gpu)
    print(probs)
    print(clas)
    # Convert categories into real names
    if cat_to_name:
        clas = [cat_to_name[str(cat)] for cat in clas]

    # Print results
    print('\nThe top {} most likely classes are:'.format(args.top_k))
    max_name_len = len(max(clas, key=len))
    row_format ="{:<" + str(max_name_len + 2) + "}{:<.4f}"
    for prob, name in zip(probs, clas):
        print(row_format.format(name, prob))
    
    # verall runtime in seconds & prints it in hh:mm:ss format
    total_time = time.time() - startTime
    print("Total Elapsed Runtime: {:.0f}m {:.0f}s".format(total_time//60, total_time % 60))

    

    #argument parser function
def getArguments():
    """
    Retrieves and parses the command line arguments created. This function returns these arguments as an
    ArgumentParser object. 
    Parameters:
      None - 
    Returns:
      parse_args() - CLI data structure 
    """
    parser = argparse.ArgumentParser()

    # Manditory arguments
    parser.add_argument('img_path', type=str, help='path to input image')
    parser.add_argument('checkpoint', type=str, help='path to a saved checkpoint')
    
    #option arguments
    parser.add_argument('--top_k', type=int, default=3, dest='top_k', help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, dest='category_names', help='path to a JSON file that maps the class values to other category names')
    parser.add_argument('--gpu', type=bool, default=False, dest='gpu', const=True, nargs='?', help='options to include cpu or cuda')

    # return parsed argument collection
    return parser.parse_args()



#Checkpoint loading function
def load_checkpoint(filepath, gpu):
    ''' 
    loads a model, classifier, state_dict and class_to_idx from a torch save
    '''
    if gpu==True:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
        
    return model


       
#defining prediction function
def predict(image_path, model, topk, gpu):  
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # DONE: Implement the code to predict the class from an image file
    image = Image.open(image_path).convert('RGB')
    image = process_image(image_path)
    image = torch.from_numpy(image).unsqueeze_(0).float()
    if gpu==True and torch.cuda.is_available():
        toCuda = torch.device("cuda:0")
        model = model.to(toCuda)
        image = image.to(toCuda)
    else:
        toCuda = torch.device("cpu")
        model.cpu()
        image.cpu()

    model.eval()
    
    # Calculate class probabilities 
    with torch.no_grad():
        outputs = model.forward(image)
    
    # Get topk probabilities and classes
    probs, class_idxs = outputs.topk(topk)
    
    probs, class_idxs = probs.to('cpu'), class_idxs.to('cpu')
    probs = probs.exp().data.numpy()[0]
    class_idxs = class_idxs.data.numpy()[0]
    #print(class_idxs)
    
    # Convert from indices to the actual class labels
    try:
        ## Convert from indices to the actual class labels
        classes = np.array([model.idx_to_class[idx] for idx in class_idxs])
       
    except KeyError:
        print("The key does not exist!")
    
    return probs, classes



# image processing function
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    image = Image.open(image_path)
 
    if image.size[0] > image.size[1]:
        image.thumbnail((4500,256)) 
    else:
        image.thumbnail((256,4500))
    
    left_margin = (image.width -224)/2
    bottom_margin = (image.height -224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    image = image.crop((left_margin,bottom_margin,right_margin,top_margin))
    
    image_new = np.array(image)/225
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    image_new = (image_new - mean)/std
    
    image_new = image_new.transpose((2,0,1))
    
    return image_new
    
    

#main function call
if __name__ == "__main__":

    main()