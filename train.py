#!/usr/bin/env python3
#                                                                        
# author: Abhishek Pandey
# date: 09-08-2020                                   
# description: Train a new network on a dataset and save the model as a checkpoint. Prints out training loss, validation loss, and validation accuracy 
#              as the network trains. Prints out the accuracy on the test set and  the total elapsed runtime after training.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py <data directory> 
#                      --save_path <file path to save the checkpoint> 
#                      --arch <model architecture>
#                      --learning_rate <learning rate>
#                       --epochs <number of training epochs>
#                      --gpu
#   Example call:
#    python train.py flowers --gpu True --epochs 2 --learning_rate 0.001 --arch 'vgg16' --hidden_units '4200, 3000, 1000' --output_units 202

#main imports
import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

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


def getArguments():
    """
    Retrieves and parses the command line arguments created. This function returns these arguments as an
    ArgumentParser object. 
    Parameters:
      None - 
    Returns:
      parse_args() - CLI data structure 
    """
    parser = argparse.ArgumentParser(description='Image classifier')
    
    # Set directory to save checkpoints: python train.py data_dir --save_dir --epochs
    parser.add_argument('data_directory', action="store", type=str, help='location to the image files')
    parser.add_argument('--save_dir', type=str, dest="save_dir", default='.', help='location where checkpont will be saved')
    parser.add_argument('--epochs', type=int, default=20, dest="epochs", help='Number of epochs to run')
    parser.add_argument('--gpu', type=bool, default=False, dest='gpu', const=True, nargs='?', help='Enter training model option. Accepted options include cpu and cuda')
    parser.add_argument('--arch', type=str, action="store", dest="arch", default='vgg16', help='option to select the required architecture from these arch-\'vgg16\', \'densenet121\', \'alexnet\'')
    parser.add_argument('--learning_rate', type=float, action="store", dest="learning_rate", default=0.001, help='Learning rate for model')
    parser.add_argument('--hidden_units', type=str, dest="hidden_units", action="store", default='550', help= "allow user to set Hidden Units, can give multiple with a comma separator, should be less than input unit")
    parser.add_argument('--output_units', type=int, default=202, dest= 'output_units', action="store", help='allow user to set Output size of the network, should be less than hidden unit')

    return parser.parse_args()



# Main program function defined below
def main():
    # Measure total program runtime by collecting start time
    start_time = time.time()
    
    # Create & retrieve Command Line Arugments
    args = getArguments()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)   
    save_dir = os.path.join(args.save_dir, 'checkpoint.pth')
    
    #working on selection of a required model architecture
    input_layers = None
    output_size = None
    
    #selecting model
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    else:
        raise ValueError('Please choose one of \'vgg16\', \'alexnet\' or , \'densenet121\' for parameter arch.')
        
    for param in model.parameters():
        param.requires_grad = False
        
    print(model)
    print(input_size)
    
    #Getting criterion, optimizer & input_size from selected model
    criterion, optimizer, classifier = modelSelection(model, input_size, args.hidden_units, args.output_units, args.learning_rate)
    
    #setting model for cuda/cpu availabilty
    #if args.gpu==True and torch.cuda.is_available():
       # cudaAvl = True
       # model.cuda()
   # else:
      # model.cpu()
    
    #calling other functions
    train_loaders, valid_loaders, test_loaders, training_datasets, validation_datasets, testing_datasets, transform_dict  = data_parser(args.data_directory)
    print(type(training_datasets))
    
    #training function is called 
    training(model, train_loaders, valid_loaders, criterion=criterion, optimizer=optimizer, epochs=int(args.epochs), gpu=args.gpu)
     
    #getting tsting accuracy
    validation_check(model, test_loaders, gpu=args.gpu)
    
    #saving the model to checkpoint as model is now trained
    checkpoint = {'input_size': input_layers,
                  'epochs': args.epochs,
                  'learning_rate':args.learning_rate,
                  'batch_size': 64,
                  'data_transforms': transform_dict['training_transforms'],
                  'model': model,
                  'classifier': classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx':  training_datasets.class_to_idx
                }

    torch.save(checkpoint, 'checkpoint.pth')



#selection of architecture
#using selected network arch ,getting classifier and optimizer
def modelSelection(model, input_size, hidden_units, output_units, learning_rate):
    drop_p = 0.5
    #check which model is selected by user
    print(model)
    #splitting the hidden units if more than one are given
    hidden_units = [int(x) for x in hidden_units.split(',')]
    #appending the output unit to hidden unit
    hidden_units.append(output_units)
    print(hidden_units)
    #confirming if hidden unit is less than input unit
    if input_size < hidden_units[0]:
        raise ValueError('Please make sure hidden unit is lesser than input size that is ' + str(input_size))
    # Add the first layer in model classifier with input unit and 1st element of hidden unit in hidden layers
    hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_units[0])])
    
    # based on number of hidden unit, find out how many layer is required, thus extend those to hidden layers
    layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
    hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    #writing an ordered dictionary
    dictnet = OrderedDict()
    #update the dictnet
    for i in range(len(hidden_layers)):
        dictnet.update({'fc{}'.format(i): hidden_layers[i]})
        
        if i+1 < len(hidden_layers):
            dictnet.update({'relu{}'.format(i): nn.ReLU()})
            dictnet.update({'dropout{}'.format(i): nn.Dropout(p=drop_p)})
    #final item is softmax function on output
    dictnet.update({'output': nn.LogSoftmax(dim=1)})
    
    model.classifier = nn.Sequential(dictnet)
    
    classifier =  model.classifier
    print(classifier)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return criterion, optimizer, classifier
    
    

#data transform function
def data_parser(data_directory):    
        train_dir = data_directory + '/train'
        valid_dir = data_directory + '/valid'
        test_dir = data_directory + '/test'
        
        batch_size = 64
        
        transform_dict = {
            'training_transforms': transforms.Compose([transforms.RandomResizedCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomRotation(30),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225]),
                                                        ]),
            'data_transforms': transforms.Compose( [transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225]),
                                                    ])}
        print(type(transform_dict))

        training_datasets = (datasets.ImageFolder(train_dir, transform=transform_dict['training_transforms']))
        validation_datasets = (datasets.ImageFolder(valid_dir, transform=transform_dict['data_transforms']))
        testing_datasets = (datasets.ImageFolder(test_dir, transform=transform_dict['data_transforms']))
        print(type(training_datasets))

        train_loaders = torch.utils.data.DataLoader(training_datasets, batch_size=64, drop_last=False, num_workers=0)
        valid_loaders =  torch.utils.data.DataLoader(validation_datasets, batch_size=64, drop_last=False, num_workers=0)
        test_loaders =    torch.utils.data.DataLoader(testing_datasets, batch_size=64, drop_last=False, num_workers=0)
        print(type(train_loaders))


        return train_loaders, valid_loaders, test_loaders, training_datasets, validation_datasets, testing_datasets, transform_dict

  
    
#training function    
def training(model, trainLoaders, validLoaders, criterion, optimizer, epochs, gpu):
    print('Training starts')

    startTime = time.time()
    cudaAvl = False
    printEvery = 30
    #assigning Cuda or CPU processing power
    if gpu==True and torch.cuda.is_available():
        cudaAvl = True
        model.cuda()
    else:
        model.cpu()
    # Set model to train mode
    model.train()
    steps = 0
    for epo in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainLoaders):
            inputs, labels = Variable(inputs), Variable(labels)
            steps += 1
            
            if cudaAvl:
                inputs, labels = inputs.cuda(), labels.cuda()
             # Clearing the gradients
            optimizer.zero_grad()
             # feeding forward
            outputs = model.forward(inputs)
             # loss
            loss = criterion(outputs, labels)
            # to backward to calculate the gradients
            loss.backward()
            # Take a step with the optimizer to update the weights
            optimizer.step()
            running_loss += loss.data[0]
            
            if steps % printEvery == 0:
                model.eval()
                accuracy = 0
                validation_loss = 0
                
                with torch.no_grad():
                    for ii, (images, labels) in enumerate(validLoaders):
                        inputs, labels = Variable(images), Variable(labels)
                        if cudaAvl:
                            inputs, labels = inputs.cuda(), labels.cuda()
                        output = model.forward(inputs)
                        validation_loss += criterion(output, labels).data[0]
                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {} / {}.. ".format(epo+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/printEvery),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validLoaders)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validLoaders)))
                running_loss = 0
                #set model back to training
                model.train()
    elapsed_time = time.time() - startTime
    print('Elapsed Time: {:.0f}m {:.0f}s'.format(elapsed_time//60, elapsed_time % 60))
  


#validation function
def validation_check(model, testLoaders, gpu):

    ''' function for do the validation on test datasets
    '''
    print('vaildation starts')
    valid_loss = 0
    accuracy = 0
    # Set model to eval mode 
    model.eval()
    #assigning Cuda or CPU processing power
    if gpu==True and torch.cuda.is_available():
        cudaAvl = True
        model.cuda()
    else:
        model.cpu()
    # gradients turning
    with torch.no_grad():
        for ii, (inputs, labels) in enumerate(testLoaders):
            if cudaAvl:
                inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda()) 
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            #feeding forward
            output = model.forward(inputs)
            #the output is log-softmax -> inverse via exponential to get probabilities
            ps = torch.exp(output).data
            #equality will result in a tensor where 1 is correct prediction and 0 is false prediction
            equality = (labels.data == ps.max(1)[1])
            #equality is a ByteTensor which has no mean method -> convert to FloatTensor
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Testing Accuracy is: {:.3f}".format(accuracy/len(testLoaders)))

    
    
#main function call    
if __name__ == '__main__':
    main()
  