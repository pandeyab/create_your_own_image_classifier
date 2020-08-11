#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Import workspace utils
#from workspace_utils import keep_awake

#main imports
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
#from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
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

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets

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

train_loaders = torch.utils.data.DataLoader(training_datasets, batch_size=64, shuffle=True, drop_last=False, num_workers=0)
valid_loaders =  torch.utils.data.DataLoader(validation_datasets, batch_size=64, shuffle=True, drop_last=False, num_workers=0)
test_loaders =    torch.utils.data.DataLoader(testing_datasets, batch_size=64, shuffle=True, drop_last=False, num_workers=0)
print(type(train_loaders))


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[5]:


# TODO: Build and train your network
#building neural network with densenet networks
model = models.vgg16(pretrained=True)
print(model)
print(type(model))
#next cell will be used for further model building and training


# In[6]:


#feeezeing paramater, here tensor dont require gradient to fine tune the pretrained network, avoid backdrop
for param in model.parameters():
    param.requires_grad = False

#updating classifier to give names to layers
classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088, 4096)),
                                       ('ReLu',nn.ReLU()),
                                       ('dropout', nn.Dropout(p=0.5)),
                                       ('fc2',nn.Linear(4096, 202)),
                                       ('output',nn.LogSoftmax(dim=1))
                                       ])
                          )
model.classifier = classifier

#save other things such as the mapping of classes to indices
#model.class_to_idx = training_datasets.class_to_idx

#defining criterion
criterion = nn.NLLLoss()

#defining  optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


# In[7]:


# Define training function
def training(criterion, optimizer, epochs):
    
    ''' training function for model
    '''
    print('Training starts')
    
    startTime = time.time()
    cudaAvl = False
    printEvery = 25
    #assigning Cuda or CPU processing power
    if torch.cuda.is_available():
        cudaAvl = True
        model.cuda()
    else:
        model.cpu()
    # Set model to train mode
    model.train()
    steps = 0
    for epo in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loaders):
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
                    for ii, (images, labels) in enumerate(valid_loaders):
                        inputs, labels = Variable(images), Variable(labels)
                        if cudaAvl:
                            inputs, labels = inputs.cuda(), labels.cuda()
                        #feeding forward
                        output = model.forward(inputs)
                        #calculating validation loss
                        validation_loss += criterion(output, labels).data[0]
                        #the output is log-softmax -> inverse via exponential to get probabilities
                        ps = torch.exp(output).data
                        #equality will result in a tensor where 1 is correct prediction and 0 is false prediction
                        equality = (labels.data == ps.max(1)[1])
                        #equality is a ByteTensor which has no mean method -> convert to FloatTensor
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {} / {}.. ".format(epo+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/printEvery),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(valid_loaders)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loaders)))
                running_loss = 0
                #set model back to training
                model.train()
    elapsed_time = time.time() - startTime
    print('Elapsed Time: {:.0f}m {:.0f}s'.format(elapsed_time//60, elapsed_time % 60))


# In[8]:


epochs = 15
#training function is called 
training(criterion=criterion, optimizer=optimizer, epochs=epochs)


#  ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[9]:


# TODO: Do validation on the test set
def validation_check(model, dataloader, criterion):
    
    ''' function for do the validation on test datasets
    '''
    print('vaildation starts')
    valid_loss = 0
    accuracy = 0
    # Set model to eval mode 
    model.eval()
   #assigning Cuda or CPU processing power
    if torch.cuda.is_available():
        cudaAvl = True
        model.cuda()
    else:
        model.cpu()
    # gradients turning
    with torch.no_grad():
        for ii, (inputs, labels) in enumerate(dataloader):
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
    #return valid_loss/len(dataloader.dataset), accuracy/len(dataloader.dataset)
    print("Testing Accuracy is: {:.3f}".format(accuracy/len(dataloader)))
#getting tsting accuracy
validation_check(model, test_loaders, criterion)


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[10]:


# TODO: Save the checkpoint 
def saving_checkpoint(model):
    
    ''' saves the checkpoint of model.
    '''
    #saving mapping of classes to indices 
    model.class_to_idx = training_datasets.class_to_idx
    
    checkpoint = {'network':"vgg16",
                  'input': 25088,
                  'otput': 1000,
                  'batch_size': 64,
                  'classifier': model.classifier,
                  'class_to_idx':model.class_to_idx,
                  'state_dict': model.state_dict()
                 }
    torch.save(checkpoint,'checkpoint.pth')

#function call
saving_checkpoint(model) 


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[11]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
def loading_checkpoint(filepath):
    ''' Loads a checkpoint and rebuilds the model.
    '''
    checkpoint = torch.load(filepath)
    #verify the NN
    if checkpoint['network'] == 'vgg16':
        print("network is recognized")
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("network is not recognized")
    #setting the mapping of classes to saved indices
    model.class_to_idx = checkpoint['class_to_idx']
    #ordering the classifier
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

#calling checkpoint load function
model = loading_checkpoint('checkpoint.pth')
print(model)
print(type(model))


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[12]:


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    image = Image.open(image_path)
 
    if image.size[0] > image.size[1]:
        image.thumbnail((5000,256)) 
    else:
        image.thumbnail((256,5000))
    
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


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[13]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[14]:


with Image.open('flowers/test/14/image_06052.jpg') as image:
    plt.imshow(image)


# In[15]:


#processed image
imshow(process_image("flowers/test/14/image_06052.jpg"));


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[16]:


# Set device to cuda if available
if torch.cuda.is_available():
        cudaAvl = True
        toCuda = torch.device("cuda:0")
        #model.cuda()
else:
       toCuda = torch.device("cpu")

# setting actual class labels convrter on probabilities
idx_to_class = dict([[v, k] for k, v in training_datasets.class_to_idx.items()])
model.idx_to_class = idx_to_class

#defining prediction function
def predict(image_path, model, topk=5):  
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # DONE: Implement the code to predict the class from an image file
    image = Image.open(image_path).convert('RGB')
    image = process_image(image_path)
    image = torch.from_numpy(image).unsqueeze_(0).float()
    
    #model = model.to(device)
    #image = image.to(device)
    if cudaAvl:
        model = model.to(toCuda)
        image = image.to(toCuda)
    else:
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
    classes = np.array([model.idx_to_class[idx] for idx in class_idxs])
    #print(classes)
    
    return probs, classes

# Predict probabilities and classes
probs, classes = predict('flowers/test/10/image_07090.jpg', model)

print(probs)
print(classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[17]:


# TODO: Display an image along with the top 5 classes

#init a empty figure plot
plt.figure(figsize=(7,11))
#assigning subplot dim
plot_1 = plt.subplot(2,1,1)

#processing an image
image = process_image('flowers/test/14/image_06052.jpg')
flower_title = cat_to_name['15']

#plotting image
imshow(image,plot_1,title=flower_title)
flower_names = [cat_to_name[i] for i in classes]

plt.subplot(2,1,2)
#plotting barplot
sns.barplot(x=probs,y=flower_names,color=sns.color_palette()[0])
plt.show()


# ## convert notebook to html
# !!jupyter nbconvert *.ipynb

# In[18]:


get_ipython().getoutput('jupyter nbconvert *.ipynb')


# In[ ]:




