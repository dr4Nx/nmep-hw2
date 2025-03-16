# HW 1 Worksheet

---

This is the worksheet for Homework 1. Your deliverables for this homework are:

- [ ] This worksheet with all answers filled in. If you include plots/images, be sure to include all the required files. Alternatively, you can export it as a PDF and it will be self-sufficient.
- [ ] Kaggle submission and writeup (details below)
- [ ] Github repo with all of your code! You need to either fork it or just copy the code over to your repo. A simple way of doing this is provided below. Include the link to your repo below. If you would like to make the repo private, please dm us and we'll send you the GitHub usernames to add as collaborators.

`https://github.com/dr4Nx/nmep-hw2`

## To move to your own repo:

First follow `README.md` to clone the code. Additionally, create an empty repo on GitHub that you want to use for your code. Then run the following commands:

```bash
$ git remote rename origin staff # staff is now the provided repo
$ git remote add origin <your repos remote url>
$ git push -u origin main
```



# Part -1: PyTorch review

Feel free to ask your NMEP friends if you don't know!

## -1.0 What is the difference between `torch.nn.Module` and `torch.nn.functional`?

`Module is a base class for all neural networks. It takes an object oriented approach to models (having built in methods for model management). Functional uses a stateless approach, meaning it doesn't store model parameters or require explicit layer definitions.`

## -1.1 What is the difference between a Dataset and a DataLoader?

`A Dataset simply represents the collection of data points. It does not handle functional behavior like batching, shuffling, etc. DataLoader effectively serves as a wrapper for a Dataset, allowing for batching, shuffling, and more efficient data access.`

## -1.2 What does `@torch.no_grad()` above a function header do?

`It disables gradient computation within a specific function, reducing memory usage (particularly to stop backprop during evals)`



# Part 0: Understanding the codebase

Read through `README.md` and follow the steps to understand how the repo is structured.

## 0.0 What are the `build.py` files? Why do we have them?**

`build.py files are used to build the model and data loaders using the configs. Generally all the config parameters are handled in the build files, which then call the appropriate class to build the model or data loader. They are used and implemented in such a way that the code can be written free of any config dependencies, since they handle the difference`

## 0.1 Where would you define a new model?

`In the /models folder, and then create a new .py for that model.`

## 0.2 How would you add support for a new dataset? What files would you need to change?

`We can support adding a new dataset by adding it into the build.py and datasets.py file in the /data folder.`

## 0.3 Where is the actual training code?

`The training code can be found in main.py`

## 0.4 Create a diagram explaining the structure of `main.py` and the entire code repo.

Be sure to include the 4 main functions in it (`main`, `train_one_epoch`, `validate`, `evaluate`) and how they interact with each other. Also explain where the other files are used. No need to dive too deep into any part of the code for now, the following parts will do deeper dives into each part of the code. For now, read the code just enough to understand how the pieces come together, not necessarily the specifics. You can use any tool to create the diagram (e.g. just explain it in nice markdown, draw it on paper and take a picture, use draw.io, excalidraw, etc.)

### Main
The 'main' function first gets the relevant dataset by using 'build_loader' from data, which builds the data loader and returns train, val, and test.

the 'main' function then uses 'build_model' from /models. build_model will then build/return the selected model. 

main will then build an optimizer using a similar method (calling 'build_optimizer' from optimizer.py). It uses cross entropy loss, and uses a cosine annealing learning rate.

After definitions are done, main now runs 'train_one_epoch' for each epoch as defined. 'train_one_epoch' simply enumrates over the data loader and runs backpropagation and the optimizer on the model. It then calls 'validate' after each epoch, 'validate' being a no grad function that checks the model against the validation dataset.

At the end of training, main calls 'evaluate' which simply checks the end accuracy of the model. 'evaluate' is also a no grad function.


# Part 1: Datasets

The following questions relate to `data/build.py` and `data/datasets.py`.

## 1.0 Builder/General

### 1.0.0 What does `build_loader` do?

`build_loader` simply acquires the relevant dataset as called and returns train, val, and test datasets. to whatever function called it.

### 1.0.1 What functions do you need to implement for a PyTorch Datset? (hint there are 3)

`One must implement __getitem__, __len__, and _get_transforms (as well as __init__ the constructor)`

## 1.1 CIFAR10Dataset

### 1.1.0 Go through the constructor. What field actually contains the data? Do we need to download it ahead of time?

`The self.dataset ultimately contains the dataset. The data was downloaded ahead of time, but it is likely that you can get the data at run time as well.`

### 1.1.1 What is `self.train`? What is `self.transform`?

`self.train simply indicates whether we are generating a training dataset or not. self.transform indicates what changes we make to the data, potentially to make it more robust at test time.`

### 1.1.2 What does `__getitem__` do? What is `index`?

`__getitem__... gets an item from the dataset. index is the index of the image in the dataset.`

### 1.1.3 What does `__len__` do?

`returns the length of the dataset`

### 1.1.4 What does `self._get_transforms` do? Why is there an if statement?

`self._get_transforms returns a transformation to perform on the data. There is an if statement because we do different transformations on train or test data. We perform more transformations on train data, likely to make the model more robust.`

### 1.1.5 What does `transforms.Normalize` do? What do the parameters mean? (hint: take a look here: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

`Normalizes an image based on (first parameter) mean and (second parameter) standard deviation.`

## 1.2 MediumImagenetHDF5Dataset

### 1.2.0 Go through the constructor. What field actually contains the data? Where is the data actually stored on honeydew? What other files are stored in that folder on honeydew? How large are they?

`self.file contains the actual data in the end. The actual data is /honey/nmep/medium-imagenet-nmep-96.hdf5. If the intended folder was /data, those contain many other datasets that are combined many GBs of data. If the intended folder is actually /honey/nmep, there are two other large files (the original zip file and database.db)`

> *Some background*: HDF5 is a file format that stores data in a hierarchical structure. It is similar to a python dictionary. The files are binary and are generally really efficient to use. Additionally, `h5py.File()` does not actually read the entire file contents into memory. Instead, it only reads the data when you access it (as in `__getitem__`). You can learn more about [hdf5 here](https://portal.hdfgroup.org/display/HDF5/HDF5) and [h5py here](https://www.h5py.org/).

### 1.2.1 How is `_get_transforms` different from the one in CIFAR10Dataset?

`It appends transforms rather than straight up adding them.`

### 1.2.2 How is `__getitem__` different from the one in CIFAR10Dataset? How many data splits do we have now? Is it different from CIFAR10? Do we have labels/annotations for the test set?

`It has to get both the item and the label from the file. We do not get labels/annotations for the test set, compared to CIFAR 10 which gives labels for everything.`

### 1.2.3 Visualizing the dataset

Visualize ~10 or so examples from the dataset. There's many ways to do it - you can make a separate little script that loads the datasets and displays some images, or you can update the existing code to display the images where it's already easy to load them. In either case, you can use use `matplotlib` or `PIL` or `opencv` to display/save the images. Alternatively you can also use `torchvision.utils.make_grid` to display multiple images at once and use `torchvision.utils.save_image` to save the images to disk.

Be sure to also get the class names. You might notice that we don't have them loaded anywhere in the repo - feel free to fix it or just hack it together for now, the class names are in a file in the same folder as the hdf5 dataset.

`in visualize.ipynb`


# Part 2: Models

The following questions relate to `models/build.py` and `models/models.py`.

## What models are implemented for you?

`lenet and resnet18`

## What do PyTorch models inherit from? What functions do we need to implement for a PyTorch Model? (hint there are 2)

`Pytorch models inherit from nn.Module. You need a __init__ and a forward function.`

## How many layers does our implementation of LeNet have? How many parameters does it have? (hint: to count the number of parameters, you might want to run the code)

`There are 5 trainable layers. In total, it has around 100k parameters.`



# Part 3: Training

The following questions relate to `main.py`, and the configs in `configs/`.

## 3.0 What configs have we provided for you? What models and datasets do they train on?

`We are provided lenet_base (trained on cifar), resnet18_base (also cifar), and resnet18_medium_imagenet (medium imagenet)`

## 3.1 Open `main.py` and go through `main()`. In bullet points, explain what the function does.

- gets data as defined in config
- gets model from config as defined
- logs info about params, flops, etc.
- builds optimizer as defined
- runs training
- runs validation after every epoc/calculates loss
- runs evaluation after training is complete
- logs information at the end

## 3.2 Go through `validate()` and `evaluate()`. What do they do? How are they different? 
> Could we have done better by reusing code? Yes. Yes we could have but we didn't... sorry...

`validate and evaluate both run the model on evaluate mode. the main difference is that evaluate simply returns predictions, while validate takes those predictions and checks loss as well as accuracy.`


# Part 4: AlexNet

## Implement AlexNet. Feel free to use the provided LeNet as a template. For convenience, here are the parameters for AlexNet:

```
Input NxNx3 # For CIFAR 10, you can set img_size to 70
Conv 11x11, 64 filters, stride 4, padding 2
MaxPool 3x3, stride 2
Conv 5x5, 192 filters, padding 2
MaxPool 3x3, stride 2
Conv 3x3, 384 filters, padding 1
Conv 3x3, 256 filters, padding 1
Conv 3x3, 256 filters, padding 1
MaxPool 3x3, stride 2
nn.AdaptiveAvgPool2d((6, 6)) # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
flatten into a vector of length x # what is x?
Dropout 0.5
Linear with 4096 output units
Dropout 0.5
Linear with 4096 output units
Linear with num_classes output units
```

> ReLU activation after every Conv and Linear layer. DO **NOT** Forget to add activatioons after every layer. Do not apply activation after the last layer.

## 4.1 How many parameters does AlexNet have? How does it compare to LeNet? With the same batch size, how much memory do LeNet and AlexNet take up while training? 
> (hint: use `gpuststat`)

1456M for AlexNet vs 272M for LeNet.
57.82324 million parameters for AlexNet vs 99 thousand for LeNet.


## 4.2 Train AlexNet on CIFAR10. What accuracy do you get?

Report training and validation accuracy on AlexNet and LeNet. Report hyperparameters for both models (learning rate, batch size, optimizer, etc.). We get ~77% validation with AlexNet.

> You can just copy the config file, don't need to write it all out again.
> Also no need to tune the models much, you'll do it in the next part.

Training accuracy: ~99.9%
Validation accuracy: ~77.5%

Config:
AUG:
  COLOR_JITTER: 0.4
DATA:
  BATCH_SIZE: 256
  DATASET: "cifar10"
  IMG_SIZE: 70
  NUM_WORKERS: 32
  PIN_MEMORY: True
MODEL:
  NAME: alexnet
  NUM_CLASSES: 200
  DROP_RATE: 0.5
TRAIN:
  EPOCHS: 50
  WARMUP_EPOCHS: 10
  LR: 3e-4
  MIN_LR: 3e-5
  WARMUP_LR: 3e-5
  LR_SCHEDULER:
    NAME: "cosine"
  OPTIMIZER:
    NAME: "adamw"
    EPS: 1e-8
    BETAS: (0.9, 0.999)
    MOMENTUM: 0.9
OUTPUT: "output/alexnet_cifar"
SAVE_FREQ: 5
PRINT_FREQ: 99999
PRINT_FREQ: 99999



# Part 5: Weights and Biases

> Parts 5 and 6 are independent. Feel free to attempt them in any order you want.

> Background on W&B. W&B is a tool for tracking experiments. You can set up experiments and track metrics, hyperparameters, and even images. It's really neat and we highly recommend it. You can learn more about it [here](https://wandb.ai/site).
> 
> For this HW you have to use W&B. The next couple parts should be fairly easy if you setup logging for configs (hyperparameters) and for loss/accuracy. For a quick tutorial on how to use it, check out [this quickstart](https://docs.wandb.ai/quickstart). We will also cover it at HW party at some point this week if you need help.

## 5.0 Setup plotting for training and validation accuracy and loss curves. Plot a point every epoch.

`PUSH YOUR CODE TO YOUR OWN GITHUB :)`

## 5.1 Plot the training and validation accuracy and loss curves for AlexNet and LeNet. Attach the plot and any observations you have below.

`YOUR ANSWER HERE`

## 5.2 For just AlexNet, vary the learning rate by factors of 3ish or 10 (ie if it's 3e-4 also try 1e-4, 1e-3, 3e-3, etc) and plot all the loss plots on the same graph. What do you observe? What is the best learning rate? Try at least 4 different learning rates.

`YOUR ANSWER HERE`

## 5.3 Do the same with batch size, keeping learning rate and everything else fixed. Ideally the batch size should be a power of 2, but try some odd batch sizes as well. What do you observe? Record training times and loss/accuracy plots for each batch size (should be easy with W&B). Try at least 4 different batch sizes.

`YOUR ANSWER HERE`

## 5.4 As a followup to the previous question, we're going to explore the effect of batch size on _throughput_, which is the number of images/sec that our model can process. You can find this by taking the batch size and dividing by the time per epoch. Plot the throughput for batch sizes of powers of 2, i.e. 1, 2, 4, ..., until you reach CUDA OOM. What is the largest batch size you can support? What trends do you observe, and why might this be the case?
You only need to observe the training for ~ 5 epochs to average out the noise in training times; don't train to completion for this question! We're only asking about the time taken. If you're curious for a more in-depth explanation, feel free to read [this intro](https://horace.io/brrr_intro.html). 

`YOUR ANSWER HERE`

## 5.5 Try different data augmentations. Take a look [here](https://pytorch.org/vision/stable/transforms.html) for torchvision augmentations. Try at least 2 new augmentation schemes. Record loss/accuracy curves and best accuracies on validation/train set.

`YOUR ANSWER HERE`

## 5.6 (optional) Play around with more hyperparameters. I recommend playing around with the optimizer (Adam, SGD, RMSProp, etc), learning rate scheduler (constant, StepLR, ReduceLROnPlateau, etc), weight decay, dropout, activation functions (ReLU, Leaky ReLU, GELU, Swish, etc), etc.

`YOUR ANSWER HERE`



# Part 6: ResNet

## 6.0 Implement and train ResNet18

In `models/*`, we provided some skelly/guiding comments to implement ResNet. Implement it and train it on CIFAR10. Report training and validation curves, hyperparameters, best validation accuracy, and training time as compared to AlexNet. 

`YOUR ANSWER HERE`

## 6.1 Visualize examples

Visualize a couple of the predictions on the validation set (20 or so). Be sure to include the ground truth label and the predicted label. You can use `wandb.log()` to log images or also just save them to disc any way you think is easy.

`YOUR ANSWER HERE`


# Part 7: Kaggle submission

To make this more fun, we have scraped an entire new dataset for you! 🎉

We called it MediumImageNet. It contains 1.5M training images, and 190k images for validation and test each. There are 200 classes distributed approximately evenly. The images are available in 224x224 and 96x96 in hdf5 files. The test set labels are not provided :). 

The dataset is downloaded onto honeydew at `/data/medium-imagenet`. Feel free to play around with the files and learn more about the dataset.

For the kaggle competition, you need to train on the 1.5M training images and submit predictions on the 190k test images. You may validate on the validation set but you may not use is as a training set to get better accuracy (aka don't backprop on it). The test set labels are not provided. You can submit up to 10 times a day (hint: that's a lot). The competition ends on __TBD__.

Your Kaggle scores should approximately match your validation scores. If they do not, something is wrong.

(Soon) when you run the training script, it will output a file called `submission.csv`. This is the file you need to submit to Kaggle. You're required to submit at least once. 

## Kaggle writeup

We don't expect anything fancy here. Just a brief summary of what you did, what worked, what didn't, and what you learned. If you want to include any plots, feel free to do so. That's brownie points. Feel free to write it below or attach it in a separate file.

**REQUIREMENT**: Everyone in your group must be able to explain what you did! Even if one person carries (I know, it happens) everyone must still be able to explain what's going on!

Now go play with the models and have some competitive fun! 🎉
