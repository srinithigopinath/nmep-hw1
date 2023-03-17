# HW 1 Worksheet

---

This is the worksheet for Homework 1. Your deliverables for this homework are:

- [ ] This worksheet with all answers filled in. If you include plots/images, be sure to include all the required files. Alternatively, you can export it as a PDF and it will be self-sufficient.
- [ ] Kaggle submission and writeup (details below)
- [ ] Github repo with all of your code! You need to either fork it or just copy the code over to your repo. A simple way of doing this is provided below. Include the link to your repo below. If you would like to make the repo private, please dm us and we'll send you the GitHub usernames to add as collaborators.

`https://github.com/srinithigopinath/nmep-hw1`

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

`torch.nn.Module are Python classses which define Module classes. torch.nn.functional uses a stateless approach. nn.Module uses network architectures like layers and connections and functions provide functions like Relu.`

## -1.1 What is the difference between a Dataset and a DataLoader?

`A Dataset will store samples and also the labels associated with the samples. DataLoaders will wrap our dataset objects and batch our data for us.`

## -1.2 What does `@torch.no_grad()` above a function header do?

`@torch.no_grad() above a function header is a decorator that makes sure a graph isn't built when we don't want to compute any gradients. This saves us some computation.`



# Part 0: Understanding the codebase

Read through `README.md` and follow the steps to understand how the repo is structured.

## 0.0 What are the `build.py` files? Why do we have them?**

`The build.py files are used to build the model and data loaders using the configs. We usually handle all parameters in the build files. They allow us to write code free of any config dependencies`

## 0.1 Where would you define a new model?

`You would define a new model in the models folder and also modify the build.py in models. You might also have to make a new config.`

## 0.2 How would you add support for a new dataset? What files would you need to change?

`You can add support for a new dataset in the data folder under datasets.py. This is where dataset loaders are defined. You would also modify the config file`

## 0.3 Where is the actual training code?

`The actual training code is in main.py. This contains the main training loop.`

## 0.4 Create a diagram explaining the structure of `main.py` and the entire code repo.

Be sure to include the 4 main functions in it (`main`, `train_one_epoch`, `validate`, `evaluate`) and how they interact with each other. Also explain where the other files are used. No need to dive too deep into any part of the code for now, the following parts will do deeper dives into each part of the code. For now, read the code just enough to understand how the pieces come together, not necessarily the specifics. You can use any tool to create the diagram (e.g. just explain it in nice markdown, draw it on paper and take a picture, use draw.io, excalidraw, etc.)

`YOUR ANSWER HERE`



# Part 1: Datasets

The following questions relate to `data/build.py` and `data/datasets.py`.

## 1.0 Builder/General

### 1.0.0 What does `build_loader` do?

`build_loader begins by checking the dataset type. It assigns dataset_train, dataset_val, and dataset_test certain values depending on the type of dataset it is. Afterwards, it create three dataLoader objects using the DataLoader class. These are data_loader_train, data_loader_val, and data_loader_test. It sets up a modified pyTorch Datasets and Dataloaders.`

### 1.0.1 What functions do you need to implement for a PyTorch Datset? (hint there are 3)

`The functions you need to implement a PyTorch Dataset are __getitem__, __len__, and _get_transforms`

## 1.1 CIFAR10Dataset

### 1.1.0 Go through the constructor. What field actually contains the data? Do we need to download it ahead of time?

`The field that actually contains the data is self.dataset. We don't need to download it ahead of time because the dataset will download from the internet if the download boolean is set to true. `

### 1.1.1 What is `self.train`? What is `self.transform`?

`self.train is a boolean value that decides where the dataset is used for training. If this is the case, colorJitter and RandomHorizontalFlip is added to the transform. This is defaultly set to True. self.transform gets the transforms and is what is the image is set to.`

### 1.1.2 What does `__getitem__` do? What is `index`?
`__getitem__ first takes the item in the data set at index, then transforms it according to _get_transforms, and returns a tuple of (transformed_image, label).`

### 1.1.3 What does `__len__` do?

`It returns the length of the dataset.`

### 1.1.4 What does `self._get_transforms` do? Why is there an if statement?
`_get_transforms sets up a chain of transformations to the image using the torchvision.transforms as well as torchvision.transforms.Compose. There is an if statement because it transforms the image differently based on whether or not it is training data. If it is training data, it first adds some randomness to the image using ColorJitter and RandomHorizontalFlip. For both training data and other data, it then converts the image into a tensor, normalizes it to some specific means and standard deviations, and doubles the image size.`

### 1.1.5 What does `transforms.Normalize` do? What do the parameters mean? (hint: take a look here: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

`transforms.Normalize normalizes a tensor given a mean and standard deviation. Because each channel needs a specific mean and standard deviation, the number of inputs it takes in are num_channels*2.`

## 1.2 MediumImagenetHDF5Dataset

### 1.2.0 Go through the constructor. What field actually contains the data? Where is the data actually stored on honeydew? What other files are stored in that folder on honeydew? How large are they?

`The field that contains the data is the filepath. The filepath is str = "/data/medium-imagenet/medium-imagenet-nmep-96.hdf5". This means that is stored in the data folder under medium-imagenet as an hdf5 file. An HDF5 is a data model, and file format for managing data. finish....`

> *Some background*: HDF5 is a file format that stores data in a hierarchical structure. It is similar to a python dictionary. The files are binary and are generally really efficient to use. Additionally, `h5py.File()` does not actually read the entire file contents into memory. Instead, it only reads the data when you access it (as in `__getitem__`). You can learn more about [hdf5 here](https://portal.hdfgroup.org/display/HDF5/HDF5) and [h5py here](https://www.h5py.org/).

### 1.2.1 How is `_get_transforms` different from the one in CIFAR10Dataset?

`self._get_transforms creates an array of transformations to the image. It then appends a normalization and a resizing to the tensor that is created. The the split value is "train" and if self.augment is set to true, a horizontal flip and color jitter will also be applied to transforms. Finally, the transforms are composed.`

### 1.2.2 How is `__getitem__` different from the one in CIFAR10Dataset? How many data splits do we have now? Is it different from CIFAR10? Do we have labels/annotations for the test set?

`__getitem__ is different than the one in the CIFAR10Dataset because now we are using data splits. The image is taken from the file and we use an if statement to check if the split is "test". If it isn't there is a split, and if it is, then the label gets changed to a value of -1. The annotation for the test set is "test". Similar to CIFAR, a tuple  of image, label is returned.`

### 1.2.3 Visualizing the dataset

Visualize ~10 or so examples from the dataset. There's many ways to do it - you can make a separate little script that loads the datasets and displays some images, or you can update the existing code to display the images where it's already easy to load them. In either case, you can use use `matplotlib` or `PIL` or `opencv` to display/save the images. Alternatively you can also use `torchvision.utils.make_grid` to display multiple images at once and use `torchvision.utils.save_image` to save the images to disk.

Be sure to also get the class names. You might notice that we don't have them loaded anywhere in the repo - feel free to fix it or just hack it together for now, the class names are in a file in the same folder as the hdf5 dataset.

`YOUR ANSWER HERE`


# Part 2: Models

The following questions relate to `models/build.py` and `models/models.py`.

## What models are implemented for you?

`The models that are implemented are lenet and resnet18.`

## What do PyTorch models inherit from? What functions do we need to implement for a PyTorch Model? (hint there are 2)

`PyTorch models inherit from nn.Module. We need to implement __init__ and forward. `

## How many layers does our implementation of LeNet have? How many parameters does it have? (hint: to count the number of parameters, you might want to run the code)

`YOUR ANSWER HERE`



# Part 3: Training

The following questions relate to `main.py`, and the configs in `configs/`.

## 3.0 What configs have we provided for you? What models and datasets do they train on?

The provided configs are "lenet_base.yaml", "resnet18_base.yaml", and "resnet18_medium_imagenet.yaml". They train on the cifar10, cifar10, and medium_imagenet data sets respectively.

## 3.1 Open `main.py` and go through `main()`. In bullet points, explain what the function does.

1. config file -->  3 datasets (dataset_ train/val/test) + 3 data loaders (data_loader_ train/val/test)
2. sets up correct device
3. counts parameters and flops and logs them in the millions
4. defines optimizer and loss function 
5. If the model has already started training (MODEL.RESUME = True), it checks the validation accuracy. Then , if the model is in eval_mode it ends the program
6. Starts a timer and the training loop for each epoch from start to total epochs. In each epoch it:
* 


## 3.2 Go through `validate()` and `evaluate()`. What do they do? How are they different? 
> Could we have done better by reusing code? Yes. Yes we could have but we didn't... sorry...

`YOUR ANSWER HERE`


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

`YOUR ANSWER HERE`

## 4.2 Train AlexNet on CIFAR10. What accuracy do you get?

Report training and validation accuracy on AlexNet and LeNet. Report hyperparameters for both models (learning rate, batch size, optimizer, etc.). We get ~77% validation with AlexNet.

> You can just copy the config file, don't need to write it all out again.
> Also no need to tune the models much, you'll do it in the next part.

`YOUR ANSWER HERE`



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

In `models/models.py`, we provided some skelly/guiding comments to implement ResNet. Implement it and train it on CIFAR10. Report training and validation curves, hyperparameters, best validation accuracy, and training time as compared to AlexNet. 

`YOUR ANSWER HERE`

## 6.1 Visualize examples

Visualize a couple of the predictions on the validation set (20 or so). Be sure to include the ground truth label and the predicted label. You can use `wandb.log()` to log images or also just save them to disc any way you think is easy. 

`YOUR ANSWER HERE


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
