# Flower_Image_Classifier
  By *Dilip Jain*

# Flower-Image-Classifier
Flower image classifier,build with pytorch neural network
# Udacity Machine learning Nanodegree project for deep learning module titled as 'Image Classifier with Deep Learning' attempts to train an image classifier to recognize different species of flowers. 

• Solution: Used torchvision to load the data. The dataset is split into three parts,
training, validation, and testing. For the training, applied transformations such 
as random scaling, cropping, and flipping. This will help the network generalize leading
to better performance. Also need to load in a mapping from category label to category name. 
Wrote inference for classification after training and testing the model.
Then processed a PIL image for use in a PyTorch model.

## Software and Libraries
This project uses the following software and Python libraries:
 * NumPy
 * Pandas
 * Matplotlib
 * Seaborn
 * Pytorch
 * Torchvision
 * Sklearn / scikit-learn
## Code File
Open file jupyter notebook imageclassifierproject.ipynb

Files Description
* Image Classifier Project.ipynb It is used to build the model using the jupyter notebook. It can be used independently to see how the model works.
* cat_to_name.json It is used in ipynb and py file to map flower number to flower names.
*  train_args.py It is used by predict.py to enable the parameter function.
* train.py It will train a new network on a dataset and save the model as a checkpoint.
* predict_args.py It is used by predict.py to enable the parameter function.
* predict.py It uses a trained network to predict the class for an input image.
