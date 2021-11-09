# Satellite-Image-Segementation
Dense-Net from https://github.com/Sharpiless/FCN-DenseNet
Other networks from https://github.com/Deeachain/Segmentation-Pytorch

## The data
Training data in 'Data/train' folder, testing data in 'Data/val' folder. Data include two parts, original image and segmentation mask.

## In Ensemble folder
To train the model, run 'train.py', you may need run several times in order to train different networks.
The training parameters will be saved at the 'parameters' folder, here we already give the trained parameters which using gray image, and 'parameters0' also gives the trained parameters using full color image.
There are two test file 'test.py' and 'test_densenet' to test each independent network. Please use 'test_densenet' to test DenseNet because its output doesn't use ont-hot coding.
If you use 'test.py' to test other network, remember to change the network name at the beginning.
'ensemble_test.py' gives three ensemble methods.

## Output
All segmentation results will be saved at 'Ensemble/output'.
Training loss saved at 'Ensemble/loss'.