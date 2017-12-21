from __future__ import print_function
import torch
import torch.optim as optim
from torch.autograd import Variable
from ... import Net
from lr_scheduler import ReduceLROnPlateau


# ToDo: select the target class
target_class =  ... # 0, 1, ..., 9

# ToDo: NEW load pretrained network Net (see question 1)
model =
# ...

# ToDo: fix the network weights
# NEW The network as it is defined in class Net contains dropout that should be deactivated. Dropout is ignored in test
# mode
# ....

# means and std used for the image normalization
mean = 0.1307
std = 0.3081

# ToDo: allocate memory for the image variable
# NOTE: we need the gradients with respect to the input images. This requires certain changes in the initialization
# of the Pytorch Variable
# NEW: check default parameters of the initialiser of the class Variable
imagevar =

# ToDo: calculate the gradients of the objective function
# notice that we will use gradient descend algorithm below, so we need to change the
# sign of the objective function
grad = torch.zeros(1, 10).type(torch.FloatTensor)
grad[0, target_class] =

# ToDo: set learning parameters of the gradient descend
LR = 10  # worked well for me
NUM_ITER =

# ToDo: start with a black image
imagevar =

# ToDo: use SGD optimizer from pytorch.
# NEW: The variables we want to optimize are elements of the imagevar
optimizer = torch.optim.SGD([...], lr=LR, momentum=0.9)
# scheduler track the changes of the objective function and reduces LR when needed
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, min_lr=0.5e-6)

for i in range(NUM_ITER):
    print('[{0:05d}/{1:d}]'.format(i+1, NUM_ITER), end='\r')

    # ToDo set gradients of the optimizer to zero
    # ...

    # New: ToDo: for the current imagevar obtain the softmax avtivations of the last layer
    #  of the network
    act_value = ...
    # backpropagate the computed gradient to image domain
    act_value.backward(grad)

    # transmit the current value of the objective function to the scheduler
    scheduler.step(act_value.data[0, target_class-1])
    # ToDo: make step of the optimizer
    # ....

    # we clip the values of the updated image to the feasible region
    imagevar.data = torch.clamp(imagevar.data, -mean/std, (1-mean)/std)

# ToDo show image
# ...
