from openvino.inference_engine import IENetwork, IECore

import numpy as np
import time

# Loading model
model_path='sep_cnn/sep_cnn'
model_weights=model_path+'.bin'
model_structure=model_path+'.xml'

# COMPLETED: Load the model
network=IENetwork(model_structure, model_weights)

core = IECore()
model = core.load_network(network=network, device_name='CPU', num_requests=1)

input_name = next(iter(network.inputs))

# Reading and Preprocessing Image
input_img=np.load('image.npy')
input_img=input_img.reshape(1, 28, 28)


# COMPLETED: Using the input image, run inference on the model for 10 iterations
input_dict={input_name:input_img}

start=time.time()
for _ in range(10):
    # input_dict: Run Inference in a Loop
    model.infer(input_dict)


# COMPLETED: Finish the print statement
print("Time taken to run 10 iterations is: {} seconds".format(time.time()-start))