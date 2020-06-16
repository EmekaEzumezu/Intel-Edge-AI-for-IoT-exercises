
import time
import numpy as np
import cv2
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
import argparse

def main(args):
    model=args.model_path
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    start=time.time()
    
    # COMPLETED: Load the model
    model=IENetwork(model_structure, model_weights)

    core = IECore()
    net = core.load_network(network=model, device_name=args.device, num_requests=1)
    
    print(f"Time taken to load model = {time.time()-start} seconds")
    
    # Get the name of the input node
    input_name=next(iter(model.inputs))
    
    # Reading and Preprocessing Image
    input_img=cv2.imread('/data/resources/car.png')
    input_img=cv2.resize(input_img, (300,300), interpolation = cv2.INTER_AREA)
    input_img=np.moveaxis(input_img, -1, 0)

    # COMPLETED: Prepare the model for inference (create input dict etc.)
    # Running Inference in a loop on the same image
    input_dict={input_name:input_img}
    
    start=time.time()
    for _ in range(10):
        # COMPLETED: Run Inference in a Loop
        net.infer(input_dict)
    
    print(f"Time Taken to run 10 Inference on FPGA is = {time.time()-start} seconds")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default=None)
    
    args=parser.parse_args() 
    main(args)