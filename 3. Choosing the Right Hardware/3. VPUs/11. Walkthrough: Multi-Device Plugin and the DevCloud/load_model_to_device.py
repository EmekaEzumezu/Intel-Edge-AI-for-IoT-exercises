
import time
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin
import argparse

def main(args):
    model=args.model_path
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    start=time.time()
    model=IENetwork(model_structure, model_weights)

    plugin = IEPlugin(device=args.device)
    
    net = plugin.load(network=model, num_requests=1)
    print(f"Time taken to load model = {time.time()-start} seconds")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default=None)
    
    args=parser.parse_args() 
    main(args)