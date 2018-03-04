import os
import sys
import ConfigParser

def INIT(config):

    if not os.path.isfile(config):
        print ' File %s not found.' %filename
        quit()
    home = '/home/akame'

    infile = ConfigParser.SafeConfigParser()
    infile.read(config)

    param= {
            'pycaffe_path':home+infile.get('CAFFE_INFO','pycaffe_path'),
            'train_img':home+infile.get('Benchmark_INFO','train_img'),
            'test_img':home+infile.get('Benchmark_INFO','test_img'),
            'train_lmdb':home+infile.get('Benchmark_INFO','train_lmdb'),
            'train_net':home+infile.get('NETWORK_INFO','train_net'),
            'deploy_net':home+infile.get('NETWORK_INFO','deploy_net'),
            'model_snap':home+infile.get('NETWORK_INFO','model_snap'),
            'train_list_usu':home+infile.get('Benchmark_INFO','train_list_usu'),
            'solver':home+infile.get('NETWORK_INFO','solver'),

    }
    sys.path.append(param['pycaffe_path'])
    return param 