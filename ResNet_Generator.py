#!/usr/bin/env python
"""
Generate the residule learning network.
Author: Yemin Shi
Email: shiyemin@pku.edu.cn

MSRA Paper: http://arxiv.org/pdf/1512.03385v1.pdf
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

word = ['a','b','c']

def parse_args():
    """Parse input arguments
    """
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('deploy_file',
                        help='Output deploy.prototxt file',
                        default='resnet_50_deploy.prototxt')
    parser.add_argument('--layer_number', nargs='*',
                        help=('Layer number for each layer stage.'),
                        default=[3, 4, 6, 3])
    args = parser.parse_args()
    return args

def generate_data_layer():
    data_layer_str = '''name: "ResNet"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 512
input_dim: 512
'''
    return data_layer_str

def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, filler="msra"):
    conv_layer_str = '''layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  convolution_param {
    num_output: %d
    pad: %d
    kernel_size: %d
    stride: %d
    bias_term: false
    weight_filler {
      type: "%s"
    }
  }
}'''%(layer_name, bottom, layer_name, kernel_num, pad, kernel_size, stride, filler)
    return conv_layer_str

def generate_bn_layer(batch_name, scale_name, bottom):
    bn_layer_str = '''layer {
	bottom: "%s"
	top: "%s"
	name: "%s"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: true
	}
}
layer {
	bottom: "%s"
	top: "%s"
	name: "%s"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}'''%(bottom, bottom, batch_name, bottom, bottom, scale_name)
    return bn_layer_str
    
def generate_activation_layer(layer_name, bottom, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  type: "%s"
  bottom: "%s"
  top: "%s"
}'''%(layer_name, act_type, bottom, bottom)
    return act_layer_str
    
def generate_pooling_layer(kernel_size, stride, pool_type, layer_name, bottom):
    pool_layer_str = '''layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}'''%(layer_name, bottom, layer_name, pool_type, kernel_size, stride)
    return pool_layer_str
    
def generate_eltwise_layer(layer_name, bottom_1, bottom_2):
    eltwise_layer_str = '''layer {
  type: "Eltwise"
  bottom: "%s"
  bottom: "%s"
  name: "%s"
  top: "%s"
}'''%(bottom_1, bottom_2, layer_name, layer_name)
    return eltwise_layer_str
    
def generate_fc_layer(num_output, layer_name, bottom, top, filler="msra"):
    fc_layer_str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
     num_output: %d
     weight_filler {
       type: "%s"
       std: 0.001
     }
     bias_filler {
       type: "constant"
       value: 0
     }
  }
}'''%(layer_name, bottom, top, num_output, filler)
    return fc_layer_str

def generate_softmax_loss(bottom):
    softmax_loss_str = '''layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  top: "loss"
}
layer {
  name: "acc/top-1"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "acc/top-1"
  include {
    phase: TEST
  }
}
layer {
  name: "acc/top-5"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "acc/top-5"
  include {
    phase: TEST
  }
  accuracy_param {
    top_k: 5
  }
}'''%(bottom, bottom, bottom)
    return softmax_loss_str

def generate_deploy():
    args = parse_args()
    network_str = generate_data_layer()
    '''before stage'''
    last_top = 'data'
    network_str += generate_conv_layer(7, 64, 2, 3, 'conv1', last_top)
    network_str += generate_bn_layer('bn_conv1', 'scale_conv1', 'conv1')
    network_str += generate_activation_layer('conv1_relu', 'conv1')
    network_str += generate_pooling_layer(3, 2, 'MAX', 'pool1', 'conv1')
    '''stage 1'''
    last_top = 'pool1'
    network_str += generate_conv_layer(1, 256, 1, 0, 'res2a_branch1', last_top)
    network_str += generate_bn_layer('bn2a_branch1', 'scale2a_branch1', 'res2a_branch1')
    last_output = 'res2a_branch1'
    for l in xrange(1, args.layer_number[0]+1):
        network_str += generate_conv_layer(1, 64, 1, 0, 'res2%s_branch2a'%word[l], last_top)
        network_str += generate_bn_layer('bn2%s_branch2a'%l, 'scale2%s_branch2a'%l, 'res2%s_branch2a'%word[l])
        network_str += generate_activation_layer('res2%s_branch2a_relu'%word[l], 'res2%s_branch2a'%word[l])
        network_str += generate_conv_layer(3, 64, 1, 1, 'res2%s_branch2b'%word[l], 'res2%s_branch2a'%word[l])
        network_str += generate_bn_layer('bn2%s_branch2b'%l, 'scale2%s_branch2b'%l, 'res2%s_branch2b'%word[l])
        network_str += generate_activation_layer('res2%s_branch2b_relu'%word[l], 'res2%s_branch2b'%word[l])
        network_str += generate_conv_layer(1, 256, 1, 0, 'res2%s_branch2c'%word[l], 'res2%s_branch2b'%word[l])
        network_str += generate_bn_layer('bn2%s_branch2c'%l, 'scale2%s_branch2c'%l, 'res2%s_branch2c'%word[l])
        network_str += generate_eltwise_layer('res2%s'%word[l], last_output, 'res2%s_branch2c'%word[l])
        network_str += generate_activation_layer('res2%s_relu'%word[l],'res2%s'%word[l])
        last_top = 'res2%s'%word[l]
        last_output = 'res2%s'%word[l]
    
    network_str += generate_conv_layer(1, 512, 2, 0, 'conv2_output', last_top, 'conv2_output')
    last_output = 'conv2_output'
    '''stage 2'''
    network_str += generate_conv_layer(1, 128, 2, 0, 'conv3_1_1', last_top, 'conv3_1_1')
    network_str += generate_bn_layer('conv3_1_1_bn', 'conv3_1_1', 'conv3_1_1_bn')
    network_str += generate_activation_layer('conv3_1_1_relu', 'conv3_1_1_bn', 'conv3_1_1_bn', 'ReLU')
    network_str += generate_conv_layer(3, 128, 1, 1, 'conv3_1_2', 'conv3_1_1_bn', 'conv3_1_2')
    network_str += generate_bn_layer('conv3_1_2_bn', 'conv3_1_2', 'conv3_1_2_bn')
    network_str += generate_activation_layer('conv3_1_2_relu', 'conv3_1_2_bn', 'conv3_1_2_bn', 'ReLU')
    network_str += generate_conv_layer(1, 512, 1, 0, 'conv3_1_3', 'conv3_1_2_bn', 'conv3_1_3')
    network_str += generate_eltwise_layer('conv3_1_sum', last_output, 'conv3_1_3', 'conv3_1_sum', 'SUM')
    network_str += generate_bn_layer('conv3_1_sum_bn', 'conv3_1_sum', 'conv3_1_sum_bn')
    network_str += generate_activation_layer('conv3_1_sum_relu', 'conv3_1_sum_bn', 'conv3_1_sum_bn', 'ReLU')
    last_top = 'conv3_1_sum_bn'
    for l in xrange(2, args.layer_number[1]+1):
        network_str += generate_conv_layer(1, 128, 1, 0, 'conv3_%d_1'%l, last_top, 'conv3_%d_1'%l)
        network_str += generate_bn_layer('conv3_%d_1_bn'%l, 'conv3_%d_1'%l, 'conv3_%d_1_bn'%l)
        network_str += generate_activation_layer('conv3_%d_1_relu'%l, 'conv3_%d_1_bn'%l, 'conv3_%d_1_bn'%l, 'ReLU')
        network_str += generate_conv_layer(3, 128, 1, 1, 'conv3_%d_2'%l, 'conv3_%d_1_bn'%l, 'conv3_%d_2'%l)
        network_str += generate_bn_layer('conv3_%d_2_bn'%l, 'conv3_%d_2'%l, 'conv3_%d_2_bn'%l)
        network_str += generate_activation_layer('conv3_%d_2_relu'%l, 'conv3_%d_2_bn'%l, 'conv3_%d_2_bn'%l, 'ReLU')
        network_str += generate_conv_layer(1, 512, 1, 0, 'conv3_%d_3'%l, 'conv3_%d_2_bn'%l, 'conv3_%d_3'%l)
        network_str += generate_eltwise_layer('conv3_%d_sum'%l, last_top, 'conv3_%d_3'%l, 'conv3_%d_sum'%l, 'SUM')
        network_str += generate_bn_layer('conv3_%d_sum_bn'%l, 'conv3_%d_sum'%l, 'conv3_%d_sum_bn'%l)
        network_str += generate_activation_layer('conv3_%d_sum_relu'%l, 'conv3_%d_sum_bn'%l, 'conv3_%d_sum_bn'%l, 'ReLU')
        last_top = 'conv3_%d_sum_bn'%l
    network_str += generate_conv_layer(1, 1024, 2, 0, 'conv3_output', last_top, 'conv3_output')
    last_output = 'conv3_output'
    '''stage 3'''
    network_str += generate_conv_layer(1, 256, 2, 0, 'conv4_1_1', last_top, 'conv4_1_1')
    network_str += generate_bn_layer('conv4_1_1_bn', 'conv4_1_1', 'conv4_1_1_bn')
    network_str += generate_activation_layer('conv4_1_1_relu', 'conv4_1_1_bn', 'conv4_1_1_bn', 'ReLU')
    network_str += generate_conv_layer(3, 256, 1, 1, 'conv4_1_2', 'conv4_1_1_bn', 'conv4_1_2')
    network_str += generate_bn_layer('conv4_1_2_bn', 'conv4_1_2', 'conv4_1_2_bn')
    network_str += generate_activation_layer('conv4_1_2_relu', 'conv4_1_2_bn', 'conv4_1_2_bn', 'ReLU')
    network_str += generate_conv_layer(1, 1024, 1, 0, 'conv4_1_3', 'conv4_1_2_bn', 'conv4_1_3')
    network_str += generate_eltwise_layer('conv4_1_sum', last_output, 'conv4_1_3', 'conv4_1_sum', 'SUM')
    network_str += generate_bn_layer('conv4_1_sum_bn', 'conv4_1_sum', 'conv4_1_sum_bn')
    network_str += generate_activation_layer('conv4_1_sum_relu', 'conv4_1_sum_bn', 'conv4_1_sum_bn', 'ReLU')
    last_top = 'conv4_1_sum_bn'
    for l in xrange(2, args.layer_number[2]+1):
        network_str += generate_conv_layer(1, 256, 1, 0, 'conv4_%d_1'%l, last_top, 'conv4_%d_1'%l)
        network_str += generate_bn_layer('conv4_%d_1_bn'%l, 'conv4_%d_1'%l, 'conv4_%d_1_bn'%l)
        network_str += generate_activation_layer('conv4_%d_1_relu'%l, 'conv4_%d_1_bn'%l, 'conv4_%d_1_bn'%l, 'ReLU')
        network_str += generate_conv_layer(3, 256, 1, 1, 'conv4_%d_2'%l, 'conv4_%d_1_bn'%l, 'conv4_%d_2'%l)
        network_str += generate_bn_layer('conv4_%d_2_bn'%l, 'conv4_%d_2'%l, 'conv4_%d_2_bn'%l)
        network_str += generate_activation_layer('conv4_%d_2_relu'%l, 'conv4_%d_2_bn'%l, 'conv4_%d_2_bn'%l, 'ReLU')
        network_str += generate_conv_layer(1, 1024, 1, 0, 'conv4_%d_3'%l, 'conv4_%d_2_bn'%l, 'conv4_%d_3'%l)
        network_str += generate_eltwise_layer('conv4_%d_sum'%l, last_top, 'conv4_%d_3'%l, 'conv4_%d_sum'%l, 'SUM')
        network_str += generate_bn_layer('conv4_%d_sum_bn'%l, 'conv4_%d_sum'%l, 'conv4_%d_sum_bn'%l)
        network_str += generate_activation_layer('conv4_%d_sum_relu'%l, 'conv4_%d_sum_bn'%l, 'conv4_%d_sum_bn'%l, 'ReLU')
        last_top = 'conv4_%d_sum_bn'%l
    network_str += generate_conv_layer(1, 2048, 2, 0, 'conv4_output', last_top, 'conv4_output')
    last_output = 'conv4_output'
    '''stage 4'''
    network_str += generate_conv_layer(1, 512, 2, 0, 'conv5_1_1', last_top, 'conv5_1_1')
    network_str += generate_bn_layer('conv5_1_1_bn', 'conv5_1_1', 'conv5_1_1_bn')
    network_str += generate_activation_layer('conv5_1_1_relu', 'conv5_1_1_bn', 'conv5_1_1_bn', 'ReLU')
    network_str += generate_conv_layer(3, 512, 1, 1, 'conv5_1_2', 'conv5_1_1_bn', 'conv5_1_2')
    network_str += generate_bn_layer('conv5_1_2_bn', 'conv5_1_2', 'conv5_1_2_bn')
    network_str += generate_activation_layer('conv5_1_2_relu', 'conv5_1_2_bn', 'conv5_1_2_bn', 'ReLU')
    network_str += generate_conv_layer(1, 2048, 1, 0, 'conv5_1_3', 'conv5_1_2_bn', 'conv5_1_3')
    network_str += generate_eltwise_layer('conv5_1_sum', last_output, 'conv5_1_3', 'conv5_1_sum', 'SUM')
    network_str += generate_bn_layer('conv5_1_sum_bn', 'conv5_1_sum', 'conv5_1_sum_bn')
    network_str += generate_activation_layer('conv5_1_sum_relu', 'conv5_1_sum_bn', 'conv5_1_sum_bn', 'ReLU')
    last_top = 'conv5_1_sum_bn'
    for l in xrange(2, args.layer_number[3]+1):
        network_str += generate_conv_layer(1, 512, 1, 0, 'conv5_%d_1'%l, last_top, 'conv5_%d_1'%l)
        network_str += generate_bn_layer('conv5_%d_1_bn'%l, 'conv5_%d_1'%l, 'conv5_%d_1_bn'%l)
        network_str += generate_activation_layer('conv5_%d_1_relu'%l, 'conv5_%d_1_bn'%l, 'conv5_%d_1_bn'%l, 'ReLU')
        network_str += generate_conv_layer(3, 512, 1, 1, 'conv5_%d_2'%l, 'conv5_%d_1_bn'%l, 'conv5_%d_2'%l)
        network_str += generate_bn_layer('conv5_%d_2_bn'%l, 'conv5_%d_2'%l, 'conv5_%d_2_bn'%l)
        network_str += generate_activation_layer('conv5_%d_2_relu'%l, 'conv5_%d_2_bn'%l, 'conv5_%d_2_bn'%l, 'ReLU')
        network_str += generate_conv_layer(1, 2048, 1, 0, 'conv5_%d_3'%l, 'conv5_%d_2_bn'%l, 'conv5_%d_3'%l)
        network_str += generate_eltwise_layer('conv5_%d_sum'%l, last_top, 'conv5_%d_3'%l, 'conv5_%d_sum'%l, 'SUM')
        network_str += generate_bn_layer('conv5_%d_sum_bn'%l, 'conv5_%d_sum'%l, 'conv5_%d_sum_bn'%l)
        network_str += generate_activation_layer('conv5_%d_sum_relu'%l, 'conv5_%d_sum_bn'%l, 'conv5_%d_sum_bn'%l, 'ReLU')
        last_top = 'conv5_%d_sum_bn'%l
    network_str += generate_pooling_layer(7, 1, 'AVE', 'pool2', last_top, 'pool2')
    network_str += generate_fc_layer(1000, 'fc', 'pool2', 'fc', 'gaussian')
    network_str += generate_softmax_loss('fc')
    return network_str

def main():
    args = parse_args()
    network_str = generate_deploy()

    fp = open(args.deploy_file, 'w')
    fp.write(network_str)
    fp.close()

if __name__ == '__main__':
    main()
