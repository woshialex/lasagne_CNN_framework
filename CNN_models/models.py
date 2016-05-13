import theano
import theano.tensor as T
import lasagne as nn
from lasagne.layers import batch_norm as bn
import os
import numpy as np

def build_cnn(input_var, shape, version=1, N_output=5):
    ret = {}
    if N_output == 1:
        output_fn = nn.nonlinearities.sigmoid;
    else:
        output_fn = nn.nonlinearities.softmax;
    if version == 1: 
        nlf = nn.nonlinearities.LeakyRectify(leakiness = 0.1);
        ret['input'] = layer = nn.layers.InputLayer(shape, input_var)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=16, filter_size=(7,7), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=(5,5), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=(5,5), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=128, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=256, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=512, filter_size=(3,3), nonlinearity = nlf))
        ret['flatten'] = layer = nn.layers.FlattenLayer(layer);
        #ret['dropout'] = layer = nn.layers.dropout(layer,0.5);
        ret['FC{}'.format(len(ret))] = layer = nn.layers.DenseLayer(layer, num_units = 64, nonlinearity = nlf);
        ret['output'] = layer = nn.layers.DenseLayer(layer, num_units=N_output, nonlinearity=output_fn)
    elif version == 2:
        nlf = nn.nonlinearities.LeakyRectify(leakiness = 0.1);
        ret['input'] = layer = nn.layers.InputLayer(shape, input_var)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=48, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=48, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=128, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=128, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=128, filter_size=(3,3), nonlinearity = nlf))
        ret['flatten'] = layer = nn.layers.FlattenLayer(layer);
        ret['dropout'] = layer = nn.layers.dropout(layer,0.5);
        ret['FC{}'.format(len(ret))] = layer = nn.layers.DenseLayer(layer, num_units = 64, nonlinearity = nlf);
        ret['output'] = layer = nn.layers.DenseLayer(layer, num_units=N_output, nonlinearity=output_fn)
    elif version == 3:#196, CV=0.66
        nlf = nn.nonlinearities.LeakyRectify(leakiness = 0.1);
        ret['input'] = layer = nn.layers.InputLayer(shape, input_var)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=16, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=16, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=128, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=164, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=164, filter_size=(3,3), nonlinearity = nlf))
        ret['flatten'] = layer = nn.layers.FlattenLayer(layer);
        ret['dropout'] = layer = nn.layers.dropout(layer,0.5);
        ret['FC{}'.format(len(ret))] = layer = nn.layers.DenseLayer(layer, num_units = 64, nonlinearity = nlf);
        ret['output'] = layer = nn.layers.DenseLayer(layer, num_units=N_output, nonlinearity=output_fn)
    elif version == 4: #196
        nlf = nn.nonlinearities.LeakyRectify(leakiness = 0.1);
        ret['input'] = layer = nn.layers.InputLayer(shape, input_var)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=128, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=128, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=256, filter_size=(3,3), nonlinearity = nlf))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=512, filter_size=(3,3), nonlinearity = nlf))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=512, filter_size=(3,3), nonlinearity = nlf))
        ret['flatten'] = layer = nn.layers.FlattenLayer(layer);
        ret['dropout'] = layer = nn.layers.dropout(layer,0.5);
        ret['FC{}'.format(len(ret))] = layer = nn.layers.DenseLayer(layer, num_units = 512, nonlinearity = nlf);
        ret['dropout'] = layer = nn.layers.dropout(layer,0.5);
        ret['FC{}'.format(len(ret))] = layer = nn.layers.DenseLayer(layer, num_units = 128, nonlinearity = nlf);
        ret['output'] = layer = nn.layers.DenseLayer(layer, num_units=N_output, nonlinearity=output_fn)
    elif version == 5:#VGG 16, LB = 0.32
        nlf = nn.nonlinearities.LeakyRectify(leakiness = 0.1);
        from lasagne.layers import Conv2DLayer as ConvLayer;
        from lasagne.layers import MaxPool2DLayer as PoolLayer;
        ret['input'] =  nn.layers.InputLayer((None,3,224,224), input_var)
        ret['conv1_1'] = ConvLayer(ret['input'], 64, 3, pad=1, flip_filters=False)
        for k,x in ret['conv1_1'].params.iteritems():
            x.remove('trainable')
        ret['conv1_2'] = ConvLayer(ret['conv1_1'], 64, 3, pad=1, flip_filters=False)
        for k,x in ret['conv1_2'].params.iteritems():
            x.remove('trainable')
        ret['pool1'] = PoolLayer(ret['conv1_2'], 2)
        ret['conv2_1'] = ConvLayer( ret['pool1'], 128, 3, pad=1, flip_filters=False)
        for k,x in ret['conv2_1'].params.iteritems():
            x.remove('trainable')
        ret['conv2_2'] = ConvLayer( ret['conv2_1'], 128, 3, pad=1, flip_filters=False)
        ret['pool2'] = PoolLayer(ret['conv2_2'], 2)
        ret['conv3_1'] = ConvLayer( ret['pool2'], 256, 3, pad=1, flip_filters=False)
        ret['conv3_2'] = ConvLayer( ret['conv3_1'], 256, 3, pad=1, flip_filters=False)
        ret['conv3_3'] = ConvLayer( ret['conv3_2'], 256, 3, pad=1, flip_filters=False)
        ret['pool3'] = PoolLayer(ret['conv3_3'], 2)
        ret['conv4_1'] = ConvLayer( ret['pool3'], 512, 3, pad=1, flip_filters=False)
        ret['conv4_2'] = ConvLayer( ret['conv4_1'], 512, 3, pad=1, flip_filters=False)
        ret['conv4_3'] = ConvLayer( ret['conv4_2'], 512, 3, pad=1, flip_filters=False)
        ret['pool4'] = PoolLayer(ret['conv4_3'], 2)
        ret['conv5_1'] = ConvLayer( ret['pool4'], 512, 3, pad=1, flip_filters=False)
        ret['conv5_2'] = ConvLayer( ret['conv5_1'], 512, 3, pad=1, flip_filters=False)
        ret['conv5_3'] = ConvLayer( ret['conv5_2'], 512, 3, pad=1, flip_filters=False)
        ret['pool5'] = PoolLayer(ret['conv5_3'], 2)
        ret['fc6'] = nn.layers.DenseLayer(ret['pool5'], num_units=4096)
        ret['fc6_dropout'] = nn.layers.dropout(ret['fc6'], p=0.5)
        ret['fc7'] = nn.layers.DenseLayer(ret['fc6_dropout'], num_units=4096)
        ret['fc7_dropout'] = nn.layers.dropout(ret['fc7'], p=0.5)
        ret['fc8'] = nn.layers.DenseLayer(ret['fc7_dropout'], num_units=1000, nonlinearity=None)
        ret['output'] = nn.layers.NonlinearityLayer(ret['fc8'], nn.nonlinearities.softmax)
        import pickle
        model = pickle.load(open(c.params_dir+'/vgg16.pkl'));
        nn.layers.set_all_param_values(ret['output'],model['param values']);
        ret.pop("output",None);
        ret.pop("fc8",None);
        ret['output'] = nn.layers.DenseLayer(ret['fc7_dropout'], num_units=N_output, nonlinearity=output_fn)
    elif version == 6:#VGG 16 # 0.31
        nlf = nn.nonlinearities.LeakyRectify(leakiness = 0.1);
        from lasagne.layers import Conv2DLayer as ConvLayer;
        from lasagne.layers import MaxPool2DLayer as PoolLayer;
        ret['input'] =  nn.layers.InputLayer((None,3,224,224), input_var)
        ret['conv1_1'] = ConvLayer(ret['input'], 64, 3, pad=1, flip_filters=False)
        for k,x in ret['conv1_1'].params.iteritems():
            x.remove('trainable')
        ret['conv1_2'] = ConvLayer(ret['conv1_1'], 64, 3, pad=1, flip_filters=False)
        for k,x in ret['conv1_2'].params.iteritems():
            x.remove('trainable')
        ret['pool1'] = PoolLayer(ret['conv1_2'], 2)
        ret['conv2_1'] = ConvLayer( ret['pool1'], 128, 3, pad=1, flip_filters=False)
        for k,x in ret['conv2_1'].params.iteritems():
            x.remove('trainable')
        ret['conv2_2'] = ConvLayer( ret['conv2_1'], 128, 3, pad=1, flip_filters=False)
        for k,x in ret['conv2_2'].params.iteritems():
            x.remove('trainable')
        ret['pool2'] = PoolLayer(ret['conv2_2'], 2)
        ret['conv3_1'] = ConvLayer( ret['pool2'], 256, 3, pad=1, flip_filters=False)
        for k,x in ret['conv3_1'].params.iteritems():
            x.remove('trainable')
        ret['conv3_2'] = ConvLayer( ret['conv3_1'], 256, 3, pad=1, flip_filters=False)
        for k,x in ret['conv3_2'].params.iteritems():
            x.remove('trainable')
        ret['conv3_3'] = ConvLayer( ret['conv3_2'], 256, 3, pad=1, flip_filters=False)
        for k,x in ret['conv3_3'].params.iteritems():
            x.remove('trainable')
        ret['pool3'] = PoolLayer(ret['conv3_3'], 2)
        ret['conv4_1'] = ConvLayer( ret['pool3'], 512, 3, pad=1, flip_filters=False)
        ret['conv4_2'] = ConvLayer( ret['conv4_1'], 512, 3, pad=1, flip_filters=False)
        ret['conv4_3'] = ConvLayer( ret['conv4_2'], 512, 3, pad=1, flip_filters=False)
        ret['pool4'] = PoolLayer(ret['conv4_3'], 2)
        ret['conv5_1'] = ConvLayer( ret['pool4'], 512, 3, pad=1, flip_filters=False)
        ret['conv5_2'] = ConvLayer( ret['conv5_1'], 512, 3, pad=1, flip_filters=False)
        ret['conv5_3'] = ConvLayer( ret['conv5_2'], 512, 3, pad=1, flip_filters=False)
        ret['pool5'] = PoolLayer(ret['conv5_3'], 2)
        ret['fc6'] = nn.layers.DenseLayer(ret['pool5'], num_units=4096)
        ret['fc6_dropout'] = nn.layers.dropout(ret['fc6'], p=0.5)
        ret['fc7'] = nn.layers.DenseLayer(ret['fc6_dropout'], num_units=4096)
        ret['fc7_dropout'] = nn.layers.dropout(ret['fc7'], p=0.5)
        ret['fc8'] = nn.layers.DenseLayer(ret['fc7_dropout'], num_units=1000, nonlinearity=None)
        ret['output'] = nn.layers.NonlinearityLayer(ret['fc8'], nn.nonlinearities.softmax)
        import pickle
        model = pickle.load(open(c.params_dir+'/vgg16.pkl'));
        nn.layers.set_all_param_values(ret['output'],model['param values']);
        ret.pop("output",None);
        ret.pop("fc8",None);
        #ret.pop("fc7_dropout",None);
        #ret.pop("fc7",None);
        #ret['fc7'] = nn.layers.DenseLayer(ret['fc6_dropout'], num_units=256, nonlinearity=nlf);
        #ret['fc7_dropout'] = nn.layers.dropout(ret['fc7'], p=0.5)
        ret['output'] = nn.layers.DenseLayer(ret['fc7_dropout'], num_units=N_output, nonlinearity=output_fn)
    elif version == 7: #inception_v3 # LB ~ 0.31
        import inception_v3 
        ret = inception_v3.build_network(input_var);
        import pickle
        model = pickle.load(open(c.params_dir+'/inception_v3.pkl'));
        nn.layers.set_all_param_values(ret['softmax'],model['param values']);
        ret.pop("softmax",None);
        ret['pool3_dropout'] = nn.layers.dropout(ret['pool3'], p=0.5)
        ret['output'] = nn.layers.DenseLayer(ret['pool3_dropout'], num_units=N_output, nonlinearity=output_fn)
        #non trainable
        for ll in nn.layers.get_all_layers(ret['conv_4']):#4->3
            if ll is not ret['input']:
                for k,x in ll.params.iteritems():
                    if 'trainable' in x:
                        x.remove('trainable')

    elif version == 8: #inception_v3
        import inception_v3 
        ret = inception_v3.build_network(input_var);
        import pickle
        model = pickle.load(open(c.params_dir+'/inception_v3.pkl'));
        nn.layers.set_all_param_values(ret['softmax'],model['param values']);
        ret.pop("softmax",None);
        ret['pool3_dropout'] = nn.layers.dropout(ret['pool3'], p=0.5) #0.5 is a little better than 0.7
        ret['output'] = nn.layers.DenseLayer(ret['pool3_dropout'], num_units=N_output, nonlinearity=output_fn)
        #non trainable
        #for ll in nn.layers.get_all_layers(ret['conv_3']): #conv_3(0.278)>mix1 (0.285)>mix3 >> mix7
        for ll in nn.layers.get_all_layers(ret['mixed_1/join']): 
            if ll is not ret['input']:
                for k,x in ll.params.iteritems():
                    if 'trainable' in x:
                        x.remove('trainable')

    elif version == 9: #res_net, no pretrain overfit. (LB ~ 0.7)
        import Deep_Residual_Learning as resnet
        ret['output'] = resnet.build_cnn(shape, input_var, n=2);
        #import pickle
        #model = pickle.load(open(c.params_dir+'/cifar_model_n5.pkl'));
        #nn.layers.set_all_param_values(ret['output'],model['param values']);

    elif version == 10:#192, add small img short cuts
        nlf = nn.nonlinearities.LeakyRectify(leakiness = 0.1);
        ret['input'] = layer = nn.layers.InputLayer(shape, input_var)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=16, filter_size=(5,5), pad = 'same', nonlinearity = nlf, flip_filters=False))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['h'] = nn.layers.Pool2DLayer(ret['input'],pool_size=(2,2),mode='average_inc_pad');
        layer = nn.layers.ConcatLayer([layer,ret['h']]);
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=(5,5), pad = 'same', nonlinearity = nlf, flip_filters=False))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['hh'] = nn.layers.Pool2DLayer(ret['h'],pool_size=(2,2),mode='average_inc_pad');
        layer = nn.layers.ConcatLayer([layer,ret['hh']]);
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=(3,3), pad = 'same', nonlinearity = nlf, flip_filters=False))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=(3,3), pad = 'same', nonlinearity = nlf, flip_filters=False))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['hhh'] = nn.layers.Pool2DLayer(ret['hh'],pool_size=(2,2),mode='average_inc_pad');
        layer = nn.layers.ConcatLayer([layer,ret['hhh']]);
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=128, filter_size=(3,3), pad = 'same', nonlinearity = nlf, flip_filters=False))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=128, filter_size=(3,3), pad = 'same', nonlinearity = nlf, flip_filters=False))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['hhhh'] = nn.layers.Pool2DLayer(ret['hhh'],pool_size=(2,2),mode='average_inc_pad');
        layer = nn.layers.ConcatLayer([layer,ret['hhhh']]);
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=256, filter_size=(3,3), pad = 'same', nonlinearity = nlf, flip_filters=False))
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=256, filter_size=(3,4), nonlinearity = nlf, flip_filters=False))
        ret['pool{}'.format(len(ret))] = layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
        ret['conv{}'.format(len(ret))] = layer = bn(nn.layers.Conv2DLayer(layer, num_filters=512, filter_size=(3,2), nonlinearity = nlf, flip_filters=False))
        ret['flatten'] = layer = nn.layers.FlattenLayer(layer);
        ret['dropout'] = layer = nn.layers.dropout(layer,0.5);
        ret['FC{}'.format(len(ret))] = layer = nn.layers.DenseLayer(layer, num_units = 256, nonlinearity = nlf);
        ret['dropout'] = layer = nn.layers.dropout(layer,0.5);
        ret['output'] = layer = nn.layers.DenseLayer(layer, num_units=N_output, nonlinearity=output_fn)

    return ret, nn.layers.get_output(ret['output']), \
            nn.layers.get_output(ret['output'], deterministic=True)

# loads params in npz
def load_params(model, fn):
    with np.load(fn) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    nn.layers.set_all_param_values(model, param_values)

def get_predict_function(m_param, weights, file_fmt, shape):
    weights = np.asarray(weights, dtype=np.float32);
    weights = weights/np.sum(weights);
    input_var = T.tensor4('input')
    expr = 0
    for ii in range(len(m_param)):
        model_file = file_fmt.format(*(m_param[ii])) 
        net, _, output_det = build_cnn(input_var, shape, m_param[ii][0])
        load_params(net['output'], model_file)
        expr = expr + output_det * weights[ii];
        print 'loaded {}'.format(model_file.split('/')[-1])
    return theano.function([input_var], expr)
