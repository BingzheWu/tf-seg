import caffe
import tensorflow as tf
import tensorflow.contrib.slim as slim
from proto.net_param import Net_Param

class caffe2tensorflow():
    def __init__(self,prototxt,caffemodel):
        self.net=caffe.Net(prototxt,caffemodel,caffe.TEST)
        self.net_param=Net_Param(prototxt)
        self.now_layer_name=''
        self.build_layers=[]
        self.layers={}
    def bias(self,layer):
        b=layer.blobs[1].data
        bias_values = slim.model_variables('bias', shape = b.shape, initializer = tf.constant_initializer(b), dtype = tf.float32)
        return bias_values
    def fc_weight(self,layer):
        w=layer.blobs[0].data
        layer=self.net_param.layer_param(self.now_layer_name)
        blob_name=layer.bottom[0]
        shape = self.net.blobs[blob_name].data.shape
        if len(shape)==4: #if the last layer output is a 4-dimensional bolb [batch,channel,weight,height]
            assert shape[1]*shape[2]*shape[3]==w.shape[1],\
                "The blob %s's shape not match the next fc layer's shape"%(blob_name)
            w=w.reshape([w.shape[0],shape[1], shape[2], shape[3]])
            w=w.transpose([2,3,1,0])
            w=w.reshape([shape[1]*shape[2]*shape[0],w.shape[3]])
        else:
            assert shape[1]==w.shape[1],"The blob %s's shape not match the next fc layer's shape"%(blob_name)
            w=w.transpose([1,0])
        fc_weights = slim.model_variables('fc_weights', shape = w.shape, initializer = tf.constant_initializer(w), dtype = tf.float32)
        return fc_weights 
    def conv_weight(self,layer):
        w=layer.blobs[0].data
        w = w.transpose((2, 3, 1, 0))
        conv_weight = slim.model_variables('weights',w.shape, initializer = tf.constant_initializer(w))
        return conv_weight
    
    def build_conv(self,input,layer):
        w=self.conv_weight(layer)
        b=self.bias(layer)
        #because vgg16 all use same padding, commit this line
        #tf.pad(input,[[0,0],[],[],[]],mode="CONSTANT")
    
        conv=tf.nn.conv2d(input, w, [1,1,1,1],padding="SAME")
        conv = slim.conv2d(input,)
        conv=tf.nn.bias_add(conv,b)
        return conv
    
    def build_fc(self,input,layer):
        w=self.fc_weight(layer)
        b=self.bias(layer)
        shape = input.get_shape().as_list()
        dim=1
        for d in shape[1:]:
            dim*=d
        x=tf.reshape(input,[-1,dim])
        fc=tf.matmul(x,w)+b
        return fc
    def build(self):
        index=0
        for layer in self.net.layers:
            self.now_layer_name=self.net._layer_names[index]
            print index,layer.type
            self.layers[self.now_layer_name]=layer
            if layer.type=="Input":
                with tf.name_scope("input"):
                    im=tf.placeholder("float", [None, 224, 224, 3], name="data")
                    self.build_layers.append(im)
            elif layer.type=="Convolution":
                with tf.name_scope("conv_%d"%index):
                    conv=self.build_conv(self.build_layers[-1],layer)
                    self.build_layers.append(conv)
            elif layer.type=="ReLU":
                with tf.name_scope("relu_%d"%index):
                    relu=tf.nn.relu(self.build_layers[-1])
                    self.build_layers.append(relu)
            elif layer.type=="Pooling":
                with tf.name_scope("pool_%d"%index):
                    pool=tf.nn.max_pool(self.build_layers[-1],ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="SAME")
                    self.build_layers.append(pool)
            elif layer.type=="Dropout":
                continue
            elif layer.type=="InnerProduct":
                with tf.name_scope("fc_%d"%index):
                    fc=self.build_fc(self.build_layers[-1],layer)
                    self.build_layers.append(fc)
            elif layer.type=="Softmax":
                with tf.name_scope("softmax_%d"%index):
                    softmax=tf.nn.softmax(self.build_layers[-1])
                    self.build_layers.append(softmax)
            else:
                assert 0,"Undefiend layer type %s"%(layer.type)
            index+=1
    def save(self,file_name):
        if len(self.build_layers)==0:
            raise RuntimeError,"empty build_layers, please run build() first"
        f=open(file_name,'wb')
        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        graph_def_s = graph_def.SerializeToString()
        f.write(graph_def_s)
        f.close()
        print "save tfmodel to %s"%(file_name)

if __name__=="__main__":
    transfer=caffe2tensorflow("caffe/vgg16.prototxt","model/vgg16.caffemodel")
    transfer.build()
    transfer.save("model/vgg16.tfmodel")
