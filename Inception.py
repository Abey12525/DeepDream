
# coding: utf-8

# In[49]:


import tensorflow as tf
import cv2
import numpy as np 
tf.__version__
import os 
data_dir = "inception/5h/"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "tensorflow_inception_graph.pb"


class Inception5h:
    """inception model is a Deep Neural Network which has already been trained for
    classifying images into 1000 different categories"""
    
    #Name of the tensor for feedin the input
    tensor_name_input_image = "input:0"
    
    #layers in Inception 
    layer_names = ['conv2d0', 'conv2d1', 'conv2d2',
                   'mixed3a', 'mixed3b',
                   'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e',
                   'mixed5a', 'mixed5b']
    def __init__(self):
        #create a new TensorFlow computational graph
        self.graph = tf.Graph()
        
        #set the new graph as default.
        with self.graph.as_default():
            
            #Open the graph-def file for binary reading
            path = os.path.join(data_dir, path_graph_def)
            
            with tf.gfile.FastGFile(path , 'rb') as file:
                #first we need to create an empty graph-def 
                graph_def = tf.GraphDef()
                
                #Then load proto-buf file into graph-def
                
                graph_def.ParseFromString(file.read())
                
                #Finally import the graph-def to default TensorFlow graph
                
                tf.import_graph_def(graph_def, name='')
                 #now self.graph holds the Inception Model from the poto-buf file.
            #get a reference to the tensor for inputting images to graph
            
            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)
                
            #Get references to the tensors for the commonly used layers
            self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]
            
    def create_feed_dict(self, image = None):
        #create and return a feed dict with an image input is a 3-dim array already decoded
        image = np.expand_dims(image, axis = 0)
        
        #image is passed in as a 3-dim array of raw pixel-values
        
        feed_dict = {self.tensor_name_input_image: image}
        
        return feed_dict
    
    def get_gradient(self, tensor):
         # get the gradient of the given tensor with respect to the input image.
         # This allows to modify the input image so as to maximize the given tensor
        
        #set the graph as default so we can add operations to it.
        with self.graph.as_default():
            #square the tensor-values.
            #You can try and remove this to see the effect
            tensor = tf.square(tensor)
            
            #average the tensor so we get a single scalar value.
            tensor_mean = tf.reduce_mean(tensor)
            
            gradient = tf.gradients(tensor_mean, self.input)[0]
        return gradient
        
print("+=+=+=+=Inception defined+=+=+=+=")
