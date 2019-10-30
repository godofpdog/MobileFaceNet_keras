import os
import math
import numpy as np  
import tensorflow as tf 
from sklearn.metrics.pairwise import cosine_similarity
"""
need to check GetDistanceTensorflow
"""

class GetDistanceTensorflow():
    def __init__(self, metric_type='cos', embedding_dim=512, epslon=1e-8, decimals=6):
        self.embedding_dim = embedding_dim
        self.epslon = epslon
        self.decimals = decimals
        self.graph = tf.Graph()
        self.gpu_options = tf.GPUOptions(allow_growth=True)
        # self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=self.gpu_options))
        if metric_type in (0, '0', 'cos', 'cosine'):
            self._cosine_distance()
        elif metric_type in (1, '1', 'euclidean'):
            self._euclidean_distance()
        else:
            raise 'Undefined distance metric %d' % distance_metric

    def _cosine_distance(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.x1 = tf.placeholder(tf.float32, shape=(None, self.embedding_dim))
                self.x2 = tf.placeholder(tf.float32, shape=(None, self.embedding_dim))
                self.x1_norm = tf.sqrt(tf.reduce_sum(tf.square(self.x1), axis=1))
                self.x2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.x2), axis=1))
                self.dot = tf.reduce_sum(tf.multiply(self.x1, self.x2), axis=1)
                self.cos = tf.divide(self.dot, tf.multiply(self.x1_norm, self.x2_norm) + self.epslon)
                self.multiplier = tf.constant(10 ** self.decimals, dtype=self.cos.dtype)
                self.cos = tf.round(self.cos * self.multiplier) / self.multiplier
                self.pi = tf.constant(math.pi)
                self.distance = tf.divide(tf.acos(self.cos), self.pi)
                
    def _euclidean_distance(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.x1 = tf.placeholder(tf.float32, shape=(None, self.embedding_dim))
                self.x2 = tf.placeholder(tf.float32, shape=(None, self.embedding_dim)) 
                self.distance = tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), 1))
        
    def infer(self, input_x1, input_x2):
        res = self.sess.run(self.distance, feed_dict={self.x1:input_x1, self.x2:input_x2})
        return res

def get_distance(x1, x2, metric_type='cos'):
    if metric_type in (0, '0', 'cos', 'cosin'):
        dot = np.sum(np.multiply(x1, x2), axis=1)
        norm = np.linalg.norm(x1, axis=1) * np.linalg.norm(x2, axis=1)
        similarity = dot / (norm + 1e-8)
        dist = np.arccos(similarity) / math.pi
        return dist
        # return np.squeeze(np.arccos(np.round(cosine_similarity(x1, x2), 4)) / math.pi)
    elif metric_type in (1, '1', 'euclidean'):
        diff = np.subtract(x1, x2)
        return np.sum(np.square(diff), 1)
    else:
        raise 'Undefined distance metric %d' % distance_metric