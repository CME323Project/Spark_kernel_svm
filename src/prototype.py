#from pyspark import SparkContext, SparkConf
import numpy as np

class RBFKernelFunc:
    def __init__(self, sigma = 1):
        self.sigma = sigma
    def __call__(self, x_1, x_2):
        return np.exp(float(-np.dot(x_1-x_2, x_1 - x_2)) / (2 * self.sigma ** 2))

class KernelSVM:
    def __init__(self, sparkContext, l, kernel = 'rbf', kernel_param_list = [1]):
        if not kernel=='rbf':
            raise ValueError('Only rbf kernel is supported!')
        self.sc = sparkContext
        self.l = l
        self.kernel = RBFKernelFunc(kernel_param_list[0])
        self.model = sc.parallelize([0])
    def train(self, data, T):
        """
        input:
            data: RDD of tuples (x, y), where x is an array (feature) and y is either -1 or 1 (response)
            T: the iterations to run for SGD
        After called, the object will represent a trained model and can be used to predict.
        """
        #index the data so that looking up is easy, and initialize support vetors.
        working_data = data.zipWithUniqueId().map(lambda x: (x[1], (x(0), 0))).persist()
        





if __name__ == '__main__':
    test = RBFKernelFunc(2)
    print test(np.array([1,2,3]), np.array([4,3,2]))

