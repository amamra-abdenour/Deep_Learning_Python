# import keras
# mnist = keras.datasets.mnist
# (x_train,y_train),(x_test,y_test) = mnist.load_data(1000)
############
def naive_relu(x):
    assert len(x.shape) == 2 #x is truely a 2d array
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = max(x[i,j],0)
    return x
############
def naive_add(x,y):
    assert x.shape == y.shape #x is truely a 2d array
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[i,j]
    return x
##
def dot_prod(x,y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z
##
def matrix_mul(x,y):
    assert x.shape[len(x.shape) - 1] == y.shape[0]
    z = np.zeros((x.shape[0],y.shape[1]))
    for i in range (x.shape[0]):
        for j in range (y.shape[1]):
            z[i,j] = dot_prod(x[i,:],y[:,j])
    return z
##########
if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(3,2)
    y = np.random.randn(2,6)
    print(x)
    # print(x.reshape(6,2))
    print(np.dot(x, y), " \n\n ")
    print(matrix_mul(x, y) - np.dot(x, y), " \n\n ")



