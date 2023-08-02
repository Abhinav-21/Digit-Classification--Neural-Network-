# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# train_path = os.path.join('Data', 'train.csv')
# test_path = os.path.join('Data', 'test.csv')

# data = np.array(pd.read_csv(train_path))
# np.random.shuffle(data)
# train_data = data[1000:].T
# test_data = data[0:1000].T

# Y_train = train_data[0]
# X_train = train_data[1:]

# Y_test = test_data[0]
# X_test = test_data[1:]


# def init_params():
    
#     W1 = np.random.rand(10, 784) - 0.5
#     b1 = np.random.rand(10, 1) - 0.5
#     W2 = np.random.rand(10, 10) - 0.5
#     b2 = np.random.rand(10, 1) - 0.5
    
#     return W1, b1, W2, b2


# def ReLU(Z):
    
#     return np.maximum(0, Z)


# def softmax(Z):
#     exp = np.exp(Z - np.max(Z))
#     return exp/exp.sum(axis=0)  


# def forward_prop(W1, b1, W2, b2, X):
    
#     Z1 = W1.dot(X) + b1
#     A1 = ReLU(Z1)
    
#     Z2 = W2.dot(A1) + b2
#     A2 = softmax(Z2)
    
#     return Z1, A1, Z2, A2


# def one_hot(Y):
    
#     one_hot_Y = np.zeros((Y.max()+1,Y.size)) 
#     one_hot_Y[Y,np.arange(Y.size)] = 1 
    
#     return one_hot_Y
    
    
# def derivative_ReLU(Z):
    
#     return Z > 0
    
    
# def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    
#     m = Y.size
#     one_hot_Y = one_hot(Y)
#     dZ2 = A2 - one_hot_Y
#     dW2 = 1 / m * dZ2.dot(A1.T)
#     db2 = 1 / m * np.sum(dZ2, 2)
#     dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
#     dW1 = 1 / m * dZ1.dot(X.T)
#     db1 = 1 / m * np.sum(dZ1, 2)
    
#     return dW1, db1, dW2, db2


# def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    
#     W1 = W1 - alpha * dW1
#     b1 = b1 - alpha * np.reshape(db1, (10,1))
#     W2 = W2 - alpha * dW2
#     b2 = b2 - alpha * np.reshape(db2, (10,1))
            
#     return W1, b1, W2, b2


# def prediction(A):
    
#     return np.argmax(A, 0)


# def accuracy(predictions, Y):
    
#     print(predictions, Y)
    
#     return np.sum(predictions == Y)*100 / Y.size


# def gradient_descent(X, Y, iterations, alpha):
    
#     W1, b1, W2, b2 = init_params()
#     for i in range(iterations):
#         Z1, A1, Z2, Z2 = forward_prop(W1, b1, W2, b2, X)
#         dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, Z2, W2, X, Y)
#         W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
#         if i % 10 == 0:
#             print(accuracy(prediction, Y))
    
#     return W1, b1, W2, b2


# W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 100, 0.1)

# def make_predictions(X, W1 ,b1, W2, b2):
#     _, _, _, A2 = forward_prop(X, W1, b1, W2, b2)
#     predictions = prediction(A2)
#     return predictions

# def show_prediction(index,X, Y, W1, b1, W2, b2):
#     # None => cree un nouvel axe de dimension 1, cela a pour effet de transposer X[:,index] qui un np.array de dimension 1 (ligne) et qui devient un vecteur (colonne)
#     #  ce qui correspond bien a ce qui est demande par make_predictions qui attend une matrice dont les colonnes sont les pixels de l'image, la on donne une seule colonne
#     vect_X = X[:, index,None]
#     prediction = make_predictions(vect_X, W1, b1, W2, b2)
#     label = Y[index]
#     print("Prediction: ", prediction)
#     print("Label: ", label)

#     current_image = vect_X.reshape((28, 28)) * 255

#     plt.gray()
#     plt.imshow(current_image, interpolation='nearest')
#     plt.show()






