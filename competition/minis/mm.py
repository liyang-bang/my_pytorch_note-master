#
# import numpy as np
# import tensorflow as tf
#
#
#
#
#
#
# def randomize(dataset, labels):
#
#    permutation = np.random.permutation(labels.shape[0])
#
#    shuffled_dataset = dataset[permutation, :, :]
#
#    shuffled_labels = labels[permutation]
#
#    return shuffled_dataset, shuffled_labels
#
# def one_hot_encode(np_array):
#
#    return (np.arange(10) == np_array[:,None]).astype(np.float32)
#
# def reformat_data(dataset, labels, image_width, image_height, image_depth):
#
#    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
#
#    np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
#
#    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
#
#    return np_dataset, np_labels
#
# def flatten_tf_array(array):
#
#    shape = array.get_shape().as_list()
#
#    return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])
#
# def accuracy(predictions, labels):
#
#    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
#
#
#
# mnist_folder = './data/mnist/'
#
# mnist_image_width = 28
#
# mnist_image_height = 28
#
# mnist_image_depth = 1
#
# mnist_num_labels = 10
# mnist_image_size = 10
#
# mndata = MNIST(mnist_folder)
#
# mnist_train_dataset_, mnist_train_labels_ = mndata.load_training()
#
# mnist_test_dataset_, mnist_test_labels_ = mndata.load_testing()
#
# mnist_train_dataset, mnist_train_labels = reformat_data(mnist_train_dataset_, mnist_train_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
#
# mnist_test_dataset, mnist_test_labels = reformat_data(mnist_test_dataset_, mnist_test_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
#
# print("There are {} images, each of size {}".format(len(mnist_train_dataset), len(mnist_train_dataset[0])))
#
# print("Meaning each image has the size of 28*28*1 = {}".format(mnist_image_size*mnist_image_size*1))
#
# print("The training set contains the following {} labels: {}".format(len(np.unique(mnist_train_labels_)), np.unique(mnist_train_labels_)))
#
# print('Training set shape', mnist_train_dataset.shape, mnist_train_labels.shape)
#
# print('Test set shape', mnist_test_dataset.shape, mnist_test_labels.shape)
#
# train_dataset_mnist, train_labels_mnist = mnist_train_dataset, mnist_train_labels
#
# test_dataset_mnist, test_labels_mnist = mnist_test_dataset, mnist_test_labels
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D
# # from tensorflow.keras.datasets import mnist
# # #### read and preprocess data
# # (x_train,y_train),(x_test,y_test)=mnist.load_data()
# # x_train,x_test=x_train.reshape([-1,28,28,1])/255.0,x_test.reshape([-1,28,28,1])/255.0
# # #### contruct the model
# # model=Sequential()
# # model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
# # model.add(MaxPool2D(pool_size=(2,2)))
# # model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
# # model.add(MaxPool2D(pool_size=(2,2)))
# # model.add(Flatten())     ##把卷积完之后的很多"小图片合并拉直"
# # model.add(Dense(10))
# # #### compile ; fit ; evaluate
# # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# #               optimizer="adam",
# #               metrics=['accuracy'])
# # model.fit(x=x_train,y=y_train,batch_size=100,epochs=20,verbose=2)
# # model.evaluate(x=x_test,y=y_test,verbose=2)