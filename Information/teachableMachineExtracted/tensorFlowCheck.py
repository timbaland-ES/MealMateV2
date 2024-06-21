import tensorflow as tf

# Simple TensorFlow example
hello = tf.constant('Hello, TensorFlow!')
sess = tf.compat.v1.Session()
print(sess.run(hello))

