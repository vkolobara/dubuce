import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## 1. definicija računskog grafa
# podatci i parametri
X = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# afini regresijski model
Y = a * X + b

# kvadratni gubitak
loss = (Y - Y_) ** 2

# optimizacijski postupak: gradijentni spust
trainer = tf.train.GradientDescentOptimizer(0.001)


# izracunaj gradijente sa tensorflowom
grads = trainer.compute_gradients(loss, [a, b])
train_op = trainer.apply_gradients(grads)

# izracunaj gradijente rucno
dB = tf.reduce_sum(2 * (Y - Y_))
dA = tf.reduce_sum(2 * (Y - Y_) * X)


# train_op = trainer.minimize(loss)


## 2. inicijalizacija parametara
sess = tf.Session()
sess.run(tf.initialize_all_variables())


## 3. učenje
# neka igre počnu!

for i in range(100):
    val_loss, _, val_a, val_b, gs, d_a, d_b = sess.run([loss, train_op, a, b, grads, dA, dB], feed_dict={X: [1, 2, 3, 4, 5, 6, 7, 8, 9], Y_: [3, 5, 7, 9, 11, 13, 15, 17, 19]})
    print(i, val_loss, val_a, val_b, gs, (d_a, d_b))
    p1 = tf.Print(loss, [grads], message="Gradijenti tensorflow: ")
    p2 = tf.Print(loss, [dA, dB], message="Gradijenti rucno: ")
    sess.run([p1, p2], feed_dict={X: [1, 2], Y_: [3, 5]})
    print("")