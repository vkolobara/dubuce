import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from lab4.utils import tile_raster_images
import math
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'jet'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, \
                     mnist.test.labels


def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)


def sample_prob(probs):
    """Uzorkovanje vektora x prema vektoru vjerojatnosti p(x=1) = probs"""
    return tf.to_float(tf.random_uniform(tf.shape(probs)) <= probs)


def draw_weights(W, shape, N, stat_shape, interpolation="bilinear"):
    """Vizualizacija težina
    W -- vektori težina
    shape -- tuple dimenzije za 2D prikaz težina - obično dimenzije ulazne slike, npr. (28,28)
    N -- broj vektora težina
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    """
    image = (tile_raster_images(
        X=W.T,
        img_shape=shape,
        tile_shape=(int(math.ceil(N / stat_shape[0])), stat_shape[0]),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)
    plt.axis('off')


def draw_reconstructions(ins, outs, states, shape_in, shape_state, N):
    """Vizualizacija ulaza i pripadajućih rekonstrukcija i stanja skrivenog sloja
    ins -- ualzni vektori
    outs -- rekonstruirani vektori
    states -- vektori stanja skrivenog sloja
    shape_in -- dimezije ulaznih slika npr. (28,28)
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    N -- broj uzoraka
    """
    plt.figure(figsize=(8, int(2 * N)))
    for i in range(N):
        plt.subplot(N, 4, 4 * i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Test input")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Reconstruction")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 3)
        plt.imshow(states[i].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("States")
        plt.axis('off')
    plt.tight_layout()


def draw_generated(stin, stout, gen, shape_gen, shape_state, N):
    """Vizualizacija zadanih skrivenih stanja, konačnih skrivenih stanja i pripadajućih rekonstrukcija
    stin -- početni skriveni sloj
    stout -- rekonstruirani vektori
    gen -- vektori stanja skrivenog sloja
    shape_gen -- dimezije ulaznih slika npr. (28,28)
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    N -- broj uzoraka
    """
    plt.figure(figsize=(8, int(2 * N)))
    for i in range(N):
        plt.subplot(N, 4, 4 * i + 1)
        plt.imshow(stin[i].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("set state")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 2)
        plt.imshow(stout[i][0:784].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("final state")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 3)
        plt.imshow(gen[i].reshape(shape_gen), vmin=0, vmax=1, interpolation="nearest")
        plt.title("generated visible")
        plt.axis('off')
    plt.tight_layout()


Nh = 100  # Broj elemenata prvog skrivenog sloja
h1_shape = (10, 10)
Nv = 784  # Broj elemenata vidljivog sloja
v_shape = (28, 28)
Nu = 5000  # Broj uzoraka za vizualizaciju rekonstrukcije

gibbs_sampling_steps = 1
alpha = 0.1

g1 = tf.Graph()
with g1.as_default():
    X1 = tf.placeholder("float", [None, 784])
    w1 = weights([Nv, Nh])
    vb1 = bias([Nv])
    hb1 = bias([Nh])

    h0_prob = tf.nn.sigmoid(tf.matmul(X1, w1) + hb1)
    h0 = sample_prob(h0_prob)
    h1 = h0

    for step in range(gibbs_sampling_steps):
        v1_prob = tf.nn.sigmoid(tf.matmul(h1, tf.transpose(w1)) + vb1)
        v1 = sample_prob(v1_prob)
        h1_prob = tf.nn.sigmoid(tf.matmul(v1, w1) + hb1)
        h1 = sample_prob(h1_prob)

    w1_positive_grad = tf.matmul(tf.transpose(X1), h0)
    w1_negative_grad = tf.matmul(tf.transpose(v1), h1)

    dw1 = (w1_positive_grad - w1_negative_grad) / tf.to_float(tf.shape(X1)[0])

    update_w1 = tf.assign_add(w1, alpha * dw1)
    update_vb1 = tf.assign_add(vb1, alpha * tf.reduce_mean(X1 - v1, 0))
    update_hb1 = tf.assign_add(hb1, alpha * tf.reduce_mean(h0 - h1, 0))

    out1 = (update_w1, update_vb1, update_hb1)

    v1_prob = tf.nn.sigmoid(tf.matmul(h1, tf.transpose(w1)) + vb1)
    v1 = sample_prob(v1_prob)

    err1 = X1 - v1_prob
    err_sum1 = tf.reduce_mean(err1 * err1)

    initialize1 = tf.global_variables_initializer()

batch_size = 100
epochs = 1
n_samples = mnist.train.num_examples
total_batch = int(n_samples / batch_size) * epochs

sess1 = tf.Session(graph=g1)
sess1.run(initialize1)

for i in range(total_batch):
    batch, label = mnist.train.next_batch(batch_size)
    err, _ = sess1.run([err_sum1, out1], feed_dict={X1: batch})

    if i % (int(total_batch / 10)) == 0:
        print(i, err)

w1s = w1.eval(session=sess1)
vb1s = vb1.eval(session=sess1)
hb1s = hb1.eval(session=sess1)
vr, h1s = sess1.run([v1_prob, h1], feed_dict={X1: teX[0:Nu, :]})

# vizualizacija težina
draw_weights(w1s, v_shape, Nh, h1_shape)

# vizualizacija rekonstrukcije i stanja
draw_reconstructions(teX, vr, h1s, v_shape, h1_shape, 5)


# vizualizacija jedne rekonstrukcije s postepenim dodavanjem doprinosa aktivnih skrivenih elemenata
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_rec(inp, title, size, Nrows, in_a_row, j):
    """ Iscrtavanje jedne iteracije u kreiranju vidljivog sloja
    inp - vidljivi sloj
    title - naslov sličice
    size - 2D dimenzije vidljiovg sloja
    Nrows - maks. broj redaka sličica
    in-a-row . broj sličica u jednom redu
    j - pozicija sličice u gridu
    """
    plt.subplot(Nrows, in_a_row, j)
    plt.imshow(inp.reshape(size), vmin=0, vmax=1, interpolation="nearest")
    plt.title(title)
    plt.axis('off')


def reconstruct(ind, states, orig, weights, biases):
    """ Slijedno iscrtavanje rekonstrukcije vidljivog sloja
    ind - indeks znamenke u orig (matrici sa znamenkama kao recima)
    states - vektori stanja ulaznih vektora
    orig - originalnalni ulazni vektori
    weights - matrica težina
    biases - vektori pomaka vidljivog sloja
    """
    j = 1
    in_a_row = 6
    Nimg = states.shape[1] + 3
    Nrows = int(np.ceil(float(Nimg + 2) / in_a_row))

    plt.figure(figsize=(12, 2 * Nrows))

    draw_rec(states[ind], 'states', h1_shape, Nrows, in_a_row, j)
    j += 1
    draw_rec(orig[ind], 'input', v_shape, Nrows, in_a_row, j)

    reconstr = biases.copy()
    j += 1
    draw_rec(sigmoid(reconstr), 'biases', v_shape, Nrows, in_a_row, j)

    for i in range(Nh):
        if states[ind, i] > 0:
            j += 1
            reconstr = reconstr + weights[:, i]
            titl = '+= s' + str(i + 1)
            draw_rec(sigmoid(reconstr), titl, v_shape, Nrows, in_a_row, j)
    plt.tight_layout()


reconstruct(0, h1s, teX, w1s, vb1s)  # prvi argument je indeks znamenke u matrici znamenki

# Vjerojatnost da je skriveno stanje uključeno kroz Nu ulaznih uzoraka
plt.figure()
tmp = (h1s.sum(0) / h1s.shape[0]).reshape(h1_shape)
plt.imshow(tmp, vmin=0, vmax=1, interpolation="nearest")
plt.axis('off')
plt.colorbar()
plt.title('vjerojatnosti (ucestalosti) aktivacije pojedinih neurona skrivenog sloja')

# Vizualizacija težina sortitranih prema učestalosti
tmp_ind = (-tmp).argsort(None)
draw_weights(w1s[:, tmp_ind], v_shape, Nh, h1_shape)
plt.title('Sortirane matrice tezina - od najucestalijih do najmanje koristenih')

# Generiranje uzoraka iz slučajnih vektora
r_input = np.random.rand(100, Nh)
r_input[r_input > 0.9] = 1  # postotak aktivnih - slobodno varirajte
r_input[r_input < 1] = 0
r_input = r_input * 20  # pojačanje za slučaj ako je mali postotak aktivnih

s = 10
i = 0
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s

out_1 = sess1.run((v1), feed_dict={X1: trX[0:5], h0: r_input})

# Emulacija dodatnih Gibbsovih uzorkovanja pomoću feed_dict
for i in range(1000):
    out_1_prob, out_1, hout1 = sess1.run((v1_prob, v1, h1), feed_dict={X1: out_1})

draw_generated(r_input, hout1, out_1_prob, v_shape, h1_shape, 5)

# DEEP BELIEF NETWORKS


Nh2 = Nh  # Broj elemenata drugog skrivenog sloja
h2_shape = h1_shape

gibbs_sampling_steps = 2
alpha = 0.1

g2 = tf.Graph()
with g2.as_default():
    X2 = tf.placeholder("float", [None, Nv])
    w1a = tf.Variable(w1s)
    vb1a = tf.Variable(vb1s)
    hb1a = tf.Variable(hb1s)
    w2 = weights([Nh, Nh2])
    hb2 = bias([Nh2])

    h1up_prob = tf.nn.sigmoid(tf.matmul(X2, w1a) + hb1a)
    h1up = sample_prob(h1up_prob)
    h2up_prob = tf.nn.sigmoid(tf.matmul(h1up, w2) + hb2)
    h2up = sample_prob(h2up_prob)
    h2down = h2up

    for step in range(gibbs_sampling_steps):
        h1down_prob = tf.nn.sigmoid(tf.matmul(h2down, tf.transpose(w2)) + hb2)
        h1down = sample_prob(h1down_prob)
        h2down_prob = tf.nn.sigmoid(tf.matmul(h1down, w2) + hb2)
        h2down = sample_prob(h2down_prob)

    w2_positive_grad = tf.matmul(tf.transpose(h1up), h2up)
    w2_negative_grad = tf.matmul(tf.transpose(h1down), h2down)

    dw2 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(h1up)[0])

    update_w2 = tf.assign_add(w2, alpha * dw2)
    update_hb1a = tf.assign_add(hb1a, alpha * tf.reduce_mean(h1up - h1down, 0))
    update_hb2 = tf.assign_add(hb2, alpha * tf.reduce_mean(h2up - h2down, 0))

    out2 = (update_w2, update_hb1a, update_hb2)

    # rekonsturkcija ulaza na temelju krovnog skrivenog stanja h3
    # ...
    # ...
    v_out_prob = tf.nn.sigmoid(tf.matmul(h1down, tf.transpose(w1a)) + vb1a)
    v_out = sample_prob(v_out_prob)

    err2 = X2 - v_out_prob
    err_sum2 = tf.reduce_mean(err2 * err2)

    initialize2 = tf.global_variables_initializer()

batch_size = 100
epochs = 1
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

sess2 = tf.Session(graph=g2)
sess2.run(initialize2)
for i in range(total_batch):
    batch, label = mnist.train.next_batch(batch_size)
    # iteracije treniranja
    # ...
    # ...
    err, _, batch = sess1.run([err_sum1, out1, v1], feed_dict={X1: batch})
    err, _ = sess2.run([err_sum2, out2], feed_dict={X2: batch})
    if i % (int(total_batch / 10)) == 0:
        print(i, err)

    w2s, hb1as, hb2s = sess2.run([w2, hb1a, hb2], feed_dict={X2: batch})
    vr2, h2downs = sess2.run([v_out_prob, h2down], feed_dict={X2: teX[0:Nu, :]})

# vizualizacija težina
draw_weights(w2s, h1_shape, Nh2, h2_shape, interpolation="nearest")

# vizualizacija rekonstrukcije i stanja
draw_reconstructions(teX, vr2, h2downs, v_shape, h2_shape, 5)

# Generiranje uzoraka iz slučajnih vektora
r_input = np.random.rand(100, Nh)
r_input[r_input > 0.9] = 1  # postotak aktivnih - slobodno varirajte
r_input[r_input < 1] = 0
r_input = r_input * 20  # pojačanje za slučaj ako je mali postotak aktivnih

s = 10
i = 0
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s

x2 = sess1.run((v1), feed_dict={X1: trX[0:5], h0:r_input})
out_2 = sess2.run((v_out), feed_dict={X2: x2})

# Emulacija dodatnih Gibbsovih uzorkovanja pomoću feed_dict
for i in range(1000):
    out_2_prob, out_2, hout2 = sess2.run((v_out_prob, v_out, h2up), feed_dict={X2: out_2})


draw_generated(r_input, hout2, out_2_prob, v_shape, h2_shape, 5)

#

beta = 0.01

g3 = tf.Graph()
with g3.as_default():
    X3 = tf.placeholder("float", [None, Nv])
    r1_up = tf.Variable(w1s)
    w1_down = tf.Variable(tf.transpose(w1s))
    w2a = tf.Variable(w2s)
    hb1_up = tf.Variable(hb1s)
    hb1_down = tf.Variable(hb1as)
    vb1_down = tf.Variable(vb1s)
    hb2a = tf.Variable(hb2s)

    # wake pass
    h1_up_prob = tf.nn.sigmoid(tf.matmul(X3, r1_up) + hb1_up)
    h1_up =  sample_prob(h1_up_prob) # s^{(n)} u pripremi
    v1_up_down_prob = tf.nn.sigmoid(tf.matmul(h1_up, w1_down) + vb1_down)
    v1_up_down = sample_prob(v1_up_down_prob) # s^{(n-1)\mathit{novo}} u tekstu pripreme

    # top RBM Gibs passes
    h2_up_prob = tf.nn.sigmoid(tf.matmul(h1_up, w2a) + hb2a)
    h2_up = sample_prob(h2_up_prob)
    h2_down = h2_up

    for step in range(gibbs_sampling_steps):
        h1_down_prob = tf.nn.sigmoid(tf.matmul(h2_down, tf.transpose(w2a)) + hb2a)
        h1_down = sample_prob(h1_down_prob)
        h2_down_prob = tf.nn.sigmoid(tf.matmul(h1_down, w2a) + hb2a)
        h2_down = sample_prob(h2_down_prob)

        # sleep pass
    v1_down_prob = tf.nn.sigmoid(tf.matmul(h1_down, w1_down) + vb1_down)
    v1_down = sample_prob(v1_down_prob) # s^{(n-1)} u pripremi
    h1_down_up_prob = tf.nn.sigmoid(tf.matmul(v1_down, r1_up) + hb1_up)
    h1_down_up = sample_prob(h1_down_up_prob) # s^{(n)\mathit{novo}} u pripremi

    # generative weights update during wake pass
    update_w1_down = tf.assign_add(w1_down, beta * tf.matmul(tf.transpose(h1_up), X3 - v1_up_down_prob) / tf.to_float(
        tf.shape(X3)[0]))
    update_vb1_down = tf.assign_add(vb1_down, beta * tf.reduce_mean(X3 - v1_up_down_prob, 0))

    # top RBM update
    w2_positive_grad = tf.matmul(tf.transpose(h1_up), h2_up)
    w2_negative_grad = tf.matmul(tf.transpose(h1_down), h2_down)
    dw3 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(h1_up)[0])
    update_w2 = tf.assign_add(w2a, beta * dw3)
    update_hb1_down = tf.assign_add(hb1_down, beta * tf.reduce_mean(h1_up - h1_down, 0))
    update_hb2 = tf.assign_add(hb2a, beta * tf.reduce_mean(h2_up - h2_down, 0))

    # recognition weights update during sleep pass
    update_r1_up = tf.assign_add(r1_up,
                                 beta * tf.matmul(tf.transpose(v1_down_prob), h1_down - h1_down_up) / tf.to_float(
                                     tf.shape(X3)[0]))
    update_hb1_up = tf.assign_add(hb1_up, beta * tf.reduce_mean(h1_down - h1_down_up, 0))

    out3 = (update_w1_down, update_vb1_down, update_w2, update_hb1_down, update_hb2, update_r1_up, update_hb1_up)

    err3 = X3 - v1_down_prob
    err_sum3 = tf.reduce_mean(err3 * err3)

    initialize3 = tf.global_variables_initializer()

batch_size = 100
epochs = 1

n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

sess3 = tf.Session(graph=g3)
sess3.run(initialize3)
for i in range(total_batch):
    # ...
    err, _, batch = sess1.run([err_sum1, out1, v1], feed_dict={X1: batch})
    err, _ = sess3.run([err_sum3, out3], feed_dict={X3: batch})

    if i % (int(total_batch / 10)) == 0:
        print(i, err)

    w2ss, r1_ups, w1_downs, hb2ss, hb1_ups, hb1_downs, vb1_downs = sess3.run(
        [w2a, r1_up, w1_down, hb2a, hb1_up, hb1_down, vb1_down], feed_dict={X3: batch})
    vr3, h2_downs, h2_down_probs = sess3.run([v1_down_prob, h2_down, h2_down_prob], feed_dict={X3: teX[0:Nu, :]})

# vizualizacija težina
draw_weights(r1_ups, v_shape, Nh, h1_shape)
draw_weights(w1_downs.T, v_shape, Nh, h1_shape)
draw_weights(w2ss, h1_shape, Nh2, h2_shape, interpolation="nearest")

# vizualizacija rekonstrukcije i stanja
Npics = 5
plt.figure(figsize=(8, 12 * 4))
for i in range(20):
    plt.subplot(20, Npics, Npics * i + 1)
    plt.imshow(teX[i].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Test input")
    plt.subplot(20, Npics, Npics * i + 2)
    plt.imshow(vr[i][0:784].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Reconstruction 1")
    plt.subplot(20, Npics, Npics * i + 3)
    plt.imshow(vr2[i][0:784].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Reconstruction 2")
    plt.subplot(20, Npics, Npics * i + 4)
    plt.imshow(vr3[i][0:784].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Reconstruction 3")
    plt.subplot(20, Npics, Npics * i + 5)
    plt.imshow(h2_downs[i][0:Nh2].reshape(h2_shape), vmin=0, vmax=1, interpolation="nearest")
    plt.title("Top states 3")
plt.tight_layout()

x3 = sess1.run((v1), feed_dict={X1: trX[0:5], h0:r_input})
out_3 = sess3.run((v1_down), feed_dict={X3: x3})

# Emulacija dodatnih Gibbsovih uzorkovanja pomoću feed_dict
for i in range(1000):
    out_3_prob, out_3, hout3 = sess3.run((v1_down_prob, v1_down, h2_up), feed_dict={X3: out_3})

draw_generated(r_input, hout3, out_3_prob, v_shape, h2_shape, 5)

plt.show()
