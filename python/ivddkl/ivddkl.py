import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy import optimize
from scipy.stats import laplace
from scipy.stats import norm

tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()


class IVDDKLKernel:

    def __init__(self, nl, c, beta=25.0, lossKLD = True, distribution = 'laplace'):

        self.nl = nl
        self.beta = tf.Variable(beta, trainable=False, name="beta")
        self.c = tf.Variable(c, trainable=False, dtype='float32')

        self.lossKLD = lossKLD
        self.distribution = distribution

        self.gamma = None

        self.centers = None
        self.radius = tf.Variable(0.0, trainable=True, name="radius")
        self.alphas = None

    #Initialization of the landmarks
    def initCenters(self, x, mode='kmeans'):
        if mode == 'random':
            self.centers = tf.Variable(tf.zeros(shape=(self.nl, x.shape[1])), name="centers", dtype='float32')
            n = x.shape[0]
            mask = np.random.choice(range(n), size=self.nl, replace=False)
            centers = x[mask, :]
        elif mode == 'kmeans':
            self.centers = tf.Variable(tf.zeros(shape=(self.nl, x.shape[1])), name="centers", dtype='float32')
            kmeans = KMeans(n_clusters=self.nl, random_state=0).fit(x)
            centers = kmeans.cluster_centers_
        else:
            print("Mode: " + mode + " is invalid.")
            centers = None
            exit(-1)

        self.centers.assign(centers)

    #Initialization of sigma
    def initSigma(self, x, mode='log', s_value=10):
        n = x.shape[0]
        dmax_landmarks = max([np.linalg.norm(c1 - c2) for c1 in self.centers.numpy() for c2 in self.centers.numpy()])
        dmax = np.max(np.triu(distance_matrix(x,x)))#max([np.linalg.norm(x1 - x2) for x1 in x for x2 in x])

        if mode == 'log':
            s = dmax / np.log(n)
            gamma = -1/(2*(s**2))
        elif mode == 'log_landmarks':
            s = dmax_landmarks / np.log(self.nl)
            gamma = -1/(2*(s**2))
        elif mode == 'sqrt':
            s = 8*(dmax / np.sqrt(2*n))
            gamma = -1/(2*(s**2))
        elif mode == 'ocsvm':
            gamma = -1.0 / (tf.math.reduce_variance(self.centers) * x.shape[1])
        elif mode == 'exact':
            s = s_value
            gamma = -1/(2*(s**2))
        elif mode == "tracecriterion":

            def h(sigma):
                N = x.shape[0]

                kmeans = KMeans(n_clusters=5, random_state=0).fit(x)
                c = kmeans.cluster_centers_
                dist = distance_matrix(c, c)**2
                den = 2 * (sigma ** 2)
                U = np.exp(-1 * (dist / den))

                U_der = (1 / (sigma ** 3)) * (dist * U)

                sum_1 = 0
                sum_2 = 0
                for i in range(N):
                    sample = tf.reshape(x[i,:], (1, x.shape[1]))
                    dist = distance_matrix(c, sample)**2
                    W = np.exp(-1 * (dist / den))

                    W_der = (1 / (sigma ** 3)) * (dist * W)

                    B = np.matmul(np.linalg.inv(U), W)

                    v_1 = np.matmul(np.transpose(B), W_der)
                    v_2 = np.matmul(np.matmul(np.transpose(B), U_der), B)

                    sum_1 += v_1
                    sum_2 += v_2

                termine_1 = (2 * sum_1) / N
                termine_2 = sum_2 / N
                res = termine_1 - termine_2
                return -res

            s = optimize.fmin(h, 1.0, maxiter=100)
            gamma = -1/(2*(s**2))
            print(s)

        elif mode == 'mean':
            s1 = dmax / np.log(n)
            s2 = dmax / np.sqrt(self.nl)

            def h(sigma):
                N = x.shape[0]

                kmeans = KMeans(n_clusters=5, random_state=0).fit(x)
                c = kmeans.cluster_centers_
                dist = distance_matrix(c, c)**2
                den = 2 * (sigma ** 2)
                U = np.exp(-1 * (dist / den))

                U_der = (1 / (sigma ** 3)) * (dist * U)

                sum_1 = 0
                sum_2 = 0
                for i in range(N):
                    sample = x[i,:].reshape(1, x.shape[1])
                    dist = distance_matrix(c, sample)**2
                    W = np.exp(-1 * (dist / den))

                    W_der = (1 / (sigma ** 3)) * (dist * W)

                    B = np.matmul(np.linalg.inv(U), W)

                    v_1 = np.matmul(np.transpose(B), W_der)
                    v_2 = np.matmul(np.matmul(np.transpose(B), U_der), B)

                    sum_1 += v_1
                    sum_2 += v_2

                termine_1 = (2 * sum_1) / N
                termine_2 = sum_2 / N
                res = termine_1 - termine_2
                return -res

            s3 = optimize.fmin(h, 1.0, maxiter=100)

            s = (s1+s2+s3)/3
            gamma = -1/(2*(s**2))

        else:
            print("Mode: " + mode + " is invalid.")
            s = -2
            exit(-2)

        self.gamma = tf.Variable(gamma, dtype='float32')

    #Inizialization of alphas
    def initAlphas(self):
        self.alphas = tf.Variable(tf.zeros(shape=(self.nl, 1)), trainable=True, name="alphas")
        values = np.empty([self.nl, 1])
        for i in range(0, self.nl):
            values[i,0] = 1/self.nl

        self.alphas.assign(values)

    #Computation of the kernel matrix
    def computeKernelMatrix(self, x, y, kernel='rbf'):
        #compute the paiwise distances between x and y
        size_x = tf.shape(x)[0]
        size_y = tf.shape(y)[0]
        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, size_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, size_x]))
        yy = tf.transpose(yy, perm=[2, 1, 0])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)
        dist = tf.sqrt(tf.reduce_sum(square_diff, 1))

        #create the RBF kernel matrix
        if kernel == 'rbf':
            kernel_matrix = tf.exp(tf.multiply(self.gamma, dist))
        else:
            kernel_matrix = dist

        return kernel_matrix

    #Decision function
    def f(self, dist):
        res = tf.subtract(dist, self.radius) #tf.divide(tf.subtract(dist, self.radius), tf.sqrt(dist)) #
        return res

    #Distances with kernel expansion
    def computeDistance(self, x):
        Kr = self.computeKernelMatrix(self.centers, self.centers)
        Ki = self.computeKernelMatrix(x, self.centers)
        Kii = tf.constant(1.0) #self.computeKernelMatrix(x, x) #tf.constant(1.0)

        first_term = tf.matmul(tf.matmul(tf.transpose(self.alphas), Kr), self.alphas)
        second_term = tf.matmul(tf.multiply(2.0, Ki), self.alphas)
        third_term = Kii
        dist = first_term - second_term + third_term
        return tf.abs(dist)

    #Probabilies from decision function - logistic regression
    def p(self, x):
        one = tf.constant(1.0)
        prob = tf.divide(one, one + tf.exp(tf.multiply(self.beta, self.f(self.computeDistance(x)))))
        return prob

    #Compute the total loss
    def lossFunction(self, x):
        #Here we assume that the probability distribution is similar to half Laplace distribution between 0 and 1
        if self.lossKLD:
            prob = tf.sort(self.p(x), axis=0)
            log_p = tf.reduce_sum(tf.math.log(prob) + 10e-6)

            steps = 1 / prob.shape[0]
            x_range = np.arange(0, 1, steps)

            if self.distribution == 'laplace':
                p = np.array(laplace.pdf(x_range, 1, 0.2), dtype="float32")
            elif self.distribution == 'gaussian':
                p = np.array(norm.pdf(x_range, 1, 0.2), dtype="float32")

            kl_divergence = tf.reduce_sum(
                tf.where(p == 0, tf.zeros(x.shape[0], tf.float32), p * tf.math.log(p / prob)))

            l2_1 = kl_divergence
            l2_2 = tf.multiply(self.c, log_p)

            l2 = tf.multiply(tf.subtract(kl_divergence, tf.multiply(self.c, log_p)), 1000)
        #Here the radius is considered in the loss function
        else:
            prob = self.p(x)
            log_p = tf.reduce_sum(tf.math.log(prob) + 10e-6)
            l2_1 = tf.pow(self.radius, 2)
            l2_2 = tf.multiply(self.c, log_p)
            l2 = tf.subtract(tf.pow(self.radius, 2), tf.multiply(self.c, log_p))

        #Deep loss - in this simple class there is not the deep part
        l1 = 0

        #Penalization term for negative radius
        l3 = tf.minimum(0, self.radius)

        return l1, l2, l3, l2_1, l2_2

    #Compute the gradient of the loss function
    def grad(self, inputs):
        trainable = []

        with tf.GradientTape() as tape:
            l1, l2, l3, l2_1, l2_2 = self.lossFunction(inputs)
            loss_value = l1 + l2 - l3
            trainable.append(self.radius)
            trainable.append(self.alphas)

        return l1, l2, l3, l2_1, l2_2, tape.gradient(loss_value, trainable)

    #Compute the loss and perform the gd
    def update(self, train_dataset, optimizer):
        loss_1_avg = tf.keras.metrics.Mean()
        loss_2_avg = tf.keras.metrics.Mean()
        loss_3_avg = tf.keras.metrics.Mean()

        loss_21_avg = tf.keras.metrics.Mean()
        loss_22_avg = tf.keras.metrics.Mean()

        for x in train_dataset:
            trainable = []

            l1, l2, l3, l2_1, l2_2, grads = self.grad(x)
            trainable.append(self.radius)
            trainable.append(self.alphas)

            optimizer.apply_gradients(zip(grads, trainable))

            loss_1_avg.update_state(l1)
            loss_2_avg.update_state(l2)
            loss_3_avg.update_state(l3)
            loss_21_avg.update_state(l2_1)
            loss_22_avg.update_state(l2_2)

        return loss_1_avg.result().numpy(), loss_2_avg.result().numpy(), loss_3_avg.result().numpy(), \
               loss_21_avg.result().numpy(), loss_22_avg.result().numpy()

    #Train the model
    def training(self, x_train, epochs=20, batch_size=100, lr=0.001, center_mode='kmeans',
                 sigma_mode='log', earlystopping=True, thr_earlystopping=[0.80,0.90]):

        #Definition of the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        #Initialization of the class attributes
        self.initCenters(x_train, mode=center_mode)
        self.initAlphas()
        self.initSigma(x_train, mode=sigma_mode)

        #Creation of the batches
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_dataset_batch = train_dataset.batch(batch_size)

        #Init variables for rollback
        p_inside_old = 0.0
        radius_old = self.radius
        beta_old = self.beta
        alphas_old = self.alphas
        p_inside = 0.0

        #Training
        for epoch in tqdm(range(1, epochs+1)):
            #Stop if the percentage of sample inside the sphere is among the requested range
            if earlystopping and p_inside >= thr_earlystopping[0] and p_inside <= thr_earlystopping[1]:
                break
            #If too much samples are inside: rollback and reduce the learning rate
            elif earlystopping and p_inside > thr_earlystopping[1]:
                self.radius.assign(radius_old)
                self.beta.assign(beta_old)
                self.alphas.assign(alphas_old)
                p_inside = p_inside_old
                lr = lr/10
                optimizer.lr.assign(lr)
            #Otherwise optimize and continue with the training
            else:
                #Training in one step:
                p_inside_old = p_inside
                radius_old = self.radius.numpy()
                beta_old = self.beta.numpy()
                alphas_old = self.alphas.numpy()

                train_dataset_batch.shuffle(x_train.shape[0], seed=123)
                loss_value_train1, loss_value_train2, loss_value_train3,\
                loss_value_train21, loss_value_train22 = self.update(train_dataset_batch, optimizer)

                p_train = self.p(x_train).numpy()
                p_inside = self._percInside(p_train)

                #print("Losses: ", loss_value_train1, loss_value_train2, loss_value_train3)
                print("Perc inside: ", p_inside)
                print("Learning rate: ", lr)

        return self.radius, p_inside

    def testing(self, x_test):

        p_test = self.p(x_test).numpy()
        s_test = self.f(self.computeDistance(x_test)).numpy()

        return p_test, s_test

    def _percInside(self, probabilities):
        sum = 0
        for el in probabilities:
            if el >= 0.5:
                sum += 1

        return sum / len(probabilities)