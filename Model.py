import random
import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, stateSize, actionSize, learningRate, name):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        self.name = name

        with tf.compat.v1.variable_scope(self.name):
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *self.stateSize], name="inputs")

            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, self.actionSize], name="actions")

            self.targetQ = tf.compat.v1.placeholder(tf.float32, [None], name="target")

            self.ISWeights_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name='ISWeights')

            self.dense1 = tf.compat.v1.layers.dense(inputs=self.inputs_,
                                                    units=16,
                                                    activation=tf.nn.elu,
                                                    kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                        scale=1.0, mode="fan_avg", distribution="uniform"),
                                                    name="dense1")
            self.dense2 = tf.compat.v1.layers.dense(inputs=self.dense1,
                                                    units=16,
                                                    activation=tf.nn.elu,
                                                    kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                        scale=1.0, mode="fan_avg", distribution="uniform"),
                                                    name="dense2")
            self.output = tf.compat.v1.layers.dense(inputs=self.dense2,
                                                    units=self.actionSize,
                                                    kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                        scale=1.0, mode="fan_avg", distribution="uniform"),
                                                    activation=None,
                                                    name="outputs")

            self.QValue = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            self.absoluteError = abs(self.QValue - self.targetQ)

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.square(self.targetQ - self.QValue))

            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learningRate).minimize(self.loss)


class PrioritisedMemory:
    e = 0.01
    a = 0.06
    b = 0.04
    bIncr = 0.001
    maxPrior = 1.0

    def __init__(self, capacity):
        self.sumTree = SumTree(capacity)
        self.capacity = capacity

    #додає досвід до пріоритизованої пам'яті.

    def store(self, experience):

        maxPriority = np.max(self.sumTree.tree[self.sumTree.indexOfFirstData:])

        if maxPriority == 0:
            maxPriority = self.maxPrior

        self.sumTree.add(maxPriority, experience)

    # вибирає пакет досвіду з пам'яті з підвищеною ймовірністю вибору досвіду з важливими прикладами.

    def sample(self, n):
        batch = []
        batchIndexes = np.zeros([n], dtype=np.int32)
        batchISWeights = np.zeros([n, 1], dtype=np.float32)

        totalPriority = self.sumTree.total_priority()
        prioritySegmentSize = totalPriority / n

        self.b += self.bIncr
        self.b = min(self.b, 1)

        minPriority = np.min(np.maximum(self.sumTree.tree[self.sumTree.indexOfFirstData:], self.e))
        minProbability = minPriority / self.sumTree.total_priority()

        maxWeight = (minProbability * n) ** (-self.b)
        for i in range(n):
            segmentMin = prioritySegmentSize * i
            segmentMax = segmentMin + prioritySegmentSize

            value = np.random.uniform(segmentMin, segmentMax)

            treeIndex, priority, data = self.sumTree.getLeaf(value)

            samplingProbability = priority / totalPriority

            batchISWeights[i, 0] = np.power(n * samplingProbability, -self.b) / maxWeight

            batchIndexes[i] = treeIndex
            experience = [data]
            batch.append(experience)

        return batchIndexes, batch, batchISWeights

    #оновлює пріоритети в сумарному дереві за допомогою абсолютних помилок.

    def batchUpdate(self, treeIndexes, absoluteErrors):
        absoluteErrors += self.e  # do this to avoid 0 values
        clippedErrors = np.minimum(absoluteErrors, self.maxPrior)

        priorities = np.power(clippedErrors, self.a)
        for treeIndex, priority in zip(treeIndexes, priorities):
            self.sumTree.update(treeIndex, priority)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 2 * capacity - 1
        self.tree = np.zeros(self.size)
        self.data = np.zeros(capacity, dtype=object)
        self.dataPointer = 0
        self.indexOfFirstData = capacity - 1

    #додає новий елемент до сумарного дерева

    def add(self, priority, data):
        treeIndex = self.indexOfFirstData + self.dataPointer
        self.data[self.dataPointer] = data
        self.update(treeIndex, priority)
        self.dataPointer += 1
        self.dataPointer = self.dataPointer % self.capacity

    #оновлює значення вузла сумарного дерева
    #та рекурсивно оновлює значення батьківських вузлів в дереві

    def update(self, index, priority):
        change = priority - self.tree[index]
        self.tree[index] = priority

        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += change

    #знаходить лист сумарного дерева, який відповідає заданому значенню

    def getLeaf(self, value):
        parent = 0
        LChild = 1
        RChild = 2

        while LChild < self.size:
            if self.tree[LChild] >= value:
                parent = LChild
            else:
                value -= self.tree[LChild]
                parent = RChild

            LChild = 2 * parent + 1
            RChild = 2 * parent + 2

        treeIndex = parent
        dataIndex = parent - self.indexOfFirstData

        return treeIndex, self.tree[treeIndex], self.data[dataIndex]

    #повертає суму значень у кореневому вузлі сумарного дерева,
    #що відповідає загальному пріоритету всього дерева

    def total_priority(self):
        return self.tree[0]
