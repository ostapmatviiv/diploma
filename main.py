from pyglet.window import key
from pyglet.gl import *
import pyglet
from Globals import *
import pygame
from Game import Game
import random
import os
import numpy as np
from collections import deque
import tensorflow as tf
import pandas as pd
from Model import DQN, PrioritisedMemory

df = pd.DataFrame(columns=["Epoch", "Reward"])

tf.compat.v1.disable_eager_execution()

vec2 = pygame.math.Vector2

all_rewards = []


class QLearning:
    def __init__(self, game):

        self.game = game
        self.game.new_episode()

        self.stateSize = [game.state_size]
        self.actionSize = game.no_of_actions
        self.learningRate = 0.00030
        self.possibleActions = np.identity(self.actionSize, dtype=int)
        self.total_reward = 0
        self.totalTrainingEpisodes = 100000
        self.maxSteps = 3600
        self.batchSize = 64
        self.memorySize = 100000
        self.maxEpsilon = 1
        self.minEpsilon = 0.01
        self.decayRate = 0.00001
        self.decayStep = 0
        self.gamma = 0.9
        self.training = False

        self.pretrainLength = self.batchSize

        self.maxTau = 10000
        self.tau = 0
        tf.compat.v1.reset_default_graph()

        self.sess = tf.compat.v1.Session()

        self.DQNetwork = DQN(self.stateSize, self.actionSize, self.learningRate, name='DQNetwork')
        self.TargetNetwork = DQN(self.stateSize, self.actionSize, self.learningRate, name='TargetNetwork')

        self.memoryBuffer = PrioritisedMemory(self.memorySize)
        # self.pretrain()

        self.state = []
        self.trainingStepNo = 0

        self.newEpisode = False
        self.stepNo = 0
        self.episodeNo = 0
        self.saver = tf.compat.v1.train.Saver()

        load = True
        loadFromEpisodeNo = 900
        if load:
            self.episodeNo = loadFromEpisodeNo
            self.saver.restore(self.sess, "./allModels/final_model/models/model.ckpt")
            # self.saver.restore(self.sess, "./allModels/model{}/models/model.ckpt".format(self.episodeNo))
            # self.saver.restore(self.sess, "./final_model_with_add_walls/models/model.ckpt".format(self.episodeNo))
        else:
            self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(self.update_target_graph())

    # копіюємо параметри з нашої мережі до таргетної мережі
    def update_target_graph(self):
        from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

        to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

        op_holder = []

        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def pretrain(self):
        for i in range(self.pretrainLength):
            if i == 0:
                state = self.game.get_state()

            action = random.choice(self.possibleActions)
            actionNo = np.argmax(action)
            reward = self.game.make_action(actionNo)
            nextState = self.game.get_state()
            self.newEpisode = False

            if self.game.is_episode_finished():
                reward = -100
                self.memoryBuffer.store((state, action, reward, nextState, True))
                self.game.new_episode()
                state = self.game.get_state()
                self.newEpisode = True
            else:
                self.memoryBuffer.store((state, action, reward, nextState, False))
                self.game.render()
                state = nextState

        print("pretrainingDone")

    def train(self):
        # Початок навчання моделі
        if self.trainingStepNo == 0:
            self.state = self.game.get_state()

        if self.newEpisode:
            self.state = self.game.get_state()

        # Проводимо навчання протягом обмеженої кількості кроків
        if self.stepNo < self.maxSteps:
            self.stepNo += 1
            self.decayStep += 1
            self.trainingStepNo += 1
            self.tau += 1

            # Обчислюємо значення епсилон для вибору випадкової або найкращої дії
            epsilon = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * np.exp(
                -self.decayRate * self.decayStep)

            # Вибираємо дію на основі епсилон-жадного підходу
            if np.random.rand() < epsilon:
                choice = random.randint(1, len(self.possibleActions)) - 1
                action = self.possibleActions[choice]

            else:
                QValues = self.sess.run(self.DQNetwork.output,
                                        feed_dict={self.DQNetwork.inputs_: np.array([self.state])})
                choice = np.argmax(QValues)
                action = self.possibleActions[choice]

            actionNo = np.argmax(action)
            reward = self.game.make_action(actionNo)

            nextState = self.game.get_state()
            if (reward > 0):
                # print("Reward {}".format(reward))
                pass
            if self.game.is_episode_finished():
                if game.drone.done:
                    # Досягнута зона призначення
                    reward = 1000
                    print("destination hit")
                    directory = "./allModels/final_model"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    save_path = self.saver.save(self.sess, "./allModels/final_model/models/model.ckpt")
                    print("Final Model Saved")
                    self.stepNo = self.maxSteps
                    df1 = pd.read_csv("epoch_reward.csv")
                    df = pd.DataFrame({"Epoch": [self.episodeNo], "Reward": [self.total_reward],
                                       "Gates passed": [game.drone.rewardNo]})
                    df.to_csv("epoch_reward.csv", mode='a', index=False, header=False)
                    print("Episode {} reward {} epsilon {} experiences stored {} rewardGates passed {}".format(
                        self.episodeNo, self.total_reward, epsilon, self.trainingStepNo, game.drone.rewardNo))
                    return
                else:
                    # Зіткнення зі стіною
                    reward = -100
                    self.stepNo = self.maxSteps
                    # print("Reward =  -100")
                try:
                    df1 = pd.read_csv("epoch_reward.csv")
                    df = pd.DataFrame({"Epoch": [self.episodeNo], "Reward": [self.total_reward],
                                       "Gates passed": [game.drone.rewardNo]})
                    df.to_csv("epoch_reward.csv", mode='a', index=False, header=False)
                except:
                    df = pd.DataFrame(columns=["Epoch", "Reward", "Gates passed"])
                    df.loc[len(df)] = [self.episodeNo, self.total_reward, game.drone.rewardNo]
                    df.to_csv("epoch_reward.csv", index=False)
                print("Episode {} reward {} epsilon {} experiences stored {} rewardGates passed {}"
                      .format(self.episodeNo, self.total_reward, epsilon, self.trainingStepNo, game.drone.rewardNo))
                self.total_reward = 0
            # Обчислюємо загальну нагороду
            if reward > 0 or (self.trainingStepNo % 50 == 0 and reward == -1):
                self.total_reward += reward

            # Зберігаємо досвід у буфер пам'яті
            self.memoryBuffer.store((self.state, action, reward, nextState, self.game.is_episode_finished()))

            self.state = nextState

            # Проводимо навчання на партії досвіду
            trIndex, batch, ISWeights = self.memoryBuffer.sample(self.batchSize)

            states = np.array([i[0][0] for i in batch])
            actions = np.array([i[0][1] for i in batch])
            rewards = np.array([i[0][2] for i in batch])
            nextStates = np.array([i[0][3] for i in batch])
            ifdead = np.array([i[0][4] for i in batch])
            targetQs = []
            qValueNextSt = self.sess.run(self.TargetNetwork.output,feed_dict={self.TargetNetwork.inputs_: nextStates})

            for i in range(self.batchSize):
                action = np.argmax(qValueNextSt[i])
                termState = ifdead[i]
                if termState:
                    targetQs.append(rewards[i])
                else:
                    target = rewards[i] + self.gamma * qValueNextSt[i][action]
                    targetQs.append(target)

            targets = np.array([t for t in targetQs])

            # Обчислюємо втрату та оновлюємо модель
            loss, _, absError = self.sess.run(
                [self.DQNetwork.loss, self.DQNetwork.optimizer, self.DQNetwork.absoluteError],
                feed_dict={self.DQNetwork.inputs_: states,
                           self.DQNetwork.actions_: actions,
                           self.DQNetwork.targetQ: targets,
                           self.DQNetwork.ISWeights_: ISWeights})
            self.memoryBuffer.batchUpdate(trIndex, absError)

        if self.stepNo >= self.maxSteps:
            # Завершення епізоду
            self.episodeNo += 1
            self.stepNo = 0
            self.newEpisode = True
            self.game.new_episode()
            if self.episodeNo >= self.totalTrainingEpisodes:
                self.training = False
            if self.episodeNo % 100 == 0:
                directory = "./allModels/model{}".format(self.episodeNo)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = self.saver.save(self.sess, "./allModels/model{}/models/model.ckpt".format(self.episodeNo))
                print("Model Saved")
        if self.tau > self.maxTau:
            # Оновлення таргетної мережі
            self.sess.run(self.update_target_graph())
            self.tau = 0
            print("Target Network Updated")

    def test(self):
        # Тестування
        self.state = self.game.get_state()
        QValues = self.sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: np.array([self.state])})
        choice = np.argmax(QValues)
        action = self.possibleActions[choice]
        actionNo = np.argmax(action)
        self.game.make_action(actionNo)
        if self.game.is_episode_finished():
            self.game.new_episode()


class Memory:
    def __init__(self, maxSize):
        self.buffer = deque(maxlen=maxSize)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batchSize):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batchSize,
                                 replace=False)
        return [self.buffer[i] for i in index]


class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)

        backgroundColor = [0, 0, 0, 1]
        glClearColor(*backgroundColor)

        self.game = game
        self.ai = ai

        self.firstClick = True

    def on_key_press(self, symbol, modifiers):
        pass

    def on_close(self):
        self.ai.sess.close()
        pass

    def on_key_release(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.ai.training = not self.ai.training

    def on_mouse_press(self, x, y, button, modifiers):
        pass

    def on_draw(self):
        window.set_size(width=displayWidth, height=displayHeight)
        self.clear()
        self.game.render()

    def update(self, dt):
        if not game.drone.done:
            for i in range(5):
                if self.ai.training:
                    self.ai.train()
                else:
                    self.ai.test()
                    return
            pass
        else:
            return


game = Game()
ai = QLearning(game)
if __name__ == "__main__":
    # ai.train()
    # while True:
    #     ai.train()
    window = MyWindow(displayWidth, displayHeight, "AI Learns to Drive", resizable=False)
    pyglet.clock.schedule_interval(window.update, 1 / frameRate)
    pyglet.app.run()
