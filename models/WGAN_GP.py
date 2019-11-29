#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 14:14
# @Author  : LLL
# @Site    : 
# @File    : WGAN-gp.py
# @Software: PyCharm

""""
实现 wgan-gradient penalty
使用梯度惩罚对GAN做限制
大体结构与DCGAN 差不多，不同的有以下几点
- 1. Discriminator 的最后一层不加sigmoid 函数(不加激活函数)
- 2. loss 不取log 而是 L = d(real_image) - d(fake_image) + gradient_penalty
- 3. gradient_penalty 的计算,
-- 1.生成新的图片  inter_image = epsilon * real_image + (1-epsilon) *fake_image
-- 2. 计算Discriminator 对 新图片的梯度 gradient(D, inter_image)的二阶范数 slop
-- 3. 设置梯度penalty的权重lambda
-- 4. gp = lambda * (slop-1)^2
- 待修复的bug
1.  不知道为啥discriminator 中去掉drop_out 就能用了，不去掉会报错
2.  对于 RandomWeightedAverage 的使用，需要固定的batch_size 是硬伤，因为对 image_generator/tf.dataset
来说如果图像数目不是batch_size的整数，那么最后一个batch是根据实际剩下的图片数量做batch的，
现在的做法是遇到batch尺寸不匹配就跳过进入下一个
编程参考自 https://blog.csdn.net/gyt15663668337/article/details/90271265
"""

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K


tf.enable_eager_execution()


# batch_size 怎么定义？
class RandomWeightedAverage(tf.keras.layers.Layer):
    # 用固定的batch_size 的隐患，当image_batch 的数量不足一个batch的时候维度相乘会出错的
    # 尝试用K.placeholder, 然后训练的时候传参数试试
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class WGAN_GP_Model():
    def __init__(self, config, image_shape):
        # 真实图像的路径
        self.config = config
        self.initial_shape = [8, 8, 256] # 生成网络最开始的大小
        self.output_shape = image_shape
        self.latent_dim = (100,)
        self.img_shape = image_shape
        self.dropout_rate = 0.3
        # 创建dataset
        self.image_dataset = self.make_dataset_from_folders()
        # 梯度惩罚的权重
        self.gp_lambda = 10
        # from_logits 表示返回的是概率
        self.config_train()


    def make_generator(self):
        """
        生成器模型结构设计
        :return:
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(self.initial_shape[0] * self.initial_shape[1] * self.initial_shape[2],
                               use_bias=False, input_shape=self.latent_dim))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # reshape 成(7,7,256)
        model.add(layers.Reshape(self.initial_shape))
        # assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

        # 利用转置卷积生成(8,8,128)
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # 利用转置卷积生成(16,16,64)
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # 利用转置卷积生成(32,32,32)
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # 利用转置卷积生成(64,64,3)
        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

        return model

    def make_discriminator(self):
        """
        判别器模型设计
        :return:
        """
        model = tf.keras.Sequential()

        # (64,64,3) => (32, 32, 32)
        model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                                input_shape=self.img_shape))
        model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(self.dropout_rate))

        # (32,32,32) => (16, 16 ,128)
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(self.dropout_rate))

        # (16,16,128) => (8, 8, 256)
        model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(self.dropout_rate))
        # flattern (8 * 8 * 256)
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation=None))

        return model

    def wasserstein_loss(self, y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def gradient_penalty_loss2(self, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        def gradient_penalty_loss_wrapper(y_true, y_pred):
            gradients = K.gradients(y_pred, averaged_samples)[0]
            # compute the euclidean norm by squaring ...
            gradients_sqr = K.square(gradients)
            #   ... summing over the rows ...
            gradients_sqr_sum = K.sum(gradients_sqr,
                                      axis=np.arange(1, len(gradients_sqr.shape)))
            #   ... and sqrt
            gradient_l2_norm = K.sqrt(gradients_sqr_sum)
            # compute lambda * (1 - ||grad||)^2 still for each single sample
            gradient_penalty = K.square(1 - gradient_l2_norm)
            # return the mean as loss over all the batch samples
            return K.mean(gradient_penalty)
        return gradient_penalty_loss_wrapper

    def config_train(self):
        # 定义生成器和判别器
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()
        # 定义优化器
        self.optimizer = keras.optimizers.Adam(1e-4)
        # 需要注意的是 disctiminator 的输入来自三部分, real_image, gen_image的loss + gp
        # 构建噪声输入
        # 训练判别器时，冻结生成器
        self.generator.trainable = False
        real_img = keras.Input(shape=self.img_shape)
        z_disc = keras.Input(shape=self.latent_dim)
        fake_img = self.generator(z_disc)
        fake_d_out = self.discriminator(fake_img)
        valid_d_out = self.discriminator(real_img)
        # 定义实际的batch大小, 不能在train_on_batch 用，扎心啊
        # self.real_batch_size = K.placeholder(shape=(None, 1, 1, 1)), 似乎不能用place_holder
        interpolated_img = RandomWeightedAverage(self.config.batch_size)([real_img, fake_img])
        interpolated_d_out = self.discriminator(interpolated_img)
        # 创建discriminator的模型
        # 定义 gradient_penalty_loss
        # 用partial 是为了减少不调用的时候不必要的interpolated_img的输入
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names
        self.discriminator_model = tf.keras.Model(inputs=[real_img, z_disc],
                                                  outputs=[valid_d_out, fake_d_out, interpolated_d_out])

        self.discriminator_model.compile(loss=[self.wasserstein_loss,
                                               self.wasserstein_loss,
                                               self.gradient_penalty_loss2(averaged_samples=interpolated_img)],
                                         metrics=['accuracy', 'accuracy', 'accuracy'],
                                         optimizer=self.optimizer,
                                         loss_weights=[1, 1, self.gp_lambda]
                                         )
        # 训练生成器，固定判别器
        # D设置不可训练，此时的网络相当于 G + D(不过D被设置成不可训练了)
        self.discriminator.trainable = False
        self.generator.trainable = True
        z_gen = tf.keras.Input(shape=self.latent_dim)
        gen_img = self.generator(z_gen)
        fake_out = self.discriminator(gen_img)
        self.generator_model  = tf.keras.Model(z_gen, fake_out)
        # 设置loss_function
        self.generator_model .compile(loss=self.wasserstein_loss,
                              optimizer=self.optimizer)

    def train(self, epoches, disc_iterval=3, log_interval=10, save_interval=50):
        """

        :param epoches:
        :param disc_iterval:
        :param log_interval:
        :param save_interval:
        :return:
        """

        logs = []
        # 保存生成器模型
        G_config = self.generator.to_json()
        with open(self.config.logdir + '/generator_model_config.json', 'w') as json_file:
            json_file.write(G_config)
        #
        d_config = self.discriminator.to_json()
        with open(self.config.logdir + '/discriminator_model_config.json', 'w') as json_file:
            json_file.write(d_config)

        # 开始训练
        for epoch in range(epoches):
            # TODO 可以修改 取数据的方式
            for index, [image_batch, _] in enumerate(self.image_dataset):
                # TODO 关于 RandomWeightedAverage的 batch_size 不兼容的问题
                if image_batch.shape[0] != self.config.batch_size:
                    continue
                valid_label = -np.ones((image_batch.shape[0], 1))
                fake_label = np.zeros((image_batch.shape[0], 1))
                dummy = np.zeros((image_batch.shape[0], 1))  # Dummy gt for gradient penalty
                # dummy用做插入图像的y_true，只是填充参数的时候需要而已，实际上并没有用处
                image_batch = self.prosess_image(image_batch)
                noise = np.random.normal(0.0, 1.0, (image_batch.shape[0], self.latent_dim[0]))
                d_loss = self.discriminator_model.train_on_batch(x=[image_batch, noise],
                                                                 y=[valid_label, fake_label, dummy])
                if index % disc_iterval == 0:
                    g_loss = self.generator_model.train_on_batch(noise, valid_label)
                    print("epoch is {0}, d_loss is {1},d_acc is{2}, g_loss is {3}".format
                          (epoch, d_loss[0], d_loss[1], g_loss))
                if index >= np.ceil(self.image_dataset.samples / self.image_dataset.batch_size):
                    break
            if epoch % log_interval == 0:
                logs.append([epoch, d_loss[0], d_loss[1], g_loss])
                self.generate_and_save_images(epoch_index=epoch)
                # 每隔几次生成图并保存模型
            if epoch % save_interval == 0:
                self.generator.save_weights(self.config.logdir + "/ep-Generator-{epoch:03d}.h5")
                self.discriminator.save_weights(self.config.logdir + "/ep-Generator-{epoch:03d}.h5")
        self.showlogs()

    def generate_and_save_images(self, epoch_index, row_num=4, col_num=4):
        """

        :param epoch_index: 指示这是在第几个epoch生成的图片
        :param row_num: 设置行生成几张图片
        :param col_num:  设置列生成几张图片
        :return:
        """
        noise = np.random.normal(0.0, 1.0, [row_num * col_num, self.latent_dim[0]])
        predictions = self.generator.predict(noise)

        fig = plt.figure(figsize=(8, 8))
        # 每次生成的图片数量
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            # 当然是要做预处理的
            image = self.anti_process_image(predictions[i, :, :, :]).astype('uint8')
            plt.imshow(image)
            plt.axis('off')

        plt.savefig(self.config.logdir + '/sample_image' + '/image_at_epoch_{:04d}.png'.format(epoch_index))
        # plt.show()
        plt.close()

    def showlogs(self, logs):
        logs = np.array(logs)
        names = ["d_loss", "d_acc", "g_loss"]
        for i in range(3):
            plt.subplot(2, 2, i + 1)
            plt.plot(logs[:, 0], logs[:, i + 1])
            plt.xlabel("epoch")
            plt.ylabel(names[i])
        plt.tight_layout()
        plt.show()

    def prosess_image(self, image_batch):
        image_batch_process = (image_batch - 127.5) / 127.5
        return image_batch_process

    def anti_process_image(self, image_batch):
        image_batch = image_batch * 127.5 + 127.5
        return image_batch

    def make_dataset_from_folders(self):
        """
        直接从文件夹中读取图片
        :return:
        """
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
        # image_datatset 是一个包含一个(x. y)的迭代生成器
        # x是一个numpy数组，shape为[batch_size, *target_size, channels]
        # y是label
        image_dataset = image_generator.flow_from_directory(self.config.image_dir,
                                                         target_size=[self.img_shape[0], self.img_shape[1]],
                                                         color_mode='rgb',
                                                         batch_size=self.config.batch_size)
        print("image num is {0}".format(image_dataset.samples))
        print("imagesize is {2}, Dataset has {0} batches with Batchsize = {1}".
              format(np.ceil(image_dataset.samples/image_dataset.batch_size), self.config.batch_size, self.img_shape))
        return image_dataset















