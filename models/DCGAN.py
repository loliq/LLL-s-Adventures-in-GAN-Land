#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/17 16:19
# @Author  : LLL
# @Site    : 
# @File    : DCGAN.py
# @Software: PyCharm
#

import tensorflow as tf
from tensorflow.keras import layers
tf.enable_eager_execution()
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os
from imageio import imread, imsave, mimsave


class DCGAN_Model():
    def __init__(self, config, image_shape):
        # 真实图像的路径
        self.config = config
        self.initial_shape = [8, 8, 256] # 生成网络最开始的大小
        self.output_shape = image_shape
        self.latent_dim = (100,)
        self.img_shape = image_shape
        self.dropout_rate = 0.3
        # from_logits 表示返回的是概率
        # 创建dataset
        self.image_dataset = self.make_dataset_from_folders()

        if not os.path.exists(self.config.logdir + '/sample_image'):
            os.makedirs(self.config.logdir + '/sample_image')
        # 配置训练
        self.config_train()

    def config_train(self):
        """
        配置训练
        ---建立图模型
        ---配置训练用的优化器
        --- 定义loss
        :return:
        """
        # 定义生成器和判别器
        self.generator = self.make_generator()
        base_discriminator = self.make_discriminator()
        print("Genenrator achitecture ")
        self.generator.summary()
        # 定义优化器
        self.optimizer = keras.optimizers.Adam(1e-4)
        # 定义更新的D 的 compile, 更新D的时候会固定G
        self.discriminator = tf.keras.Model(
            inputs=base_discriminator.inputs,
            outputs=base_discriminator.outputs)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        self.frozen_D = tf.keras.Model(
            inputs=base_discriminator.inputs,
            outputs=base_discriminator.outputs)
        # D设置不可训练，此时的网络相当于 G + D(不过D被设置成不可训练了)
        self.frozen_D.trainable = False
        z = tf.keras.Input(shape=(self.latent_dim))
        img = self.generator(z)
        valid = self.frozen_D(img)
        self.combined = tf.keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        # 还有不用中间变量过度直接设置trainable = False 直接搞得(https://blog.csdn.net/theonegis/article/details/80115340)
        # 这篇文章也是直接设置trainable = False, 然后直接建生成模型得



    def make_generator(self):
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
        model = tf.keras.Sequential()

        # (64,64,3) => (32, 32, 32)
        model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                                input_shape=self.img_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.dropout_rate))

        # (32,32,32) => (16, 16 ,128)
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.dropout_rate))

        # (16,16,128) => (8, 8, 256)
        model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.dropout_rate))

        # flattern (8 * 8 * 256)
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        return model


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
        print("imagesize is {2}, Dataset has {0} batches with Batchsize = {1}".
              format(np.ceil(image_dataset.samples/image_dataset.batch_size), self.config.batch_size, self.img_shape))
        return image_dataset

    def train(self, epoches, disc_iterval=3, log_interval=10, save_interval=5):
        # logs = {"d_loss":[], "d_acc":[], "g_loss":[]}
        logs = []
        # 保存生成器模型, 在服务器保存模型会报错，不知道为啥，因为有两张显卡吗？
        # self.discriminator.save(self.config.logdir + "/epoch{0}-Generator.h5".format(00))

        # G_config = self.generator.to_json()
        # with open(self.config.logdir + '/generator_model_config.json', 'w') as json_file:
        #     json_file.write(G_config)
        #
        # d_config = self.discriminator.to_json()
        # with open(self.config.logdir + '/discriminator_model_config.json', 'w') as json_file:
        #     json_file.write(d_config)

        # 配置训练

        for epoch in range(epoches):
            # 训练每一个epoch的数据
            for index, [image_batch, _] in enumerate(self.image_dataset):
                # 对image 做预处理
                print(image_batch.shape[0])
                image_batch = self.prosess_image(image_batch)
                # 具体的维度以输入的维度为准
                noise = np.random.normal(0.0, 1.0, [image_batch.shape[0], self.latent_dim[0]])
                # noise = tf.random.normal([self.config.batch_size, self.latent_dim])
                gen_imgs = self.generator.predict(noise)
                # 将真的数据输入进去做梯度下降
                # 返回的值是scalar(标量值) [loss, metric]
                # 在这边因为metric 是accuracy, 所以返回值为[loss,acc]
                valid_label = np.ones((image_batch.shape[0], 1))
                fake_label = np.zeros((image_batch.shape[0], 1))
                d_loss_real = self.discriminator.train_on_batch(x=image_batch, y=valid_label)
                # 将假的数据输入进去做梯度下降
                d_loss_fake = self.discriminator.train_on_batch(x=gen_imgs, y=fake_label)
                # 将real_data 和 generated 的acc 和 loss 做均值
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # 每隔几个迭代轮会更新一次generator
                if index % disc_iterval == 0:
                    # 训练G使得判别器能判定生成图像为true
                    g_loss = self.combined.train_on_batch(noise, valid_label)
                    print("epoch is {0}, d_loss is {1},d_acc is{2}, g_loss is {3}".format
                          (epoch, d_loss[0], d_loss[1], g_loss))
                # 需要注意的是因为用的是flow_from_dictionary所以不会自己记住什么时候跳出循环，所以要加跳出循环的点
                if index >= np.ceil(self.image_dataset.samples / self.image_dataset.batch_size):
                    break
            if epoch % log_interval == 0:
                logs.append([epoch, d_loss[0], d_loss[1], g_loss])
            # 每隔几次生成图并保存模型
            if epoch % save_interval == 0:
                self.generate_and_save_images(epoch_index=epoch)
                # self.generator.save(self.config.logdir + "/epoch{0}-Generator.h5".format(epoch))
                # self.discriminator.save_weights(self.config.logdir + "/epoch{0}-discriminator.h5".format(epoch))
        self.showlogs(logs)


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

























