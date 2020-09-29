import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_datasets as tfds


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(7*7*256, activation="relu", use_bias=False, input_shape=(100,))
        self.conv1 = tf.keras.layers.Conv2DTranspose(128, 5, 1, padding='same', use_bias=False, activation='relu')
        self.conv2 = tf.keras.layers.Conv2DTranspose(64, 5, 2, padding='same', use_bias=False, activation='relu')
        self.conv3 = tf.keras.layers.Conv2DTranspose(1, 5, 2, padding='same', use_bias=False, activation='tanh')
        
        
    def call(self, input):
    
        x = self.dense(input)
        x = self.bn1(x)
        x = tf.reshape(x,(-1, 7, 7, 256))
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        return self.conv3(x)



class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.lrelu1 = tf.keras.layers.LeakyReLU(0.2)
        self.lrelu2 = tf.keras.layers.LeakyReLU(0.2)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
        self.conv1 = tf.keras.layers.Conv2D(64, 5, 2, padding='same', input_shape=[28, 28, 1], use_bias=False)
        self.conv2 = tf.keras.layers.Conv2D(128, 5, 2, padding='same', use_bias=False)
        
                
    def call(self, input):
    
        x = self.conv1(input)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.bn(x)
        x = tf.keras.layers.Flatten()(x)
        return self.dense(x)


def discriminator_loss(loss_object, real_output, fake_output):
    #here = tf.ones_like(????) or tf.zeros_like(????)  -> tf.zeros_like와 tf.ones_like에서 선택하고 (???)채워주세요
    real_loss = loss_object(tf.ones_like(real_output), real_output)  # y_test와 y_pred를 비교하는 것처럼 진짜 이미지에 대한 예측확률이 1로부터 얼마나 가까운지를 알아본다.
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output) # 가짜 이미지에 대한 예측확률이 0으로부터 얼마나 가까운지를 알아본다.
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output) # 가짜 이미지의 예측확률이 1에 가까울수록 생성자는 감별자를 잘 속인 것이다.

def normalize(x):
    image = tf.cast(x['image'], tf.float32)
    image = (image / 127.5) - 1
    return image


def save_imgs(epoch, generator, noise):
    gen_imgs = generator(noise, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig.savefig("images/mnist_%d.png" % epoch)

def train():
    data, info = tfds.load("mnist", with_info=True, data_dir='/data/tensorflow_datasets')
    train_data = data['train']
    

    if not os.path.exists('./images'):
        os.makedirs('./images')

    # settting hyperparameter
    latent_dim = 100  # 노이즈의 dimension
    epochs = 2
    batch_size = 10000
    buffer_size = 6000
    save_interval = 1

    generator = Generator()
    discriminator = Discriminator()

    gen_optimizer = tf.keras.optimizers.Adam(0.0002,beta_1 = 0.5, beta_2 = 0.999)  #강의자료에 나와있는 옵션값을 주었다.
    disc_optimizer = tf.keras.optimizers.Adam(0.0002,beta_1 = 0.5, beta_2 = 0.999) 

    train_dataset = train_data.map(normalize).shuffle(buffer_size).batch(batch_size)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(True) #인자로 들어온 두 데이터, 이를테면 y_test와 y__pred가 얼마나 같은지를 판별하는 loss 함수를 정의해준다.

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dim])  # 처음 생성자에게 주는 랜덤 노이즈값

        with tf.GradientTape(persistent=True) as tape: # gradient값이 쓰레기값이 나올 때까지 persistent
            generated_images = generator(noise) # 가짜 이미지 생성

            real_output = discriminator(images) # 진짜 이미지 판별결과
            generated_output = discriminator(generated_images) # 가짜 이미지 판별결과

            gen_loss = generator_loss(cross_entropy, generated_output) 
            disc_loss = discriminator_loss(cross_entropy, real_output, generated_output) 

        grad_gen = tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables)) # 생성자 최적화
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables)) # 판별자 최적화

        return gen_loss, disc_loss

    seed = tf.random.normal([16, latent_dim])

    for epoch in range(epochs):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0

        for images in train_dataset:
            gen_loss, disc_loss = train_step(images)

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, total_gen_loss / batch_size, total_disc_loss / batch_size))
        if epoch % save_interval == 0:
            save_imgs(epoch, generator, seed)


if __name__ == "__main__":
    train()