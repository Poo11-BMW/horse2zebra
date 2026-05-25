"""
CycleGAN — Horse ↔ Zebra
========================
Fixes applied vs. the original notebook:
  1. [Critical]  Removed sigmoid from discriminator output so raw logits match
                 BinaryCrossentropy(from_logits=True).
  2. [Arch]      Added InstanceNormalization to generator and discriminator for
                 training stability.
  3. [Arch]      Discriminator now uses LeakyReLU(0.2) instead of ReLU.
  4. [Arch]      Bottleneck Conv blocks are proper residual blocks (skip connections).
  5. [Quality]   Removed all unused imports (LSTM, VGG16, ResNet50, pandas, tqdm …).
  6. [Quality]   Removed unused variables (image_input1, image_input2).
  7. [Training]  Added checkpoint saving so training survives interruptions.
  8. [Training]  Added per-epoch loss logging and a final loss-curve plot.
  9. [Training]  generate_images() uses training=False and can optionally save to disk.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

dataset, metadata = tfds.load(
    'cycle_gan/horse2zebra', with_info=True, as_supervised=True
)
train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses,  test_zebras  = dataset['testA'],  dataset['testB']

# ──────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
BUFFER_SIZE = 1000
BATCH_SIZE  = 1
IMG_WIDTH   = 256
IMG_HEIGHT  = 256


def random_crop(image):
    return tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])


def normalize(image):
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1.0          # → [-1, 1]


def random_jitter(image):
    image = tf.image.resize(
        image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image


def preprocess_image_train(image, label):
    return random_jitter(normalize(image))


def preprocess_image_test(image, label):
    return normalize(image)


train_horses = (
    train_horses.cache()
    .map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)
train_zebras = (
    train_zebras.cache()
    .map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)
test_horses = (
    test_horses
    .map(preprocess_image_test, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)
test_zebras = (
    test_zebras
    .map(preprocess_image_test, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))

# ──────────────────────────────────────────────────────────────────────────────
# 3. Instance Normalization  (Fix #2)
#    Standard for CycleGAN — normalises per sample per channel, not per batch.
# ──────────────────────────────────────────────────────────────────────────────
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization (Ulyanov et al., 2016)."""

    def __init__(self, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        self.scale  = self.add_weight('scale',  shape=(channels,),
                                      initializer='ones',  trainable=True)
        self.offset = self.add_weight('offset', shape=(channels,),
                                      initializer='zeros', trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(var + self.epsilon)
        return self.scale * x_norm + self.offset


# ──────────────────────────────────────────────────────────────────────────────
# 4. Generator  (Fixes #2, #4)
#    • InstanceNormalization after every Conv layer
#    • Bottleneck uses true residual blocks (skip connections)
# ──────────────────────────────────────────────────────────────────────────────
def residual_block(x, filters: int):
    """Two Conv + InstanceNorm layers with an additive skip connection."""
    skip = x
    x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same')(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same')(x)
    x = InstanceNormalization()(x)
    return tf.keras.layers.Add()([skip, x])   # ← skip connection


def generator():
    inp = tf.keras.layers.Input(shape=(256, 256, 3))

    # Downsampling
    x = tf.keras.layers.Conv2D(64,  7, strides=1, padding='same')(inp)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Residual bottleneck (6 blocks with real skip connections)
    for _ in range(6):
        x = residual_block(x, 256)

    # Upsampling
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(64,  3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    out = tf.keras.layers.Conv2DTranspose(3, 7, strides=1, padding='same',
                                          activation='tanh')(x)

    return tf.keras.Model(inputs=inp, outputs=out)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Discriminator  (Fixes #1, #2, #3)
#    • No sigmoid → raw logits to match BinaryCrossentropy(from_logits=True)
#    • InstanceNormalization on inner layers
#    • LeakyReLU(0.2) throughout (standard PatchGAN)
# ──────────────────────────────────────────────────────────────────────────────
def discriminator():
    inp = tf.keras.layers.Input(shape=(256, 256, 3))

    x = tf.keras.layers.Conv2D(64,  4, strides=2, padding='same')(inp)
    x = tf.keras.layers.LeakyReLU(0.2)(x)                # no norm on first layer

    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Raw logit output — no sigmoid (Fix #1)
    out = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(x)

    return tf.keras.Model(inputs=inp, outputs=out)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Models & Optimizers  (Fix #6: unused image_input1/2 removed)
# ──────────────────────────────────────────────────────────────────────────────
generator_G     = generator()     # Horse → Zebra
generator_F     = generator()     # Zebra → Horse
discriminator_X = discriminator() # judges Horses
discriminator_Y = discriminator() # judges Zebras

generator_G_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_F_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_X_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_Y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Checkpointing  (Fix #7)
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = './checkpoints'

checkpoint = tf.train.Checkpoint(
    generator_G=generator_G,
    generator_F=generator_F,
    discriminator_X=discriminator_X,
    discriminator_Y=discriminator_Y,
    generator_G_optimizer=generator_G_optimizer,
    generator_F_optimizer=generator_F_optimizer,
    discriminator_X_optimizer=discriminator_X_optimizer,
    discriminator_Y_optimizer=discriminator_Y_optimizer,
)
ckpt_manager = tf.train.CheckpointManager(
    checkpoint, CHECKPOINT_DIR, max_to_keep=5
)

if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print(f'[INFO] Restored from checkpoint: {ckpt_manager.latest_checkpoint}')
else:
    print('[INFO] Starting training from scratch.')

# ──────────────────────────────────────────────────────────────────────────────
# 8. Loss Functions  (Fix #1: from_logits=True is now consistent with raw logits)
# ──────────────────────────────────────────────────────────────────────────────
LAMBDA   = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real),       real)
    fake_loss = loss_obj(tf.zeros_like(generated), generated)
    return (real_loss + fake_loss) * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))


def identity_loss(real_image, same_image):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))


# ──────────────────────────────────────────────────────────────────────────────
# 9. Training Step
# ──────────────────────────────────────────────────────────────────────────────
@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        # Forward passes
        fake_y   = generator_G(real_x, training=True)
        cycled_x = generator_F(fake_y, training=True)

        fake_x   = generator_F(real_y, training=True)
        cycled_y = generator_G(fake_x, training=True)

        same_x   = generator_F(real_x, training=True)   # identity
        same_y   = generator_G(real_y, training=True)   # identity

        disc_real_x = discriminator_X(real_x, training=True)
        disc_real_y = discriminator_Y(real_y, training=True)
        disc_fake_x = discriminator_X(fake_x, training=True)
        disc_fake_y = discriminator_Y(fake_y, training=True)

        # Losses
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle = (calc_cycle_loss(real_x, cycled_x) +
                       calc_cycle_loss(real_y, cycled_y))

        total_gen_g_loss = gen_g_loss + total_cycle + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Apply gradients
    generator_G_optimizer.apply_gradients(
        zip(tape.gradient(total_gen_g_loss, generator_G.trainable_variables),
            generator_G.trainable_variables)
    )
    generator_F_optimizer.apply_gradients(
        zip(tape.gradient(total_gen_f_loss, generator_F.trainable_variables),
            generator_F.trainable_variables)
    )
    discriminator_X_optimizer.apply_gradients(
        zip(tape.gradient(disc_x_loss, discriminator_X.trainable_variables),
            discriminator_X.trainable_variables)
    )
    discriminator_Y_optimizer.apply_gradients(
        zip(tape.gradient(disc_y_loss, discriminator_Y.trainable_variables),
            discriminator_Y.trainable_variables)
    )

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss


# ──────────────────────────────────────────────────────────────────────────────
# 10. Visualisation  (Fix #9: training=False, optional save-to-disk)
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_images(model, test_input, epoch=None, save: bool = False):
    prediction = model(test_input, training=False)   # ← inference mode
    plt.figure(figsize=(12, 6))
    for i, (img, title) in enumerate(
        [(test_input[0], 'Input Image'), (prediction[0], 'Predicted Image')]
    ):
        plt.subplot(1, 2, i + 1)
        plt.title(title)
        plt.imshow(img * 0.5 + 0.5)
        plt.axis('off')
    if save and epoch is not None:
        plt.savefig(os.path.join(OUTPUT_DIR, f'epoch_{epoch:03d}.png'),
                    bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 11. Training Loop  (Fix #8: loss logging)
# ──────────────────────────────────────────────────────────────────────────────
EPOCHS = 5
history = {'gen_g': [], 'gen_f': [], 'disc_x': [], 'disc_y': []}

for epoch in range(EPOCHS):
    start = time.time()
    step_losses = {'gen_g': [], 'gen_f': [], 'disc_x': [], 'disc_y': []}

    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        g_g, g_f, d_x, d_y = train_step(image_x, image_y)
        step_losses['gen_g'].append(float(g_g))
        step_losses['gen_f'].append(float(g_f))
        step_losses['disc_x'].append(float(d_x))
        step_losses['disc_y'].append(float(d_y))

    # Average losses across all steps in the epoch
    for key in history:
        history[key].append(np.mean(step_losses[key]))

    elapsed = time.time() - start
    print(
        f"Epoch {epoch + 1:>3}/{EPOCHS} | "
        f"Gen_G: {history['gen_g'][-1]:.4f} | "
        f"Gen_F: {history['gen_f'][-1]:.4f} | "
        f"Disc_X: {history['disc_x'][-1]:.4f} | "
        f"Disc_Y: {history['disc_y'][-1]:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    generate_images(generator_G, sample_horse, epoch=epoch + 1, save=True)
    ckpt_manager.save()

# ──────────────────────────────────────────────────────────────────────────────
# 12. Loss Curves  (Fix #8)
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['gen_g'], label='Generator G (H→Z)')
plt.plot(history['gen_f'], label='Generator F (Z→H)')
plt.title('Generator Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['disc_x'], label='Discriminator X (Horse)')
plt.plot(history['disc_y'], label='Discriminator Y (Zebra)')
plt.title('Discriminator Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'), bbox_inches='tight')
plt.show()
