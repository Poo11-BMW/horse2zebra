"""
CycleGAN — Horse ↔ Zebra
========================
Previous fixes (bug-fix pass):
  1. Removed sigmoid / from_logits mismatch in discriminator + loss.
  2. Added InstanceNormalization to generator and discriminator.
  3. Discriminator uses LeakyReLU(0.2) instead of ReLU.
  4. Bottleneck Conv blocks are proper residual blocks (skip connections).
  5. Removed unused imports and variables.
  6. Checkpoint saving, loss logging, generate_images uses training=False.

Level-1 research upgrades:
  A. [ImagePool]  50-image replay buffer fed to the discriminator — prevents
                  the discriminator from overfitting to the latest batch and
                  stabilises adversarial training (Shrivastava et al., 2017).
  B. [LSGAN]      Replaced BinaryCrossentropy with MeanSquaredError (targets
                  1.0 / 0.0). LSGAN provides smoother, non-saturating gradients
                  and reduces mode collapse (Mao et al., 2017).
  C. [LR Decay]   Learning rate is held constant for the first half of training
                  then linearly decayed to zero — exactly as in the original
                  CycleGAN paper (Zhu et al., 2017).

Level-2 research upgrades:
  D. [SelfAttention]           Non-local self-attention (Zhang et al., SAGAN 2019)
                               inserted after the generator bottleneck.
  E. [MultiScaleDiscriminator] Two PatchGAN discriminators at 256×256 and 128×128
                               with averaged losses (Wang et al., pix2pixHD 2018).
  F. [SpectralNormalization]   Every Conv in the discriminator is SpectralNorm-
                               wrapped (Miyato et al., 2018).

Level-3 research upgrades (this pass):
  G. [PatchNCE]       Contrastive patch-level loss (Park et al., CUT 2020).
                      The generator encoder is exposed via a shared Keras model
                      (build_generator() now returns (gen, encoder)).  For each
                      spatial patch in the generated image, the matching patch
                      in the source is the positive key; all others are negatives.
                      A 2-layer PatchMLP projects patches to a normalised
                      embedding space before the InfoNCE cross-entropy is applied.
  H. [FrequencyLoss]  FFT magnitude-spectrum loss applied on the cycle path.
                      Penalises the generator for losing high-frequency detail
                      (sharp edges, fine texture) that pixel-space L1 loss
                      often overlooks (Jiang et al., 2021).
  I. [PerceptualLoss] Frozen VGG-16 feature-matching loss (Johnson et al., 2016)
                      on the cycle-reconstructed image vs the original.
                      Augments pixel-space cycle consistency with semantic
                      consistency at three VGG depths simultaneously.
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
    return (image / 127.5) - 1.0


def random_jitter(image):
    image = tf.image.resize(
        image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = random_crop(image)
    return tf.image.random_flip_left_right(image)


def preprocess_image_train(image, label):
    return random_jitter(normalize(image))


def preprocess_image_test(image, label):
    return normalize(image)


train_horses = (
    train_horses.cache()
    .map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)
train_zebras = (
    train_zebras.cache()
    .map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)
test_horses = (
    test_horses.map(preprocess_image_test, num_parallel_calls=AUTOTUNE)
    .cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)
test_zebras = (
    test_zebras.map(preprocess_image_test, num_parallel_calls=AUTOTUNE)
    .cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)

sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))

# ──────────────────────────────────────────────────────────────────────────────
# 3. Instance Normalization
# ──────────────────────────────────────────────────────────────────────────────
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization (Ulyanov et al., 2016)."""

    def __init__(self, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        C = input_shape[-1]
        self.scale  = self.add_weight('scale',  shape=(C,),
                                      initializer='ones',  trainable=True)
        self.offset = self.add_weight('offset', shape=(C,),
                                      initializer='zeros', trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return self.scale * (x - mean) / tf.sqrt(var + self.epsilon) + self.offset


# ──────────────────────────────────────────────────────────────────────────────
# 4. Self-Attention  [Level-2 upgrade D]
#    Non-local block with gamma=0 init (identity at start, learns gradually).
# ──────────────────────────────────────────────────────────────────────────────
class SelfAttention(tf.keras.layers.Layer):
    """Non-local self-attention (Zhang et al., SAGAN 2019)."""

    def build(self, input_shape):
        C = input_shape[-1]
        self.q        = tf.keras.layers.Conv2D(C // 8, 1, padding='same', use_bias=False)
        self.k        = tf.keras.layers.Conv2D(C // 8, 1, padding='same', use_bias=False)
        self.v        = tf.keras.layers.Conv2D(C,       1, padding='same', use_bias=False)
        self.out_proj = tf.keras.layers.Conv2D(C,       1, padding='same', use_bias=False)
        self.gamma    = self.add_weight('gamma', shape=(), initializer='zeros',
                                        trainable=True)

    def call(self, x):
        B, H, W, C = (tf.shape(x)[i] for i in range(4))
        Ck = C // 8
        q  = tf.reshape(self.q(x), [B, H * W, Ck])
        k  = tf.reshape(self.k(x), [B, H * W, Ck])
        v  = tf.reshape(self.v(x), [B, H * W, C])
        attn = tf.nn.softmax(
            tf.matmul(q, k, transpose_b=True) * tf.cast(Ck, tf.float32) ** -0.5
        )
        out = tf.reshape(tf.matmul(attn, v), [B, H, W, C])
        return self.gamma * self.out_proj(out) + x


# ──────────────────────────────────────────────────────────────────────────────
# 5. Generator  [Level-3 upgrade G]
#
#  build_generator() returns TWO models that SHARE WEIGHTS via Keras
#  functional API:
#    gen  — full generator (H×W×3 → H×W×3)
#    enc  — encoder-only feature extractor (H×W×3 → [feat0, feat1, feat2])
#
#  Because both are built from the same layer graph, any update to gen's
#  weights is instantly reflected in enc.  enc is used exclusively by the
#  PatchNCE loss (section 8) and never updated independently.
#
#  Exposed feature maps (used as NCE anchors):
#    feat0  256×256×64   — after first Conv+IN+ReLU
#    feat1  128×128×128  — after first downsampling stride
#    feat2   64×64×256   — after second downsampling stride
# ──────────────────────────────────────────────────────────────────────────────
def residual_block(x, filters: int):
    """Two Conv + InstanceNorm with an additive skip connection."""
    skip = x
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = InstanceNormalization()(x)
    return tf.keras.layers.Add()([skip, x])


def build_generator():
    """Return (generator, encoder) sharing weights."""
    inp = tf.keras.layers.Input(shape=(256, 256, 3))

    # ── Encoder (features exposed for NCE) ───────────────────────────────────
    x = tf.keras.layers.Conv2D(64,  7, strides=1, padding='same')(inp)
    x = InstanceNormalization()(x);  x = tf.keras.layers.ReLU()(x)
    feat0 = x                               # 256×256×64

    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x);  x = tf.keras.layers.ReLU()(x)
    feat1 = x                               # 128×128×128

    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x);  x = tf.keras.layers.ReLU()(x)
    feat2 = x                               # 64×64×256

    # ── Bottleneck ────────────────────────────────────────────────────────────
    for _ in range(6):
        x = residual_block(x, 256)
    x = SelfAttention()(x)                  # non-local attention at bottleneck

    # ── Decoder ──────────────────────────────────────────────────────────────
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x);  x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(64,  3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x);  x = tf.keras.layers.ReLU()(x)

    out = tf.keras.layers.Conv2DTranspose(3, 7, strides=1, padding='same',
                                          activation='tanh')(x)

    gen = tf.keras.Model(inputs=inp, outputs=out)
    enc = tf.keras.Model(inputs=inp, outputs=[feat0, feat1, feat2])  # shared weights
    return gen, enc


# ──────────────────────────────────────────────────────────────────────────────
# 6. Discriminator  [Level-2 upgrades E + F]
#    SpectralNorm on every Conv, two scales, losses averaged.
# ──────────────────────────────────────────────────────────────────────────────
_SN = tf.keras.layers.SpectralNormalization


def _make_discriminator() -> tf.keras.Model:
    """Single-scale PatchGAN with SpectralNorm + InstanceNorm + LeakyReLU."""
    inp = tf.keras.layers.Input(shape=(None, None, 3))

    x = _SN(tf.keras.layers.Conv2D(64,  4, strides=2, padding='same'))(inp)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = _SN(tf.keras.layers.Conv2D(128, 4, strides=2, padding='same'))(x)
    x = InstanceNormalization()(x);  x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = _SN(tf.keras.layers.Conv2D(256, 4, strides=2, padding='same'))(x)
    x = InstanceNormalization()(x);  x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = _SN(tf.keras.layers.Conv2D(128, 4, strides=2, padding='same'))(x)
    x = InstanceNormalization()(x);  x = tf.keras.layers.LeakyReLU(0.2)(x)

    out = _SN(tf.keras.layers.Conv2D(1, 4, strides=1, padding='same'))(x)
    return tf.keras.Model(inputs=inp, outputs=out)


class MultiScaleDiscriminator(tf.keras.Model):
    """Two PatchGAN discriminators at 256×256 and 128×128 (pix2pixHD)."""
    NUM_SCALES = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disc_scales = [_make_discriminator() for _ in range(self.NUM_SCALES)]
        self.downsample  = tf.keras.layers.AveragePooling2D(
            pool_size=2, strides=2, padding='same'
        )

    def call(self, x, training: bool = False) -> list:
        outputs = []
        for i, disc in enumerate(self.disc_scales):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x, training=training))
        return outputs


# ──────────────────────────────────────────────────────────────────────────────
# 7. Image Replay Buffer  [Level-1 upgrade A]
# ──────────────────────────────────────────────────────────────────────────────
class ImagePool:
    """50-image history buffer (Shrivastava et al., 2017)."""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.pool: list = []

    def query(self, image: tf.Tensor) -> tf.Tensor:
        if len(self.pool) < self.max_size:
            self.pool.append(image)
            return image
        if np.random.rand() > 0.5:
            idx = np.random.randint(len(self.pool))
            stored = self.pool[idx]
            self.pool[idx] = image
            return stored
        return image


pool_fake_x = ImagePool()
pool_fake_y = ImagePool()

# ──────────────────────────────────────────────────────────────────────────────
# 8. PatchNCE Loss  [Level-3 upgrade G]
#
#  PatchMLP — 2-layer MLP projector.  Maps encoder feature vectors to an
#             L2-normalised embedding space of dimension 256.  One instance
#             per encoder layer per generator direction (6 total).
#
#  patch_nce_loss — for each spatial position i in the generated image:
#    • positive key  = embedding of position i in the SOURCE image
#    • negative keys = embeddings of all other positions in the SOURCE image
#    InfoNCE cross-entropy is averaged over layers and positions.
#
#  Gradient flow:
#    nce_loss → enc(fake_y) → fake_y → generator weights (decoder + encoder)
#    nce_loss → enc(real_x) → encoder weights (only via the source branch)
#    nce_loss → PatchMLP weights
#  The generator therefore learns to preserve content-patch identity while
#  changing domain style — without needing a full cycle round-trip.
# ──────────────────────────────────────────────────────────────────────────────
NCE_NUM_PATCHES = 256    # patches sampled per layer per image
NCE_TEMP        = 0.07   # InfoNCE temperature
LAMBDA_NCE      = 1.0    # weight of NCE loss relative to other losses


class PatchMLP(tf.keras.Model):
    """2-layer MLP projector for PatchNCE embeddings."""

    def __init__(self, hidden: int = 256, out: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.fc1  = tf.keras.layers.Dense(hidden)
        self.relu = tf.keras.layers.ReLU()
        self.fc2  = tf.keras.layers.Dense(out)

    def call(self, x, training: bool = False):
        return tf.math.l2_normalize(self.fc2(self.relu(self.fc1(x))), axis=-1)


def patch_nce_loss(enc, real_x, fake_y, mlps, training: bool = False) -> tf.Tensor:
    """
    Compute PatchNCE loss across all encoder layers.

    Args:
        enc     : encoder model (shared with the generator)
        real_x  : source image batch  (B, H, W, 3)
        fake_y  : generated image batch (B, H, W, 3)
        mlps    : list of PatchMLP — one per encoder layer
        training: passed through to enc and mlps

    Returns:
        Scalar NCE loss averaged over all layers.
    """
    src_feats = enc(real_x, training=training)   # [feat0, feat1, feat2]
    gen_feats = enc(fake_y, training=training)   # same structure

    total = tf.constant(0.0)
    for src_f, gen_f, mlp in zip(src_feats, gen_feats, mlps):
        B       = tf.shape(src_f)[0]
        H       = tf.shape(src_f)[1]
        W       = tf.shape(src_f)[2]
        C       = tf.shape(src_f)[3]
        N_total = H * W
        N       = tf.minimum(NCE_NUM_PATCHES, N_total)

        # ── Sample the SAME N positions from both source and generated ────────
        idx = tf.random.shuffle(tf.range(N_total))[:N]   # (N,)

        src_patches = tf.reshape(
            tf.gather(tf.reshape(src_f, [B, N_total, C]), idx, axis=1), [-1, C]
        )  # (B*N, C)
        gen_patches = tf.reshape(
            tf.gather(tf.reshape(gen_f, [B, N_total, C]), idx, axis=1), [-1, C]
        )  # (B*N, C)

        # ── Project to embedding space ────────────────────────────────────────
        src_emb = tf.reshape(mlp(src_patches, training=training), [B, N, -1])
        gen_emb = tf.reshape(mlp(gen_patches, training=training), [B, N, -1])

        # ── Scaled dot-product logits: (B, N, N) ─────────────────────────────
        #    logits[b, i, j] = similarity(gen[b,i], src[b,j])
        #    Target: diagonal (position i ↔ position i is the positive pair)
        logits = tf.matmul(gen_emb, src_emb, transpose_b=True) / NCE_TEMP
        labels = tf.eye(N, batch_shape=[B])   # one-hot on diagonal

        layer_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )
        total = total + layer_loss

    return total / tf.cast(len(mlps), tf.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Frequency-Domain Loss  [Level-3 upgrade H]
#
#  Computes L1 difference of FFT magnitude spectra on the luminance channel.
#  Pixel-space L1 (cycle loss) weights all frequencies equally; the FFT loss
#  explicitly penalises high-frequency mismatch — the detail that makes
#  translated images look crisp rather than blurry.
#  Applied on the cycle path: frequency_loss(real_x, cycled_x).
# ──────────────────────────────────────────────────────────────────────────────
LAMBDA_FREQ = 0.5


def frequency_loss(real: tf.Tensor, generated: tf.Tensor) -> tf.Tensor:
    """L1 difference of FFT magnitude spectra (luminance channel)."""
    real_gray = tf.reduce_mean(real,      axis=-1)   # (B, H, W)
    fake_gray = tf.reduce_mean(generated, axis=-1)
    real_fft  = tf.signal.rfft2d(real_gray)           # complex (B, H, W//2+1)
    fake_fft  = tf.signal.rfft2d(fake_gray)
    return tf.reduce_mean(tf.abs(tf.abs(real_fft) - tf.abs(fake_fft)))


# ──────────────────────────────────────────────────────────────────────────────
# 10. Perceptual Loss  [Level-3 upgrade I]
#
#  A frozen VGG-16 network extracts features at three depths:
#    block1_conv2 → low-level texture (edges, colours)
#    block2_conv2 → mid-level patterns (fur, grass)
#    block3_conv3 → high-level semantics (body parts)
#
#  The loss is the average L1 distance between features of the real source
#  image and its cycle-reconstructed version.  Unlike pixel-level cycle loss,
#  this is insensitive to small spatial shifts but very sensitive to semantic
#  content — encouraging the model to preserve what the horse *is*, not just
#  how each pixel looks.
# ──────────────────────────────────────────────────────────────────────────────
LAMBDA_PERC = 0.1

_vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
_vgg.trainable = False   # frozen: gradients flow through but weights never update

_perceptual_extractor = tf.keras.Model(
    inputs=_vgg.input,
    outputs=[
        _vgg.get_layer('block1_conv2').output,   # low-level
        _vgg.get_layer('block2_conv2').output,   # mid-level
        _vgg.get_layer('block3_conv3').output,   # semantic
    ]
)


def perceptual_loss(real: tf.Tensor, generated: tf.Tensor) -> tf.Tensor:
    """Average L1 VGG feature distance across three depths."""
    prep = tf.keras.applications.vgg16.preprocess_input
    r_feats = _perceptual_extractor(prep((real      + 1.) * 127.5), training=False)
    g_feats = _perceptual_extractor(prep((generated + 1.) * 127.5), training=False)
    return sum(tf.reduce_mean(tf.abs(rf - gf))
               for rf, gf in zip(r_feats, g_feats)) / 3.


# ──────────────────────────────────────────────────────────────────────────────
# 11. Models & Optimizers
#
#  build_generator() returns (gen, enc) sharing weights.
#  encoder_G / encoder_F are only called during NCE loss computation;
#  they are never optimised independently.
#
#  nce_mlps_G / nce_mlps_F — one PatchMLP per encoder layer per direction.
#  Their variables are updated by the generator optimizers alongside gen weights.
# ──────────────────────────────────────────────────────────────────────────────
generator_G, encoder_G = build_generator()   # Horse → Zebra
generator_F, encoder_F = build_generator()   # Zebra → Horse
discriminator_X        = MultiScaleDiscriminator()   # judges Horses
discriminator_Y        = MultiScaleDiscriminator()   # judges Zebras

ENCODER_DIMS = [64, 128, 256]   # feature channels at each exposed encoder layer
nce_mlps_G   = [PatchMLP() for _ in ENCODER_DIMS]
nce_mlps_F   = [PatchMLP() for _ in ENCODER_DIMS]

generator_G_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_F_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_X_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_Y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# ──────────────────────────────────────────────────────────────────────────────
# 12. Checkpointing
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = './checkpoints'

checkpoint = tf.train.Checkpoint(
    generator_G=generator_G,
    generator_F=generator_F,
    discriminator_X=discriminator_X,
    discriminator_Y=discriminator_Y,
    nce_mlp_G0=nce_mlps_G[0], nce_mlp_G1=nce_mlps_G[1], nce_mlp_G2=nce_mlps_G[2],
    nce_mlp_F0=nce_mlps_F[0], nce_mlp_F1=nce_mlps_F[1], nce_mlp_F2=nce_mlps_F[2],
    generator_G_optimizer=generator_G_optimizer,
    generator_F_optimizer=generator_F_optimizer,
    discriminator_X_optimizer=discriminator_X_optimizer,
    discriminator_Y_optimizer=discriminator_Y_optimizer,
)
ckpt_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print(f'[INFO] Restored checkpoint: {ckpt_manager.latest_checkpoint}')
else:
    print('[INFO] Starting training from scratch.')

# ──────────────────────────────────────────────────────────────────────────────
# 13. Core Loss Functions  [Level-1: LSGAN  |  Level-2: multi-scale]
# ──────────────────────────────────────────────────────────────────────────────
LAMBDA   = 10
loss_obj = tf.keras.losses.MeanSquaredError()


def _to_list(x):
    return x if isinstance(x, list) else [x]


def discriminator_loss(real, generated):
    pairs = list(zip(_to_list(real), _to_list(generated)))
    total = sum((loss_obj(tf.ones_like(r), r) +
                 loss_obj(tf.zeros_like(g), g)) * 0.5
                for r, g in pairs)
    return total / len(pairs)


def generator_loss(generated):
    outs = _to_list(generated)
    return sum(loss_obj(tf.ones_like(g), g) for g in outs) / len(outs)


def calc_cycle_loss(real_image, cycled_image):
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))


def identity_loss(real_image, same_image):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))


# ──────────────────────────────────────────────────────────────────────────────
# 14. Training Steps
#
#  generator_step():
#    Computes ALL generator losses — adversarial, cycle, identity, NCE,
#    frequency, perceptual — and updates generator + NCE MLP weights together.
#    NCE MLP variables are appended to gen_*_vars so a single optimizer call
#    covers both generator and projector weights.
#
#  discriminator_step():
#    Unchanged in structure; still uses buffered fakes from ImagePool.
#
#  Note on encoder_G / encoder_F inside @tf.function:
#    enc(real_x)  → one encoder-forward pass on the source
#    enc(fake_y)  → one encoder-forward pass on the generated image
#    Both are valid TF ops and trace correctly.  Gradients from NCE flow
#    through enc(fake_y) → fake_y → generator, and through enc(real_x) →
#    encoder weights (subset of generator weights).
# ──────────────────────────────────────────────────────────────────────────────

@tf.function
def generator_step(real_x, real_y):
    """Update generators + NCE MLPs; return fresh fakes and all sub-losses."""
    # Variables to optimise: generator weights + NCE MLP weights
    gen_g_vars = (generator_G.trainable_variables +
                  [v for m in nce_mlps_G for v in m.trainable_variables])
    gen_f_vars = (generator_F.trainable_variables +
                  [v for m in nce_mlps_F for v in m.trainable_variables])

    with tf.GradientTape(persistent=True) as tape:
        # ── Forward passes ────────────────────────────────────────────────────
        fake_y   = generator_G(real_x, training=True)
        cycled_x = generator_F(fake_y, training=True)

        fake_x   = generator_F(real_y, training=True)
        cycled_y = generator_G(fake_x, training=True)

        same_x   = generator_F(real_x, training=True)
        same_y   = generator_G(real_y, training=True)

        # Discriminators in inference mode (update them in discriminator_step)
        disc_fake_x = discriminator_X(fake_x, training=False)
        disc_fake_y = discriminator_Y(fake_y, training=False)

        # ── Standard CycleGAN losses ──────────────────────────────────────────
        adv_g = generator_loss(disc_fake_y)
        adv_f = generator_loss(disc_fake_x)
        cyc   = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
        id_g  = identity_loss(real_y, same_y)
        id_f  = identity_loss(real_x, same_x)

        # ── PatchNCE loss [Level-3 upgrade G] ────────────────────────────────
        nce_g = patch_nce_loss(encoder_G, real_x, fake_y, nce_mlps_G, training=True)
        nce_f = patch_nce_loss(encoder_F, real_y, fake_x, nce_mlps_F, training=True)

        # ── Frequency loss [Level-3 upgrade H] ───────────────────────────────
        freq = (frequency_loss(real_x, cycled_x) +
                frequency_loss(real_y, cycled_y)) * 0.5

        # ── Perceptual loss [Level-3 upgrade I] ──────────────────────────────
        perc = (perceptual_loss(real_x, cycled_x) +
                perceptual_loss(real_y, cycled_y)) * 0.5

        # ── Total generator losses ────────────────────────────────────────────
        total_g = adv_g + cyc + id_g + LAMBDA_NCE * nce_g + LAMBDA_FREQ * freq + LAMBDA_PERC * perc
        total_f = adv_f + cyc + id_f + LAMBDA_NCE * nce_f + LAMBDA_FREQ * freq + LAMBDA_PERC * perc

    generator_G_optimizer.apply_gradients(
        zip(tape.gradient(total_g, gen_g_vars), gen_g_vars)
    )
    generator_F_optimizer.apply_gradients(
        zip(tape.gradient(total_f, gen_f_vars), gen_f_vars)
    )

    return fake_x, fake_y, total_g, total_f, nce_g, nce_f, freq, perc


@tf.function
def discriminator_step(real_x, real_y, buffered_fake_x, buffered_fake_y):
    """Update both MultiScaleDiscriminators using (possibly historical) fakes."""
    with tf.GradientTape(persistent=True) as tape:
        disc_real_x = discriminator_X(real_x,          training=True)
        disc_real_y = discriminator_Y(real_y,          training=True)
        disc_fake_x = discriminator_X(buffered_fake_x, training=True)
        disc_fake_y = discriminator_Y(buffered_fake_y, training=True)
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    discriminator_X_optimizer.apply_gradients(
        zip(tape.gradient(disc_x_loss, discriminator_X.trainable_variables),
            discriminator_X.trainable_variables)
    )
    discriminator_Y_optimizer.apply_gradients(
        zip(tape.gradient(disc_y_loss, discriminator_Y.trainable_variables),
            discriminator_Y.trainable_variables)
    )
    return disc_x_loss, disc_y_loss


# ──────────────────────────────────────────────────────────────────────────────
# 15. Linear LR Decay Schedule  [Level-1 upgrade C]
# ──────────────────────────────────────────────────────────────────────────────
def get_lr(epoch: int, total_epochs: int, initial_lr: float = 2e-4) -> float:
    decay_start = total_epochs // 2
    if epoch < decay_start:
        return initial_lr
    progress = (epoch - decay_start) / max(1, total_epochs - decay_start)
    return float(initial_lr * (1.0 - progress))


# ──────────────────────────────────────────────────────────────────────────────
# 16. Visualisation
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_images(model, test_input, epoch=None, save: bool = False):
    prediction = model(test_input, training=False)
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
# 17. Training Loop
#     generator_step now returns 8 values; all are logged per epoch.
# ──────────────────────────────────────────────────────────────────────────────
EPOCHS  = 5
history = {k: [] for k in
           ['gen_g', 'gen_f', 'disc_x', 'disc_y', 'nce_g', 'nce_f', 'freq', 'perc']}

for epoch in range(EPOCHS):
    start = time.time()

    # ── Linear LR Decay [Level-1 upgrade C] ──────────────────────────────────
    lr = get_lr(epoch, EPOCHS)
    for opt in [generator_G_optimizer, generator_F_optimizer,
                discriminator_X_optimizer, discriminator_Y_optimizer]:
        opt.learning_rate.assign(lr)

    step_losses = {k: [] for k in history}

    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        # Step 1 — generators + NCE MLPs
        fake_x, fake_y, g_g, g_f, nce_g, nce_f, freq, perc = \
            generator_step(image_x, image_y)

        # Step 2 — replay buffer [Level-1 upgrade A]
        buffered_fake_x = pool_fake_x.query(fake_x)
        buffered_fake_y = pool_fake_y.query(fake_y)

        # Step 3 — discriminators
        d_x, d_y = discriminator_step(image_x, image_y,
                                      buffered_fake_x, buffered_fake_y)

        for k, v in zip(history.keys(),
                        [g_g, g_f, d_x, d_y, nce_g, nce_f, freq, perc]):
            step_losses[k].append(float(v))

    for key in history:
        history[key].append(np.mean(step_losses[key]))

    elapsed = time.time() - start
    h = history
    print(
        f"Epoch {epoch + 1:>3}/{EPOCHS} | lr={lr:.2e} | "
        f"G_G:{h['gen_g'][-1]:.3f} G_F:{h['gen_f'][-1]:.3f} | "
        f"D_X:{h['disc_x'][-1]:.3f} D_Y:{h['disc_y'][-1]:.3f} | "
        f"NCE_G:{h['nce_g'][-1]:.3f} NCE_F:{h['nce_f'][-1]:.3f} | "
        f"Freq:{h['freq'][-1]:.2f} Perc:{h['perc'][-1]:.2f} | "
        f"Time:{elapsed:.1f}s"
    )

    generate_images(generator_G, sample_horse, epoch=epoch + 1, save=True)
    ckpt_manager.save()

# ──────────────────────────────────────────────────────────────────────────────
# 18. Loss Curves
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

axes[0].plot(history['gen_g'],  label='Gen G (H→Z)')
axes[0].plot(history['gen_f'],  label='Gen F (Z→H)')
axes[0].plot(history['disc_x'], label='Disc X')
axes[0].plot(history['disc_y'], label='Disc Y')
axes[0].set_title('Adversarial + Cycle Losses')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history['nce_g'], label='PatchNCE G')
axes[1].plot(history['nce_f'], label='PatchNCE F')
axes[1].set_title('PatchNCE Losses  [Level-3 G]')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend()

axes[2].plot(history['freq'], label='Frequency')
axes[2].plot(history['perc'], label='Perceptual (VGG)')
axes[2].set_title('Frequency & Perceptual Losses  [Level-3 H+I]')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'), bbox_inches='tight')
plt.show()
