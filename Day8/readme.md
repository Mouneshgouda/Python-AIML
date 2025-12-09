
```python
!git clone https://github.com/Mouneshgouda/FakeFace.git

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Conv2D, Conv2DTranspose, Flatten, Reshape,
    BatchNormalization, LeakyReLU, Input
)


BATCH_SIZE = 64
IMG_SIZE = 128
LATENT_DIM = 100
EPOCHS = 400
LR = 2e-4

train_ds = tf.keras.utils.image_dataset_from_directory(
    "/content/FakeFace",
    label_mode=None,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
).map(lambda x: (tf.cast(x, tf.float32) / 127.5) - 1.0).prefetch(tf.data.AUTOTUNE)



def build_generator(latent_dim=100):
    return tf.keras.Sequential([
        Input((latent_dim,)),
        Dense(8*8*512, use_bias=False),
        Reshape((8, 8, 512)),

        Conv2DTranspose(256, 4, 2, "same", use_bias=False), BatchNormalization(), LeakyReLU(),
        Conv2DTranspose(128, 4, 2, "same", use_bias=False), BatchNormalization(), LeakyReLU(),
        Conv2DTranspose(64, 4, 2, "same", use_bias=False),  BatchNormalization(), LeakyReLU(),

        Conv2DTranspose(3, 4, 2, "same", activation="tanh")
    ], name="Generator")



def build_discriminator(img_size=128):
    return tf.keras.Sequential([
        Input((img_size, img_size, 3)),
        Conv2D(64, 4, 2, "same"),  LeakyReLU(0.2),
        Conv2D(128, 4, 2, "same"), LeakyReLU(0.2),
        Conv2D(256, 4, 2, "same"), LeakyReLU(0.2),
        Conv2D(512, 4, 2, "same"), LeakyReLU(0.2),
        Flatten(),
        Dense(1)
    ], name="Discriminator")


G, D = build_generator(), build_discriminator()



loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_opt = tf.keras.optimizers.Adam(LR, 0.5)
d_opt = tf.keras.optimizers.Adam(LR, 0.5)



@tf.function
def train_step(real):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake = G(noise, training=True)
        r_out, f_out = D(real, training=True), D(fake, training=True)

        g_loss = loss_fn(tf.ones_like(f_out), f_out)
        d_loss = loss_fn(tf.ones_like(r_out), r_out) + loss_fn(tf.zeros_like(f_out), f_out)

    g_opt.apply_gradients(zip(g_tape.gradient(g_loss, G.trainable_variables), G.trainable_variables))
    d_opt.apply_gradients(zip(d_tape.gradient(d_loss, D.trainable_variables), D.trainable_variables))
    return g_loss, d_loss



def show_images(epoch, seed):
    preds = G(seed, training=False)
    preds = (preds + 1) / 2.0   # [-1,1] → [0,1]
    fig, axes = plt.subplots(4,4, figsize=(6,6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(preds[i].numpy())
        ax.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.show()


seed = tf.random.normal([16, LATENT_DIM])
for e in range(1, EPOCHS+1):
    for real in train_ds:
        g_loss, d_loss = train_step(real)
    print(f"Epoch {e}/{EPOCHS} | G: {g_loss:.3f} D: {d_loss:.3f}")
    if e % 5 == 0: show_images(e, seed)




