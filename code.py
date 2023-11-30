import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import glob
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

# Below are the condition attributes we will be using for our cGAN
time_deltas = ['1 Hour', '6 Hours']
cell_cycle_phases = ['G1', 'S', 'G2', 'M']
dna_content = ['Normal', 'Doubled']
mitotic_index = ['Low', 'Medium', 'High']
cyclin_levels = ['Low', 'Medium', 'High']
cytoskeletal_marker_levels = ['Moderate', 'High', 'Very High']

# As we know, the batch size should match the number of images that we are processing.
batch_size = 8
# z_index is the noise vector. It is independent of value of size of any other input. We can assign the value based on the 
# complexity and the amount of accuracy that we are expecting from the model.
noise_vector_for_generator = 100
epochs = 1000
img_shape = (137, 121, 3)  

# A function to load the images, and pre-process them befor sending it to our model.
def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(img_shape[:2])
    return np.array(img) / 255.0


# Since we are using cGAN, we need couple of attributes to leverage the power of cGAN with the help of the features that are 
# available in our dataset. As cGAN only supports categorical features, we need to convert numerical features(if any) to 
# categorical features. 
def encode_labels():
    all_labels = np.array([cell_cycle_phases, dna_content, mitotic_index, cyclin_levels, cytoskeletal_marker_levels])
    encoder = OneHotEncoder()
    # We transpose the labels as the OneHotEncoder, will expect it's input to be in a definite shape. Also, it will be processing
    # the data row wise but not column wise. That is the reason the dataset will be transposed, and the feature will be scanned 
    # accordingly in a row wise fashion. The place where the data is not there, a filler value "0" will be assigned.
    encoder.fit(all_labels.T)
    return encoder

# Building the Generator
def build_generator(noise_vector_for_generator):
    model = tf.keras.Sequential([
        # Dense layer with appropriate input dimension
        layers.Dense(256, activation='relu', input_dim=noise_vector_for_generator),
        layers.Reshape((1, 1, 256)),
        
        # Transposed convolution layers

        # We are using the "Transposed" version because in a traditional GANs, we will introduce the noise, then slowly train the
        # model to got to the level of generating accurate images. So we are basically "upsampling" unlike the normal Conv2D approach
        # where we "downsample", to make the image more contrast, clear, and big enough for the Generator to slowly produce a good 
        # and accurate image.
        layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='valid', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='valid', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='valid', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding='valid', activation='tanh'),  # 3 for RGB channels
    ])
    return model

# Building the Discriminater
def build_discriminator(img_shape):
    model = tf.keras.Sequential([
        # Convolutional layers with appropriate input shape
        layers.Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding='same', activation='relu'),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
        ])
    return model

# cGAN Model Construction
generator = build_generator(noise_vector_for_generator)
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

discriminator.trainable = False
cgan_input = layers.Input(shape=(noise_vector_for_generator,))
cgan_output = discriminator(generator(cgan_input))
cgan = models.Model(cgan_input, cgan_output)
cgan.compile(loss='binary_crossentropy', optimizer='adam')

# Training Function
def train_cgan(generator, discriminator, cgan, epochs, batch_size, noise_vector_for_generator, image_files, conditioned_labels):
    images = np.array([load_image(img_path) for img_path in image_files])
    for epoch in range(epochs):
        for batch in range(len(images) // batch_size):
            # Noise vector and conditioned labels for generator
            noise = np.random.normal(0, 1, (batch_size, noise_vector_for_generator))
            idx = np.random.randint(0, conditioned_labels.shape[0], batch_size)
            batch_conditioned_labels = conditioned_labels[idx]

            # Generate images
            gen_imgs = generator.predict([noise, batch_conditioned_labels])

            # Real images and labels
            real_imgs = images[idx]

 
            # From both of the below line, you are just telling the discriminator that - Anything that is coming from the real dataset 
            # should be considered as the absolute true information with no inaccuracies in it. It is perfect data in other words. 

            # On the other hand, about the fake_labels, just as the case for the real_labels, we are directly, again telling the 
            # discriminator that the anything that is coming from the generator has to be considered as "FAKE". 

            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            # So, we are just "reinforcing" the discriminator by giving it the ability to be able to analyze the data, and respond
            # appropriately 

            # Train the discriminator (real)

            # The train_on_batch method comes "in-built" with the tensorflow keras API. So, if we have build the network of a model
            # sequentialy or in any manner, we will automaticaly have access to use any method available in the keras API.
            # And it does take two arguments. The first argument consist of the input data (the data and labels). The second argument
            # consists of the output data(Target output)
            d_loss_real = discriminator.train_on_batch([real_imgs, batch_conditioned_labels], real_labels)

            # Train the discriminator (fake)
            d_loss_fake = discriminator.train_on_batch([gen_imgs, batch_conditioned_labels], fake_labels)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            g_loss = cgan.train_on_batch([noise, batch_conditioned_labels], real_labels)

            print(f'Epoch: {epoch} Batch: {batch} D Loss: {d_loss[0]} G Loss: {g_loss}')


# Main Execution
image_files = sorted(glob.glob('D:\\ValuableStuff\\Everything about FAU\\Artificial Intelligence in Medical Healthcare - FALL 2023\\cell_imagescell_images\\*.png'))

# We are subjecting our categorical features to the OneHotEncoder, and getting the format which our model(cGAN) expects to receive
encoder = encode_labels()

dataset = np.array([
    ['1 Hour', 'G1', 'Normal', 'Low', 'Low', 'Moderate'],
    ['6 Hour', 'S', 'Doubled', 'Medium', 'High', 'High'],
    ['1 Hour', 'G2', 'Doubled', 'Medium', 'Medium', 'High'],
    ['6 Hour', 'M', 'Doubled', 'High', 'High', 'Very High'],
    ['1 Hour', 'G1', 'Normal', 'Low', 'Low', 'Moderate'],
    ['6 Hour', 'S', 'Doubled', 'High', 'High', 'High'],
    ['1 Hour', 'S', 'Normal', 'Low', 'Medium', 'Moderate'],
    ['6 Hour', 'G2', 'Doubled', 'Medium', 'Medium', 'High'],
])
conditioned_labels = encoder.transform(dataset).toarray()

train_cgan(generator, discriminator, cgan, epochs, batch_size, noise_vector_for_generator, image_files, conditioned_labels)

