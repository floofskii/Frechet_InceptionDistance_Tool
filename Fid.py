import numpy as np
from numpy import cov, trace, iscomplexobj, asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.transform import resize
import os
import pickle

# Function to load a batch of CIFAR data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # Convert image to float32 to reduce memory usage
        image = image.astype(np.float32)
        # Resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, anti_aliasing=True)
        # Store
        images_list.append(new_image)
    return asarray(images_list)

# Calculate Frechet Inception Distance
def calculate_fid(model, images1, images2):
    # Calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # Calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # Check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # Calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Load CIFAR-10 data
data_path = 'C:\\Users\\winni\\Downloads\\cifar-10-python\\cifar-10-batches-py'
batch_1_filename = 'test_batch'  # Adjusted to the new file name provided

# Load the first batch of CIFAR-10 data
data_batch = unpickle(os.path.join(data_path, batch_1_filename))
images = data_batch[b'data']
labels = data_batch[b'labels']
images = images.reshape((len(images), 3, 32, 32)).transpose(0, 2, 3, 1)

# Set the path to the weights file
local_weights_file = 'C:\\Users\\winni\\Downloads\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Initialize the InceptionV3 model with weights set to None to prevent automatic download
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3), weights=None)

# Load the weights from the file you've downloaded manually
model.load_weights(local_weights_file)

# Scale CIFAR-10 images to the required size for InceptionV3
images = scale_images(images, (299, 299, 3))

# Pre-process images
images = preprocess_input(images)

# Calculate FID between two sets of images (using the same set for demonstration)
fid = calculate_fid(model, images, images)
print('FID (same): %.3f' % fid)
