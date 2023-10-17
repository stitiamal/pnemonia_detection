#!/usr/bin/env python
# coding: utf-8

# In[ ]:


data_location = 'C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray'

#i start by defining a function that takes as argument 'data_location' and initialize an empty list to store invalid images
def remove_invalid_images(dataset_path):
    invalid_images = []
    #with the foor loop, i will check all the directories and sub directories in my dataset folder
    for root, dirs, files in os.walk(data_location):
        for f in files: #for each file found, i will use the root_directory and the file name to constructs the file_path
            file_path = os.path.join(root, f)
            try:
                with Image.open(file_path) as img: #i will try to open the image using the file path created above
                    img.verify() #i use the verify() method to check if the image is valid or no 
            except (IOError, SyntaxError):
                print(f"Invalid image: {file_path}") #i will indicate through a print message that the image is invalid if so
                invalid_images.append(file_path) #i append the file path to the invalid_images list initilized in the begining of the function
                os.remove(file_path)
    
    return invalid_images #return the list of invalid images

invalid_images = remove_invalid_images(data_location) #calling the function
print(f"Removed {len(invalid_images)} invalid images.") #print the list of results


# In[ ]:


from PIL import Image, ImageFilter
import os

train_pneumonia_set = "C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray/train/PNEUMONIA"
train_normal_set = "C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray/train/NORMAL"

test_pneumonia_set = "C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray/test/PNEUMONIA"
test_normal_set = "C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray/test/NORMAL"

val_pneumonia_set = "C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray/val/PNEUMONIA"
val_normal_set = "C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray/val/NORMAL"


train_gauss = 'C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray/train_gauss'
test_gauss = 'C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray/test_gauss'
val_gauss = 'C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray/val_gauss'


def apply_gaussian_blur(input_directory, output_directory):
    size = (224, 224)
    os.makedirs(output_directory, exist_ok=True)

    for image_name in os.listdir(input_directory):
        if image_name.endswith(".jpeg"):
            image_path = os.path.join(input_directory, image_name)
            image = Image.open(image_path)
            img_gauss = image.filter(ImageFilter.GaussianBlur(radius=2))
            resized_img = img_gauss.resize(size, Image.ANTIALIAS)
            output = os.path.join(output_directory, image_name)
            resized_img.save(output)

apply_gaussian_blur(val_pneumonia_set, os.path.join(val_gauss, "pneumonia"))
apply_gaussian_blur(val_normal_set, os.path.join(val_gauss, "normal"))

apply_gaussian_blur(train_pneumonia_set, os.path.join(train_gauss, "pneumonia"))
apply_gaussian_blur(train_normal_set, os.path.join(train_gauss, "normal"))

apply_gaussian_blur(test_pneumonia_set, os.path.join(test_gauss, "pneumonia"))
apply_gaussian_blur(test_normal_set, os.path.join(test_gauss, "normal"))


# In[ ]:


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    np_image = np.array(image)
    noisy_image = np_image.copy()
    salt = np.random.rand(*np_image.shape) < salt_prob
    pepper = np.random.rand(*np_image.shape) < pepper_prob
    noisy_image[salt] = 255
    noisy_image[pepper] = 0
    return Image.fromarray(noisy_image)

def add_speckle_noise(image, intensity):
    np_image = np.array(image)
    noise = np.random.randn(*np_image.shape) * intensity
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_poisson_noise(image, scale):
    np_image = np.array(image)
    noisy_image = np.random.poisson(np_image / 255.0 * scale) * 255.0 / scale
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_uniform_noise(image, intensity):
    np_image = np.array(image)
    noise = np.random.uniform(low=-intensity, high=intensity, size=np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Define the path to your base directory
base_directory = 'C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray'

# Define the subdirectories
subdirectories = ['train', 'test', 'val']

# Define the classes
class_names = ['pneumonia', 'normal']

# Define noise parameters
salt_prob = 0.01
pepper_prob = 0.01
speckle_intensity = 0.1
poisson_scale = 0.1
uniform_intensity = 0.1

for subdirectory in subdirectories:
    for class_name in class_names:
        source_directory = os.path.join(base_directory, subdirectory, class_name)
        destination_directory = os.path.join(base_directory, subdirectory + '_noisy', class_name)
        os.makedirs(destination_directory, exist_ok=True)

        for image_name in os.listdir(source_directory):
            image_path = os.path.join(source_directory, image_name)
            image = Image.open(image_path)

            noisy_image = add_salt_and_pepper_noise(image, salt_prob, pepper_prob)
            noisy_image.save(os.path.join(destination_directory, 'salt_pepper_' + image_name))

            noisy_image = add_speckle_noise(image, speckle_intensity)
            noisy_image.save(os.path.join(destination_directory, 'speckle_' + image_name))

            noisy_image = add_poisson_noise(image, poisson_scale)
            noisy_image.save(os.path.join(destination_directory, 'poisson_' + image_name))

            noisy_image = add_uniform_noise(image, uniform_intensity)
            noisy_image.save(os.path.join(destination_directory, 'uniform_' + image_name))

