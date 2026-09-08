path = "/kaggle/input/dog-breed-image-dataset/dataset"

def get_features_dog(folder,size=(150, 150)):
    images = []
    labels = []
    for dog_name in os.listdir(folder):
        dog_class = os.path.join(folder, dog_name)
        for img_name in os.listdir(dog_class):
            img_path = os.path.join(dog_class, img_name)
            img = Image.open(img_path)
            img = img.resize(size)
            img = np.array(img)
            images.append(img)
            labels.append(dog_name)
            
    return np.array(images), np.array(labels)