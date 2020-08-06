import numpy as np
import cv2
from matplotlib import pyplot as plt

image_width = 190
image_length = 190
total_pixels = image_width * image_length

images = 7
variants = 2
total_images = images * variants

pet_image_vector = []

for i in range(1, total_images + 1):
    pet_image = cv2.cvtColor(cv2.imread("C:/training/" + str(i) + ".jpg"), cv2.COLOR_RGB2GRAY)
    pet_image = pet_image.reshape(total_pixels, )
    pet_image_vector.append(pet_image)

pet_image_vector = np.asarray(pet_image_vector)
pet_image_vector = pet_image_vector.transpose()

avg_pet_vector = pet_image_vector.mean(axis=1)
avg_pet_vector = avg_pet_vector.reshape(pet_image_vector.shape[0], 1)
normalized_pet_vector = pet_image_vector - avg_pet_vector


covariance_matrix = np.cov(np.transpose(normalized_pet_vector))
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

k = 2
k_eigen_vectors = eigen_vectors[0:k, :]
eigen_pairs = k_eigen_vectors.dot(np.transpose(normalized_pet_vector))


weights = np.transpose(normalized_pet_vector).dot(np.transpose(eigen_pairs))

print("Введите номер желаемого изображения")
photo=input()
test_add = "C:/training/" + photo + ".jpg"
test_img = cv2.imread(test_add)
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

test_img = test_img.reshape(total_pixels, 1)
test_normalized_pet_vector = test_img - avg_pet_vector
test_weight = np.transpose(test_normalized_pet_vector).dot(np.transpose(eigen_pairs))

showphoto = avg_pet_vector.reshape(190, 190)
plt.imshow(showphoto, cmap='gray', interpolation='bicubic')
#plt.show()

index = np.argmin(np.linalg.norm(test_weight - weights, axis=1))

print(index)
print('Загружено изображение с номером', photo)
if (index >= 0 and index <= 6):
    print("Вероятнее, это собака")
if (index > 6 and index <=13):
    print("Вероятнее, это кошка")
