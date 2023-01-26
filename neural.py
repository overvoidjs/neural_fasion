from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Список с названиями классов
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 
           'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

# Загружаем модель
model = load_model('fashion.h5')
model.summary()

# Загружаем изображение
img = image.load_img("./shorts.jpg", target_size=(28, 28), color_mode="grayscale")

# Преобразуем картинку в массив
x = image.img_to_array(img)
# Меняем форму массива в плоский вектор
x = x.reshape(1, 784)
# Инвертируем изображение
x = 255 - x
# Нормализуем изображение
x /= 255

# Запускаем распознавание
prediction = model.predict(x)

# Выводим ответ
prediction = np.argmax(prediction)
print("Номер класса:", prediction)
print("Название класса:", classes[prediction])