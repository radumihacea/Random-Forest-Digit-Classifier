#random-forest-real-image-classifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL import Image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test= (X_train[:50000].reshape(-1,784),
                  X_test[:10000].reshape(-1,784))
y_train,y_test= y_train[:50000], y_test[:10000]

rf = RandomForestClassifier(n_estimators=50,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

accuracy = accuracy_score(y_test, rf.predict(X_test))
print(f"Accuracy: {accuracy:.2%}")

def classify_image(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    rows_with_white = np.where(img_array.max(axis=1) > 0)[0]
    cols_with_white = np.where(img_array.max(axis=0) > 0)[0]
    
    if len(rows_with_white) == 0:
        print("The image is completely black!")
        return

    top, bottom = rows_with_white[0], rows_with_white[-1]
    left, right = cols_with_white[0], cols_with_white[-1]
    cropped_img = img.crop((left, top, right, bottom))
    
    cropped_img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    
    final_image = Image.new('L', (28, 28), color=0)
    pos_x = (28 - cropped_img.width) // 2
    pos_y = (28 - cropped_img.height) // 2
    final_image.paste(cropped_img, (pos_x, pos_y))
    
    img_flatten = np.array(final_image).flatten()
    pred = rf.predict([img_flatten])[0]
    print(f"{image_path}: Predicted digit is {pred}")

classify_image("c.png")

    