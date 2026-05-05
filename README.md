# Random Forest MNIST Classifier 🧠🔢

A lightweight Machine Learning pipeline that trains a Random Forest model to recognize handwritten digits using the classic MNIST dataset. The project features a custom image-processing pipeline that allows the model to accurately classify real-world, user-drawn images.

## ✨ Features
* **Fast Training:** Utilizes Scikit-Learn's `RandomForestClassifier` for quick training and inference.
* **High Accuracy:** Achieves **~96.50%** accuracy on the MNIST testing set.
* **Custom Image Support:** Includes an automated preprocessing function that crops, centers, and resizes custom hand-drawn images to perfectly match the model's expected input format.

## 🛠️ Tech Stack
* **Python 3.x**
* **Scikit-Learn:** Model creation, training, and accuracy evaluation.
* **TensorFlow (Keras):** Used exclusively as a fast, reliable way to fetch the MNIST dataset.
* **NumPy:** For flattening images and matrix manipulations.
* **Pillow (PIL):** For real-world image processing (grayscale conversion, cropping, and resizing).

## 🚀 Getting Started

### 1. Install Dependencies
Clone the repository and install the required Python libraries:
`pip install numpy scikit-learn tensorflow pillow`

### 2. Prepare a Custom Test Image
To test the model with your own handwriting:
1. Open a drawing app (like MS Paint).
2. Set the background to **Black**.
3. Draw a single digit (0-9) using a **White** brush (medium thickness).
4. Save the image in the project directory (e.g., `my_digit.png`).

### 3. Run the Classifier
Update the target image path at the very bottom of the `random-forest-real-image-classifier.py` file:
`classify_image("my_digit.png")`

Then, execute the script in your terminal:
`python random-forest-real-image-classifier.py`

## 🧠 Under the Hood: Why Preprocessing Matters
Random Forest algorithms differ from Convolutional Neural Networks (CNNs). While CNNs learn spatial features (like curves and edges) and can scan an image, a Random Forest relies on fixed pixel locations (e.g., *"Is the pixel at coordinate [14, 14] bright or dark?"*).

Because of this, if a user draws a "3" slightly off-center, the white pixels won't align with the patterns the model learned from the MNIST dataset (which consists of perfectly centered digits). 

**The Solution in this Code:**
To bridge the gap between "laboratory data" and "real-world input," this script includes a smart pre-processing pipeline. Before predicting, the code:
1. Scans the custom image to find the exact boundaries of the drawn white pixels.
2. Crops out the dead black space.
3. Resizes the resulting digit bounding-box to 20x20 pixels.
4. Pastes it precisely in the mathematical center of a new 28x28 black canvas.

This ensures your custom drawings perfectly mimic the MNIST format, drastically improving the model's real-world accuracy!
