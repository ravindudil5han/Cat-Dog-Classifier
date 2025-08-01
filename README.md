
---

# 🐾 Cat vs. Dog Classifier – Project Documentation

This project is a **Convolutional Neural Network (CNN)**-based image classifier designed to distinguish between images of cats and dogs. It includes a complete training pipeline and a Flask-based web application for real-time image prediction.

---

## 📦 1. Project Setup

### ✅ Prerequisites

Ensure the following are installed on your machine:

* [Python 3.10](https://www.python.org/downloads/)
* [Git](https://git-scm.com/)

---

### 📥 Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/ravindudil5han/Cat-Dog-Classifier.git
cd Cat-Dog-Classifier
```

---

### 🛠️ Create a Virtual Environment

Using a virtual environment isolates project dependencies:

```bash
python -m venv venv
```

Activate it:

* **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

* **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

---

### 📚 Install Dependencies

With the virtual environment activated, install the required libraries:

```bash
pip install tensorflow matplotlib flask numpy pillow
```

---

## 🗂️ 2. Dataset Preparation

This project uses a **Cats vs. Dogs** dataset. Download and organize the dataset as shown below:

> 📥 **Dataset used:**
> [Kaggle - Cat and Dog Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

### 📁 Required Folder Structure

```
.
├── training_set
│   ├── cats
│   │   ├── cat.1.jpg
│   │   ├── cat.2.jpg
│   │   └── ...
│   └── dogs
│       ├── dog.1.jpg
│       ├── dog.2.jpg
│       └── ...
└── test_set
    ├── cats
    │   ├── cat.1001.jpg
    │   ├── cat.1002.jpg
    │   └── ...
    └── dogs
        ├── dog.1001.jpg
        ├── dog.1002.jpg
        └── ...
```

Ensure your dataset matches this structure before proceeding.

---

## 🧠 3. Training the Model

Run the training script using:

```bash
python main.py
```

This will:

1. Load and preprocess the dataset.
2. Train a CNN model for up to **100 epochs**, using early stopping if accuracy reaches **99.9%**.
3. Visualize training and validation performance.
4. Save the model as `cat_dog_model.h5`.

> 🧪 **Note:** Adjust training parameters in `main.py` if needed for custom experiments.

---

## 🌐 4. Running the Web Application

The web interface is powered by **Flask**. To launch the application:

```bash
python app.py
```

Once the server is running, open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 🖼️ 5. Using the Web App

1. Open the web app in your browser.
2. Click **"Choose File"** to upload a cat or dog image.
3. Click **"Predict"** to classify the image.
4. The model will display:

   * 🐱 "Cat" or 🐶 "Dog"
   * 📊 Prediction confidence score

---

## 🚀 Project Highlights

* ✅ Clean modular codebase
* 🧠 CNN trained from scratch
* 🖼️ Live image upload interface with Flask
* 📈 Real-time prediction results

---

For any questions or improvements, feel free to contribute or open an issue on the [GitHub repository](https://github.com/ravindudil5han/Cat-Dog-Classifier).

---
