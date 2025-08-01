
-----

### 1\. Project Setup

#### Prerequisites

Before you begin, make sure you have the following installed:

  * **Python 3.10**
  * **Git**

-----

#### Cloning the Repository

First, clone the repository to your local machine using the following command in your terminal:

```bash
git clone https://github.com/ravindudil5han/Cat-Dog-Classifier.git
cd Cat-Dog-Classifier
```

-----

#### Creating a Virtual Environment

It's a good practice to use a virtual environment to manage project dependencies. This prevents conflicts with other Python projects.

```bash
python -m venv venv
```

Then, activate the virtual environment:

  * **On Windows:**
    ```bash
    venv\Scripts\activate
    ```
  * **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

-----

#### Installing Dependencies

With the virtual environment active, install all the required libraries using pip:

```bash
pip install tensorflow matplotlib flask numpy pillow
```

-----

### 2\. Dataset Preparation

Your project uses a **Cat vs. Dog** dataset. You need to download and organize this dataset in the correct directory structure. The training and test data should be placed in `training_set` and `test_set` directories, respectively.

The required folder structure is as follows:

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

You can find a suitable dataset on Kaggle or other public sources. Download the dataset and extract it to match the structure above.

-----

### 3\. Training the Model

The `train.py` script is responsible for building and training the Convolutional Neural Network (CNN) model. It will save the trained model as `cat_dog_model.h5`.

To start the training process, run:

```bash
python train.py
```

This script will:

1.  Load the dataset from the `training_set` and `test_set` directories.
2.  Train a CNN model over 100 epochs, but will stop early if it reaches **99.9% accuracy**.
3.  Display plots of the training and validation accuracy and loss.
4.  Save the final trained model as `cat_dog_model.h5`.

-----

### 4\. Running the Web Application

The `app.py` script serves a simple web interface for predicting whether an image is a cat or a dog. It uses the `cat_dog_model.h5` model saved in the previous step.

To run the web app, simply execute:

```bash
python app.py
```

After running this command, you should see a message in your terminal indicating that the Flask server is running, likely on **`http://127.0.0.1:5000`**.

-----

### 5\. Usage

1.  Open your web browser and navigate to the address provided by the `app.py` script (e.g., `http://127.0.0.1:5000`).
2.  You will be presented with a simple interface where you can upload an image.
3.  Click the "Choose File" button to select a picture of a cat or a dog.
4.  Click the "Predict" button.
5.  The application will then display the prediction (Cat or Dog) along with the confidence score.
