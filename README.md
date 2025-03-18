# 📰 Fake News Detection using Passive Aggressive Classifier
This project is a Fake News Detection System that uses TF-IDF vectorization and a Passive Aggressive Classifier to classify news articles as Fake or Real.

## 📌 Features
- Preprocesses text data using TF-IDF Vectorization
- Trains a Passive Aggressive Classifier
- Evaluates model performance using Accuracy Score

## 🚀 Installation & Setup
```bash ### 1️⃣ Clone the Repository
git clone https://github.com/thenewjapzzz/fake-news-detection.git
cd fake-news-detection
```

```bash ### 2️⃣ Install Dependencies
pip install -r requirements.txt
```

``` base ### 3️⃣ Run the Program
python main.py
```

## 📜 How It Works
- Loads and splits the dataset into training (80%) and testing (20%) sets.
- Converts text into numerical features using TF-IDF Vectorization.
- Trains a Passive Aggressive Classifier.
- Predicts the news category and prints the accuracy.
