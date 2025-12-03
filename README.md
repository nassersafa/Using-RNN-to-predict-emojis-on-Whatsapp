# 🌟 Emoji Sentiment Classifier using RNN
This project builds a Recurrent Neural Network (RNN) that predicts the appropriate emoji for a given text message. The model learns emotional patterns from WhatsApp-style messages and maps them to emojis such as 😊, 😢, ❤️, 😂, 😡, and 😴.
##  Project Overview
The goal is to classify text messages into emotional categories using an RNN. The model is trained on a synthetic dataset of 500 messages labeled with emojis that represent different emotional tones.
Examples of predictions:
| Input Text | Predicted Emoji |
|------------|------------------|
| "I miss you so much" | ❤️ |
| "I'm so tired today" | 😴 |
| "This is hilarious" | 😂 |
| "I'm really sad today" | 😢 |
##  Project Structure
emoji-rnn/
│
├── emoji_rnn_project.ipynb   # Main notebook (training + prediction)
└── README.md                 # Project documentation
##  How It Works
1. Data Preparation  
A custom dataset of messages was generated across 6 categories (happy, sad, love, funny, angry, tired). Each message is paired with its emoji.
2. Text Preprocessing  
- Tokenization (convert words → integers)  
- Padding (make all sequences length 10)  
- Emoji labels encoded as numbers  

##  Model Architecture
The RNN is built using TensorFlow/Keras:
Embedding (input_dim=5000, output_dim=32, input_length=10)  
SimpleRNN (64 units)  
Dense (32 units, ReLU)  
Dense (6 units, Softmax)  
Loss: sparse_categorical_crossentropy  
Optimizer: Adam  

## Training
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

##  Predicting Emojis
Use this function:
predict_emoji("I am so tired today")
Example output: 😴  
You can input any sentence, and the model predicts the most likely emoji.

## Results
- The RNN successfully learns emotional patterns.  
- More training data improves performance.  
- Training for 20–30 epochs gives better stability.  
- Adding more angry/sad sentences improves accuracy.  

##  Technologies Used
Python, TensorFlow/Keras, NumPy, Pandas, Scikit-learn, Google Colab  

