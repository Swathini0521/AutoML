# AutoML
🔥 AutoML with Genetic Algorithm – Streamlit App
Automatically discover the best machine learning model and hyperparameters using a Genetic Algorithm — all through a clean, interactive Streamlit interface.

📌 Features
🧠 Auto-selects from classifiers: Logistic Regression, Random Forest, SVM, Decision Tree, and Naive Bayes.

🔧 Auto-tunes hyperparameters using a Genetic Algorithm.

📊 Evaluates models using accuracy_score.

📁 Simple CSV upload for your dataset.

🖥️ Intuitive web interface powered by Streamlit.

📂 Folder Structure
bash
Copy
Edit
.
├── automl_streamlit.py         # Main Streamlit app
├── requirements.txt            # Python dependencies
└── README.md                   # You're here!
🚀 How to Run Locally
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run automl_streamlit.py
Your browser will open at: http://localhost:8501

📋 How to Use the App
Upload a .csv file.

Choose the target column (the one you want to predict).

Click “Run AutoML”.

View:

✅ Best model selected

🛠️ Optimal hyperparameters found

💡 Example Use Cases
Predicting customer churn

Loan approval classification

Disease diagnosis from health data

🛠️ Built With
Python

Streamlit

Scikit-learn

Genetic Algorithm (custom logic)

📄 License
MIT License — free to use and modify.

