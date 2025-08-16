# 📩 Spam Classifier Web App  

A **Streamlit-based web application** that classifies SMS/email messages as **Spam** or **Not Spam**.  
It uses a **StackingClassifier** with multiple base learners and a Random Forest meta-classifier, trained on TF-IDF features.  

---

## 🚀 Features  
- Classify individual text messages as Spam/Not Spam  
- Show prediction probabilities  
- Preprocesses text using **NLTK** (stopwords removal, lemmatization, cleaning URLs/HTML/special characters)  
- Uses **TF-IDF (max_features=3000)** for text vectorization  
- Powered by a **StackingClassifier** combining multiple learners  

---

## 🧠 Machine Learning Model  

- **Vectorizer**: `TfidfVectorizer(max_features=3000)`  
- **Base Learners**:  
  - `KNeighborsClassifier()`  
  - `BernoulliNB()`  
  - `XGBClassifier(n_estimators=100)`  
  - `ExtraTreesClassifier(n_estimators=100)`  
- **Final Estimator (Meta-learner)**: `RandomForestClassifier()`  

---


