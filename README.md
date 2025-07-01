# 🧠 Student Math Score Predictor

A full-stack Machine Learning web application that predicts a student's **math score** based on demographic and academic inputs. This project is built using a **modular Python architecture**, wrapped with **Flask**, and **deployed on AWS** with complete CI/CD automation using **Elastic Beanstalk**.

[🔗 Live App](http://student-env.eba-puvqpaf8.eu-north-1.elasticbeanstalk.com/predictdata)

---

## 🚀 Key Highlights

- ✅ **Modular ML Pipeline**: Separated logic for data ingestion, transformation, training, and prediction
- 🔁 **CI/CD**: Deployed using AWS CodePipeline + CodeBuild with Elastic Beanstalk
- 🎯 **Model Tuning**: Evaluated 6 regression algorithms across 1000+ hyperparameter combinations
- 🌐 **Flask Web App**: Intuitive frontend to input features and get real-time predictions
- 📦 **Production-Ready Structure**: Scalable and maintainable file structure using OOP and clean coding principles

---

## 🧩 Input Features

The model predicts math scores based on:

- Gender  
- Race or Ethnicity  
- Parental Level of Education  
- Lunch Type  
- Test Preparation Course  
- Writing Score  
- Reading Score  

---
### 📁 Project Structure
```
├── .ebextensions/
│   └── python.config               # WSGI path config for deployment
├── artifacts/
│   ├── model.pkl                   # Trained model
│   ├── preprocessor.pkl            # Transformation pipeline
│   ├── train.csv, test.csv         # Training datasets
├── notebook/
│   └── data/
│       └── stud.csv               # Raw input dataset
├── logs/                           # Training & inference logs
├── src/
│   ├── components/                 # Data ingestion, transformation
│   ├── pipelines/                  # Training & prediction logic
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging setup
│   └── utils.py                    # Helper utilities
├── templates/
│   ├── index.html                  # Input form UI
│   └── home.html                   # Output result page
├── application.py                  # Flask entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
├── notes.txt, .gitignore           # Misc files
```
## 🧑‍💻 Author

- 🧭 **Portfolio**: [divyanshsingh.xyz](https://divyanshsingh.xyz)  
- 💼 **LinkedIn**: [divyanshsinghnrj](https://www.linkedin.com/in/divyanshsinghnrj) 
