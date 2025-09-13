# IT Support Chatbot System

An intelligent IT support chatbot system that combines Retrieval-Augmented Generation (RAG) with hybrid machine learning classification to provide automated IT support assistance.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Components](#components)
- [Logging](#logging)
- [Model Training](#model-training)
- [Troubleshooting](#troubleshooting)

## 🌟 Features

- **RAG-based Chatbot**: Uses knowledge base for contextual responses
- **Hybrid Classification**: Combines LightGBM, Logistic Regression, and XGBoost
- **Persistent Storage**: Caches embeddings and models for faster subsequent runs
- **Structured Logging**: Detailed conversation logs with timestamps and categories
- **Two API Options**: 
  - OpenRouter API (external)
  - LM Studio local API (offline)
- **Multi-step Interaction**: Sequential questioning approach for better issue diagnosis

## 📁 Project Structure

```
project/
├── kb/
│   └── kb_consolidated_1.txt          # Knowledge base text file
├── data/
│   └── filtered_data.csv              # Training data for classification
├── embeddings/
│   ├── faiss_index.bin               # Vector embeddings index (generated)
│   └── chunks.pkl                    # Text chunks (generated)
├── logs/
│   └── conversation_logs.txt         # Conversation logs (generated)
├── models/
│   └── hybrid_classifier.pkl         # Trained classifier (generated)
├── .env                             # API keys and configuration
├── build_embeddings.py              # Embedding generation script
├── train_classifier.py              # Classification model training
├── rag_chatbot.py                   # OpenRouter API version chatbot
├── rag_chatbot_LM.py                 # LM Studio local API version chatbot
├── main.py                          # Main execution script (OpenRouter)
├── main_LM.py                        # Main execution script (LM Studio)
└── requirements.txt                 # Python dependencies
```

## 🛠️ Prerequisites

### For OpenRouter Version:
- Python 3.7+
- OpenRouter API key
- Internet connection

### For LM Studio Version:
- Python 3.7+
- LM Studio installed and running
- `openai/gpt-oss-20b` model loaded in LM Studio
- LM Studio API server enabled (default port: 1234)

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Required packages:**
```
openai
faiss-cpu
scikit-learn
lightgbm
xgboost
python-dotenv
pandas
numpy
tqdm
sentence-transformers
requests
```

## ⚙️ Setup

### 1. Knowledge Base
Place your knowledge base file at:
```
kb/kb_consolidated_1.txt
```

### 2. Training Data
Place your classification training data at:
```
data/filtered_data.csv
```
Must contain columns: `Summary` (input) and `Request Type` (target)

### 3. API Configuration

**For OpenRouter Version (.env file):**
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**For LM Studio Version:**
- Start LM Studio
- Load `openai/gpt-oss-20b` model
- Enable API server (Settings → API Server)
- Default URL: `http://localhost:1234/v1`

## ▶️ Usage

### OpenRouter Version:
```bash
python main.py
```

### LM Studio Version:
```bash
python main_LM.py
```

### First Run Process:
1. Builds embeddings from knowledge base (one-time setup)
2. Trains hybrid classification model (one-time setup)
3. Starts interactive chatbot session

### Subsequent Runs:
- Reuses cached embeddings and trained models
- Appends new conversations to logs

## 🤖 Chatbot Workflow

1. **Initial Greeting**: Bot greets user and asks for issue description
2. **First Question**: Bot asks follow-up to narrow down issue
3. **Second Question**: Bot asks another follow-up based on conversation history
4. **Final Response**: Bot provides 3-4 quick fixes/diagnostics
5. **Resolution**: User indicates if issue is solved
6. **Logging**: Complete conversation logged with category prediction

## 📊 Logging

### Log Location:
```
logs/conversation_logs.txt
```

### Log Format:
```
[2024-01-15 14:30:45] User: System is running slow
[2024-01-15 14:30:45] Bot: Can you specify whether it is due to some specific application or for all?

[2024-01-15 14:31:10] User: For some specific application
[2024-01-15 14:31:10] Bot: Can you tell me which application is causing the issue?

[2024-01-15 14:31:35] User: It's Chrome browser
[2024-01-15 14:31:35] Bot: Try these solutions: 1) Close unused tabs...

[2024-01-15 14:32:00] User: Issue solved
[2024-01-15 14:32:00] Bot: Solved by Bot | Category: Performance Issue

============================================================
```

## 🧠 Model Training

### Automatic Training:
- First run automatically trains the hybrid classifier
- Uses filtered_data.csv for training
- Combines LightGBM, Logistic Regression, and XGBoost

### Manual Retraining:
```bash
python train_classifier.py
```

### Hybrid Model Architecture:
- **Base Models**: LightGBM, Logistic Regression, XGBoost
- **Meta Model**: Logistic Regression
- **Ensemble Method**: Stacking with 5-fold cross-validation

## 🔧 Troubleshooting

### Common Issues:

**1. Embeddings not building:**
```bash
# Delete existing embeddings and rebuild
rm -rf embeddings/
python build_embeddings.py
```

**2. Classification model issues:**
```bash
# Delete existing model and retrain
rm -rf models/hybrid_classifier.pkl
python train_classifier.py
```

**3. LM Studio connection errors:**
- Ensure LM Studio is running
- Verify API server is enabled
- Check if model is loaded
- Confirm port 1234 is available

**4. API key errors (OpenRouter):**
- Verify .env file exists
- Check API key validity
- Ensure internet connection

**5. Slow responses:**
- First run will be slower due to setup
- Large knowledge base increases processing time
- Local models may take longer to respond

### Clear All Cached Data:
```bash
rm -rf embeddings/
rm -rf models/
rm -rf logs/conversation_logs.txt
```

## 📈 Performance Tips

1. **Knowledge Base Optimization:**
   - Keep chunks concise but informative
   - Regular updates improve accuracy

2. **Classification Training:**
   - More training data improves category prediction
   - Balanced dataset recommended

3. **Embedding Generation:**
   - One-time cost on first run
   - Cached for subsequent runs

4. **API Usage:**
   - LM Studio version works offline
   - OpenRouter version requires internet

## 🆘 Support

For issues, please check:
1. All required files are in correct locations
2. Dependencies are properly installed
3. API services are running (if applicable)
4. Review logs for error messages

## 📄 License

This project is for internal use. All rights reserved. Under AIR.

## 🙋‍♂️ Authors

NoxRe

---
