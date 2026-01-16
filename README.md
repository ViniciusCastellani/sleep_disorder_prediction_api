# Sleep Disorder Prediction System

A machine learning-powered API that predicts sleep disorders (Insomnia, Sleep Apnea, or None) based on health and lifestyle data extracted from natural language input.

## Features

- **Natural Language Processing**: Extract health metrics from conversational text in Portuguese or English
- **Dual Model Architecture**: Combines Decision Tree and Neural Network models for optimal predictions
- **REST API**: Flask-based API for easy integration
- **Local LLM Integration**: Uses Ollama with Gemma 2B for data extraction
- **Database Support**: MySQL integration for training data storage

## Architecture

The system uses a hybrid prediction approach:

1. **Primary Model**: Decision Tree classifier with 85% confidence threshold
2. **Fallback Model**: Neural Network (MLP) for cases where Decision Tree confidence is low
3. **Data Extraction**: Ollama-powered LLM extracts structured data from natural language

## Prerequisites

- Python 3.8+
- MySQL Server
- Ollama with Gemma 2B model installed

### Installing Ollama and Gemma

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Pull Gemma 2B model
ollama pull gemma:2b

# Start Ollama server
ollama serve
```

## Installation

1. **Clone the repository**
```bash
git clone <https://github.com/ViniciusCastellani/sleep-disorder-prediction-api>
cd <project-directory>
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` with your MySQL credentials:
```env
DB_HOST=localhost
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=sleep_db
```

5. **Set up MySQL database**
```sql
CREATE DATABASE sleep_db;
USE sleep_db;

CREATE TABLE sleep_data (
    person_id INT PRIMARY KEY AUTO_INCREMENT,
    gender VARCHAR(10),
    age INT,
    occupation VARCHAR(50),
    sleep_duration FLOAT,
    quality_of_sleep INT,
    physical_activity_level INT,
    stress_level INT,
    bmi_category VARCHAR(20),
    blood_pressure VARCHAR(10),
    heart_rate INT,
    daily_steps INT,
    sleep_disorder VARCHAR(20)
);
```

6. **Load training data** (optional)
```bash
# If you have the CSV file
python scripts/send_csv_data_to_sql.py
```

7. **Train models**
```bash
python -m scripts.train_models
```

## Usage

### Start the API Server

```bash
python main.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### POST `/predict`

Predicts sleep disorder from natural language input.

**Request:**
```json
{
  "text": "I'm a 35 year old male engineer. I sleep 6.5 hours per night and my sleep quality is about 7/10. I exercise 45 minutes daily and walk 8000 steps. My stress level is 6/10, BMI is normal, blood pressure is 128/85, and heart rate is 72 bpm."
}
```

**Response (Success):**
```json
{
  "status": "success",
  "input": {
    "Gender": "Male",
    "Age": 35,
    "Occupation": "Engineer",
    "Sleep Duration": 6.5,
    "Quality of Sleep": 7,
    "Physical Activity Level": 45,
    "Stress Level": 6,
    "BMI Category": "Normal",
    "Heart Rate": 72,
    "Daily Steps": 8000,
    "systolic": 128,
    "diastolic": 85
  },
  "prediction": {
    "class_id": 1,
    "class_name": "None",
    "probability": 0.8945,
    "model_used": "decision_tree"
  }
}
```

**Response (Incomplete Data):**
```json
{
  "status": "incomplete",
  "missing_fields": ["Heart Rate", "Daily Steps"]
}
```

#### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "online"
}
```

## Input Fields

The system extracts the following fields from natural language:

- **Gender**: Male, Female
- **Age**: Integer (years)
- **Occupation**: Doctor, Nurse, Engineer, Teacher, Accountant, Other
- **Sleep Duration**: Float (hours per day)
- **Quality of Sleep**: Integer (1-10 scale)
- **Physical Activity Level**: Integer (minutes per day of exercise)
- **Stress Level**: Integer (1-10 scale)
- **BMI Category**: Underweight, Normal, Overweight, Obese
- **Blood Pressure**: String format "systolic/diastolic" (e.g., "120/80")
- **Heart Rate**: Integer (beats per minute)
- **Daily Steps**: Integer (steps per day)

## Project Structure

```
.
├── ai_module/              # LLM integration for data extraction
│   ├── ollama_client.py    # Ollama API client
│   ├── selector.py         # Main extraction logic
│   ├── validation.py       # Field validation and normalization
│   └── prompt/             # YAML prompt templates
├── data/                   # Database configuration and fetching
├── models/                 # Trained model files (.joblib)
├── predict/                # Prediction pipeline
├── preprocessing/          # Data preprocessing modules
├── scripts/                # Utility scripts
│   ├── train_models.py     # Model training script
│   └── expand_csv.py       # Dataset augmentation
├── training/               # Training pipelines
├── main.py                 # Flask API server
├── requirements.txt        # Python dependencies
└── .env.example           # Environment variables template
```

## Model Details

### Decision Tree
- **Criterion**: Entropy
- **Max Depth**: 5
- **Class Weight**: Balanced
- **Use Case**: High-confidence predictions (≥85%)

### Neural Network (MLP)
- **Architecture**: 490-490 hidden layers
- **Activation**: ReLU
- **Solver**: Adam
- **Regularization**: L2 (alpha=1e-8)
- **Use Case**: Fallback for uncertain cases

Both models are trained with SMOTE oversampling to handle class imbalance.

## Development

### Retraining Models

```bash
# After updating training data
python scripts/train_models.py
```

### Expanding Dataset

```bash
# Generate synthetic data
python scripts/expand_csv.py
```

## Troubleshooting

**Ollama Connection Error**
- Ensure Ollama is running: `ollama serve`
- Check if Gemma 2B is installed: `ollama list`

**MySQL Connection Error**
- Verify credentials in `.env`
- Ensure MySQL server is running
- Check database exists: `SHOW DATABASES;`

**Model Loading Error**
- Run training script: `python scripts/train_models.py`
- Ensure `models/` directory contains all `.joblib` files

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.
