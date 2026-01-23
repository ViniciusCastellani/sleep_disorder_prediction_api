# Sleep Disorder Prediction System

A machine learning-powered API that predicts sleep disorders (Insomnia, Sleep Apnea, or None) based on health and lifestyle data extracted from natural language input.

## Features

- **Natural Language Processing**: Extract health metrics from conversational text in Portuguese or English
- **Dual Model Architecture**: Combines Decision Tree and Neural Network models for optimal predictions
- **REST API**: Flask-based API for easy integration
- **Google Gemini AI Integration**: Uses Gemini 2.0 Flash for intelligent data extraction
- **Database Support**: MySQL integration for training data storage

## Architecture

The system uses a hybrid prediction approach:

1. **Primary Model**: Decision Tree classifier with 85% confidence threshold
2. **Fallback Model**: Neural Network (MLP) for cases where Decision Tree confidence is low
3. **Data Extraction**: Google Gemini AI extracts structured data from natural language

## Prerequisites

- Python 3.8+
- MySQL Server
- Google Gemini API Key

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Get API Key" or "Create API Key"
4. Copy your API key (keep it secure!)

**Note**: The Gemini API has a free tier with generous limits suitable for development and testing.

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/ViniciusCastellani/sleep-disorder-prediction-api
cd sleep-disorder-prediction-api
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

Edit `.env` with your configuration:
```env
# Database Configuration
DB_HOST=localhost
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=sleep_db

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important**: Never commit your `.env` file to version control. It's already in `.gitignore`.

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

The API will be available at `http://127.0.0.1:5000`

### API Endpoints

#### POST `/predict`

Predicts sleep disorder from natural language input.

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am a 35 year old male doctor. I sleep 7.5 hours per night and my sleep quality is 8 out of 10. I exercise for 45 minutes daily and walk 8000 steps per day. My stress level is 6. I am overweight. My blood pressure is 128/85 and my heart rate is 72 bpm."
  }'
```

**Response (Success):**
```json
{
  "status": "success",
  "original_text": "I am a 35 year old male doctor...",
  "extracted_fields": {
    "Gender": "Male",
    "Age": 35,
    "Occupation": "Doctor",
    "Sleep Duration": 7.5,
    "Quality of Sleep": 8,
    "Physical Activity Level": 45,
    "Stress Level": 6,
    "BMI Category": "Overweight",
    "Heart Rate": 72,
    "Daily Steps": 8000,
    "systolic": 128,
    "diastolic": 85
  },
  "model_input": {
    "gender": "Male",
    "age": 35,
    "occupation": "Doctor",
    "sleep_duration": 7.5,
    "quality_of_sleep": 8,
    "physical_activity_level": 45,
    "stress_level": 6,
    "bmi_category": "Overweight",
    "heart_rate": 72,
    "daily_steps": 8000,
    "systolic": 128,
    "diastolic": 85
  },
  "prediction": {
    "class_id": 1,
    "class_name": "None",
    "probability": 1.0,
    "model_used": "decision_tree"
  }
}
```

**Response (Incomplete Data):**
```json
{
  "status": "incomplete",
  "original_text": "I am 28 years old and sleep 6 hours.",
  "extracted": {
    "Age": 28,
    "Sleep Duration": 6.0
  },
  "missing_fields": [
    "Gender",
    "Occupation",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Blood Pressure",
    "Heart Rate",
    "Daily Steps"
  ]
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
├── ai_module/              # Gemini AI integration for data extraction
│   ├── gemini_client.py    # Google Gemini API client
│   ├── selector.py         # Main extraction logic
│   ├── validation.py       # Field validation and normalization
│   └── prompt/             # YAML prompt templates
│       └── extract_sleep.yaml
├── data/                   # Database configuration and fetching
├── models/                 # Trained model files (.joblib)
├── predict/                # Prediction pipeline
│   ├── predict_combined.py
│   └── feature_mapper.py
├── preprocessing/          # Data preprocessing modules
├── scripts/                # Utility scripts
│   ├── train_models.py     # Model training script
│   └── expand_csv.py       # Dataset augmentation
├── training/               # Training pipelines
├── main.py                 # Flask API server
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── .gitignore             # Git ignore file
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

## Google Gemini Integration

The system uses Google's Gemini 2.0 Flash model for natural language understanding:

- **Model**: `gemini-2.0-flash-exp`
- **Temperature**: 0.0 (deterministic output)
- **Output Format**: JSON
- **Capabilities**: Multilingual support (English and Portuguese)

The Gemini API is configured to extract structured health data from conversational text while strictly following validation rules to ensure data quality.

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

### Testing the API

```bash
# Health check
curl -X GET http://127.0.0.1:5000/health

# Prediction with complete data
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Sou mulher de 42 anos, enfermeira. Durmo 6 horas, qualidade 5. Exercício 30 minutos, 10000 passos. Estresse 8. Peso normal. PA 118/78, FC 68."}'
```

## Troubleshooting

**Gemini API Error**
- Verify your API key in `.env` file
- Check if you have API quota remaining at [Google AI Studio](https://aistudio.google.com/)
- Ensure internet connection is stable

**MySQL Connection Error**
- Verify credentials in `.env`
- Ensure MySQL server is running
- Check database exists: `SHOW DATABASES;`

**Model Loading Error**
- Run training script: `python scripts/train_models.py`
- Ensure `models/` directory contains all `.joblib` files

**JSON Parsing Error**
- The system automatically cleans Gemini's JSON output
- Check `ai_module/prompt/extract_sleep.yaml` for prompt configuration

## API Rate Limits

**Google Gemini Free Tier:**
- 15 requests per minute
- 1,500 requests per day
- 1 million tokens per day

For production use, consider upgrading to a paid plan.

## Security Notes

- Never expose your `GEMINI_API_KEY` publicly
- Keep `.env` file out of version control
- Use environment variables in production deployments
- Regularly rotate API keys

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review the troubleshooting section

## Acknowledgments

- Google Gemini AI for natural language processing
- scikit-learn for machine learning models
- Flask for web framework