# Retail Sales Data API

This Flask API processes retail sales data and provides endpoints for data analysis and sending data to Ull.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python app.py
```

## API Endpoints

### POST /api/process-data
Processes the retail sales data and returns similarity metrics.

Request body:
```json
{
    "customerID": "optional_customer_id",
    "date_range": {
        "start": "YYYY-MM-DD",
        "end": "YYYY-MM-DD"
    }
}
```

### GET /api/health
Health check endpoint to verify the API is running.

## Data Format

The API expects retail sales data in CSV format with the following columns:
- customerID
- item
- amount_usd
- date
- review
- payment
- month
