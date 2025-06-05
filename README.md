# content_recom
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)

A hybrid recommendation system that combines content-based filtering using cosine similarity and collaborative filtering. The system allows you to fine-tune recommendation results by adjusting alpha parameters in the Recommender class.

## Table of Contents
- [Installation](#installation)
- [Development](#development)
  - [Azure Function Setup](#create-an-azure-function)
  - [Local Settings Configuration](#fetch-local-settings-of-the-azure-function)
  - [Blob Storage Management](#update-file-on-blob)
  - [Function Deployment](#deploy-an-azure-function)
- [Running the Application](#launch-streamlit-app)

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv 
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development

### Create an Azure Function
Prerequisites:
- Valid Azure Cloud account
- Azure CLI installed
- Azure Functions Core Tools installed
- Storage account name: contentrecom
- Resource group: poc

```bash
az login
az functionapp create --name contentrecomapp-new --storage-account contentrecom --resource-group poc --consumption-plan-location westeurope --runtime python --runtime-version 3.10 --functions-version 4 --os-type linux
```

### Fetch Local Settings of the Azure Function
```bash
func azure functionapp fetch-app-settings contentrecomapp-new --output local.settings.json
```

### Update File on Blob Storage
```bash
az storage blob upload --account-name poc --container-name uploads --name index.hnsw --file ../index.hnsw
```

### Deploy an Azure Function
```bash
func azure functionapp publish contentrecomapp-new --build remote
```

## Running the Application

To launch the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` by default.
