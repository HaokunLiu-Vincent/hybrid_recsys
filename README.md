# Movie Recommendation System
## End-to-End ML Pipeline on AWS SageMaker
**Hybrid Recommendation: Collaborative Filtering + Content-Based + NLP Transformer**

## Project Overview
This project demonstrates a complete, production-ready machine learning pipeline on AWS SageMaker.
I built a **three-branch hybrid recommendation system** that fuses:
- **Collaborative Filtering (NMF Matrix Factorization)**
- **Content-Based Filtering**
- **NLP Transformer-based semantic understanding (SBERT)**

The focus is on **real-world ML engineering**: environment stabilization, dependency fixing, GPU training, CPU inference deployment, cost control, and full pipeline delivery.

## Project Structure
```
.
├── 01_EDA_From_S3.ipynb                # Exploratory Data Analysis
├── 02_Model_Development.ipynb         # Local model development & testing
├── 03_SageMaker_Pipeline.ipynb           # SageMaker training & deployment pipeline
├─ src/
│   ├── train.py             # Training script for SageMaker
│   ├── inference.py         # Inference script for endpoint
│   └── requirements.txt     # Fixed dependencies
├── setup_env.sh          # conda env set up script
├── env_set_up.txt        # documentation for conda env set up
└── README.md
```

## Actual Workflow
1. Upload dataset to AWS S3
2. Perform **Exploratory Data Analysis (EDA)**
3. Build and validate the hybrid model **locally**
4. Build a scalable training pipeline on SageMaker
5. Train the model on **GPU instances**
6. Deploy a lightweight real-time inference endpoint on **CPU instances**
7. Test via API, then clean up resources to avoid cost

## Tech Stack
- Python: **3.11**
- PyTorch: **2.4.0**
- ML: scikit-learn, pandas, numpy, joblib
- NLP: transformers, sentence-transformers (SBERT)
- Cloud: AWS SageMaker, S3
- Training Instance: **ml.g4dn.xlarge (GPU)**
- Inference Instance: **ml.t2.medium (CPU)**

## Stable Environment
### Python Version
- **Python 3.11**
  (Python 3.9 deprecated by AWS in April 2026; 3.10 had compatibility issues)

### requirements.txt
```
numpy==1.26.4
pandas==2.1.4
joblib==1.3.2
matplotlib==3.10.8
seaborn==0.13.2
scikit-learn==1.2.1
torch==2.4.0
sentence-transformers==3.0.1
transformers==4.40.2
boto3==1.34.159
botocore==1.34.159
sagemaker==2.236.0
s3fs==2024.3.1
fsspec==2024.3.1
ipywidgets==8.1.2
ipykernel==6.29.3
```

## Steps to Run
1. Upload dataset to S3
2. Run the environment & kernel `setup-env.sh` by following the instructions in  `env_set_up.txt`
3. Select kernal named "Python 3.11 (SageMaker V2)" for all three notebooks, run `EDA.ipynb` for data insights
4. Run `Model_Development.ipynb` to validate model 
5. Run `Pipeline.ipynb` for SageMaker training
   - Training instance: `ml.g4dn.xlarge (GPU)`
6. Deploy inference endpoint
   - Inference instance: `ml.t2.medium (CPU)`
7. Send user ID → get Top-N movie recommendations (movie ID, title, genre)
8. **Delete endpoint immediately after testing**

## Key Challenges & Solutions (Interview Highlights)
1. **Python & PyTorch version conflict**
   - Fixed with **Python 3.11 + PyTorch 2.4.0**
2. **Dependency version hell**
   - Pinned all library versions to ensure stability
3. **SageMaker container path misunderstanding**
   - Used `/opt/ml/input` and `/opt/ml/model` correctly
4. **Invalid instance type for inference**
   - Switched to supported, low-cost `ml.t2.medium`
5. **GPU training & CPU inference separation**
   - Used `ml.g4dn.xlarge (GPU)` for fast Transformer training
   - Used `ml.t2.medium (CPU)` for cost-efficient inference
6. **Full pipeline in cloud API**
   - Delivered end-to-end system without relying on pre-built solutions

## Project Value
- Built a **high-performance hybrid recommender system**
- Completed full cloud ML lifecycle on AWS SageMaker
- Experience with GPU training & CPU inference optimization
- Solved real-world engineering & environment issues
- Capable of shipping ML models to production

✅ Status: **Full Pipeline Success | EDA ✔ | Model Dev ✔ | GPU Training ✔ | CPU Inference ✔ | API Test ✔**