# S3 Data Requirements

This document describes the required S3 bucket structure and data format for the mineral prospectivity framework.

## S3 Bucket Structure

```
s3://your-bucket-name/
├── data/
│   ├── training/
│   │   ├── features.npy
│   │   └── labels.npy
│   ├── validation/
│   │   ├── features.npy
│   │   └── labels.npy
│   ├── test/
│   │   ├── features.npy
│   │   └── labels.npy
│   └── features_metadata.json
├── experiments/
│   └── {experiment_name}/
│       ├── config_{timestamp}.json
│       ├── models/
│       │   └── {timestamp}/
│       │       ├── model_0.pt
│       │       ├── model_1.pt
│       │       └── ...
│       └── results/
│           └── {timestamp}/
│               ├── history_0.json
│               ├── history_1.json
│               └── ...
```

## Data Format Requirements

### 1. Feature Arrays (features.npy)

NumPy arrays with shape `(n_samples, n_features)` containing geospatial features:

```python
import numpy as np

# Example: Create synthetic training features
n_samples = 10000
n_features = 50

features = np.random.randn(n_samples, n_features).astype(np.float32)

# Save as NumPy binary format
np.save('features.npy', features)
```

**Feature types typically include:**
- Geophysical data (magnetic intensity, gravity anomalies)
- Geochemical data (element concentrations)
- Geological data (rock types, structural features)
- Remote sensing data (spectral indices)
- Topographic data (elevation, slope, aspect)
- Proximity features (distance to faults, contacts)

### 2. Label Arrays (labels.npy)

NumPy arrays with shape `(n_samples,)` or `(n_samples, 1)` containing binary labels:

```python
# Example: Create synthetic labels (0 = no deposit, 1 = deposit)
labels = np.random.binomial(1, 0.05, n_samples).astype(np.float32)

np.save('labels.npy', labels)
```

**Label conventions:**
- `1` = Known mineral deposit location
- `0` = Non-deposit location or background

### 3. Features Metadata (features_metadata.json)

JSON file describing the features:

```json
{
  "n_features": 50,
  "feature_names": [
    "magnetic_intensity",
    "gravity_anomaly",
    "cu_concentration",
    "au_concentration",
    "elevation",
    "slope",
    "distance_to_fault",
    ...
  ],
  "feature_types": {
    "geophysical": [0, 1],
    "geochemical": [2, 3],
    "topographic": [4, 5],
    "proximity": [6]
  },
  "units": {
    "magnetic_intensity": "nT",
    "gravity_anomaly": "mGal",
    "cu_concentration": "ppm",
    "au_concentration": "ppb",
    "elevation": "m",
    "slope": "degrees",
    "distance_to_fault": "km"
  },
  "normalization": {
    "method": "robust",
    "description": "Features normalized using RobustScaler"
  },
  "spatial_info": {
    "coordinate_system": "EPSG:4326",
    "spatial_resolution": "1km"
  },
  "data_source": "Survey data from XYZ project",
  "date_collected": "2023-01-01",
  "preprocessing": [
    "Outlier clipping at 1st and 99th percentiles",
    "Missing value imputation using median",
    "RobustScaler normalization"
  ]
}
```

## Uploading Data to S3

### Using AWS CLI

```bash
# Upload training data
aws s3 cp features.npy s3://your-bucket/data/training/features.npy
aws s3 cp labels.npy s3://your-bucket/data/training/labels.npy

# Upload validation data
aws s3 cp val_features.npy s3://your-bucket/data/validation/features.npy
aws s3 cp val_labels.npy s3://your-bucket/data/validation/labels.npy

# Upload test data
aws s3 cp test_features.npy s3://your-bucket/data/test/features.npy
aws s3 cp test_labels.npy s3://your-bucket/data/test/labels.npy

# Upload metadata
aws s3 cp features_metadata.json s3://your-bucket/data/features_metadata.json
```

### Using Python

```python
import boto3
import numpy as np

s3_client = boto3.client('s3')
bucket_name = 'your-bucket'

# Upload training features
s3_client.upload_file(
    'features.npy',
    bucket_name,
    'data/training/features.npy'
)

# Upload training labels
s3_client.upload_file(
    'labels.npy',
    bucket_name,
    'data/training/labels.npy'
)
```

### Using the Framework

```python
from mineral_prospectivity.data.s3_loader import S3DataLoader
import numpy as np

# Create dummy data
features = np.random.randn(1000, 50).astype(np.float32)
labels = np.random.binomial(1, 0.05, 1000).astype(np.float32)

# Save locally first
np.save('features.npy', features)
np.save('labels.npy', labels)

# Upload using S3DataLoader
s3_loader = S3DataLoader(
    bucket_name='your-bucket',
    experiment_name='my-experiment'
)

s3_loader.upload_file('features.npy', 'data/training/features.npy')
s3_loader.upload_file('labels.npy', 'data/training/labels.npy')
```

## Data Preparation Recommendations

### 1. Data Splitting

Split your data before uploading:

```python
from sklearn.model_selection import train_test_split

# Split data: 70% train, 15% val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    features, labels, test_size=0.15, stratify=labels, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42  # 0.176 ≈ 0.15/0.85
)

# Save splits
np.save('train_features.npy', X_train)
np.save('train_labels.npy', y_train)
np.save('val_features.npy', X_val)
np.save('val_labels.npy', y_val)
np.save('test_features.npy', X_test)
np.save('test_labels.npy', y_test)
```

### 2. Preprocessing

Apply preprocessing before uploading:

```python
from mineral_prospectivity.data.preprocessor import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(
    scaler_type='robust',
    imputation_strategy='median',
    handle_outliers=True
)

# Fit on training data
preprocessor.fit(X_train)

# Transform all splits
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor for later use
preprocessor.save('preprocessor.pkl')
s3_loader.upload_file('preprocessor.pkl', 'data/preprocessor.pkl')
```

### 3. Class Imbalance

Handle class imbalance if needed:

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

## Data Quality Checks

Before uploading, verify data quality:

```python
from mineral_prospectivity.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
quality_report = preprocessor.check_data_quality(features)

print(f"Data quality report:")
print(f"  Samples: {quality_report['n_samples']}")
print(f"  Features: {quality_report['n_features']}")
print(f"  Missing values: {quality_report['missing_values']['fraction']:.2%}")
print(f"  Constant features: {len(quality_report['constant_features'])}")
```

## Verification

After uploading, verify data accessibility:

```python
from mineral_prospectivity.aws_batch.job_launcher import BatchJobLauncher

launcher = BatchJobLauncher(
    job_queue='mineral-prospectivity-queue',
    job_definition='mineral-prospectivity-training',
    s3_bucket='your-bucket'
)

# Check data requirements
requirements = launcher.check_s3_data_requirements()

for component, exists in requirements.items():
    status = "✓" if exists else "✗"
    print(f"{status} {component}")
```

## Example: Complete Data Preparation Script

```python
import numpy as np
from sklearn.model_selection import train_test_split
from mineral_prospectivity.data.preprocessor import DataPreprocessor
from mineral_prospectivity.data.s3_loader import S3DataLoader
import json

# Load your raw data
features = np.load('raw_features.npy')
labels = np.load('raw_labels.npy')

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(
    features, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

# Preprocess
preprocessor = DataPreprocessor()
preprocessor.fit(X_train)

X_train = preprocessor.transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# Save locally
np.save('train_features.npy', X_train)
np.save('train_labels.npy', y_train)
np.save('val_features.npy', X_val)
np.save('val_labels.npy', y_val)
np.save('test_features.npy', X_test)
np.save('test_labels.npy', y_test)

# Create metadata
metadata = {
    "n_features": X_train.shape[1],
    "n_train_samples": len(X_train),
    "n_val_samples": len(X_val),
    "n_test_samples": len(X_test),
    "positive_fraction_train": float(y_train.mean()),
    "preprocessing": "RobustScaler with outlier clipping"
}

with open('features_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Upload to S3
s3_loader = S3DataLoader(
    bucket_name='your-bucket',
    experiment_name='experiment-1'
)

# Upload all files
for split in ['train', 'val', 'test']:
    s3_loader.upload_file(f'{split}_features.npy', f'data/{split}ing/features.npy')
    s3_loader.upload_file(f'{split}_labels.npy', f'data/{split}ing/labels.npy')

s3_loader.upload_file('features_metadata.json', 'data/features_metadata.json')

print("Data uploaded successfully!")
```
