# How to Find Your GCP Project Zone/Region

## Quick Check

### 1. Check gcloud Configuration
```bash
gcloud config get-value project
# Output: pbm-nonprod-myclaims
```

### 2. Check Application Configuration

**Default in code:**
- **Project ID**: `pbm-poc-coderev-genai-poc` (default in `config/config.py`)
- **Location**: `us-central1` (default in `config/config.py`)

**Your actual project:**
- **Project ID**: `pbm-nonprod-myclaims` (from gcloud config)
- **Location**: Check `.env` file or runtime logs

### 3. Check .env File
```bash
grep -E "PROJECT_ID|LOCATION" .env
```

If not set, the application uses defaults from `config/config.py`.

---

## For Vertex AI (Gemini/Embeddings)

**Important**: Vertex AI uses **regions**, not zones.

Common Vertex AI regions:
- `us-central1` (Iowa)
- `us-east1` (South Carolina)
- `us-east4` (Northern Virginia)
- `us-west1` (Oregon)
- `europe-west1` (Belgium)
- `asia-southeast1` (Singapore)

---

## How to Find Which Region You're Using

### Method 1: Check Application Logs

When the application starts, look for:
```
✅ Google Cloud Vertex AI Embeddings initialized successfully
   Project: pbm-nonprod-myclaims
   Region: us-central1
```

### Method 2: Check .env File

Create or check `.env` file:
```bash
PROJECT_ID=pbm-nonprod-myclaims
LOCATION=us-central1
```

### Method 3: Check gcloud Vertex AI Locations

```bash
gcloud ai locations list --project=pbm-nonprod-myclaims
```

### Method 4: Check GCP Console

1. Go to [GCP Console](https://console.cloud.google.com)
2. Select project: `pbm-nonprod-myclaims`
3. Navigate to **Vertex AI** → **Model Garden** or **Workbench**
4. Check which region your resources are in

---

## Setting the Location

### Option 1: Environment Variable (Recommended)

Create/update `.env` file:
```bash
PROJECT_ID=pbm-nonprod-myclaims
LOCATION=us-east1
```

### Option 2: Update config.py

Edit `config/config.py`:
```python
project_id: str = "pbm-nonprod-myclaims"
location: str = "us-east1"  # Change to your desired region
```

---

## Available Zones vs Regions

**Zones** (for Compute Engine):
- `us-east1-b`, `us-east1-c`, `us-east1-d`
- `us-east4-b`, `us-east4-c`

**Regions** (for Vertex AI):
- `us-east1` (South Carolina)
- `us-east4` (Northern Virginia)
- `us-central1` (Iowa) - default in config

---

## Quick Commands

```bash
# Check current gcloud project
gcloud config get-value project

# Check available Vertex AI regions
gcloud ai locations list --project=pbm-nonprod-myclaims

# Check compute zones
gcloud compute zones list --project=pbm-nonprod-myclaims

# Check compute regions
gcloud compute regions list --project=pbm-nonprod-myclaims
```

---

## Summary

- **Your Project**: `pbm-nonprod-myclaims`
- **Default Location**: `us-central1` (if not overridden in .env)
- **To Change**: Set `LOCATION=your-region` in `.env` file
- **For Vertex AI**: Use region (e.g., `us-east1`), not zone (e.g., `us-east1-b`)

