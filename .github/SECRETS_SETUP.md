# GitHub Secrets Setup Guide

This guide explains how to store Azure credentials in GitHub Secrets so they're available in CI/CD pipelines without sharing them with the team.

## How It Works

The application already reads credentials from **environment variables** (via Pydantic Settings). GitHub Actions can inject secrets as environment variables, so **no code changes are needed**.

## Setting Up GitHub Secrets

### Step 1: Navigate to Repository Settings

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**

### Step 2: Add Azure Credentials

Add the following secrets (choose **Option 1** OR **Option 2**):

#### Option 1: Azure OpenAI API Key Authentication

| Secret Name | Description | Example |
|------------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | `https://your-resource.openai.azure.com` |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key | `abc123...` |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Embedding model deployment name | `text-embedding-ada-002` |
| `AZURE_OPENAI_API_VERSION` | API version | `2024-02-01` |

#### Option 2: Azure AD Service Principal (Alternative)

| Secret Name | Description | Example |
|------------|-------------|---------|
| `AZURE_TENANT_ID` | Azure AD tenant ID | `12345678-1234-1234-1234-123456789012` |
| `AZURE_CLIENT_ID` | Service principal client ID | `87654321-4321-4321-4321-210987654321` |
| `AZURE_CLIENT_SECRET` | Service principal client secret | `secret-value...` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | `https://your-resource.openai.azure.com` |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Embedding model deployment name | `text-embedding-ada-002` |
| `AZURE_OPENAI_API_VERSION` | API version | `2024-02-01` |

### Step 3: Use in GitHub Actions Workflow

The workflow file (`.github/workflows/ci.yml`) will automatically use these secrets:

```yaml
env:
  AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
  AZURE_OPENAI_KEY: ${{ secrets.AZURE_OPENAI_KEY }}
  # ... etc
```

## How the Code Reads Credentials

The application uses Pydantic Settings which automatically reads from:

1. **Environment variables** (highest priority) ← GitHub Actions sets these
2. **`.env` file** (if present) ← For local development
3. **Default values** (lowest priority) ← Empty strings in `config.py`

**No code changes needed!** The existing code in `config/config.py` already supports this:

```python
class Settings(BaseSettings):
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret: str = ""
    
    class Config:
        env_file = ".env"  # Also reads from environment variables
        case_sensitive = False
```

## Security Benefits

✅ **Secrets are encrypted** - GitHub encrypts secrets at rest and in transit  
✅ **Access control** - Only repository admins can view/manage secrets  
✅ **Audit trail** - GitHub logs when secrets are accessed  
✅ **No code exposure** - Credentials never appear in code, logs, or PRs  
✅ **Team-friendly** - Team members can run CI/CD without knowing credentials  

## Local Development

For local development, team members can still use `.env` file:

```bash
# .env (not committed to git)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-key-here
```

Or set environment variables directly:

```bash
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_KEY=your-key-here
```

## Testing Locally

To test that secrets work correctly:

```bash
# Set environment variables (simulating GitHub Actions)
export AZURE_OPENAI_ENDPOINT=${{ secrets.AZURE_OPENAI_ENDPOINT }}
export AZURE_OPENAI_KEY=${{ secrets.AZURE_OPENAI_KEY }}

# Run your application
python main.py
```

## Troubleshooting

### Secrets not working in GitHub Actions?

1. **Check secret names** - Must match exactly (case-sensitive in GitHub UI, but code is case-insensitive)
2. **Check workflow file** - Ensure `env:` section includes all required secrets
3. **Check logs** - GitHub Actions logs will show if environment variables are set (but not their values)

### Local development not working?

1. **Check `.env` file** - Ensure it exists and has correct variable names
2. **Check environment variables** - Run `env | grep AZURE` to see what's set
3. **Check config.py** - Ensure `env_file = ".env"` is in Config class

## Next Steps

1. Add secrets to GitHub repository (Settings → Secrets → Actions)
2. Copy `.github/workflows/ci.yml.example` to `.github/workflows/ci.yml`
3. Customize the workflow for your deployment needs
4. Commit and push - secrets will be automatically available in CI/CD


