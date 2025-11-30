#!/usr/bin/env python3
"""
Check GCP Project and Location Configuration
Shows which project and location/zone the application is using
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.config import settings
    
    print("=" * 60)
    print("GCP Configuration Check")
    print("=" * 60)
    print(f"\n📋 Project ID: {settings.project_id}")
    print(f"📍 Location/Region: {settings.location}")
    print(f"\n💡 Note: For Vertex AI, 'location' refers to a region, not a zone")
    print(f"   Common regions: us-central1, us-east1, us-west1, etc.")
    print("\n" + "=" * 60)
    
    # Check if .env file exists
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_file):
        print(f"\n✅ .env file found: {env_file}")
        with open(env_file, 'r') as f:
            env_content = f.read()
            if 'PROJECT_ID' in env_content or 'LOCATION' in env_content:
                print("   Contains PROJECT_ID or LOCATION overrides")
            else:
                print("   No PROJECT_ID or LOCATION overrides (using config.py defaults)")
    else:
        print(f"\n⚠️  No .env file found - using config.py defaults")
        print(f"   Default project: {settings.project_id}")
        print(f"   Default location: {settings.location}")
    
    print("\n" + "=" * 60)
    print("To check available Vertex AI regions, run:")
    print("  gcloud ai locations list --project=pbm-nonprod-myclaims")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

