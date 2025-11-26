"""
Generate and save embeddings for all 600 intent examples
Run this ONCE to create the embeddings file, then use it for fast initialization
"""

import numpy as np
import pickle
import logging
from typing import Dict
from classifiers.embedded_classifier import CVS_INTENT_EXAMPLES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import embeddings utility
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.config import settings
    
    if getattr(settings, 'use_google_embeddings', False):
        from services.google_embeddings import get_google_embeddings as get_embeddings
        EMBEDDINGS_PROVIDER = "Google Cloud Vertex AI"
    else:
        from services.azure_embeddings import get_azure_embeddings as get_embeddings
        EMBEDDINGS_PROVIDER = "Azure OpenAI"
    
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Embeddings not available: {e}")
    EMBEDDINGS_AVAILABLE = False
    exit(1)

def generate_and_save_embeddings():
    """Generate embeddings for all 600 intent examples and save to file"""
    
    print("=" * 80)
    print("🧠 GENERATING INTENT EMBEDDINGS")
    print("=" * 80)
    
    print(f"\n📊 Total intents: {len(CVS_INTENT_EXAMPLES)}")
    print(f"📊 Total examples: {sum(len(examples) for examples in CVS_INTENT_EXAMPLES.values())}")
    
    # Initialize embeddings service
    print(f"\n⏳ Initializing {EMBEDDINGS_PROVIDER} Embeddings...")
    
    # Force fresh initialization by clearing singleton
    if EMBEDDINGS_PROVIDER == "Google Cloud Vertex AI":
        import services.google_embeddings as emb_module
        emb_module._google_embeddings = None
    else:
        import services.azure_embeddings as emb_module
        emb_module._azure_embeddings = None
    
    embeddings_service = get_embeddings()
    print(f"✅ Embeddings service ready ({EMBEDDINGS_PROVIDER})")
    print(f"   Client status: {'Connected' if embeddings_service.client else 'NOT CONNECTED (using mocks)'}")
    print(f"   Auth method: {embeddings_service.auth_method}")
    
    # Generate embeddings for all intents
    intent_embeddings = {}
    
    print("\n⏳ Generating embeddings (using batch processing)...")
    print(f"   This will make ~30 {EMBEDDINGS_PROVIDER} API calls (one per intent)")
    print("   Estimated time: ~30-40 seconds\n")
    
    for i, (intent, examples) in enumerate(CVS_INTENT_EXAMPLES.items(), 1):
        print(f"   [{i}/{len(CVS_INTENT_EXAMPLES)}] {intent}: {len(examples)} examples...", end='')
        
        # Batch embed all examples for this intent
        embeddings = embeddings_service.embed(examples)
        
        # Convert to numpy array
        intent_embeddings[intent] = np.array(embeddings)
        
        print(f" ✅")
    
    print(f"\n✅ All embeddings generated!")
    
    # Save to file in the classifiers directory (where embedded_classifier.py lives)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    classifiers_dir = os.path.join(script_dir, "classifiers")
    output_file = os.path.join(classifiers_dir, "intent_embeddings_cache.pkl")
    print(f"\n⏳ Saving to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(intent_embeddings, f)
    
    print(f"✅ Embeddings saved successfully!")
    
    # Verify file
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"   File size: {file_size:.2f} MB")
    print(f"   Location: {output_file}")
    
    print("\n" + "=" * 80)
    print("🎉 DONE!")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("   1. The embeddings are now cached in intent_embeddings_cache.pkl")
    print("   2. The classifier will automatically load from this file")
    print("   3. Initialization will be INSTANT (no API calls needed!)")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    generate_and_save_embeddings()

