#!/bin/bash
# setup_cert.sh - Provision & verify CVS root certificate trust (behind Zscaler/corporate proxy)
# Idempotent, self-validating, builds combined CA bundle for Python requests + gcloud.
set -euo pipefail

echo "================================"
echo "Corporate Root Certificate Trust Setup"
echo "================================"

CERTS_DIR="certs"
CERT_FILE_ORIG="$CERTS_DIR/CVSHealthRoot.cer"
COMBINED_CA="$CERTS_DIR/combined-ca.pem"
CERT_URL="http://crl.cvshealth.com/CVSHealthRoot.cer"

mkdir -p "$CERTS_DIR"

download_cert() {
    if [ -f "$CERT_FILE_ORIG" ] && [ -s "$CERT_FILE_ORIG" ]; then
        echo "🔁 Existing CVS root cert found: $CERT_FILE_ORIG"
        return 0
    fi
    echo "⬇️  Downloading CVS root certificate from $CERT_URL"
    if command -v curl >/dev/null 2>&1; then
        curl -k -L "$CERT_URL" -o "$CERT_FILE_ORIG"
    elif command -v wget >/dev/null 2>&1; then
        wget --no-check-certificate "$CERT_URL" -O "$CERT_FILE_ORIG"
    else
        echo "❌ Neither curl nor wget available"; exit 1
    fi
}

verify_cert() {
    if openssl x509 -in "$CERT_FILE_ORIG" -text -noout >/dev/null 2>&1; then
        echo "✅ Certificate format verified"
        openssl x509 -in "$CERT_FILE_ORIG" -subject -issuer -dates -noout
    else
        echo "❌ Certificate validation failed"; exit 1
    fi
}

build_bundle() {
    local certifi_path
    certifi_path=$(python -c 'import certifi,sys; sys.stdout.write(certifi.where())' 2>/dev/null || true)
    if [ -z "$certifi_path" ] || [ ! -f "$certifi_path" ]; then
        echo "⚠️  certifi bundle not found; using corporate cert only"
        cp "$CERT_FILE_ORIG" "$COMBINED_CA"
    else
        echo "🔧 Building combined CA bundle: $COMBINED_CA"
        cat "$certifi_path" "$CERT_FILE_ORIG" > "$COMBINED_CA"
    fi
}

export_env() {
    export SSL_CERT_FILE="$COMBINED_CA"
    export REQUESTS_CA_BUNDLE="$COMBINED_CA"
    echo "🌱 Exported SSL_CERT_FILE & REQUESTS_CA_BUNDLE"
}

verify_https() {
    echo "🌍 Verifying outbound TLS to oauth2.googleapis.com"
    python - <<'PY'
import os, requests, sys
print('CA bundle:', os.environ.get('REQUESTS_CA_BUNDLE'))
try:
        r = requests.get('https://oauth2.googleapis.com', timeout=8)
        print('Status:', r.status_code)
        if r.status_code in (200,404):
                print('✅ TLS handshake succeeded')
        else:
                print('⚠️ Unexpected status code; TLS appears OK')
except Exception as e:
        print('❌ TLS verification failed:', e)
        sys.exit(1)
PY
}

configure_gcloud() {
    if command -v gcloud >/dev/null 2>&1; then
        echo "🛠  Configuring gcloud custom_ca_certs_file"
        gcloud config set core/custom_ca_certs_file "$COMBINED_CA" >/dev/null || echo "⚠️  gcloud config update failed"
    else
        echo "ℹ️  gcloud not installed; skipping"
    fi
}

download_cert
verify_cert
build_bundle
export_env
verify_https
configure_gcloud

echo "================================"
echo "✅ Trust setup complete"
echo "   Corporate cert: $CERT_FILE_ORIG"
echo "   Combined bundle: $COMBINED_CA"
echo "================================"
cat <<'PERSIST'
To persist across sessions, append to ~/.zshrc (or ~/.bashrc):
    export SSL_CERT_FILE="$(pwd)/certs/combined-ca.pem"
    export REQUESTS_CA_BUNDLE="$(pwd)/certs/combined-ca.pem"
    # Optional: gcloud config set core/custom_ca_certs_file "$(pwd)/certs/combined-ca.pem"

Retest Vertex AI:
    python test_vertex_ai_connection.py

If issues persist:
    - Confirm Zscaler interception still active (visit any HTTPS site)
    - Re-run this script after dependency installs (pip may install certifi later)
    - Inspect bundle head: head certs/combined-ca.pem
PERSIST

