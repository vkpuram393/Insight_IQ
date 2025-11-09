#!/bin/bash

# Script to download CVS Health Root Certificate
# This certificate is required when behind CVS proxy/Zscaler

CERT_DIR="certs"
CERT_FILE="$CERT_DIR/CVSHealthRoot.cer"
CERT_URL="http://crl.cvshealth.com/CVSHealthRoot.cer"

echo "================================"
echo "CVS Root Certificate Setup"
echo "================================"

# Create certs directory if it doesn't exist
mkdir -p "$CERT_DIR"

# Try to download the certificate
echo "Attempting to download CVS root certificate..."
echo "URL: $CERT_URL"

if command -v curl &> /dev/null; then
    echo "Using curl..."
    curl -k -L "$CERT_URL" -o "$CERT_FILE"
elif command -v wget &> /dev/null; then
    echo "Using wget..."
    wget --no-check-certificate "$CERT_URL" -O "$CERT_FILE"
else
    echo "ERROR: Neither curl nor wget found!"
    exit 1
fi

# Check if download was successful
if [ -f "$CERT_FILE" ] && [ -s "$CERT_FILE" ]; then
    echo "✓ Certificate downloaded successfully to $CERT_FILE"
    echo "  File size: $(ls -lh "$CERT_FILE" | awk '{print $5}')"

    # Verify it's a valid certificate
    if openssl x509 -in "$CERT_FILE" -text -noout &> /dev/null; then
        echo "✓ Certificate appears to be valid"
        echo ""
        echo "Certificate details:"
        openssl x509 -in "$CERT_FILE" -subject -issuer -dates -noout
    else
        echo "⚠ Warning: Downloaded file may not be a valid certificate"
    fi

    echo ""
    echo "Next steps:"
    echo "1. The .env file has been configured to use this certificate"
    echo "2. Install Python dependencies: pip install -r requirements.txt"
    echo "3. Run the application: python main.py"
else
    echo "✗ Failed to download certificate"
    echo ""
    echo "Manual download instructions:"
    echo "1. Open in browser: $CERT_URL"
    echo "2. Save the file as: $CERT_FILE"
    echo "3. Or contact IT support to obtain the CVS root certificate"
    exit 1
fi

echo "================================"

