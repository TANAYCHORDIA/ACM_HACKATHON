#!/bin/bash

# ==========================================
# Pneumonia Detection API Test Script
# ==========================================

BASE_URL="http://localhost:8000"
IMAGE_FILE="image.jpg"
INVALID_FILE="invalid.pdf"

echo "=========================================="
echo " Pneumonia Detection API - Test Suite"
echo "=========================================="
echo ""

# ------------------------------------------
# Helper function
# ------------------------------------------
print_divider() {
  echo ""
  echo "------------------------------------------"
  echo ""
}

# ------------------------------------------
# 1. Root Endpoint Test
# ------------------------------------------
echo "1️⃣ Testing Root Endpoint (GET /)"
ROOT_RESPONSE=$(curl -s "$BASE_URL/")
echo "$ROOT_RESPONSE" | jq .
print_divider

# ------------------------------------------
# 2. Health Check Endpoint Test
# ------------------------------------------
echo "2️⃣ Testing Health Endpoint (GET /health)"
HEALTH_RESPONSE=$(curl -s "$BASE_URL/health")
echo "$HEALTH_RESPONSE" | jq .
print_divider

# ------------------------------------------
# 3. Prediction Endpoint – Valid Image
# ------------------------------------------
if [ -f "$IMAGE_FILE" ]; then
  echo "3️⃣ Testing Prediction Endpoint (POST /predict) with valid image"
  PREDICT_RESPONSE=$(curl -s -X POST "$BASE_URL/predict" \
    -H "accept: application/json" \
    -F "file=@${IMAGE_FILE}")

  echo "$PREDICT_RESPONSE" | jq .
else
  echo "⚠️  Image file '$IMAGE_FILE' not found. Skipping valid prediction test."
fi
print_divider

# ------------------------------------------
# 4. Prediction Endpoint – Invalid File Type
# ------------------------------------------
echo "4️⃣ Testing Prediction Endpoint with invalid file type"

# Create a dummy invalid file if it doesn't exist
if [ ! -f "$INVALID_FILE" ]; then
  echo "This is not an image" > "$INVALID_FILE"
fi

INVALID_RESPONSE=$(curl -s -X POST "$BASE_URL/predict" \
  -H "accept: application/json" \
  -F "file=@${INVALID_FILE}")

echo "$INVALID_RESPONSE" | jq .
print_divider

# ------------------------------------------
# 5. Prediction Endpoint – Missing File
# ------------------------------------------
echo "5️⃣ Testing Prediction Endpoint with missing file"
MISSING_FILE_RESPONSE=$(curl -s -X POST "$BASE_URL/predict" \
  -H "accept: application/json")

echo "$MISSING_FILE_RESPONSE" | jq .
print_divider

# ------------------------------------------
# Expected Responses Summary
# ------------------------------------------
echo "=========================================="
echo " ✅ EXPECTED RESPONSES SUMMARY"
echo "=========================================="
echo ""

cat <<EOF
1. GET /
----------------------------------
{
  "message": "Pneumonia Detection API",
  "status": "online",
  "version": "1.0.0",
  "docs": "/docs"
}

2. GET /health
----------------------------------
{
  "status": "healthy",
  "timestamp": "<ISO_TIMESTAMP>",
  "model_loaded": false
}

3. POST /predict (valid image)
----------------------------------
- HTTP 200
- success: true
- prediction: "Normal" | "Bacterial Pneumonia" | "Viral Pneumonia"
- confidence: 0.0 – 1.0
- probabilities object with all 3 classes
- priority_score: 0 – 100
- urgency_level: LOW | MEDIUM | HIGH | CRITICAL
- severity_level: None | Mild | Moderate | Severe
- consultation_recommended: true | false

4. POST /predict (invalid file type)
----------------------------------
- HTTP 400
{
  "detail": "Invalid file type: application/pdf. Must be an image."
}

5. POST /predict (missing file)
----------------------------------
- HTTP 422
{
  "detail": [
    {
      "loc": ["body", "file"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}

==========================================
 🎉 TESTING COMPLETE
==========================================
EOF
