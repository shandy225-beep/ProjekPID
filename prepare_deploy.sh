#!/bin/bash
# Script untuk prepare deployment ke Streamlit Cloud

echo "🚀 Preparing for Streamlit Cloud Deployment..."
echo "================================================"

# Check required files
echo ""
echo "📋 Checking required files..."

REQUIRED_FILES=(
    "dashboard/app.py"
    "requirements.txt"
    ".streamlit/config.toml"
    "data/processed/transformed_data.pkl"
    "models/evaluation_results.json"
    "models/metadata.json"
)

all_exists=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file - MISSING!"
        all_exists=false
    fi
done

if [ "$all_exists" = false ]; then
    echo ""
    echo "⚠️  Some required files are missing!"
    echo "Run: python run_pipeline.py"
    exit 1
fi

echo ""
echo "📊 Checking data files size..."
ls -lh data/processed/transformed_data.pkl
ls -lh models/evaluation_results.json
ls -lh models/metadata.json

echo ""
echo "🔍 Git status:"
git status --short

echo ""
echo "================================================"
echo "✅ All files ready for deployment!"
echo ""
echo "Next steps:"
echo "1. Commit and push all files:"
echo "   git add ."
echo "   git commit -m 'Prepare for Streamlit deployment'"
echo "   git push origin main"
echo ""
echo "2. Go to: https://share.streamlit.io"
echo "3. Deploy with:"
echo "   - Repository: shandy225-beep/ProjekPID"
echo "   - Branch: main"
echo "   - Main file: dashboard/app.py"
echo ""
echo "📚 See DEPLOY.md for detailed instructions"
echo "================================================"
