#!/bin/bash
# Voice Emotion Detector - AWS Deployment Script
# Deploys the application to AWS using CDK

set -e

echo "=========================================="
echo "Voice Emotion Detector - AWS Deployment"
echo "=========================================="

# Check for AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured!"
    echo "Please run: aws configure"
    exit 1
fi

echo "✅ AWS credentials found"
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-ap-southeast-2}
echo "   Account: $AWS_ACCOUNT"
echo "   Region: $AWS_REGION"

# Navigate to CDK directory
cd "$(dirname "$0")/cdk"

# Install CDK dependencies if needed
if [ ! -d "node_modules" ]; then
    echo ""
    echo "📦 Installing CDK dependencies..."
    npm install
fi

# Build TypeScript
echo ""
echo "🔨 Building CDK..."
npm run build

# Bootstrap CDK if needed (first time only)
echo ""
echo "🚀 Checking CDK bootstrap..."
npx cdk bootstrap aws://$AWS_ACCOUNT/$AWS_REGION 2>/dev/null || true

# Deploy
echo ""
echo "☁️ Deploying to AWS..."
npx cdk deploy --require-approval never

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Note the Elastic IP from the output above"
echo "2. Wait 5-10 minutes for the app to start"
echo "3. Access your app at http://<elastic-ip>"
echo ""
echo "To check logs on the EC2 instance:"
echo "  ssh -i <your-key.pem> ec2-user@<elastic-ip>"
echo "  sudo journalctl -u voice-emotion -f"
echo ""
echo "To destroy the stack:"
echo "  cd cdk && npm run destroy"
