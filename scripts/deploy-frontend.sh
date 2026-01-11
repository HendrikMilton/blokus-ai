#!/bin/bash
set -e

# Build frontend
cd frontend && bun run build && cd ..

# Get Terraform outputs
cd infrastructure
S3_BUCKET=$(terraform output -raw s3_bucket_name)
CF_DISTRIBUTION=$(terraform output -raw cloudfront_distribution_id)
cd ..

# Sync to S3
aws s3 sync frontend/dist s3://$S3_BUCKET --delete

# Invalidate CloudFront cache
aws cloudfront create-invalidation --distribution-id $CF_DISTRIBUTION --paths "/*"

echo "Deployed! URL: $(cd infrastructure && terraform output -raw website_url)"
