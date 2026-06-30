#!/bin/bash
# Deploy script for the application

set -euo pipefail

APP_NAME="myapp"
DEPLOY_DIR="/opt/${APP_NAME}"
LOG_FILE="/var/log/${APP_NAME}/deploy.log"

function check_prerequisites() {
    command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { echo "kubectl required"; exit 1; }
}

function build_image() {
    echo "Building Docker image..."
    docker build -t "${APP_NAME}:latest" .
}

function deploy_to_k8s() {
    echo "Deploying to Kubernetes..."
    kubectl apply -f k8s/
    kubectl rollout status deployment/"${APP_NAME}"
}

function run_migrations() {
    echo "Running database migrations..."
    kubectl exec deploy/"${APP_NAME}" -- python manage.py migrate
}

# Main
check_prerequisites
build_image
deploy_to_k8s
run_migrations

echo "Deploy complete!"
