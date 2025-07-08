# 🚀 CI/CD Pipeline Guide

This document explains the comprehensive CI/CD pipeline for the Clinical Churn & CLV Prediction project.

## 📋 Overview

Our CI/CD pipeline consists of multiple workflows that ensure code quality, security, and reliable deployments:

### 🔄 Workflows

1. **Main CI/CD Pipeline** (`.github/workflows/ci.yml`)
   - Runs on every push/PR to `main` and `develop` branches
   - Tests, trains models, builds Docker images, and deploys

2. **Release Pipeline** (`.github/workflows/release.yml`)
   - Triggers when a GitHub release is published
   - Creates versioned Docker images and release notes

3. **Security Scan** (`.github/workflows/security.yml`)
   - Weekly automated security scans
   - Manual trigger available
   - Checks for vulnerabilities and outdated dependencies

4. **Deploy Pipeline** (`.github/workflows/deploy.yml`)
   - Manual deployment to staging/production
   - Includes rollback capabilities

## 🏗️ Pipeline Stages

### 1. **Test & Lint** (`test` job)
- **Code Quality**: Black formatting, Flake8 linting
- **Security**: Bandit security scan, Safety vulnerability check
- **Testing**: Unit tests with coverage reporting
- **Dependencies**: Cached pip installations

### 2. **Model Training** (`train-model` job)
- **Training**: Runs model training pipeline
- **Validation**: Ensures model files are generated
- **Artifacts**: Uploads trained models and MLflow runs

### 3. **Docker Build** (`build-docker` job)
- **Multi-stage builds**: API and Streamlit images
- **Registry**: Pushes to GitHub Container Registry
- **Caching**: Optimized build times with layer caching
- **Tagging**: Automatic version tagging

### 4. **Integration Tests** (`integration-test` job)
- **End-to-end testing**: Full service stack testing
- **API validation**: Health checks and prediction endpoints
- **Cleanup**: Proper resource cleanup

### 5. **Deployment** (`deploy-staging`, `deploy-production`)
- **Environment-specific**: Separate staging and production
- **Health checks**: Post-deployment validation
- **Notifications**: Success/failure alerts

## 🚀 How to Use

### Automatic Triggers

The pipeline runs automatically on:
- **Push to `main`**: Full pipeline including production deployment
- **Push to `develop`**: Full pipeline including staging deployment
- **Pull Requests**: Test and validation only
- **GitHub Releases**: Versioned releases with Docker images

### Manual Deployment

1. **Go to GitHub Actions** → **Deploy** workflow
2. **Click "Run workflow"**
3. **Select environment**: `staging` or `production`
4. **Enter version tag**: e.g., `v1.0.0`, `latest`
5. **Click "Run workflow"**

### Creating a Release

1. **Create a new release** on GitHub
2. **Tag version**: e.g., `v1.0.0`
3. **Publish release**
4. **Pipeline automatically**:
   - Builds versioned Docker images
   - Creates release notes
   - Tags containers with version

## 🔧 Configuration

### Environment Variables

Set these in GitHub repository settings:

```bash
# Required for Docker registry
GITHUB_TOKEN  # Automatically provided

# Optional for notifications
SLACK_WEBHOOK_URL  # For Slack notifications
EMAIL_SMTP_SERVER  # For email notifications
```

### Environment Protection

Configure environment protection rules in GitHub:

1. **Go to Settings** → **Environments**
2. **Create environments**: `staging`, `production`
3. **Add protection rules**:
   - Required reviewers
   - Wait timer
   - Deployment branches

### Secrets Management

Store sensitive data as GitHub secrets:

```bash
# Database connections
DATABASE_URL
REDIS_PASSWORD

# API keys
MLFLOW_TRACKING_URI
MODEL_REGISTRY_URI

# External services
SLACK_WEBHOOK_URL
EMAIL_CREDENTIALS
```

## 📊 Monitoring & Alerts

### Pipeline Metrics

- **Build times**: Tracked in GitHub Actions
- **Success rates**: Available in Actions tab
- **Coverage reports**: Uploaded to Codecov
- **Security reports**: Generated as artifacts

### Notifications

Configure notifications for:
- **Pipeline failures**: Immediate alerts
- **Security vulnerabilities**: Weekly reports
- **Deployment success**: Team notifications
- **Dependency updates**: Automated issues

## 🔒 Security Features

### Automated Scans

- **Code security**: Bandit static analysis
- **Dependency vulnerabilities**: Safety checks
- **Container scanning**: Built-in Docker security
- **Secrets detection**: Prevents credential leaks

### Best Practices

- **Least privilege**: Minimal permissions for jobs
- **Secret rotation**: Regular credential updates
- **Immutable tags**: Version-specific deployments
- **Rollback capability**: Quick recovery from failures

## 🛠️ Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check logs
   # Verify dependencies
   # Test locally first
   ```

2. **Deployment Issues**
   ```bash
   # Check environment variables
   # Verify Docker images
   # Test health endpoints
   ```

3. **Security Failures**
   ```bash
   # Review security reports
   # Update vulnerable dependencies
   # Fix code issues
   ```

### Debug Commands

```bash
# Test locally
docker-compose up --build

# Check logs
docker-compose logs

# Validate configuration
docker-compose config

# Run security scan locally
bandit -r .
safety check
```

## 📈 Performance Optimization

### Build Optimization

- **Multi-stage Docker builds**: Smaller images
- **Layer caching**: Faster rebuilds
- **Parallel jobs**: Concurrent execution
- **Dependency caching**: Reuse pip cache

### Deployment Optimization

- **Blue-green deployments**: Zero downtime
- **Health checks**: Fast failure detection
- **Resource limits**: Prevent resource exhaustion
- **Auto-scaling**: Handle traffic spikes

## 🔄 Rollback Strategy

### Automatic Rollback

- **Health check failures**: Automatic rollback
- **Deployment timeouts**: Rollback to previous version
- **Error thresholds**: Configurable failure limits

### Manual Rollback

1. **Go to Deploy workflow**
2. **Select previous version**
3. **Redeploy to environment**
4. **Verify rollback success**

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Security Scanning Tools](https://owasp.org/www-project-devsecops-guideline/)
- [MLOps Best Practices](https://mlops.community/)

## 🤝 Contributing

When contributing to the CI/CD pipeline:

1. **Test changes locally** first
2. **Update documentation** for new features
3. **Follow security best practices**
4. **Add appropriate tests**
5. **Review with team** before merging

---

**Need help?** Create an issue or contact the DevOps team! 