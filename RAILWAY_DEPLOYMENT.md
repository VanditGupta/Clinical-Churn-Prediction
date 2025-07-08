# 🚂 Railway Deployment Guide

## Quick Start (5 minutes)

### Step 1: Sign Up
1. Go to [railway.app](https://railway.app)
2. Click "Sign Up" and choose "Continue with GitHub"
3. Authorize Railway to access your GitHub account

### Step 2: Deploy Your App
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Find and select your `clinical_churn_clv` repository
4. Click "Deploy Now"

### Step 3: Configure Environment Variables
Railway will automatically detect your `docker-compose.yml` and deploy all services. You may need to set these environment variables:

```bash
# In Railway Dashboard → Your Project → Variables tab:

# API Service
REDIS_HOST=redis
MLFLOW_TRACKING_URI=file:./mlruns
MODEL_PATH=models/churn_model.pkl

# Streamlit App  
API_BASE_URL=https://your-api-service-url.railway.app
```

### Step 4: Access Your App
Railway will provide URLs for each service:
- **API**: `https://your-api-service.railway.app`
- **Streamlit**: `https://your-streamlit-service.railway.app`
- **MLflow**: `https://your-mlflow-service.railway.app`

---

## 🔧 Configuration Details

### What Railway Does Automatically:
- ✅ Detects `docker-compose.yml`
- ✅ Builds Docker images
- ✅ Creates services for each container
- ✅ Handles networking between services
- ✅ Provides HTTPS endpoints
- ✅ Manages environment variables

### Service Architecture:
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │     Redis       │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│   (Cache)       │
│   Port: 8501    │    │   Port: 8000    │    │   Port: 6379    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 📊 Monitoring & Logs

### View Logs:
1. Go to Railway Dashboard
2. Select your project
3. Click on any service
4. Go to "Logs" tab

### Health Checks:
- API health endpoint: `GET /health`
- Automatic monitoring by Railway
- Email notifications for failures

### Metrics:
- Request count
- Response times
- Error rates
- Resource usage

---

## 🔄 Automatic Deployments

### Enable Auto-Deploy:
1. In Railway Dashboard → Settings
2. Enable "Deploy on Push"
3. Select branch (usually `main`)

### Deployment Triggers:
- Push to main branch
- Manual deployment from dashboard
- GitHub Actions integration

---

## 🚨 Troubleshooting

### Common Issues:

#### 1. Redis Connection Failed
```bash
# Check environment variables
REDIS_HOST=redis  # Should be the service name, not localhost
```

#### 2. Model Not Found
```bash
# Ensure model file is committed to Git
git add models/churn_model.pkl
git commit -m "Add trained model"
git push
```

#### 3. Port Conflicts
```bash
# Railway automatically assigns ports
# Use $PORT environment variable in your code
```

#### 4. Build Failures
```bash
# Check Dockerfile syntax
# Ensure all dependencies are in requirements.txt
# Verify file paths are correct
```

### Debug Commands:
```bash
# View service logs
railway logs

# Check service status
railway status

# Restart services
railway restart
```

---

## 💰 Cost Management

### Free Tier Limits:
- **$5 credit** per month
- **Shared resources** (CPU/RAM)
- **Automatic scaling** based on usage

### Cost Optimization:
1. **Monitor usage** in Railway Dashboard
2. **Scale down** during development
3. **Use sleep mode** for non-critical services
4. **Optimize Docker images** for faster builds

### Upgrade When Needed:
- **Pro plan**: $20/month for dedicated resources
- **Team plan**: $20/user/month for collaboration
- **Enterprise**: Custom pricing for large teams

---

## 🔐 Security Best Practices

### Environment Variables:
- ✅ Store secrets in Railway Variables
- ❌ Never commit API keys to Git
- ✅ Use different values for dev/prod

### Network Security:
- ✅ HTTPS enabled by default
- ✅ Automatic SSL certificates
- ✅ Isolated service networks

### Access Control:
- ✅ GitHub OAuth integration
- ✅ Team member management
- ✅ Service-specific permissions

---

## 📱 Testing Your Deployment

### Health Check:
```bash
curl https://your-api-service.railway.app/health
```

### Prediction Test:
```bash
curl -X POST https://your-api-service.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": "Male", 
    "income": 60000,
    "location": "Urban",
    "study_type": "Phase II",
    "condition": "Diabetes",
    "visit_adherence_rate": 0.7,
    "tenure_months": 12,
    "last_visit_gap_days": 15,
    "num_medications": 3,
    "has_side_effects": false,
    "transport_support": true,
    "monthly_stipend": 400,
    "contact_frequency": 3.0,
    "support_group_member": false,
    "language_barrier": false,
    "device_usage_compliance": 0.6,
    "survey_score_avg": 7.0
  }'
```

### Streamlit App:
- Open your Streamlit URL in browser
- Test the dashboard functionality
- Verify data visualization

---

## 🎉 Success Checklist

- [ ] All services deployed successfully
- [ ] Health checks passing
- [ ] API endpoints responding
- [ ] Streamlit dashboard accessible
- [ ] Redis connection working
- [ ] Model predictions working
- [ ] Environment variables set
- [ ] Auto-deploy enabled
- [ ] Monitoring configured
- [ ] Team access granted (if needed)

---

## 🆘 Support

### Railway Support:
- **Documentation**: [docs.railway.app](https://docs.railway.app)
- **Discord**: [discord.gg/railway](https://discord.gg/railway)
- **Email**: support@railway.app

### Project Issues:
- Check logs in Railway Dashboard
- Verify environment variables
- Test locally with `docker compose up`
- Review this deployment guide

Happy deploying! 🚂✨ 