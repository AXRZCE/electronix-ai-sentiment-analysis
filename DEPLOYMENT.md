# 🚀 Deployment Guide

## Local Deployment

### Prerequisites
- Docker & Docker Compose
- 4GB+ RAM available
- 2GB+ disk space

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd electronix-ai-sentiment-analysis

# Start application
docker-compose up --build

# Access application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Development Mode
```bash
# Backend
pip install -r requirements.txt
cd backend && python main.py

# Frontend (new terminal)
cd frontend && npm install && npm run dev
# Access: http://localhost:5173
```

## Cloud Deployment Options

### 1. Railway (Recommended for Backend)

**Steps:**
1. Fork this repository
2. Connect Railway to your GitHub
3. Deploy backend service:
   ```bash
   # Railway will auto-detect Dockerfile
   # Set environment variables if needed
   ```
4. Update frontend API URL to Railway backend URL

**Pros:**
- Easy Docker deployment
- Automatic HTTPS
- Good for ML models

**Cons:**
- Paid service required for production
- Cold starts possible

### 2. Render

**Backend Deployment:**
1. Connect GitHub repository
2. Choose "Web Service"
3. Set build command: `docker build -f backend/Dockerfile .`
4. Set start command: `python backend/main.py`

**Frontend Deployment:**
1. Choose "Static Site"
2. Build command: `cd frontend && npm install && npm run build`
3. Publish directory: `frontend/dist`

### 3. Vercel (Frontend Only)

**Steps:**
```bash
cd frontend
npm install -g vercel
vercel --prod
```

**Configuration (vercel.json):**
```json
{
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

### 4. Heroku

**Backend (Dockerfile deployment):**
```bash
# Install Heroku CLI
heroku create your-app-name
heroku container:push web -a your-app-name
heroku container:release web -a your-app-name
```

**Frontend:**
```bash
# Build pack deployment
heroku create your-frontend-app
git subtree push --prefix frontend heroku main
```

## Production Considerations

### Environment Variables
```bash
# Backend
PYTHONUNBUFFERED=1
MODEL_PATH=/app/model
QUANTIZED_MODEL=false

# Frontend
VITE_API_URL=https://your-backend-url.com
```

### Model Optimization
```bash
# Quantize model for faster inference
python quantize_model.py --model_path ./model --output_dir ./model_quantized

# Use quantized model
QUANTIZED_MODEL=true docker-compose up
```

### Security Checklist
- [ ] Remove debug mode in production
- [ ] Set proper CORS origins
- [ ] Add rate limiting
- [ ] Use HTTPS
- [ ] Secure API keys (if added)
- [ ] Monitor resource usage

### Scaling Options
1. **Horizontal Scaling**: Multiple backend instances
2. **Load Balancing**: Nginx/HAProxy
3. **Caching**: Redis for frequent predictions
4. **CDN**: For frontend static assets

## Monitoring

### Health Checks
```bash
# Backend health
curl https://your-backend-url.com/health

# Expected response
{
  "status": "healthy",
  "model_loaded": true
}
```

### Logging
```bash
# Docker logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Application logs
tail -f backend/logs/app.log
```

### Metrics to Monitor
- Response time
- Memory usage
- CPU usage
- Error rate
- Request volume

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Check model files exist
ls -la model/
# Should contain: config.json, model.safetensors, tokenizer files
```

**2. Memory Issues**
```bash
# Reduce batch size in fine-tuning
python finetune.py --batch_size 4

# Use quantized model
QUANTIZED_MODEL=true docker-compose up
```

**3. CORS Errors**
```bash
# Update backend CORS settings
# In backend/main.py, add your frontend URL to allow_origins
```

**4. Port Conflicts**
```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Backend
  - "3001:3000"  # Frontend
```

### Performance Optimization

**Backend:**
- Use quantized models
- Enable model caching
- Optimize batch processing
- Use GPU if available

**Frontend:**
- Enable gzip compression
- Use CDN for assets
- Implement lazy loading
- Optimize bundle size

**Database (if added):**
- Connection pooling
- Query optimization
- Indexing
- Caching layer

## Cost Estimation

### Free Tier Options
- **Render**: 750 hours/month free
- **Railway**: $5/month starter
- **Vercel**: Generous free tier for frontend
- **Heroku**: Limited free tier (deprecated)

### Paid Recommendations
- **Small Scale**: Railway ($5-20/month)
- **Medium Scale**: AWS/GCP ($50-200/month)
- **Large Scale**: Kubernetes cluster ($200+/month)

## Backup & Recovery

### Model Backup
```bash
# Backup fine-tuned models
tar -czf model-backup-$(date +%Y%m%d).tar.gz model/
```

### Database Backup (if applicable)
```bash
# PostgreSQL example
pg_dump database_name > backup.sql
```

### Disaster Recovery
1. Keep model files in cloud storage
2. Automate deployments with CI/CD
3. Monitor service health
4. Have rollback procedures ready
