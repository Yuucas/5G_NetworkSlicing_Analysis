# Getting Started Guide

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- PostgreSQL 15+ (if running without Docker)
- Redis 7+ (if running without Docker)
- CUDA-capable GPU (optional, for training)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd 5G_NetworkSlicing_Analysis
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### 3. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

### 4. Prepare Data

```bash
# Create data directories
make setup-data

# The dataset should be in: dataset/Quality of Service 5G.csv
# Preprocess data
python scripts/preprocess_data.py
```

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

Services will be available at:
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000
- Grafana: http://localhost:3000

### Option 2: Local Development

#### Start Infrastructure Services

```bash
# Start PostgreSQL
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=network_slicing \
  -p 5432:5432 \
  postgres:15-alpine

# Start Redis
docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine

# Start Kafka (optional for streaming)
# Use docker-compose for Kafka stack
```

#### Run Application

```bash
# Terminal 1: Start API
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Dashboard
streamlit run src/monitoring/dashboard.py

# Terminal 3: Start MLflow
mlflow server --backend-store-uri postgresql://postgres:postgres@localhost:5432/network_slicing \
  --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

## Training Models

### Train Reinforcement Learning Agent

```bash
# Train DQN agent
python scripts/train_rl_agent.py --algorithm dqn --episodes 10000

# Train PPO agent
python scripts/train_rl_agent.py --algorithm ppo --episodes 5000
```

### Train Deep Learning Models

```bash
# Train LSTM forecaster
python scripts/train_dl_model.py --model lstm --epochs 100

# Train Transformer model
python scripts/train_dl_model.py --model transformer --epochs 100
```

### Monitor Training

```bash
# View training progress in MLflow
# Open http://localhost:5000 in browser

# View TensorBoard (if enabled)
tensorboard --logdir=logs/tensorboard
```

## Using the API

### Test API Health

```bash
curl http://localhost:8000/health
```

### Request Resource Allocation

```bash
curl -X POST "http://localhost:8000/api/v1/allocate/request" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "User_123",
    "application_type": "Video_Call",
    "signal_strength_dbm": -75,
    "latency_ms": 30,
    "required_bandwidth_mbps": 10,
    "priority": 7
  }'
```

### Predict Resource Demand

```bash
curl -X POST "http://localhost:8000/api/v1/predict/resource_demand" \
  -H "Content-Type: application/json" \
  -d '{
    "network_states": [
      {
        "signal_strength_dbm": -75,
        "latency_ms": 30,
        "required_bandwidth_mbps": 10,
        "application_type": "Video_Call",
        "user_id": "User_123"
      }
    ],
    "model_type": "dqn"
  }'
```

### Get Metrics

```bash
# Current metrics
curl http://localhost:8000/api/v1/monitor/metrics

# Performance history
curl http://localhost:8000/api/v1/monitor/performance/history?hours=24

# SLA compliance
curl http://localhost:8000/api/v1/monitor/sla_compliance
```

## Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Generate coverage report
make coverage
```

## Development Workflow

### Code Formatting

```bash
# Format code with black and isort
make format

# Run linters
make lint
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Monitoring and Observability

### Access Dashboards

1. **Grafana** (http://localhost:3000)
   - Username: admin
   - Password: admin (change in production)
   - Pre-configured dashboards for network metrics

2. **MLflow** (http://localhost:5000)
   - Track experiments
   - Compare model performance
   - Model registry

3. **Streamlit Dashboard** (http://localhost:8501)
   - Real-time metrics
   - Interactive visualizations
   - Model performance monitoring

### View Logs

```bash
# API logs
docker-compose logs -f api

# Application logs (local)
tail -f logs/app.log

# All service logs
docker-compose logs -f
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check which process is using the port
   lsof -i :8000  # On Linux/Mac
   netstat -ano | findstr :8000  # On Windows
   ```

2. **Database connection error**
   ```bash
   # Verify PostgreSQL is running
   docker ps | grep postgres

   # Check connection
   psql -h localhost -U postgres -d network_slicing
   ```

3. **Model not found**
   ```bash
   # Ensure models are trained
   ls checkpoints/

   # Train models if needed
   make train
   ```

4. **Out of memory during training**
   - Reduce batch size in config/model_config.yaml
   - Use CPU instead of GPU for smaller models
   - Enable gradient checkpointing

### Getting Help

- Check logs: `docker-compose logs -f [service_name]`
- Review documentation in `docs/`
- Open an issue on GitHub
- Check FAQ: `docs/FAQ.md`

## Next Steps

- Read the [Architecture Guide](docs/architecture/ARCHITECTURE.md)
- Explore [API Documentation](http://localhost:8000/docs)
- Review [Model Training Guide](docs/MODEL_TRAINING.md)
- Check out [Deployment Guide](docs/DEPLOYMENT.md)
- Contribute: See [CONTRIBUTING.md](CONTRIBUTING.md)
