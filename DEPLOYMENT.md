# Deployment Guide for Agentic Research Assistant

## Overview
This guide covers deployment options for the Agentic Research Assistant application.

## Prerequisites
- Python 3.8+
- API keys for OpenRouter and Tavily
- Git (for version control)

## Local Development

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd agentic-research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Start the Application
```bash
# Start backend server
uvicorn backend.main:app --reload

# In a new terminal, start frontend
streamlit run frontend/app.py
```

### 3. Access the Application
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Production Deployment

### Backend Deployment (Render.com)
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables from your .env file
6. Deploy

### Frontend Deployment (Streamlit Cloud)
1. Push your code to GitHub
2. Go to streamlit.io/cloud
3. Connect your GitHub repository
4. Select the frontend/app.py file
5. Add environment variables (secrets) for API keys
6. Deploy

### Alternative: Docker Deployment
Create a Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY backend/requirements.txt .
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
                  -r backend/requirements.txt \
                  -r frontend/requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
```

Build and run:
```bash
docker build -t agentic-research-assistant .
docker run -p 8000:8000 -p 8501:8501 --env-file .env agentic-research-assistant
```

## Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| OPENROUTER_API_KEY | API key for OpenRouter (LLM access) | Yes |
| TAVILY_API_KEY | API key for Tavily Search | Yes |
| MODEL_NAME | LLM model to use (deepseek-chat/deepseek-reasoner) | No, defaults to deepseek-chat |
| EMBEDDING_MODEL | Embedding model name | No, defaults to all-MiniLM-L6-v2 |
| CHROMA_DB_PATH | Path to ChromaDB database | No, defaults to ./chroma_db |
| MEMORY_DB_PATH | Path to memory database | No, defaults to ./memory_db |
| BACKEND_HOST | Backend host | No, defaults to 0.0.0.0 |
| BACKEND_PORT | Backend port | No, defaults to 8000 |
| FRONTEND_PORT | Frontend port | No, defaults to 8501 |

## Data Persistence
- ChromaDB: Stores document embeddings (in ./chroma_db by default)
- Memory: Stores chat history and user preferences (in ./memory_db by default)
- Exports: Stores generated reports (in ./exports by default)
- Uploads: Stores uploaded documents (in ./uploads by default)

To persist data in production, mount volumes or use cloud storage solutions.

## Scaling Considerations
- The application is designed for single-instance deployment
- For high availability, consider using a load balancer with multiple instances
- ChromaDB can be scaled to a clustered deployment
- Memory storage could be moved to Redis or a database for shared state

## Monitoring and Logging
- Backend logs: Standard output from Uvicorn
- Frontend logs: Streamlit logging
- Consider adding structured logging and monitoring in production

## Security Considerations
- Keep API keys secure and never commit them to version control
- Use HTTPS in production
- Implement rate limiting for API endpoints
- Regularly update dependencies
- Consider adding authentication for multi-user scenarios

## Troubleshooting
1. **API Connection Issues**: Verify API keys and network connectivity
2. **ChromaDB Errors**: Ensure sufficient disk space and permissions
3. **Memory Issues**: Check available RAM, especially with large document sets
4. **Slow Responses**: Consider using faster models or optimizing chunk sizes
5. **Export Failures**: Verify reportlab and docx installations

## Support
For issues and questions, please refer to the README.md file or open an issue on GitHub.