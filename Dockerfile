# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./src ./src

# copy trained model
COPY ./models ./models

# ✅ Add src to PYTHONPATH so that 'data_processing' is discoverable
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
# ✅ Add models to PYTHONPATH so that 'rfm_clustering_model.pkl' is discoverable
# ENV PYTHONPATH="${PYTHONPATH}:/app/models"
# Expose port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
