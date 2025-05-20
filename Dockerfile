FROM python:3.10-slim

# Set environment variables
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy necessary files
COPY bank_campaign_model_training.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the training script
CMD ["python", "bank_campaign_model_training.py"]
