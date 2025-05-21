# Use an official Python runtime as the base image.
FROM python:3.10-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements.txt file first to leverage Docker caching.
COPY requirements.txt .

# Install Python dependencies.
# Explicitly install numpy and scikit-learn first with upgrade and force-reinstall
# to mitigate potential binary incompatibility issues, then install the rest.
RUN pip install --no-cache-dir --upgrade --force-reinstall numpy scikit-learn && \
    pip install --no-cache-dir -r requirements.txt

# Copy all your application code (including training script and test files)
# from the current build context (airflow-model-training-repo) into the container.
COPY . .

# No CMD needed here, as Cloud Build steps will explicitly define what to run.
# If you were deploying this image to a service like Cloud Run, you would add a CMD here
# to start your application (e.g., CMD ["python", "bank_campaign_model_training.py"]
# or CMD ["gunicorn", "..."] if it were a web app).
