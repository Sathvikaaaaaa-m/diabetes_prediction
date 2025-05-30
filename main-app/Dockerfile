# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the app source code to the container
COPY . /app/

# Expose the port your app runs on
EXPOSE 5000

# Run the application with Gunicorn, binding to 0.0.0.0:5000
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000"]
