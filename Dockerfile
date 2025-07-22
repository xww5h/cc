# Using python:3.12-slim as it's a recent stable version with a small footprint.
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install system dependencies (build tools for the C compiler, wget for downloading),
# upgrade pip, install Python packages, and finally remove the build tools to keep the image lean.
# build-essential provides gcc, which is required by llama-cpp-python.
# ca-certificates is added for robust network access.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip cmake && \
    pip install --no-cache-dir -r requirements.txt

RUN apt-get purge -y --auto-remove build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Copy the application code into the container
COPY app.py .

# Create a directory for the models, download the model, and clean up.
# This will make the Docker image large, but self-contained.
# NOTE: As requested, this uses the Qwen3 model. Please ensure this URL is valid, as the build will fail if wget cannot download the file.
RUN mkdir -p /app/models && \
    wget -O /app/models/Qwen3-8B-Q8_0.gguf "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf?download=true"

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define an environment variable for the model path inside the container.
# The model is now baked into the image.
ENV MODEL_PATH=/app/models/Qwen3-8B-Q8_0.gguf

# Run app.py when the container launches
CMD ["python", "app.py"]