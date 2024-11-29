# Start from the official PyTorch image with CUDA (or CPU if preferred)
FROM pytorch/pytorch

# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libpq-dev libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt /workspace

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of your local code into the container
COPY . /workspace

# Default command to run when the container starts
CMD ["python", "main.py"]
