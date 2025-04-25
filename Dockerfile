FROM python:3.10-slim
ENV LANG C.UTF-8

# Set working directory
WORKDIR /app

# Copy all code and data into the container
COPY . /app

# Install Python dependencies (CPU-only)
RUN pip install --no-cache-dir \
  streamlit \
  torch \
  numpy \
  pandas \
  chromadb \
  scikit-learn

# working dir to src to resolve "../" in paths
WORKDIR /app/src

# expose port
EXPOSE 8501

# default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]