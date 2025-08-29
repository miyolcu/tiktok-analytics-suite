FROM python:3.9-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Install package
COPY setup.py .
COPY README.md .
RUN pip install -e .

EXPOSE 8501

# Default command
CMD ["streamlit", "run", "src/dashboard/web_dashboard.py"]
