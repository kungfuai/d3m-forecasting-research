FROM python:3.7.9

WORKDIR /workspace

# Install pip packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

