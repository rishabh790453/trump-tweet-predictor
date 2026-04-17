FROM python:3.11-slim

WORKDIR /app

# Install dashboard deps
COPY dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt anthropic

# Copy dashboard code
COPY dashboard/ ./dashboard/

# Copy data files needed at runtime
COPY trump_tweets_finance_with_market_data.csv ./

# Working directory is dashboard (app.py lives there)
WORKDIR /app/dashboard

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
