# 🚀 Trump Tweet Prediction Model - Deployment Guide

## Quick Start

### Option 1: Use From GitHub (Simple)

```bash
# Clone the repo
git clone https://github.com/yourusername/trump-tweets-combined.git
cd trump-tweets-combined

# Install dependencies
pip install -r requirements.txt

# Generate predictions
python3 generate_llama.py --headlines "Trump imposes new tariffs" "Markets rally"
```

---

## Model Types & How to Use

### 1️⃣ **Fine-tuned Llama 3** (Best Quality)

**What it does:** Generates authentic Trump-style tweets using a neural network fine-tuned on 86,000+ Trump tweets.

**Use it:**
```python
from generate_llama import generate_tweets

headlines = [
    "Trump imposes new tariffs on China",
    "Stock market hits record high",
    "NATO expands membership"
]

tweets = generate_tweets(headlines, num_tweets=5)
for tweet in tweets:
    print(tweet)
```

**Requirements:**
- PyTorch with CUDA (GPU recommended)
- transformers library
- 15GB+ disk space for model

**Command line:**
```bash
python3 generate_llama.py \
    --headlines "Trump tariff announcement" "Markets react" \
    --n 5 \
    --temp 0.85
```

---

### 2️⃣ **RAG (Retrieval-Augmented Generation)** (Most Accurate)

**What it does:** Retrieves similar Trump tweets from training data and generates variations using Llama.

**Use it:**
```python
from generate_rag import generate_with_rag

headlines = [
    "China hikes tariffs on US goods",
    "Fed cuts interest rates"
]

tweets = generate_with_rag(headlines, k=5, num_tweets=3)
```

**Why it's best:**
- ✅ Most accurate predictions
- ✅ Grounded in actual Trump tweets
- ✅ Authentic voice & style
- ✅ Lower hallucination rate

**Files needed:**
- `rag_index.faiss` (85 MB) - Vector index
- `rag_data.jsonl` (44 MB) - Training examples
- `trump_llama_v3` (15 GB) - Model weights

**Setup:**
```bash
# Download models (included in repo)
# Or build from scratch:
python3 scripts/build_rag_index.py --data finetune_data.jsonl
```

---

### 3️⃣ **Markov Chain** (Fast, No GPU)

**What it does:** Statistical word chain generation - fast but basic.

**Use it:**
```python
from markov_predictor import MarkovTweetGenerator

gen = MarkovTweetGenerator(order=2)
tweets = gen.generate_with_topic(
    keywords=["tariff", "China", "trade"],
    num_tweets=5
)
```

**Pros:**
- ✅ No GPU needed
- ✅ Very fast
- ✅ No dependencies

**Cons:**
- ❌ Lower quality
- ❌ Less authentic
- ❌ Grammar issues

---

### 4️⃣ **N-gram Model** (Fast, No GPU)

**What it does:** Probabilistic language model using n-grams.

**Use it:**
```python
from ngram_predictor import NgramTweetGenerator

gen = NgramTweetGenerator(n=3)
tweets = gen.generate_with_keywords(
    keywords=["economy", "jobs", "growth"],
    num_tweets=5
)
```

**Similar to Markov** - good for quick predictions without GPU.

---

## Integration Examples

### Web API (Flask)

```python
from flask import Flask, request, jsonify
from generate_rag import generate_with_rag

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    headlines = data.get('headlines', [])
    
    tweets = generate_with_rag(headlines, num_tweets=3)
    
    return jsonify({
        'headlines': headlines,
        'predictions': tweets
    })

if __name__ == '__main__':
    app.run(port=5000)
```

**Usage:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"headlines": ["Trump announces new policy", "Markets rally"]}'
```

---

### Python Package

```python
from trump_tweets import TrumpTweetPredictor

# Initialize (loads model once)
predictor = TrumpTweetPredictor(model_type='rag')

# Generate predictions
headlines = ["Breaking: Trump meets Xi Jinping"]
tweets = predictor.predict(headlines, num_tweets=5)

# Batch processing
news_articles = [
    ["Tariffs rise on China imports"],
    ["Stock market hits record high"],
    ["Fed announces rate cut"]
]

all_predictions = predictor.batch_predict(news_articles)
```

---

### Docker Container

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY trump_llama_v3 trump_llama_v3/
COPY rag_index.faiss .
COPY rag_data.jsonl .
COPY generate_rag.py .

EXPOSE 5000

CMD ["python3", "-m", "flask", "run", "--host", "0.0.0.0"]
```

**Build & run:**
```bash
docker build -t trump-tweet-predictor .
docker run --gpus all -p 5000:8000 trump-tweet-predictor
```

---

### AWS Lambda

**Requirements:**
- Model files in S3
- Lambda with GPU support (G4 instances)
- ~15 min cold start

```python
import json
import boto3
from generate_rag import generate_with_rag

s3 = boto3.client('s3')

def lambda_handler(event, context):
    headlines = json.loads(event['body'])['headlines']
    
    tweets = generate_with_rag(headlines, num_tweets=3)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'tweets': tweets})
    }
```

---

## Installation by Use Case

### 🏃 **Quick Demo (No GPU)**
```bash
pip install -r requirements-minimal.txt
python3 generate_markov.py
```

### 💻 **Local Development (With GPU)**
```bash
pip install -r requirements-full.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 generate_rag.py --headlines "..."
```

### 🚀 **Production (GPU Server)**
```bash
# Ubuntu 22.04 + NVIDIA GPU
bash install-prod.sh

# Start prediction service
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### 🌐 **Cloud (AWS/GCP/Azure)**
```bash
# See cloud-specific guides in docs/
./deploy-to-aws.sh
./deploy-to-gcp.sh
./deploy-to-azure.sh
```

---

## Model Performance

| Model | Quality | Speed | GPU Needed | Size |
|-------|---------|-------|-----------|------|
| **RAG + Llama** | ⭐⭐⭐⭐⭐ | Slow | ✅ Yes | 15GB |
| **Fine-tuned Llama** | ⭐⭐⭐⭐ | Slow | ✅ Yes | 15GB |
| **N-gram** | ⭐⭐ | Fast | ❌ No | 50MB |
| **Markov** | ⭐⭐ | Fast | ❌ No | 30MB |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or use CPU-only (slower)
pip install torch torchvision torchaudio
```

### "CUDA out of memory"
```bash
# Use a smaller model or batch size
python3 generate_rag.py --batch-size 1 --max-length 128
```

### "Model not found"
```bash
# Download model weights
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
# Or use pretrained
python3 scripts/download-models.py
```

---

## API Reference

### `generate_with_rag(headlines, k=5, num_tweets=3, temperature=0.85)`

**Parameters:**
- `headlines` (list): News headlines to generate tweets from
- `k` (int): Number of similar examples to retrieve (default: 5)
- `num_tweets` (int): Number of tweets to generate (default: 3)
- `temperature` (float): Creativity level 0-1 (default: 0.85)

**Returns:**
- `list`: Generated tweet strings

---

## Contributing

Want to improve the model? 

```bash
# 1. Add new training data
python3 scripts/add_training_data.py your_data.json

# 2. Fine-tune
python3 scripts/finetune_llama.py --epochs 3 --lr 1e-5

# 3. Test
python3 -m pytest tests/test_predictions.py

# 4. Submit PR
git commit -am "Improve model accuracy"
git push origin feature-branch
```

---

## License

MIT - Feel free to use in commercial projects

---

## Support

- 📧 Email: support@example.com
- 💬 Issues: https://github.com/yourrepo/issues
- 📖 Docs: https://docs.example.com

