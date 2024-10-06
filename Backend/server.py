from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
app = FastAPI()

class TextRequest(BaseModel):
    text: str


@app.get('/')
def root():
    return {"message": "The API is working fine"}


@app.post('/sentiment')
def analysis(request: TextRequest):
    score = sentiment_analysis(request.text)
    print(score[0]['label'], score[0]['score'])
    
    if score[0]['label'] == 'NEGATIVE' and score[0]['score'] > 0.85:
        score[0]['label'] = 'Very Negative'
    
    elif score[0]['label'] == 'POSITIVE' and score[0]['score'] > 0.85:
        score[0]['label'] = 'Very Positive'

    return {"message": score}