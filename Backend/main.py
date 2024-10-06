from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)


class TextRequest(BaseModel):
    text: str


@app.get('/')
def root():
    return {"message": "The API is working fine"}



@app.post('/sentiment')
async def analyze_sentiment(request: TextRequest):
    # Perform sentiment analysis using the model
    result = distilled_student_sentiment_classifier(request.text)

    # Initialize variables to track the maximum score and corresponding label
    max_label = ""
    max_score = 0.0

    # Iterate through the list of sentiment scores and modify labels if necessary
    for sentiment in result[0]:  # Assuming result is a list of lists like your example
        label = sentiment['label']
        score = sentiment['score']

        # Apply the logic to rename labels if score > 0.85
        if label == 'positive' and score > 0.85:
            label = 'Very Positive'
        elif label == 'negative' and score > 0.85:
            label = 'Very Negative'
        elif score < 0.6:
            label = 'Neutral'

        # Check if the current score is the highest
        if score > max_score:
            max_score = score
            max_label = label

    return {"label": max_label, "score": max_score}



