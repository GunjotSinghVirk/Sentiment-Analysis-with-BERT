import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Load pre-trained model and tokenizer
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(text):
    # Encode the text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    # Get the input IDs and attention mask from the encoding
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get the predicted class
    predictions = softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(predictions).item()
    
    # The model predicts sentiment on a scale of 1 to 5
    return predicted_class + 1

# Example usage
text = "I really enjoyed this movie. The acting was superb and the plot was engaging."
sentiment = predict_sentiment(text)
print(f"Sentiment: {sentiment}/5")
