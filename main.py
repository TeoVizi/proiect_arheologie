from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

text = "This is a sample sentence."
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)
