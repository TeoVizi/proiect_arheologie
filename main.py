from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

model_name = "distilbert-base-uncased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

context = """
Everyday I wake up at 10am in order to go to school and my name is Alex.
"""

question = "What is my name?"

inputs = tokenizer(question, context, return_tensors="pt")

outputs = model(**inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1

answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])
print(question)
print(answer)
