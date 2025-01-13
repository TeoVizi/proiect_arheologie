import torch
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import quantization

model_name = "distilbert-base-uncased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

context = """
Everyday I wake up at 10am in order to go to school and my name is Alex. I am a student and a worker
"""

question = "What are my roles?"

inputs = tokenizer(question, context, return_tensors="pt")

outputs = model(**inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1

answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])
print(question)
print(answer)

def quantize_weights(model):
    
    with open("weights.txt", "w") as f:
        for name, param in model.named_parameters():
            if param.requires_grad:  
                weight_array = param.data.cpu().numpy().astype(np.float32)

                quantized_data, scale, zero_point = quantization.quantize(weight_array)

                f.write(f"Layer: {name}\n")
                f.write(f"Quantized Data: {quantized_data}\n")
                f.write(f"Scale: {scale} | Zero Point: {zero_point}\n\n")

                print(f"Quantized {name} | Scale: {scale} | Zero Point: {zero_point}")

def quantize_activations_layer_by_layer(model, inputs, file_name):
   
    hooks = []

    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, tuple):  
                for idx, out in enumerate(output):  
                    if isinstance(out, torch.Tensor):
                        activation_array = out.cpu().detach().numpy().astype(np.float32)
                        quantized_data, scale, zero_point = quantization.quantize(activation_array)

                        with open(file_name, "a") as f:
                            f.write(f"Layer: {name} | Output Index: {idx}\n")
                            f.write(f"Quantized Activation: {quantized_data}\n")
                            f.write(f"Scale: {scale} | Zero Point: {zero_point}\n\n")

                        print(f"Layer {name} | Output Index: {idx} | Quantized activation | Scale: {scale} | Zero Point: {zero_point}")
            elif isinstance(output, torch.Tensor):  
                activation_array = output.cpu().detach().numpy().astype(np.float32)
                quantized_data, scale, zero_point = quantization.quantize(activation_array)

                with open(file_name, "a") as f:
                    f.write(f"Layer: {name}\n")
                    f.write(f"Quantized Activation: {quantized_data}\n")
                    f.write(f"Scale: {scale} | Zero Point: {zero_point}\n\n")

                print(f"Layer {name} | Quantized activation | Scale: {scale} | Zero Point: {zero_point}")
        return fn

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn(name)))

    model(**inputs)

    for hook in hooks:
        hook.remove()

quantize_weights(model)

quantize_activations_layer_by_layer(model, inputs, "activations.txt")
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])

print(question)
print(f"Answer: {answer}")
