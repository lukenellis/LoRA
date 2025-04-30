import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Load your trained LoRA model ===
# Replace this with your output_dir (where model was saved after training)
model_dir = "../text-classification/model"  # or wherever your model is saved
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# === Example sentence for CoLA (acceptability judgment) ===
sentence = "The ship sank beneath the waves."

# === 1. Preprocessing ===
start_pre = time.time()
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
end_pre = time.time()

# === 2. Model Inference ===
start_inf = time.time()
with torch.no_grad():
    outputs = model(**inputs)
end_inf = time.time()

# === 3. Postprocessing ===
start_post = time.time()
predicted_class = torch.argmax(outputs.logits, dim=-1).item()
end_post = time.time()

# === Print predictions and time ===
print(f"\nPrediction: {predicted_class} (1 = acceptable, 0 = not acceptable)")
print(f"Preprocessing time: {end_pre - start_pre:.4f} sec")
print(f"Inference time:     {end_inf - start_inf:.4f} sec")
print(f"Postprocessing time:{end_post - start_post:.4f} sec\n")

# === Plotting the breakdown ===
times = [end_pre - start_pre, end_inf - start_inf, end_post - start_post]
labels = ['Preprocessing', 'Inference', 'Postprocessing']

plt.bar(labels, times)
plt.ylabel('Seconds')
plt.title('Inference Pipeline Timing')
plt.show()
