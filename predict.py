
from transformers import BertTokenizer
from model import BertRegressor
import torch
import dateparser

model_path = "./priority_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertRegressor.from_pretrained(model_path)
model.eval()

user_boost_keywords = {
    "emir": ["submit", "final", "report", "project", "asap", "today"],
    "ayse": ["walk", "relax", "meditate"],
    "default": ["urgent", "now", "deadline"]
}

def extract_time_from_text(text):
    parsed = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
    if parsed:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return None

def predict_priority_and_time(text, user_id="default"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        output = model(**inputs)
        score = output["logits"].item()

    keywords = user_boost_keywords.get(user_id, user_boost_keywords["default"])
    if any(word in text.lower() for word in keywords):
        score += 0.05

    score = max(0.0, min(1.0, score))
    return {
        "priority": round(score, 2),
        "scheduled_time": extract_time_from_text(text)
    }

if __name__ == "__main__":
    print(predict_priority_and_time("Submit the final report today by 6pm", user_id="emir"))
