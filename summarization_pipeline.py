
from transformers import pipeline
import csv
from transformers import PegasusTokenizer


model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model =model_name, tokenizer=tokenizer)

with open('topic_2.csv', newline='') as f:
    reader = csv.reader(f)
    docs_raw = list(reader)

summary = summarizer(docs_raw, min_length=10, max_length=50)
print(summary)