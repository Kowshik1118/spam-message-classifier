import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

df = pd.read_csv("spam.csv", encoding="latin-1")
if {"v1", "v2"}.issubset(df.columns):
    df = df.rename(columns={"v1": "label", "v2": "message"})[["label", "message"]]
df = df.dropna(); df["label"] = df["label"].str.lower().str.strip()
model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
model.fit(df["message"], df["label"])

def classify():
    text = input_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter a message."); return
    prediction = model.predict([text])[0]
    confidence = model.predict_proba([text]).max()
    result_var.set(f"Prediction: {prediction.upper()} | Confidence: {confidence:.2%}")

root = tk.Tk(); root.title("Spam Message Classifier"); root.geometry("650x420")
tk.Label(root, text="Spam Message Classifier", font=("Arial", 18, "bold")).pack(pady=10)
tk.Label(root, text="Enter message:").pack(anchor="w", padx=15)
input_box = tk.Text(root, height=8, font=("Arial", 12)); input_box.pack(fill="both", expand=True, padx=15, pady=5)
tk.Button(root, text="Classify Message", command=classify, font=("Arial", 12, "bold"), bg="#dc2626", fg="white").pack(pady=10)
result_var = tk.StringVar(); tk.Label(root, textvariable=result_var, font=("Arial", 14, "bold")).pack(pady=10)
root.mainloop()
