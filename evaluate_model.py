# evaluate_model.py:Kaydedilen modeli açar, test verisinde özet üretir, kaliteyi ROUGE ile ölçer

import re
import nltk
import evaluate
from nltk.corpus import stopwords
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
#set: Listeyi kümeye çevirir. Çünkü kümeler daha hızlı arama sağlar.


# 🔄 KAYDEDİLMİŞ MODELİ YÜKLE
model = AutoModelForSeq2SeqLM.from_pretrained("./saved_model")
tokenizer = AutoTokenizer.from_pretrained("./saved_model")
#train and save'de eğitilen model ve tokenizer ./saved_model klasöründen geri yüklenir.

# 🔽 VERİ
print("📥 Test verisi yükleniyor...")
dataset = load_dataset("cnn_dailymail", "3.0.0")
test_data = dataset["test"].select(range(50))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # linkleri kaldırdık
    text = re.sub(r"@\S+", "", text)     # mentionları kaldırdık
    text = re.sub(r"#\S+", "", text)     # hashtagi kaldırdık
    text = re.sub(r"\d+", "", text)      # sayıları kaldırdık
    text = re.sub(r"[^\w\s]", "", text)  # noktalama işaretlerinikaldırd   ık
    return text

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopwords])

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

def onisleme(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text.strip()

def augment_shuffle(text):
    words = text.split()
    random.shuffle(words)
    return " ".join(words)

def apply_augmentation(text):
    methods = [augment_shuffle]
    return random.choice(methods)(text)

# 🔄 ÖN İŞLEME
def preprocess_batch(batch):
    inputs = ["summarize: " + clean_text(doc) for doc in batch["article"]] 
    #Her test örneğinin başına "summarize: " eklenerek inputları hazırladık.
    targets = [clean_text(summary) for summary in batch["highlights"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    #Tokenizer ile giriş ve hedef (labels) verileri dönüştürülür.
    return model_inputs

print("🔄 Test verisi işleniyor...")
tokenized_test = test_data.map(preprocess_batch, batched=True)
#Tüm test verisine bu ön işlemeyi uyguladk sırasıyla
"""------------------------------------------------------------------------------------"""
# 🔎 ROUGE Hesaplama
print("📊 ROUGE skoru hesaplanıyor...")
rouge = evaluate.load("rouge")
#ROUGE metriği yükledik (otomatik özetlemeyi değerlendirmek için).

inputs = ["summarize: " + clean_text(x) for x in test_data["article"]]
input_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
#Modelin anlayacağı şekilde tüm test verisi encode edilir.

outputs = model.generate(input_ids, max_length=128, num_beams=4)
#Model her haber için özet üretir (generate() ile).

decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
decoded_labels = [clean_text(x) for x in test_data["highlights"][:len(decoded_preds)]]
#Üretilen özetler ve gerçek özetler decode edilir ve karşılaştırmaya hazır hale gelir.

rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
print("✅ ROUGE Skorları:", rouge_scores)

# 🔍 5 Örnek Göster
print("\n📌 Örnek Çıktılar:")
for i in range(5):
    print(f"\n🔹 Haber {i+1}")
    print("📘 Haber (ilk 300 karakter):", test_data[i]["article"][:300])
    print("✅ Gerçek Özet:", test_data[i]["highlights"])
    print("🧠 Model Özeti :", decoded_preds[i])
"""
İlk 5 haber için:

Haber metni (kısaltılmış)
gerçek özet
Modelin ürettiği özet yazdırılır

"""