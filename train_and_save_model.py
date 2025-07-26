#train and save model:	Veriyi temizler, modeli eğitir, sonucu ./saved_model klasörüne kaydeder
# VERİ YÜKLEME
import re
import random
import nltk
import evaluate
import spacy
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer


dataset = load_dataset("cnn_dailymail", "3.0.0")  

nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("english")
nlp = spacy.load("en_core_web_sm")  # Lemmatization için
#spaCy kütüphanesinin küçük İngilizce modelini lemmatizasyon için yükledik.



# ver temizleme fonkları

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
    #Kelimeleri köklerine çevirdik

def onisleme(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text.strip()
#Yukarıdaki üç fonku sırayla uygular ve temiz metni output olarak verir.


def apply_augmentation(text):
    methods = [augment_shuffle]
    return random.choice(methods)(text)
#bu fonksiyon bir çok augmantation çeşidinden rastgele seçim yapabilir ancak burada özellikle shuffle olanı seçtik



# MODEL YÜKLEME (BART)
from transformers import BartTokenizer, BartForConditionalGeneration
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
#Facebook tarafından sunulan BART modelinin tokenizer ve modeli yükledik.

# VERİ HAZIRLIK
train_data = dataset["train"].select(range(1000))  # 1000 örnek
val_data = dataset["validation"].select(range(200))
#her birkaç adımda bir validation verisi ile test edilir.

def onisleme(batch):
    inputs = tokenizer(batch["article"], max_length=512, truncation=True, padding="max_length")
    # haber metinlerini tokenizer aracılığıyla sayılara dönüştürdüü (tokenize eder).
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["highlights"], max_length=128, truncation=True, padding="max_length")
        #highlights kısmı da aynı şekilde tokenize edilir 
        #as_target_tokenizer()->modelin hedef (label) olarak beklediği özel formatta tokenlar üretir.
        #T5 veya BART gibi encoder-decoder modellerde output'un nasıl tokenlanacağını belirtmek için kullanılır.
        # tokenizer hem giriş hem çıkış için kullanılabilir ama çıkış için ayrı bir biçimde çalışması gerekebilir.
    inputs["labels"] = labels["input_ids"]
    #Eğitim sırasında model kendi özetini üretir ve labels ile karşılaştırır.
    #labels, eğitim sırasında modelin çıktısıyla karşılaştırılarak kayıp (loss) hesaplanmasını sağlar.
    return inputs

tokenized_train = train_data.map(onisleme, batched=True)
tokenized_val = val_data.map(onisleme, batched=True)
test_data = dataset["test"].select(range(50))
tokenized_test = test_data.map(onisleme, batched=True)
"""--------------------------------------------------------------------------------------"""
#EĞİTİM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir="./results",       #	Eğitim çıktılarının (ağırlıklar, loglar vs.) kaydedileceği klasör oluşturdu
    eval_strategy="steps",       # Değerlendirme (validation) işlemi adım sayısına (steps) bağlı yapılacak.
    eval_steps=150,             # Her 150 adımda değerlendirme
    save_steps=150,              #	150 adımda bir model kaydedilir (checkpoint).
    per_device_train_batch_size=2,   #Her bir cihazda (GPU/CPU) eğitim sırasında 2 örneklik batch kullanılacak. K
                                     #küçük tuttuk hızlı olması için
    
    per_device_eval_batch_size=2,     #	Doğrulama sırasında da 2 örneklik batch.
    num_train_epochs=3,               
    # Hızlı test için 1 epoch sonrasında 3 epoch yapılmasına rağmen değişiklik olmadı rouge değerlerinde
    logging_dir="./logs",
    logging_steps=10,    #	Her 10 adımda bir log bilgisi yazılır (loss, accuracy gibi).
    save_total_limit=1,
    disable_tqdm=False          # İlerleme çubuğu
)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model) 
#	Verileri otomatik olarak doğru formatta paketleyen yardımcı. 
#  #DataCollatorForSeq2Seq özetleme için uygun olanıdır.
)
#Trainer nesnesi oluşturulur ve model ile veriler bir araya 


print("Eğitim başlıyor...")
trainer.train()


# MODELİ KAYDET
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
print("✅ Model ve tokenizer kaydedildi: ./saved_model")
