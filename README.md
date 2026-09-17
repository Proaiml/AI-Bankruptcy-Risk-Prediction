# AI Bankruptcy Risk Prediction (Yapay Sinir Ağı ile İflas Riski Tahmini)

Bu proje, şirketlerin finansal verilerini kullanarak **"Bankrupt?" (İflas Riski)** ikili sınıflandırması (binary classification) yapan, önceden eğitilmiş PyTorch yapay sinir ağı modelini (`model_weights.pth`) modern ve kurumsal bir web arayüzü ile sunar.

> **Not:** Kendi yazdığınız eğitim scriptiniz olan kök dizindeki `main.py` dosyasına **dokunulmamış**, tüm web arayüzü ve backend servisi bağımsız olarak **`web_app/`** klasörü içerisine yerleştirilmiştir.

---

## 🏗️ Model Mimarisi

Model mimarisi aşağıdaki katmanlardan oluşmaktadır:
```
95 Giriş Özelliği → 30 Nöron (ReLU) → 20 Nöron (ReLU) → 1 Çıktı (Logit)
```
- **Loss Fonksiyonu (Eğitimde):** `BCEWithLogitsLoss`
- **Inference Sırasında:** Modelin sonuna ek bir Sigmoid katmanı eklenmez; tahmin anında `torch.sigmoid(logit)` ile olasılık hesaplanır.
- **Karar Kuralı:**
  - `probability >= 0.5` $\rightarrow$ **Bankrupt (1)**
  - `probability < 0.5` $\rightarrow$ **Non-Bankrupt (0)**
- **Feature Sıralaması:** 95 finansal özelliğin sırası doğrudan `data.csv` dosyasından okunur ve birebir korunur.
- **Model Yükleme:** Model ağırlıkları uygulama başlangıcında yalnızca **bir kez** hafızaya alınır (`model.eval()`).

---

## 📁 Proje Dosya Yapısı

```
ödev1/
│
├── main.py                 # Sizin orijinal eğitim ve görselleştirme kodunuz (dokunulmadı)
├── model_weights.pth       # Eğitilmiş PyTorch model ağırlıkları
├── data.csv                # Veri seti (95 özellik + 1 hedef sütun)
├── requirements.txt        # Gerekli kütüphaneler
├── README.md               # Kurulum ve çalıştırma kılavuzu
│
└── web_app/                # Web Arayüzü ve Servis Klasörü
    ├── main.py             # FastAPI backend servisi & PyTorch model inference
    ├── templates/
    │   └── index.html      # Modern, finansal AI dashboard şablonu
    └── static/
        ├── style.css       # Koyu lacivert finans/AI teması, responsive 3 sütunlu grid
        └── script.js       # Form doğrulama, hızlı test butonları, API haberleşmesi
```

---

## 🚀 Kurulum ve Çalıştırma Adımları

### 1. Gereksinimlerin Yüklenmesi
Terminal veya PowerShell üzerinden proje klasöründe şu komutu çalıştırın:

```bash
pip install -r requirements.txt
```

### 2. Web Arayüzünü Başlatma

Arayüzü başlatmak için en pratik yöntem:

#### Yöntem 1 (Tek Tıkla Başlatma - Tavsiye Edilen):
Kök dizindeki **`baslat.bat`** (veya `start.bat`) dosyasına çift tıklayın.
- Python ortamını ve kütüphaneleri otomatik tespit eder.
- FastAPI sunucusunu başlatır.
- Tarayıcınızı (`http://127.0.0.1:8000`) otomatik olarak açar!

#### Yöntem 2 (Terminal / Komut Satırı ile):
```bash
cd web_app
uvicorn main:app --reload
```
veya kök dizinden:
```bash
uvicorn web_app.main:app --reload
```

Sunucu varsayılan olarak `http://127.0.0.1:8000` adresinde çalışmaya başlayacaktır.

### 3. Tarayıcıdan Erişilmesi
Web tarayıcınızı açarak aşağıdaki adrese gidin:

👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🖥️ Web Arayüzü Özellikleri

1. **Otomatik 95 Feature Girişi:** `data.csv`'deki 95 özelliğin tamamı gerçek adlarıyla ve doğru sırayla input alanı olarak oluşturulur.
2. **Responsive Tasarım:** Masaüstü ekranlarda **3 sütunlu grid**, mobil cihazlarda **tek sütunlu** kompakt görünüm.
3. **Hızlı Test Butonları (Quick Test Data):**
   - **Load Healthy Sample:** `data.csv`'den gerçek bir sağlıklı şirket örneğini otomatik doldurur.
   - **Load Bankrupt Sample:** `data.csv`'den gerçek bir iflas riski olan şirket örneğini otomatik doldurur.
   - **Fill Averages:** Veri setinin ortalama değerlerini doldurur.
   - **Clear All:** Tüm alanları temizler.
4. **Canlı Özellik Arama:** 95 özellik arasında aradığınız metriği (örneğin *ROA*, *Margin*, *Debt*) anında filtrelemenizi sağlayan arama çubuğu.
5. **Görsel Risk Raporu:** Tahmin sonucunda iflas durumu (**Bankrupt** veya **Non-Bankrupt**), risk yüzdesi (%87.34 gibi) ve renkli ilerleme çubuğu (progress gauge) anında gösterilir.
6. **Validasyon & Hata Kontrolü:** Boş bırakılan alanlar, sayısal olmayan veya NaN/sonsuz değerler hem frontend hem backend tarafında yakalanarak kullanıcı uyarılır.

---

## 🔌 API Endpoint'leri

### 1. `POST /predict`
95 özelliğin değerlerini alır, `[1, 95]` boyutunda `torch.float32` tensör oluşturur ve model tahminini döner.

**Örnek İstek (Request Body):**
```json
{
  "features": {
    " ROA(C) before interest and depreciation before interest": 0.370594,
    "... (95 özellik)": 0.0
  }
}
```

**Örnek Yanıt (Response):**
```json
{
  "prediction": 1,
  "probability": 0.8734,
  "percentage": 87.34
}
```

### 2. `GET /api/sample?type={bankrupt|non_bankrupt|mean}`
Arayüzden tek tıkla test yapılabilmesi için veri setinden örnek veri döner.

---

## ⚠️ Bilgilendirme ve Uyarı
Sayfada yer alan model mimarisi ve uyarı metinleri:
- **Model Architecture:** `95 Inputs → 30 ReLU → 20 ReLU → 1 Output`
- **Classification:** `Neural Network Binary Classification Model`
- **Uyarı:** *"This model is an experimental decision-support prototype and should not be considered financial advice."*
