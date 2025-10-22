# Career Mentor Chatbot (RAG Powered by Gemini)

Bu proje, kullanıcılara kariyer hedefleri ve mesleki gelişimleri konusunda kişiselleştirilmiş rehberlik sunan bir yapay zeka destekli **Geri Alma-Artırılmış Üretim (RAG)** mimarisiyle güçlendirilmiş kariyer mentor robotudur. Kullanıcıların sorularına yanıt vermek için bir kariyer veri setinden bilgi çeker ve doğal dil işleme yeteneklerini kullanır.

Sistem, **dört farklı kariyer veri setinden elde edilen zengin ve ilişkisel bilgileri** kapsamlı bir bilgi tabanı kullanıyor. LangChain ve HuggingFace Embeddings kullanılarak kariyer veri setindeki Soru-Cevap içeriklerini indeksler. Kullanıcı sorusu geldiğinde, **ChromaDB**'den en alakalı bilgi parçalarını (context) alır ve bu parçaları **Google Gemini 2.0 Flash** modeline bir prompt ile besler.

---

## Temel Özellikler

* **Gelişmiş RAG Mimarisi:** Bilgiyi sadece veri setinden çekerek doğru ve güvenilir cevaplar üretir.
* **Akıllı Cevap Formatlama:** Sohbet geçmişinin olup olmamasına göre dinamik yanıt formatı uygular:
    * *Yeni Sorular:* Uygun meslek alanları, üniversite bölümleri, beceriler ve adım adım hazırlık planı sunar.
    * *Devam Eden Sohbetler:* Konuyu tekrarlamadan, kısa ve odaklı bilgilerle konuşmayı ilerletir.
* **Sohbet Geçmişi Yönetimi:** Konuşmanın bağlamını koruyarak tutarlı bir mentorluk deneyimi sağlar.
* **Kullanıcı Dostu Arayüz:** Streamlit ile hızlıca geliştirilmiş sezgisel ve etkileşimli sohbet arayüzü.

---

## Teknolojiler

* **Python**
* **Streamlit:** Web arayüzü için.
* **LangChain:** LLM entegrasyonu, RAG ve sohbet hafızası yönetimi için.
* **Google Gemini (gemini-2.0-flash):** Büyük Dil Modeli (LLM) olarak.
* **HuggingFace Embeddings (all-MiniLM-L6-v2):** Metin gömme (embedding) işlemleri için.
* **ChromaDB:** Vektör veritabanı olarak.

--- 

### Ön Koşullar

* Python 3.8+
* pip (Python paket yöneticisi)
* Gemini modellerine erişimi olan geçerli bir Google API Key

---

### Kurulum

1.  **Depoyu Klonlayın :**
    ```bash
    git clone https://github.com/melisaonl/career-mentor-chatbot.git
    cd career-mentor-chatbot
    ```

2.  **Gerekli Kütüphaneleri Yükleme:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ortam Değişkenlerini Ayarlama:**
    Projenizin ana dizininde `.env` adında bir dosya oluşturun ve Google API anahtarınızı aşağıdaki gibi içine ekleyin:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    ```
    * `YOUR_GOOGLE_GEMINI_API_KEY` kısmını kendi Google Gemini API anahtarınızla değiştirin.

4.  **Veri Kümeleri Hazırlığı (Ön İşleme):**
    Bu projenin temel bilgi havuzu, dört farklı ham veri setinden özel olarak işlenmiş ve birleştirilmiş tek bir CSV dosyası (`data/career_chatbot_dataset.csv`) üzerine kuruludur. Aşağıdaki orijinal veri setleri Kaggle platformundan temin edilmiş, ilişkilendirilerek ve ilgili sütunlar birleştirilerek nihai veri seti oluşturulmuştur:

    * **Job Skill Set:** [Kaggle Linki](https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set)
    * **Field of Study vs Occupation:** [Kaggle Linki](https://www.kaggle.com/datasets/jahnavipaliwal/field-of-study-vs-occupation)
    * **Job Descriptions 2025:** [Kaggle Linki](https://www.kaggle.com/datasets/adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles)
    * **LinkedIn Job Posts Insights Dataset:** [Kaggle Linki](https://www.kaggle.com/datasets/sindhumadhurii/linkedin-job-posts-insights-dataset)

    Bu kapsamlı veri birleştirme ve ön işleme adımı, botun çeşitli kariyer sorularına daha zengin ve ilişkisel yanıtlar verebilmesini sağlamak amacıyla projenin başlangıç aşamasında gerçekleştirilmiştir. Nihai `career_chatbot_dataset.csv` dosyası `data/` klasöründe bulunmaktadır.

5.  **Veritabanını Oluşturma:**
    Vektör veritabanını oluşturun (ilk çalıştırmada veya veriler/model değiştiğinde)
    ```bash
    python create_database.py
    ```
    Bu komut, `career_chroma_db` klasörünü oluşturacak ve içine kariyer verilerini işleyip gömerek kaydedecektir.

6.  **Uygulamayı Çalıştırma:**
    Veritabanı oluşturulduktan sonra Streamlit uygulamasını başlatabilirsiniz:
    ```bash
    streamlit run app.py
    ```
    Uygulama, varsayılan web tarayıcınızda açılacaktır.

7.  Daha sonra tarayıcınızı şu adresten açın: http://localhost:8501/

---

### Deploy Link
[Deploy Linki](https://career-mentor-chatbot-tavabspfadykmdabwxxy4m.streamlit.app/))

![Career Mentor Chatbot Demo](https://github.com/user-attachments/assets/2a8f6908-75d3-43be-b36d-7486a71483e3)
