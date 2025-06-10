# Face Swap Web Uygulaması

## Genel Bakış

Bu proje, videolardaki yüzleri tespit edip ayırt ederek yüz değiştirme işlemleri için altyapı sağlayan Flask tabanlı bir web uygulamasıdır. Yüz tespiti için MTCNN, yüz tanıma için ise InceptionResnetV1 modelleri kullanılır ve GPU desteği ile yüksek performans sunulur. Kullanıcı dostu bir arayüz üzerinden video yükleyerek yüzlerin çıkarılmasını ve kaydedilmesini sağlayabilirsiniz.

---

## İçindekiler

- [Özellikler](#özellikler)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [GPU ve CUDA Desteği](#gpu-ve-cuda-desteği)
- [Fonksiyon Detayları](#fonksiyon-detayları)
- [Proje Yapısı](#proje-yapısı)
- [Sıkça Sorulan Sorular](#sıkça-sorulan-sorular)
- [Lisans](#lisans)
- [İletişim](#iletişim)

---

## Özellikler

- Videolardan farklı yüzlerin tespiti ve ayrıştırılması
- MTCNN ile yüksek doğrulukta yüz tespiti (GPU destekli)
- InceptionResnetV1 ile yüz tanıma ve karşılaştırma
- Flask tabanlı kullanıcı dostu web arayüzü
- İşlem ilerlemesini takip etme (progress_status)
- Tespit edilen yüzlerin otomatik olarak çıktı klasörüne kaydedilmesi
- CUDA desteği ile GPU hızlandırması, yoksa CPU ile çalışma
- Esnek yapı: Maksimum kare sayısı ve tolerans parametreleri özelleştirilebilir

---

## Kullanılan Teknolojiler

- **Python**: 3.8 veya üzeri
- **Flask**: Web uygulaması çerçevesi
- **PyTorch**: Derin öğrenme modelleri için (CUDA destekli önerilir)
- **facenet-pytorch**: MTCNN ve InceptionResnetV1 modelleri
- **OpenCV**: Görüntü ve video işleme
- **Pillow (PIL)**: Görüntü manipülasyonu
- **NumPy**: Matematiksel işlemler

---

## Kurulum

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/kullanici/face-swap-web.git
cd face-swap-web
```

### 2. Sanal Ortam Oluşturun

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.\.venv\Scripts\activate.bat
```

### 3. Bağımlılıkları Yükleyin

CUDA destekli PyTorch için (örneğin, CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CUDA desteği olmayan sistemler için:

```bash
pip install torch torchvision torchaudio
```

Diğer bağımlılıklar:

```bash
pip install facenet-pytorch opencv-python flask pillow numpy
```

**Not**: Sisteminizdeki CUDA sürümüne uygun PyTorch sürümünü seçin. CUDA sürümünüzü kontrol etmek için `nvidia-smi` komutunu kullanabilirsiniz.

---

## Kullanım

### 1. Flask Sunucusunu Başlatın

```bash
# Linux/macOS
export FLASK_APP=app.py
export FLASK_ENV=development
flask run

# Windows PowerShell
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
flask run
```

### 2. Videoyu Yükleyin

- Web arayüzüne gidin (varsayılan: `http://127.0.0.1:5000`).
- Bir video dosyası yükleyin.
- Uygulama, videodaki yüzleri tespit eder ve farklı yüzleri `output_faces` klasörüne kaydeder.

### 3. Çıktılar

- Tespit edilen yüzler, `output_faces` klasörüne PNG formatında kaydedilir.
- Her yüz için kare numarası ve yüz konumu bilgileri döndürülür.

---

## GPU ve CUDA Desteği

Uygulama, başlangıçta sistemdeki GPU'yu kontrol eder ve CUDA kullanılabilirse modeller GPU üzerinde çalışır. Aksi takdirde, CPU kullanılır. GPU durumu konsolda şu şekilde görüntülenir:

```yaml
CUDA kullanılabilir: Evet
Toplam GPU sayısı: 1
GPU 0: NVIDIA GeForce RTX 3060
Cihaz: cuda:0
```

**Not**: CUDA desteği için NVIDIA GPU ve uygun sürücülerin yüklü olması gerekir.

---

## Fonksiyon Detayları

### `extract_distinct_faces(video_path, output_folder, max_frames=50, tolerance=0.6, progress_status=None)`

#### Parametreler

- `video_path` (str): İşlenecek video dosyasının yolu.
- `output_folder` (str): Tespit edilen yüzlerin kaydedileceği klasör.
- `max_frames` (int): İşlenecek maksimum kare sayısı (varsayılan: 50).
- `tolerance` (float): Yüz tanıma için eşik değeri (varsayılan: 0.6).
- `progress_status` (dict, opsiyonel): İşlem ilerlemesini takip için dictionary.

#### İşleyiş

1. Video kare kare işlenir (maksimum `max_frames` kadar).
2. MTCNN ile yüz tespiti yapılır.
3. Tespit edilen yüzler, InceptionResnetV1 ile gömülür (embedding).
4. Yeni yüzler, önceki yüzlerle karşılaştırılır ve eşik değerine göre sınıflandırılır.
5. Yeni yüzler `output_folder` altına kaydedilir.
6. İlerleme durumu `progress_status` dictionary'sine yazılır.

#### Dönüş Değeri

- `face_map`: Yüz ID'si, kare numarası ve yüz konumu bilgilerini içeren dictionary.

---

## Proje Yapısı

```plaintext
face-swap-web/
│
├── app.py                # Flask uygulaması ana dosyası
├── face_extractor.py     # Yüz çıkarma ve tanıma fonksiyonları
├── templates/            # HTML şablonları
│   └── index.html        # Ana web arayüzü
├── static/               # CSS, JS ve yüklenen dosyalar
│   ├── css/
│   ├── js/
│   └── uploads/          # Yüklenen videolar
├── output_faces/         # Çıkarılan yüzlerin kaydedildiği klasör
├── requirements.txt      # Proje bağımlılıkları
└── README.md             # Bu dosya
```

---

## Sıkça Sorulan Sorular (SSS)

**S: CUDA desteği nasıl etkinleştirilir?**  
C: Sisteminizde NVIDIA GPU ve uygun CUDA sürücüsü yüklü olmalıdır. PyTorch'un CUDA destekli sürümünü kurduğunuzdan emin olun.

**S: Yüz algılama ne kadar doğru?**  
C: MTCNN, yüksek doğruluk sağlar. Ancak düşük çözünürlüklü videolar veya kötü aydınlatma koşulları performansı etkileyebilir.

**S: Maksimum kare sayısı nasıl değiştirilir?**  
C: `extract_distinct_faces` fonksiyonundaki `max_frames` parametresini ayarlayabilirsiniz.

**S: Yüz değiştirme özelliği var mı?**  
C: Bu proje, yüz tespiti ve ayırt etme sağlar. Yüz değiştirme için ek modüller entegre edilmelidir.

**S: Hangi video formatları destekleniyor?**  
C: OpenCV desteklediği tüm formatlar (MP4, AVI, vb.) kullanılabilir.

---

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

---

Teşekkürler!


---