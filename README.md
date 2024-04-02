#pca-analysis

# PCA Yüz Analizi Projesi

Bu proje, PCA (Principal Component Analysis - Temel Bileşen Analizi) kullanarak yüz görüntülerinden öznitelik çıkarma işlemini gerçekleştirmektedir.

## Kurulum

Projeyi çalıştırmak için aşağıdaki adımları takip ediniz.

### Sanal Ortamın Oluşturulması

Python sanal ortamını oluşturmak için terminalinize veya komut satırınıza şu komutları giriniz:

```sh
# Sanal ortamı oluştur
cd model
python -m venv pca-env

# Sanal ortamı aktif et
# Windows'ta:
pca-env\\Scripts\\activate
# MacOS/Linux'ta:
source pca-env/bin/activate
```
Gerekli kütüphaneleri yüklemek için requirements.txt dosyasını pip ile çalıştırın 

```sh
pip install -r requirements.txt
```
## Kullanım

Bu bölüm, `main.py` dosyasını çalıştırmak için gerekli adımları içermektedir. Projeyi başarıyla çalıştırmak için aşağıdaki talimatları izleyin:

### Ayarlar

1. `main.py` dosyasını bir metin editörü ile açın.
2. `dir_path` değişkenini, yüz görüntülerinizin bulunduğu klasörün yolu ile güncelleyin. Bu, projenin yüz görüntülerini nereden yükleyeceğini belirler.
   Örneğin:
   ```python
   dir_path = 'path/to/your/images/directory'


categories listesini, analiz etmek istediğiniz görüntü klasör isimleriyle güncelleyin. Bu klasörler, dir_path içinde bulunmalıdır.
```python
categories = ['mutlu', 'uzgun']
```
## Sonuçlar 

`eigenfaces_combined.png`: En yüksek 10 öz değere karşılık gelen öz yüzlerin birleştirilmiş görüntüsü.

`covariance_matrix.txt`: Kovaryans matrisinin kaydedildiği dosya.

`projected_data.txt`: Görüntülerin PCA uzayına iz düşürülmüş hallerinin kaydedildiği dosya.

