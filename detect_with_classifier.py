"""görüntü sınıflandırması için eğitilmiş derin bir sinir ağını
alıp bir nesne algılayıcısına dönüştürmek için kullanalım."""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from metehanserce.detection_helpers import sliding_window
from metehanserce.detection_helpers import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2


#construct the argument parse and parse the arguments =>Bağımsız değişkenleri yapılandırcağımız bölüm

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image") #input olarak hangi resmi seçtiğimizi belirtiyoruz

ap.add_argument("-s", "--size", type=str, default="(200, 150)",
                help="ROI size (in pixels)") #Kayan penceremizin hangi boyutta olacağını belirtiyoruz

ap.add_argument("-c", "--min-conf", type=float, default=0.9,
                help="minimum probability to filter weak detections") #Zayıf algılamarı bulmak için filtre değeri belirtiyoruz

ap.add_argument("-v", "--visualize", type=int, default=-1,
                help="whether or not to show extra visualizations for debugging") #Hata ayıklama için ek görselleştirmelerin gösterilip gösterilmeyeceğini belirleyen bir anahtar
args = vars(ap.parse_args())

# nesne algılama prosedürü için kullanılan değişkenlerimizi tanımlıyoruz


WIDTH = 600 #Görüntülerimizin tutarlı bir başlangıç genişliğine sahip olması için değer veriyoruz
PYR_SCALE = 1.5 #Görüntü piramidi ölçek faktörümüzü belirliyoruz.Büyük ölçek demek daha az katman demek dolayısıyla daha hızlı çalışmak demek
WIN_STEP = 16 #Her iki (x, y) yönünde de kaç piksel "atlayacağımızı" belirten kayan pencere adım boyutumuz. Adım boyutumuz ne kadar küçük olursa o kadar fazla pencere incelemiş oluruz bu yüzden zaman kaybı çok olur.4 VE 8 ideal değerlerdir
ROI_SIZE = eval(args["size"])#Bunu argümanlara input olarak biz giriyoruz. Algılamak istediğimiz nesnelerin en boy oranını kontrol etmek için.En-boy oranını ayarlarken bir hata yapılırsa, nesneleri tespit etmek neredeyse imkansız olacaktır
INPUT_SIZE = (224, 224) #Cnn için ideal input-size olduğunu unutmayalım çünkü biz ResNet50 kullandık

#load our network weights from disk
print("[INFO] loading... network....")
model = ResNet50(weights="imagenet", include_top=True)

## giriş görüntüsünü diskten yüklüyoruz,
# verilen genişliğe sahiptir ve ardından boyutlarını alır
orig = cv2.imread(args["image"]) #resim yolunu argüman olarak gireceğiz
orig = imutils.resize(orig, width=WIDTH) #Önceden belirlediğimiz genişliği resize içinde kullanıyoruz
(H, W) = orig.shape[:2]

#görüntü piramidini başlatıyoruz
pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE) #girdi olarak okunacak resim=> Resim için ölçek faktörümüz=>Ve En boy oranını parametre olarak veriyoruz

#Görüntülerden üretilen ROI'leri tutan listemiz
rois = []

#Oluşturduğumuz piramitler üzerinde kayan penceleri tutan listemiz:
locs = [] #ROI'nin orijinal görüntüde olduğu yerin # (x, y) koordinatları

# görüntü piramidi katmanları üzerinde döngü yapmanın ne kadar sürdüğü ve
# sürgülü pencere konumu
start=time.time() #verdiğimiz değerlere göre süreyi ölçeceğiz böylelikle en ideal değeri bulabiliriz.


"""
Döngüdeki ilk adımımız, piramidimizin orijinal görüntü boyutları (W) ve mevcut katman boyutları (image.shape [1])
arasındaki ölçek faktörünü hesaplamaktır.Nesne sınırlayıcı kutularımızı daha sonra yükseltmek için bu değere ihtiyacımız var.

Şimdi, görüntü piramidimizdeki bu özel katmandan kayan pencere döngümüze kademeli olarak gireceğiz. Kayar pencere oluşturucumuz,
görüntümüzde yan yana ve yukarı aşağı bakmamızı sağlar.Oluşturduğu her "Roi" için, yakında görüntü sınıflandırması uygulayacağız.

Ön işleme, CNN’nin gerekli INPUT_SIZE değerine yeniden boyutlandırılmasını, görüntünün dizi biçimine dönüştürülmesini ve Keras’ın 
ön işleme kolaylık işlevinin uygulanmasını içerir. Bu, bir toplu boyut eklemeyi, RGB'den BGR'ye dönüştürmeyi ve ImageNet veri 
setine göre renk kanallarını sıfır merkezlemeyi içerir.ROIS ve ilişkili konum koordinatlarının listesini güncelliyoruz

"""
#Piramidimizin ürettiği her görüntünün üzerinden geçelim:
for image in pyramid:

    #"orijinal" görüntü arasındaki ölçek faktörünü belirliyoruz piramidin boyutları ve "mevcut" katmanı
    scale = W / float(image.shape[1])

    #görüntü piramidinin her katmanı için kayan nokta üzerinde döngü ve pencere konumu
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE): # kayan pencerelerimizin üzerindeki döngümüzü tanımlar.
        # ROI'nin (x, y) koordinatlarını, ve
        # * orijinal * resim boyutları belirliyoruz
        x = int(x * scale)
        y = int(y * scale)            #Koordinatları ölçekliyoruz
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)

        # ROI'yi alıyoruz ve önceden işliyoruz, böylece daha sonra sınıflandırabiliriz
        # Keras / TensorFlow kullanan bölge
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)   #Ön işleme, CNN’nin gerekli INPUT_SIZE değerine yeniden boyutlandırılmasını sağlıyoruz
        roi = preprocess_input(roi)

        # ROI listemizi ve ilişkili koordinatları güncelliyoruz
        rois.append(roi)
        locs.append((x, y, x + w, y + h))

        #Ayrıca isteğe bağlı görselleştirmeyi de gerçekleştiriyoruz:
        #her bir kaymayı görselleştirip görselleştirmediğimizi kontrol edin.
        #görüntü piramidindeki pencereler:

        """
        Burada, hem orijinal görüntüyü "nereye baktığımızı" gösteren yeşil bir kutuyla hem de sınıflandırma için hazır olan 
        yeniden boyutlandırılmış ROI'yi görselleştiriyoruz . Gördüğünüz gibi, 
        yalnızca bayrak komut satırı aracılığıyla ayarlandığında --görselleştireceğiz.
        """
        if args["visualize"] > 0:
            # orijinal görüntüyü klonlayın ve ardından bir sınırlayıcı kutu çizin
            # mevcut bölgeyi çevreleyen şekilde yani
            clone = orig.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            #görselleştirmeyi ve mevcut "Roi" yi gösterir
            cv2.imshow("Visualization", clone)
            cv2.imshow("ROI", roiOrig)
            cv2.waitKey(0)
        """
        Daha sonra, (1) piramit + kayan pencere sürecindeki karşılaştırmamızı kontrol edeceğiz, (2) tüm 
        yatırım getirilerimizi toplu olarak sınıflandıracağız ve (3) tahminlerin kodunu çözeceğiz  
         """

## görüntü piramidi katmanları üzerinde dönmenin ne kadar sürdüğünü ve
#sürgülü pencere konumunu bulacağız
"""
İlk olarak, piramit + sürgülü pencere zamanlayıcımızı sonlandırıyoruz ve işlemin ne kadar sürdüğünü gösteriyoruz Ardından, 
ROI'leri alır ve bunları (toplu olarak) önceden eğitilmiş görüntü sınıflandırıcımızdan (yani ResNet) tahmin yoluyla geçiririz
.Gördüğünüz gibi, çıkarım süreci için burada da bir ölçüt yazdırıyoruz.Son olarak,her ROI için yalnızca en iyi tahmini 
alarak tahminleri çözer.Sınıf etiketlerini (anahtarları) bu etiketle (değerler) ilişkili ROI konumlarıyla 
eşlemek için bir araca ihtiyacımız olacak; etiket sözlüğü bu amaca hizmet eder

"""
end = time.time()
print("[BİLGİ] piramit/pencere üzerinden döngü {:.5f} saniye sürdü".format(end-start))

# ROI'leri NumPy dizisine dönüştürme işlemi yapıyoruz
rois = np.array(rois, dtype="float32")

#ResNet kullanarak önerilen "ROI" lerinin her birini sınıflandırın ve sonra nasıl yapıldığını gösterin.
# sınıflandırmaların süresini kontrol ediyoruz

print("[BİLGİ].... ROI'ler sınıflandırılıyor")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("[BİLGİ] ROI'leri sınıflandırmak {: .5f} saniye sürdü".format(end-start))

## tahminlerin kodunu çöz ve sınıfı eşleyen bir sözlüğü başlat
#Bu etiketle (değerler) ilişkili herhangi bir "Roi" için # etiket (anahtar) ata
preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}

#Şimdi devam edip etiketler sözlüğümüzü dolduralım:
#tahminler üzerinden döngü yapıyoruz
for (i, p) in enumerate(preds):
    (imagenetID, label, prob) = p[0] #mevcut "ROİ" için tahmin bilgilerini yakalıyoruz
    # tahmin edilen olasılığı sağlayarak zayıf tespitleri filtreliyoruz
    # minimum olasılıktan büyük olanları seçiyoruz
    if prob >= args["min_conf"]:
        #tahminle ilişkili sınırlayıcı kutuyu yakalıyoruz ve koordinatları dönüştürüyoruz
        box = locs[i]
        #etiket için tahminlerin listesini alıyoruz ve sınırlayıcı kutu ve olasılık listesine ekliyoruz
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

"""


Satır 157'den başlayarak tahminlerin üzerinden geçerek, önce ImageNet ID, sınıf etiketi ve olasılık dahil olmak üzere
tahmin bilgilerini alıyoruz.Oradan, asgari güvenin karşılanıp karşılanmadığını kontrol ediyoruz. Böyle varsayarsak, 
etiket sözlüğünü sınırlayıcı kutu ve her bir sınıf etiketi(anahtar) ile ilişkili prob skor tuple(değer) ile güncelleriz.

Özet olarak, şimdiye kadar elimizde:
Görüntü piramidimizle ölçekli görüntüler oluşturuldu
Görüntü piramidimizin her katmanı (ölçeklenmiş görüntü) için kayan pencere yaklaşımı kullanılarak oluşturulan ROI'ler
Her ROI için sınıflandırma gerçekleştirdik ve sonuçları etiket listemize yerleştirdikGörüntü sınıflandırıcımızı Keras,
TensorFlow ve OpenCV ile bir nesne algılayıcısına dönüştürmeyi henüz tamamlamadık. Şimdi sonuçları görselleştirmemiz
gerekiyor.Bu, sonuçlarla (etiketler) faydalı bir şey yapmak için mantığı uygulayacağınız zamandır, halbuki bizim 
durumumuzda, biz sadece nesnelere açıklama ekleyeceğiz. Ayrıca, üst üste binen algılamalarımızı maksimum olmayan
bastırma (NMS) aracılığıyla ele almamız gerekecek.
"""

#Şimdi etiket listemizdeki tüm anahtarların üzerinden geçelim:
#görüntüdeki algılanan nesnelerin her biri için etiketlerin üzerinde döngü yapıyoruz
for label in labels.keys():
    #orijinal resmi klonlayın, böylece üzerine çizim yapabiliriz
    print("[INFO] '{}' için sonuçları gösteriyor".format(label))
    clone = orig.copy()
    #Geçerli etiket için tüm sınırlayıcı kutular üzerinde döngü yapıyoruz
    for (box, prob) in labels[label]:
        #resmin üzerine sınırlayıcı kutuyu çiziyoruz
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        #maksimum olmayan bastırma uygulamasından önce sonuçları gösteriyoruz
        #görseli tekrar klonlıyoruz, böylece sonuçları daha sonra görüntüleyebiliriz
        cv2.imshow("Before", clone)
        clone = orig.copy()

#Şimdi NMS'yi uygulayalım ve "after" NMS görselleştirmemizi gösterelim:NMS=>Maksimum olmayanı bastırma
# sınırlayıcı kutuları ve ilgili tahmini çıkarın
# tahmine göre, ardından maksimum olmayan bastırma uygulayın

        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)
        #Nms uygulandıktan sonra tutulan tüm sınırlayıcı kutular üzerinde döngü yapıyoruz
        for (startX, startY, endX, endY) in boxes:
            #sınırlayıcı kutuyu ve etiketi resmin üzerine çiziyoruz
            cv2.rectangle(clone, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        #maksimum olmayan bastırma uyguladıktan sonra çıktıyı gösteriyoruz
        cv2.imshow("After",clone)
        cv2.waitKey(0)



"""
NMS'yi uygulamak için, önce sınırlayıcı kutuları ve ilgili tahmin olasılıklarını (proba) 210 ve 211. satırlardan 
çıkarıyoruz. Daha sonra bu sonuçları NMS'nin imultils uygulamasına aktarıyoruz (Satır 161). Maksimum olmayan bastırma 
hakkında daha fazla ayrıntı için blog yazıma baktığınızdan emin olun.NMS uygulandıktan sonra, 214-220 Satırları 
sınırlayıcı kutu dikdörtgenlerine ve "sonraki" görüntüdeki etiketlere açıklama ekler. 219 ve 220 satırları, bir tuşa 
basılıncaya kadar sonuçları görüntüler, bu noktada tüm GUI pencereleri kapanır ve komut dosyası çıkar.İyi iş! 

Kodu çalıştırmak için terminale;
python detect_with_classifier.py --image images/stingray.jpg --size "(300, 150)"
python detect_with_classifier.py --image images/hummingbird.jpg --size "(250, 250)"
python detect_with_classifier.py --image images/lawn_mower.jpg --size "(200, 200)"
python detect_with_classifier.py --image images/lawn_mower.jpg --size "(200, 200)" --min-conf 0.95
python detect_with_classifier.py --image images/maymun.jpg --size "(200,200)" --min-conf 0.95
yazabilirsiniz
"""

