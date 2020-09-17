import imutils

def sliding_window(image, step, ws):
    #slide a window across the image=>görüntü boyunca bir pencere kaydırın
    for y in range(0, image.shape[0] - ws[1], step): #  y değerleri satırlar üzerindeki döngümüzdür
        for x in range(0, image.shape[1] - ws[0], step): # x değerleri sütunlar üzerindeki döngümüz

            #yield the current window => mecvut pencereyi aç
    #sonuçta (x, y) değerlerine, pencere boyutuna (ws) ve adım boyutuna göre resmimizin  (yani ROI) verir.
            yield (x, y, image[y:y + ws[1], x:x + ws[0]]) # kısacası return gibi davranır bizim için kayan pencereleri oluşturur
    """
    image:Döngü yapacağımız ve pencereler oluşturacağımız girdi görüntüsü.
    
    step:Her iki (x, y) yönünde kaç piksel "atlayacağımızı" belirten adım boyutumuz.
    Pratikte, yaygın olarak biradım nın 4 -e 8 olduğu varsayılır.. Unutmayın, adım boyutunuz ne kadar küçükse, 
    incelemeniz gereken daha fazla pencere vardır.
    
    ws:pencere boyutu,penceremizden çıkaracağımız pencerenin genişliğini ve yüksekliğini(piksel cinsinden) tanımlar
    """

#Artık kayan pencere rutinimizi başarıyla tanımladığımıza göre,
#Şimdi bizim image_pyramid bir giriş görüntüsünün çok ölçekli bir gösterimini oluşturacağız

def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    #yield to original image=> Orjinal resmi return edeceğiz
    yield image #ilk kez piramidimizin bir katmanını oluşturması istendiğinde orijinal, değiştirilmemiş görüntüyü verir.

    #keep looping over the image pyramid=>görüntü piramidinin üzerinden geçmeye devam ediyoruz
    while True: #Sonraki oluşturulan görüntüler ölçeklenmeye başlar
        #piramitteki sonraki görüntünün boyutlarını hesaplıyoruz
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        #yeniden boyutlandırılan görüntü sağlanan minimum değeri karşılamıyorsa;piramidi oluşturmayı bırakıyoruz
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        #yiel the next image in the pyramid=> piramitteki bir sonraki resme geç
        yield image


        """
        image:=>Çok ölçekli gösterimler oluşturmak istediğimiz için input olarak aldığımız resim
        scale:=>Ölçek faktörümüz , görüntünün her katmanda ne kadar yeniden boyutlandırılacağını kontrol eder
                Daha küçük ölçek değerleri piramitte daha fazla katman sağlar ve daha büyük ölçek değerleri daha az katman sağlar.
        minsize:=> Çıktı görüntüsünün minimum boyutunu kontrol eder.Bu önemlidir çünkü Sonsuza kadar küçük ölçek oluşturabilir kodumuz.
                   Doğru yerde while içinden çıkmamız gerek      
        """





