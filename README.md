# Cnn_object_detection
# 1-python detect_with_classifier.py --image images/stingray.jpg --size "(300, 150)"
![stingray](https://user-images.githubusercontent.com/64548477/93517317-f8245380-f933-11ea-93c3-b23c844bf0f0.gif)


![stingray](https://user-images.githubusercontent.com/64548477/93517156-babfc600-f933-11ea-8308-c9721e80e6ef.png)

# 2- python detect_with_classifier.py --image images/hummingbird.jpg --size "(250, 250)"
![hummingbird](https://user-images.githubusercontent.com/64548477/93517854-b21bbf80-f934-11ea-8310-c28885d84a30.gif)
![hummingbird](https://user-images.githubusercontent.com/64548477/93517646-71bc4180-f934-11ea-9f09-f4a4e7545976.png)

# 3-python detect_with_classifier.py --image images/lawn_mower.jpg --size "(200, 200)"
Resimlerde görüldüğü gibi doğru tespiti yapamadık bu resmimiz için.Fakat sonra ki adımda --min-conf 0.95 kullnarak sistemimizin daha güvenilir tahmin yapmasını sağlayacağız ve sonucu hep birlikte göreceğiz

![lawn_mower](https://user-images.githubusercontent.com/64548477/93518394-6a496800-f935-11ea-96f8-4a3f8100cc39.gif)
![lawn_mower](https://user-images.githubusercontent.com/64548477/93518200-2f473480-f935-11ea-92bf-6488e4c9d64c.png)


# 4-python detect_with_classifier.py --image images/lawn_mower.jpg --size "(200, 200)" --min-conf 0.95
Şimdi çıktımıza tekrar bakalım ve tahmin için label'a baktığımız da az önceki gibi half_truck yazısını almadık çünkü güvenilirliği arttırdık.
![lawn_mower_min_conf](https://user-images.githubusercontent.com/64548477/93519359-d5e00500-f936-11ea-9bae-1973b1d2d43e.gif)
![lawn_mower_min_conf](https://user-images.githubusercontent.com/64548477/93519085-6cf88d00-f936-11ea-90ce-8b54dc536a8c.png)

https://github.com/mserce2/Cnn_object_detection/projects/2#column-10898766
