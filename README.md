Bu proje, Titanic veri setini kullanarak yolcuların hayatta kalma durumlarını tahmin etmek için R'da makine öğrenimi modellerinin nasıl oluşturulacağını gösterir. 
Projede hem Random Forest hem de XGBoost algoritmaları kullanılmış ve performansları karşılaştırılmıştır.

Projeye ait veri seti, R'daki titanic paketinden alınan titanic_train veri setidir. 
Veri seti, 891 yolcuya ait çeşitli bilgileri (yaş, cinsiyet, sınıf, bilet ücreti vb.) ve hayatta kalma durumlarını içerir.

# Gereksinimler
Bu projeyi yerel makinenizde çalıştırabilmek için aşağıdaki R paketlerinin yüklü olması gerekir. Paketleri, R veya RStudio konsolunu kullanarak yükleyebilirsiniz.
install.packages("titanic")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("naniar")
install.packages("fastDummies")
install.packages("caret")
install.packages("ranger")
install.packages("pROC")
install.packages("xgboost")

Alternatif olarak, kodun başında yer alan if (!require()) fonksiyonu ile de otomatik olarak yüklenebilirler.

