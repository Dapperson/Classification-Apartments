# Laporan Proyek Pertama: Predictive Analytics - Roni Merdiansah

## Domain Proyek

Saat ini tinggal di sebuah apartemen sudah menjadi gaya hidup masyarakat urban, hal ini disebabkan karena banyaknya fasilitas yang ditawarkan di sebuah apartemen. Apalagi saat ini banyaknya apartemen yang dibangun di pusat kota ataupun pusat bisnis, sehingga dapat memudahkan bagi mereka para profesional muda untuk beraktivitas [(Tim Editorial Rumah.com, 2022)](https://www.rumah.com/panduan-properti/bisnis-apartemen-69876). Kelengkapan fasilitas yang dimiliki tiap apartemen sangat mempengaruhi dalam mengklasifikasikan apartemen itu sendiri. Proyek yang dilakukan saat ini akan mempelajari klasifikasi yang sudah ada sebelumnya pada dataset dengan pendekatan Machine Learning.

## Business Understanding

### Problem Statements

Sebuah hal penting untuk mengethaui fasilitas seperti apa yang dapat mempengaruhi kualitas sebuah apartemen. Hal ini akan sangat membantu seseorang yang ingin memulai bisnis apartemen. Berikut beberapa rumusan masalah yang akan kita cari tahu:
- Seberapa akurat klasifikasi yang ada sebelumnya pada dataset?
- Fasilitas apa saja yang sangat berpengaruh terhadap peng-klasifikasian itu?

### Goals

Adapun tujuan dari proyek ini sebagai berikut:
- Melatih model klasifikasi dengan jumlah 10 model dan mencaritahu model mana yang memiliki akurasi paling bagus terhadap kolom target, dan menjadikan kolom lain sebagai fitur.
- Mencaritahu ada berapa fitur penting yang memiliki presentase tinggi terhadap kolom target, dengan cara mengambil model yang memiliki kemampuan membaca presentase fitur dengan baik.

## Data Understanding

Data ini merupakan data imajiner [Paris Housing](https://www.kaggle.com/datasets/aleshagavrilov/parishousing) (Perumahan Paris) dari tahun 1990-2021. Meskipun imajiner data ini cukup bagus untuk digunakan dalam proyek ini. Data ini berisi karakteristik dari setiap perumahan di Paris seperti lokasi, fasilitas, dan harga. Data ini memiliki total 1000 data.

### Variabel

Paris Housing memiliki 17 variabel sebagai berikut:

1. `squareMeters` : Luas Tanah (numerik-m2)
2. `numberOfRooms` : Jumlah Ruangan per-Apartemen (Teks Angka dalam bahasa inggris)
3. `floors` : Jumlah Lantai Apartemen (numerik)
4. `cityCode` : Kode Pos (numerik-Kategorikal)
5. `cityPartRange` : Jarak ke Kota (numerik-m2)
6. `numPrevOwners` : Berapa Kali sudah ditempati sebelumnya (numerik)
7. `made` : Tahun dibangun (numerik-Tahun)
8. `isNewBuilt` : Baru/Direnovasi (Biner-Ya/Tidak)
9. `hasStormProtector` : Penangkal Petir (Biner-Ya/Tidak)
10. `basement` : Ruang Bawah tanah (numerik)
11. `attic` : Loteng (numerik)
12. `garage` : Garasi (numerik)
13. `hasStorageRoom` : Gudang (numerik)
14. `hasGuestRoom` : Ruang Tamu (numerik)
15. `price` : Harga (numerik)
16. `category` : Kategori (numerik)
17. `PoolAndYard` : Kolam Renang & Taman (Kategorikal)

### Exploratory Data Analysis

1. analisis kolom kategori
![](https://raw.githubusercontent.com/Dapperson/Classification-Apartments/main/Picture/Variabel%20Kategorikal.png)

2. analisis kolom numerik
![](https://raw.githubusercontent.com/Dapperson/Classification-Apartments/main/Picture/Variabel%20Numerik.png)

3. analisis kolom kategorikal & biner
![](https://raw.githubusercontent.com/Dapperson/Classification-Apartments/main/Picture/Presentase%20Category.png)


Dari beberapa visualisasi yang dilakukan didapatkan beberapa insight sebagai berikut:

- Mayoritas `category` adalah *Basic* dengan presentase sebesar 87%, dan *Luxury* adalah 13%
- Hampir semua fitur pada kolom numerikal memiliki rata-rata yang sama antara `category` *Basic* dan *Luxury* seperti perbandingan pada presentase sebelumnya, tidak terdapat perbedaan yang signifikan pada nilai tertentu
- Begitu juga pada kolom kategorikal & biner, tidak terdapat perbedaan yang signifikan antar sub kolom, hampir semuanya memiliki perbandingan yang sama.


## Data Preparation

### Data Cleaning

Pada tahap ini dilakukan pembersihan pada data-data yang tidak memiliki informasi yang berarti, seperti bernilai null atau kosong. Setelah dilakukan pemeriksaan pada dataset, didapatkan bahwa datanya bersih dan tidak ada yang null, atau cacat.

### Feature Engineering

#### Delete Unnecessary Features

Karena model yang akan dibuat disini adalah klasifikasi, maka diperlukan mengeliminasi beberapa fitur yang tidak diperlukan. adapaun fitur yang akan dieliminasi adalah `cityCode` dan `made`.

#### Reformatting String
Karena sebelumnya pada fitur `numberOfRooms` merupakan Teks Angka dalam bahasa inggris, maka harus diubah dahulu menjadi numerik agar fitur dapat digunakan pada saat pemodelan Machine Learning

#### Encoding

Encoding perlu dilakukan agar fitur selain numerik dapat ikut digunakan pada saat pemodelan Machine Learning dengan cara melabeli setiap nilainya, karena Machine Learning hanya bisa menggunakan nilai numerik. Untuk fitur `category` yang merupakan kolom target akan diaplikasikan dengan *LabelEncoder*, dimana nilai *Basic* menjadi `0` dan nilai *Luxury* menjadi `1`

Sedangkan untuk kolom kategorikal dan biner lainnya akan diaplikasikan dengan *OneHotEncoder*, karena kolom-kolom ini merupakan input feature. Didapatkan beberapa fitur baru hasil *OneHotEncoder* sebagai berikut:

- `isNewBuilt_False`
- `isNewBuilt_True`
- `hasStormProtector_False`
- `hasStormProtector_True`
- `hasStorageRoom_False`
- `hasStorageRoom_True`
- `PoolAndYard_has pool and has yard`	
- `PoolAndYard_has pool and no yard`	
- `PoolAndYard_no pool and has yard`	
- `PoolAndYard_no pool and no yard`

Dimana setiap fitur memiliki nilai `0` dan `1`

### Feature Selection

Adapun fitur target yang digunakan seperti yang sudah dibahas sebelumnya yaitu kolom `category`. Karena pada tahap sebelumnya sudah dilakukan eliminasi fitur, dan juga telah melakukan tahap *Encoding* sehingga terdapat penambahan fitur, sehingga jumlah akhir dari fitur input adalah 20.


## Modeling

### Split Data

Pembagian untuk data latih adalah 80% atau 8000 data, dan data test adalah 20% atau 2000 data.

### Basis Model

Disini akan menggunakan 10 basis model untuk mencari tahu perbandingan model mana yang memiliki akurasi paling bagus, adapun 10 model itu sebagai berikut:

- DecisionTreeClassifier
- LogisticRegression
- KNeighborsClassifier
- GaussianNB
- SVC
- LinearSVC
- RandomForestClassifier
- GradientBoostingClassifier
- ExtraTreesClassifier
- XGBClassifier

Karena data pada kolom target sebelumnya tidak seimbang (imbalance) jumlah antara *Basic* dan *Luxury*, maka pada proyek ini menggunakan _Cross Validation - Stratified K Fold_. Konsep dari Cross Validation - Stratified K Fold sendiri adalah validasi silang antara data latih dan data test, dimana sejumlah data yang pernah menjadi data latih akan di posisikan menjadi data test, begitu pun sebaliknya. Validasi silang ini dilakukan tergantung jumlah _n_ yang dimasukkan. Pada proyek ini menggunakan jumlah _n_ sebanyak 2. Setelah membuat basis model didaptkan 5 model terbaik yang memiliki akurasi sempurna sebagai berikut:

- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- ExtraTreesClassifier
- XGBClassifier

Confusion Matrix
![](https://raw.githubusercontent.com/Dapperson/Classification-Apartments/main/Picture/Confusion%20Matrix.png)

Kelima model tersebut dapat membaca model dengan baik dengan nilai
True Positive = `4367`
True Negative = `633`
False Positive = `0`
False Negative = `0`

Karena salah satu tujuan proyek ini adalah mencari tahu fitur penting yang sangat berpengaruh terhadap klasifikasi, maka diantara kelima model tersebut yang paling banyak mendeteksi fitur penting adalah *RandomForestClassifier*

![](https://raw.githubusercontent.com/Dapperson/Classification-Apartments/main/Picture/Feature%20Importance%20RFC.png)


## Evaluasi

Untuk memastikan kembali model yang dipilih tidak ada kesalahan, maka pada tahap evaluasi ini melakukan _Hyperparameter Tuning_ dengan menggunakan _GridSearchCV_. GridSearchCV berfungsi untuk melakukan permutasi terhadap semua parameter yang kita inputkan pada model untuk mencari parameter terbaik. Setelah melakukan Hyperparameter Tuning didapatkan parameter terbaik 

Best Score: 1.0
Best Hyperparameters: 
- criterion : gini
- max_depth : 5
- max_features : auto
- n_estimators : 200}

dengan matriks kebingungan dari data test sebagai berikut

![](https://raw.githubusercontent.com/Dapperson/Classification-Apartments/main/Picture/Confusion%20Matrix%20HT.png)

True Positive = `1744`
True Negative = `256`
False Positive = `0`
False Negative = `0`

Dengan demikian dapat disimpulkan bahwa kolom `category` meng-klasifikasikan apartemen dengan sempurna antara apartemen *Basic* (Biasa) dengan apartemen *Luxury* (Mewah), dimana faktor yang sangat berpengaruh dalam peng-klasifikasian itu adalah dari *apakah apartemen itu baru di bangun/direnovasi* dan juga *apakah apartemen itu memiliki fasilitas taman ataupun kolam renang*.
