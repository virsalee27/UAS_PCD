# judul

Disusun Oleh : <br>
Gilang Arbiansyah 2206074 <br>
Virzza Rahmaliyadi 2206103 <br>

# BAB I PENDAHULUAN <br>
Tumor adalah suatu kondisi yang ditandai dengan pertumbuhan sel abnormal yang membentuk massa atau neoplasma, yang sering kali menyerupai pembengkakan (Resnet & Saputra, 1907). Tumor dapat berkembang di berbagai organ tubuh manusia, termasuk otak (Candra et al., 2024). Berdasarkan data epidemiologi dari tinjauan sistematis, insidensi tumor otak di seluruh dunia tercatat sebesar 10,82 per 100.000 penduduk per tahun, dengan rentang antara 0,01 hingga 25,95 per 100.000 penduduk per tahun (Pratama et al., 2024). Tumor otak dapat dibedakan menjadi dua jenis, yaitu tumor primer yang berkembang langsung di otak dan tumor sekunder yang merupakan hasil metastasis dari organ lain (Otak et al., n.d.). Glioma merupakan jenis tumor otak primer yang paling sering ditemukan, di mana sekitar 78% dari total kasus tumor otak ganas termasuk dalam kategori ini (Septipalan et al., 2024). Selain itu, data dari Central Brain Tumor Registry of the United States (CBTRUS) menunjukkan bahwa meningioma adalah tumor otak yang paling sering terdiagnosis secara histologis dengan angka 36,8%, diikuti oleh tumor pituitari sebesar 16,2% (Candra et al., 2024).  <br>

Untuk mendeteksi keberadaan tumor otak secara akurat, pasien umumnya disarankan menjalani pemeriksaan pencitraan medis seperti CT Scan atau MRI (Pratama et al., 2024). Dari hasil pencitraan medis tersebut, tumor dapat diklasifikasikan berdasarkan lokasi dan jenisnya. Namun, klasifikasi secara manual oleh tenaga medis sering kali membutuhkan waktu yang lama dan memiliki potensi kesalahan. Oleh karena itu, diperlukan suatu metode berbasis kecerdasan buatan yang dapat membantu mengklasifikasikan tumor otak dengan lebih efisien dan akurat. Salah satu metode yang saat ini banyak digunakan dalam analisis pencitraan medis adalah Convolutional Neural Network (CNN) (Otak et al., n.d.).  <br>

CNN adalah teknik dalam deep learning yang sangat efektif dalam mengenali pola pada citra, termasuk pencitraan medis. Dengan menggunakan CNN, proses klasifikasi tumor otak dapat dilakukan secara otomatis berdasarkan karakteristik visual dari citra MRI. Salah satu model arsitektur CNN yang telah terbukti efektif dalam tugas klasifikasi citra adalah VGG-16. Model ini dikembangkan oleh K. Simonyan dan A. Zisserman dari Universitas Oxford dan berhasil mencapai kinerja yang sangat baik dalam pengenalan gambar pada dataset skala besar (Resnet & Saputra, 1907).  <br>

Dalam penelitian ini, metode CNN dengan model VGG-16 diterapkan untuk mengklasifikasikan jenis tumor otak berdasarkan citra MRI. Tumor otak akan dikategorikan ke dalam empat kelas, yaitu Glioma Tumor, Meningioma Tumor, No Tumor, dan Pituitary Tumor. Tujuan dari penelitian ini adalah untuk mengembangkan sistem klasifikasi berbasis deep learning yang dapat membantu tenaga medis dalam mendiagnosis tumor otak dengan lebih cepat dan akurat. Dengan adanya sistem ini, diharapkan dapat meningkatkan efisiensi diagnosis serta membantu dalam upaya deteksi dini tumor otak, sehingga penanganan dapat dilakukan lebih tepat dan efektif(Septipalan et al., 2024).<br>

# BAB II METODE PENELITIAN <br>
1.	Persiapan Data <br>
Langkah pertama dalam pembuatan model ini adalah menyiapkan data yang akan digunakan untuk melatih dan menguji model. Dataset yang digunakan terdiri dari gambar MRI otak yang dibagi menjadi dua kategori, yaitu Tumor dan Normal. Data gambar ini kemudian diubah menjadi format yang dapat diproses oleh model menggunakan ImageDataGenerator.<br>

2.  Preprocessing dan Augmentasi Data <br>
Sebelum digunakan untuk pelatihan, gambar-gambar diubah ukurannya menjadi 224x224 piksel karena ukuran tersebut cocok dengan arsitektur model VGG16 yang akan digunakan. Selain itu, gambar juga diproses agar nilai pikselnya berada dalam rentang [0, 1] dengan melakukan normalisasi. Augmentasi gambar juga dilakukan untuk menambah variasi data dan mencegah overfitting, misalnya dengan rotasi gambar, pergeseran, dan pembalikan horizontal.<br>

3.  Arsitektur Model <br>
Dalam model ini, digunakan arsitektur VGG16 yang sudah dilatih sebelumnya pada dataset ImageNet. Kami memanfaatkan transfer learning dengan menggunakan model VGG16 tanpa lapisan klasifikasinya, karena lapisan tersebut tidak dibutuhkan untuk dataset kita. Kemudian, lapisan klasifikasi baru ditambahkan di atasnya, terdiri dari lapisan dense dengan fungsi aktivasi ReLU dan sigmoid untuk output biner (Tumor atau Normal).<br>

4.	Pelatihan Model <br>
Setelah arsitektur model siap, langkah berikutnya adalah pelatihan. Proses pelatihan dilakukan selama beberapa epoch, di mana model belajar untuk memprediksi kategori gambar berdasarkan data pelatihan. Kami menggunakan optimizer Adam dengan learning rate yang sangat kecil agar proses pelatihan berjalan lebih stabil. Setiap epoch, hasil model dievaluasi menggunakan data validasi untuk memeriksa seberapa baik model bekerja pada data yang tidak terlihat sebelumnya. <br>

5.	Evaluasi Model<br>
Setelah pelatihan selesai, evaluasi dilakukan pada model untuk mengukur seberapa akurat model dalam mengklasifikasikan gambar. Pengukuran dilakukan menggunakan akurasi, yaitu persentase gambar yang diklasifikasikan dengan benar. Selain itu, confusion matrix juga digunakan untuk menunjukkan bagaimana model mengklasifikasikan gambar pada setiap kategori. <br>

# BAB III HASIL DAN PEMBAHASAN <br>
## 3.1

## 3.2

## 3.3 

## 3.4

## 3.5

# KESIMPULAN

# DAFTAR PUSTAKA
