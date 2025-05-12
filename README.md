##  Hate Speech and Offensive Language Detection

###  Giới thiệu

Dự án này nhằm phát hiện và phân loại các phát ngôn thù ghét (*hate speech*) và ngôn từ xúc phạm (*offensive language*) trên văn bản tiếng Anh. Chúng tôi áp dụng mô hình BERT kết hợp với các kiến trúc mạng nơ-ron CNN và LSTM để tăng cường khả năng học đặc trưng ngữ nghĩa.

Mục tiêu là phân loại văn bản thành ba loại:

* **0 - Normal**: văn bản bình thường.
* **1 - Offensive**: ngôn từ gây xúc phạm.
* **2 - Hate Speech**: phát ngôn thù ghét.

---

##  Cấu trúc thư mục

```
.
├── create_balanced_data.ipynb      # Xử lý và cân bằng dữ liệu đầu vào
├── BERT_CNN_+_BERT_LSTM_+_BERT.ipynb   # Huấn luyện và đánh giá các mô hình BERT
├── README.md
├── /models                         
└── /data                           
```

---

##  Xử lý dữ liệu

Notebook `create_balanced_data.ipynb` thực hiện các bước sau:

1. **Đọc dữ liệu từ file gốc** (`.csv`):

   * Gồm 3 nhãn: "hate", "offensive", "normal".

2. **Tiền xử lý văn bản**:

   * Xoá dấu câu và ký tự đặc biệt.
   * Chuyển về chữ thường.
   * Xoá stopwords (danh sách từ dừng tiếng Anh như "the", "is", "at",...).

3. **Chuyển nhãn thành số**:

   * `"normal"` → `0`
   * `"offensive"` → `1`
   * `"hate"` → `2`

4. **Cân bằng dữ liệu (Balancing)**:

   * Sử dụng kỹ thuật under-sampling để giảm số lượng mẫu ở lớp chiếm đa số.
   * Mục tiêu: số lượng lớp gần bằng nhau để giảm bias mô hình.

5. **Lưu dữ liệu ra file**: chuẩn bị cho bước huấn luyện.

---

##  Mô hình và Kiến trúc

Notebook `BERT_CNN_+_BERT_LSTM_+_BERT.ipynb` thử nghiệm 3 kiến trúc:

### 1. **BERT**

* Sử dụng `bert-base-uncased` từ Hugging Face Transformers.
* Fine-tune trực tiếp đầu ra `CLS` token với một dense layer để phân loại.

### 2. **BERT + LSTM**

* Đầu ra từ BERT (chuỗi embedding) được đưa vào một lớp LSTM.
* Sau đó là dense layer để phân loại.
* Kiến trúc:
* ![image](https://github.com/user-attachments/assets/5c7bb519-5679-4784-80e1-f57fc8f1ab2e)


### 3. **BERT + CNN**

* Đầu ra từ BERT được đưa qua Conv1D + GlobalMaxPooling.
* Sau đó là dense layer để phân loại.
* Kiến trúc: 
* ![image](https://github.com/user-attachments/assets/81de390d-8654-4efc-b605-3e7b796ddc63)


Tất cả mô hình đều sử dụng:

* Loss: `CrossEntropyLoss`
* Optimizer: `AdamW`
* Evaluation: Accuracy, Precision, Recall, F1-score
* Huấn luyện bằng TensorFlow 2.x

---

##  Kết quả 

Chúng tôi đánh giá ba mô hình: **BERT**, **BERT + LSTM** và **BERT + CNN** trên tập kiểm tra, cả trước và sau khi áp dụng các kỹ thuật **augmentation** và tiền xử lý nâng cao. Kết quả được đo bằng độ chính xác (`Accuracy`), độ chính xác theo lớp (`Precision`), độ nhạy (`Recall`) và F1-score.

###  Trước khi áp dụng augmentation

| Model        | Acc | F1 Score (class 0) | F1 Score (class 1) | F1 Score (class 2) |
|--------------|-----|------------------|------------------------|--------------|
| BERT + CNN   | 0.88 | 0.83        | 0.87                   | 0.93     |
| BERT + LSTM  | 0.89 | 0.85        | 0.88                   | 0.93     |
| BERT         | 0.86 | 0.80            | 0.84                   | 0.89         |

###  Sau khi áp dụng augmentation và tiền xử lý 

| Model        | Acc | F1 Score (class 0) | F1 Score (class 1) | F1 Score (class 2) |
|--------------|-----|------------------|------------------------|--------------|
| BERT + CNN   | 0.90 | 0.85        | 0.89               | 0.95     |
| BERT + LSTM  | 0.92 | 0.87    | 0.90               | 0.96     |
| BERT         | 0.88     | 0.82        | 0.87                   | 0.92         |

 **Nhận xét:**
- Sau khi áp dụng các kỹ thuật làm sạch và tăng cường dữ liệu, mọi mô hình đều được cải thiện đáng kể.
- Mô hình **BERT + LSTM** cho kết quả tốt nhất trên cả ba lớp, đặc biệt trong việc nhận diện **hate speech** (F1 = 0.87) và đạt **accuracy = 92%**.
- **BERT + CNN** cũng cải thiện rõ rệt, đặc biệt ở lớp **offensive language** và **neither**.

---
## DEMO

[!DEMO](https://youtu.be/Ukwrk8WAQPU)
