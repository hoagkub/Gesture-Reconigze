# Đề tài Nhận diện bàn tay
# Các file - file đó để làm gì?
*  _plot-acc-line-CV.png - là đường accuracy của model khi áp dụng K-Fold CV
*  _plot-loss-line-CV.png - là đường loss của model khi áp dụng K-Fold
*  acc-measure-by-conf-mat.png - là kết quả tính accuracy bằng confusion matrix
*  acc-measure-by-func.png - là kết quả tính accuracy bằng hàm
*  conf_mat.png - là confusion matrix của model khi đã áp dụng K-Fold CV
*  con_mat_each_label.png - là confusion matrix của từng label khi đã áp dụng K-Fold CV
*  fold0.png --> fold4.png - là các đường loss của từng split (fold0 nghĩa là lấy fold 0 làm val set, các fold còn lại làm train set)
*  plot-acc-line-noCV.png - là đường accuracy của model khi chưa dùng K-Fold CV
*  plot-loss-line-final-model.png - là đường loss của model khi dùng K-Fold CV
*  plot-loss-line-noCV.png - là đường loss của model khi chưa dùng K-Fold CV. Đường loss này cho thấy khi chưa dùng K-Fold CV thì dữ liệu đem vào train chưa đủ để thể hiện hết toàn bộ dữ liệu (Unpresentative Train Dataset)
*  Hãy xem file "train_model_CV.html" hoặc "train_model_CV.pdf" để xác nhận bạn có chạy code "train_model_CV.ipynb" đúng hay không. Vì hai file này là hai file được nhóm xuất ra khi đã chạy trên colab
*  train_model_no_CV.ipynb - là code khi chưa áp dụng K-Fold CV
*  train_model_CV.ipynb - là code khi đã dùng K-Fold CV
*  Trong quá trình chạy code, bạn thiếu thư viện nào thì cài thư viện đó
