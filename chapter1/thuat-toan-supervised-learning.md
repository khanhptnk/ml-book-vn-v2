# Thuật toán Supervised Learning

1. **Sử dụng tập phát triển để tinh chỉnh hyperameter của model**: với mỗi tập giá trị của các hyperparameter \(ví dụ thử chọn $$\lambda$$ trong các giá trị 0.1, 0.01, 0.001\):

   a. **Huấn luyện**: tìm $$w$$ để tối thiểu hóa _objective function_. Theo dõi learning curve để áp dụng early stopping.

   b. **Đánh giá trên development set**: thông báo gía trị của evaluation function trên development set.

2. **Đánh giá trên tập kiểm tra**: với model $$w^*$$ cho kết quả tốt nhất ở development set, thông báo giá trị của evaluation function trên test set.

