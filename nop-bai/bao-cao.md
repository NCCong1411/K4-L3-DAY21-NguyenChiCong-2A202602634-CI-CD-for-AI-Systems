# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Chí Công |
| MSSV | 2A202602634 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/NCCong1411/K4-L3-DAY21-NguyenChiCong-2A202602634-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---:|---:|---:|---:|---:|
| 1 | 50 | 0.05 | 2 | 0.7083 | 0.8600 |
| 2 | 100 | 0.10 | 3 | 0.7407 | 0.8600 |
| 3 | 200 | 0.10 | 5 | 0.7368 | 0.8600 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`.

**Lý do:** Lần 2 đạt F1 cao nhất 0.7407 và vượt ngưỡng 0.65. Một thử nghiệm khác đạt accuracy cao nhất 0.8760 nhưng F1 chỉ 0.7328, chứng minh accuracy cao chưa đồng nghĩa nhận diện lớp dương tốt. Tăng số cây và độ sâu còn tốn thời gian mà không cải thiện F1, nên cấu hình đã chọn cân bằng chất lượng và độ phức tạp.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Lớp thu nhập trên 50K chỉ chiếm 24.8%. Mô hình luôn đoán “thu nhập thấp” vẫn đạt accuracy 75.2% dù không tìm được mẫu dương nào, nên accuracy dễ gây hiểu nhầm. F1 lớp dương kết hợp precision và recall, đo đồng thời khả năng hạn chế dự đoán sai và tìm đủ người thu nhập cao. Không dùng `weighted` vì lớp đông sẽ chi phối; không dùng `macro` vì mục tiêu là lớp 1. Với chiến dịch tiếp cận khách hàng tiềm năng, false negative nguy hiểm hơn do bỏ sót khách hàng giá trị, vì vậy recall lớp 1 rất quan trọng.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow lỗi SQLite | SQLAlchemy 2.1 không tương thích | Pin SQLAlchemy 2.0.36 |
| AWS từ chối OIDC | Subject mới có owner ID và repo ID | Sửa trust policy đúng repo, nhánh main |
| Push không chạy workflow | Actions của fork chưa bật đủ | Enable bằng API và kiểm tra lại |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---:|---:|
| Bước 2 (chỉ `train_batch1`) | 0.7407 | 0.8600 |
| Bước 3 (thêm `train_batch2`) | 0.7345 | 0.8800 |

**Nhận xét:** Accuracy tăng 0.0200 nhưng F1 giảm 0.0062 do recall lớp dương giảm. Pipeline vẫn xanh đủ bốn job; rollback guard giữ model Bước 2 vì candidate kém production.

---

## 5. Phần Bonus Đã Thực Hiện

- [ ] Bonus 1 - DagsHub: workflow sẵn sàng, chờ cấu hình secrets.
- [x] Bonus 2 - Quét threshold 0.10–0.90; production chọn 0.30.
- [x] Bonus 3 - Tạo `detail.txt`, confusion matrix và metric từng lớp.
- [x] Bonus 4 - Candidate kém hơn không thay production 0.7407.
- [x] Bonus 5 - Ghi tỷ lệ lớp dương và cảnh báo lệch trên 5 điểm phần trăm.
