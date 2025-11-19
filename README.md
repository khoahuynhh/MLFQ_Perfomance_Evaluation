# MLFQ Simulator

Trình mô phỏng **Multi-Level Feedback Queue (MLFQ)** này được viết bằng Python/SimPy để đánh giá các chiến lược lập lịch CPU đa cấp trên các bộ xử lý đa lõi. Hệ thống tạo tải công việc Poisson, xử lý bằng ba hàng đợi ưu tiên (Q1, Q2, Q3) và thu thập thống kê thời gian chờ cùng thời gian hoàn tất cho từng tiến trình.

## Tính năng chính
- Ba mức ưu tiên với RR ở Q1/Q2 và FCFS ở Q3, hỗ trợ khấu trừ thời lượng theo quantum.
- Cơ chế ưu tiên định kỳ (priority boost) nhằm tránh đói CPU cho các tiến trình mức thấp.
- Mô phỏng đa lõi: mỗi `Core` chạy như một tiến trình SimPy độc lập, có thể bị ngắt/chiếm quyền.
- Tùy chọn mô hình I/O: xác suất tiến trình phát sinh I/O và thời gian I/O theo phân bố mũ.
- Bộ thu thập thống kê `StatisticsCollector` trả về thời gian chờ, thời gian hoàn tất trung bình và số job hoàn thành.

## Cấu trúc
```
.
├── mlfq_sim.py   # Toàn bộ logic mô phỏng
├── test.py       # Ví dụ/kiểm thử nhanh (nếu cần)
└── Design_...pdf # Tài liệu mô tả hệ thống
```

## Yêu cầu
- Python 3.9+
- [SimPy](https://simpy.readthedocs.io/) (`pip install simpy`)

## Cách chạy
```powershell
python -u mlfq_sim.py
```
Mặc định, script chạy `simulate()` với các tham số mẫu và in:
- `average_wait_time`
- `average_turnaround_time`
- `num_jobs`

Để tùy chỉnh, bạn có thể gọi trực tiếp `simulate()` hoặc dùng script `test.py`. Ví dụ:
```python
from mlfq_sim import simulate

result = simulate(
    num_cpus=8,
    lam=3.0,
    mu=1.2,
    q1=0.02,
    q2=0.08,
    s_period=0.5,
    simulation_time=200.0,
    seed=42,
    io_probability=0.25,
    io_rate=1.5,
)
print(result)
```

## Điều chỉnh tham số
- `num_cpus`: số lõi CPU.
- `lam` / `mu`: tốc độ đến và phục vụ (Poisson/Exp).
- `q1`, `q2`: quantum cho hai hàng đầu.
- `s_period`: chu kỳ boost ưu tiên.
- `io_probability`, `io_rate`: xác suất và tốc độ I/O.
- `simulation_time`: thời lượng mô phỏng.
- `seed`: tái lập kết quả.

## Kiểm thử & mở rộng
- Dùng `test.py` để viết các kịch bản kiểm thử tùy chỉnh.
- Có thể mở rộng bằng cách:
  1. Thêm chính sách ưu tiên khác (ví dụ aging động).
  2. Ghi log chi tiết hơn (CSV/JSON) để phân tích hậu kỳ.
  3. Tích hợp giao diện CLI để quét nhiều cấu hình và xuất biểu đồ.

## Giấy phép
Chưa có giấy phép cụ thể; cập nhật theo nhu cầu của bạn trước khi public.

