# Ghi chú phát hành Mage-Flow-Turbo-Native-Inference v1.0.0

> 🌐 Ngôn ngữ / Language: **Tiếng Việt** | [English](RELEASE-NOTES-v1.0.0.md)

## Đây là gì

Mage-Flow-Turbo-Native-Inference là **bộ công cụ suy luận native di động cho Mage-Flow-Turbo**. Python đảm nhận cấu hình, kiểm tra, điều phối CLI/REST, vòng đời và thu thập bằng chứng; việc suy luận thực tế do runtime native `stable-diffusion.cpp` (`sd-cli`) thực hiện. Đây không phải một mô hình mới và không huấn luyện hay tinh chỉnh gì.

v1.0.0 là bản công khai đầu tiên dự kiến của dòng này. Tag/Release chỉ được tạo sau khi exact public source head vượt qua các gate qualification CPU/CUDA bắt buộc và evidence cuối đã được xác minh.

## Mục tiêu qualification (đang chờ)

- **Linux x86-64 CPU** — đang chờ qualification.
- **NVIDIA CUDA** (`cuda0`) — đang chờ qualification.
- **Notebook Kaggle CPU** — đang chờ qualification tích hợp (adapter).
- **Notebook Kaggle T4/T4x2** — đang chờ qualification tích hợp dùng `cuda0`.

Vulkan, Metal, ROCm, SYCL, Windows và multi-GPU là các khả năng được tài liệu hóa upstream nhưng **không phải** mục tiêu qualification của v1.0.0.

## Điểm nổi bật

- Model manifest JSON chung với xác minh SHA-256 từng thành phần, fail closed khi thiếu/mơ hồ/khớp sai.
- Runtime manager di động: resolve/build/xác minh `sd-cli` đã pin cho CPU và CUDA.
- Backend/placement cấu hình được (`cpu`, `cuda0`, chuỗi placement đã kiểm tra, `max_vram`, `split_mode`, `auto_fit`).
- CLI `mageflow-native`: `doctor`, `verify`, `generate`, `serve`, `runtime build`.
- REST API loopback với các endpoint ổn định và single-flight.
- Core chung không có phụ thuộc cứng `/kaggle/*`; Kaggle chỉ là adapter mỏng.
- Bằng chứng portability layout cục bộ và harness qualification CPU/CUDA.
- CI không trọng số mô hình và audit publication surface (Actions v7).

## Bằng chứng

Sau khi các gate external bắt buộc PASS, bằng chứng acceptance CPU/CUDA, notebook đã chạy, PNG acceptance, manifest và runtime provenance sẽ được phát hành dưới dạng release assets. Không kèm trọng số mô hình. Giá trị đo chỉ được lấy từ evidence qualification cuối đã xác minh.
