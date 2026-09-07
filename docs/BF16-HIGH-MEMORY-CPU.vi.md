# Qualification CPU nhiều RAM BF16

> 🌐 Language / Ngôn ngữ: [English](BF16-HIGH-MEMORY-CPU.md) | **Tiếng Việt**

Tài liệu này định nghĩa track qualification CPU BF16 dạng opt-in cho Mage-Flow-Turbo-Native-Inference. Track này không thay thế profile `q8-reference` đã đóng băng và không thay đổi contract benchmark CPU ↔ T4 của v1.0.0. Profile BF16 là hạ tầng qualification thử nghiệm và **chưa release-qualified** cho tới khi có evidence từ một CPU run thật.

## Contract profile

`bf16-high-memory-cpu` dùng một mirror Kaggle model duy nhất cho cả transformer lẫn VAE:

```text
mage-flow-community-mage-flow-turbo / PyTorch / default
```

mirror này cung cấp cả hai file:

- `transformer/diffusion_pytorch_model.safetensors`
- `vae/diffusion_pytorch_model.safetensors`

Không cần input `pytorch/vae-only` riêng cho profile BF16.

- Transformer Mage-Flow-Turbo: SafeTensors BF16 chính thức từ mirror `PyTorch / default` duy nhất, SHA-256 `6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d`.
- Text encoder Qwen3-VL-4B: artifact GGUF `Q4_K_M` reference hiện tại.
- Mage VAE: artifact SafeTensors reference hiện tại từ cùng mirror `PyTorch / default`.
- Backend: chỉ `cpu`.
- RAM hiển thị tối thiểu: 27 GiB.
- Headroom runtime quan sát bắt buộc: `MemAvailable` tối thiểu 3 GiB trong canonical generation.

Profile fail-closed. CUDA, accelerator không hỗ trợ, RAM không đủ, thiếu telemetry, sai identity hoặc thiếu model artifact đều không được fallback về Q8.

## Gate canonical đầu tiên

Chỉ chạy đúng một generation 512×512 trước khi chạy matrix lớn hơn:

```bash
python -m integrations.kaggle.qualification \
  --backend cpu \
  --profile bf16-high-memory-cpu \
  --input-root /kaggle/input \
  --work-root /kaggle/working/mageflow-bf16-qualification \
  --repo-dir "$PWD"
```

Canonical request vẫn giữ seed 42, 4 steps, CFG 1.0, 4 threads và prompt cáo đỏ đã đóng băng. Evidence ghi exact model identities, runtime identity, tổng RAM, RAM khả dụng trước khi chạy, RAM khả dụng thấp nhất, peak RSS của `sd-cli`, elapsed time và PNG identity.

Gate 1 512 đã chạy thành công trên phiên Kaggle CPU-only mới tại source HEAD `ee9119e8831558353dd514ef41fe867808e327b9`: model/runtime verification PASS, backend CPU, artifact là PNG 512×512 hợp lệ, `MemAvailable` thấp nhất quan sát được vẫn từ 3 GiB trở lên.

## Qualification matrix theo resolution

Lệnh matrix resolve model, verify model artifact và verify runtime **đúng một lần cho mỗi tiến trình matrix**, sau đó chạy canonical request đã đóng băng tuần tự 512 → 640 → 768 → 1024:

```bash
python -m integrations.kaggle.qualification_matrix \
  --backend cpu \
  --profile bf16-high-memory-cpu \
  --input-root /kaggle/input \
  --work-root /kaggle/working/mageflow-bf16-matrix \
  --repo-dir "$PWD"
```

Matrix chỉ là evidence qualification về feasibility, hành vi bộ nhớ, latency và artifact correctness; matrix **không phải release qualification** và không so sánh chất lượng hình ảnh. Chỉ width và height thay đổi giữa các lượt; prompt, seed, steps, CFG và threads bị đóng băng.

Matrix yêu cầu runtime CPU prebuilt tường minh qua `MAGE_CPU_PREBUILT_SD_CLI` và **không build từ source**. Setup chỉ chạy một lần: đo RAM, profile preflight, build manifest, verify manifest, resolve `sd-cli` prebuilt, verify runtime identity và SHA của binary. Setup timing telemetry được ghi riêng để phân biệt overhead verify artifact, overhead verify runtime và latency inference thật.

Fail-fast RAM policy được giữ nguyên: RAM hiển thị tối thiểu 27 GiB và `MemAvailable` tối thiểu quan sát được 3 GiB. Nếu headroom xuống dưới 3 GiB, resolution hiện tại bị ghi là failed, partial evidence được ghi và matrix dừng ngay; không chạy resolution nào sau đó. Matrix 640/768/1024 phải có evidence thật trước khi công bố bất kỳ so sánh timing hoặc bộ nhớ giữa các resolution.

## Acceptance

Gate 512×512 chỉ được chấp nhận khi model/runtime verification PASS, `sd-cli` thoát thành công, PNG hợp lệ, backend được chọn là CPU và minimum available memory quan sát được vẫn từ 3 GiB trở lên.

Chỉ sau khi gate này PASS mới chạy fresh CPU session với cùng matrix 512 → 640 → 768 → 1024 để so sánh trực tiếp Q8 và BF16. Cho tới khi có real BF16 evidence, profile này chỉ là hạ tầng qualification thử nghiệm và không được mô tả là đã release-qualified.
