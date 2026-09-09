# Qualification CPU nhiều RAM BF16

> 🌐 Language / Ngôn ngữ: [English](BF16-HIGH-MEMORY-CPU.md) | **Tiếng Việt**

Tài liệu này định nghĩa track qualification CPU BF16 dạng opt-in cho Mage-Flow-Turbo-Native-Inference. Track này không thay thế profile `q8-reference` đã đóng băng và không thay đổi contract benchmark CPU ↔ T4 của v1.0.0. BF16 là profile CPU high-memory dạng opt-in, experimental nhưng được support. Đã có real CPU evidence và benchmark paired 768×768, nhưng BF16 không thay thế profile canonical `q8-reference` và không được nâng lên default/release reference từ evidence này. Track qualification/publication BF16 đã đóng: không cần thêm matrix resolution BF16, test scaling nhiều core hoặc test ở resolution cao hơn cho closeout hiện tại của dự án.

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

## Matrix resolution tùy chọn cho nghiên cứu

Matrix 512 → 640 → 768 → 1024 chỉ còn là công cụ cho nghiên cứu follow-up tùy chọn. Matrix này không bắt buộc cho closeout BF16, contract release v1.0.0 hoặc workflow fresh evidence production-demo Q8 canonical.

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

Đã có real BF16 CPU evidence và track qualification/publication BF16 đã đóng. Tuy vậy profile vẫn là opt-in và experimental: benchmark paired 768×768 đã hoàn tất không chứng minh lợi thế hình ảnh vượt trội nhất quán đủ để thay Q8 làm profile canonical/default. Fresh matrix Q8-vs-BF16 512 → 640 → 768 → 1024 không còn là gate còn thiếu; chỉ chạy nếu sau này chủ động mở một nghiên cứu BF16 mới với scope tách biệt.

## Nghiên cứu tùy chọn so sánh Q8 và BF16 trên cùng CPU

Nếu sau này chủ động mở một nghiên cứu CPU cùng máy với scope tách biệt, chạy cả hai matrix trong cùng một session trên cùng một host, cùng `sd-cli` prebuilt, cùng source HEAD, cùng request đóng băng (seed 42, 4 steps, CFG 1.0, 4 threads, prompt fox) và cùng dải resolution 512 → 640 → 768 → 1024. Dùng hai work root sạch riêng biệt và không chạy đồng thời. Quy trình này chỉ là nghiên cứu tùy chọn; không phải gate BF16 còn chờ, không thuộc contract release v1.0.0 và không thuộc workflow fresh evidence production-demo Q8 canonical:

```bash
python -m integrations.kaggle.qualification_matrix \
  --backend cpu --profile q8-reference \
  --input-root /kaggle/input \
  --work-root /kaggle/working/mageflow-q8-matrix-paired \
  --repo-dir "$PWD"
```

```bash
python -m integrations.kaggle.qualification_matrix \
  --backend cpu --profile bf16-high-memory-cpu \
  --input-root /kaggle/input \
  --work-root /kaggle/working/mageflow-bf16-matrix-paired \
  --repo-dir "$PWD"
```

Matrix harness chỉ hỗ trợ CPU và từ chối mọi backend không phải `cpu` trước khi resolve model hoặc generation. Cả profile `q8-reference` lẫn `bf16-high-memory-cpu` dùng chung một `sd-cli` CPU prebuilt; không build từ source, không CMake, không biên dịch.

Mỗi profile ghi record theo từng resolution gồm `mem_available_before_run_kb` mới lấy ngay trước generation, `mem_available_after_run_kb` (tuỳ chọn, ghi sau khi child process thoát), `minimum_mem_available_kb`, `peak_sd_cli_rss_kb`, `elapsed_ms` và identity của artifact. Aggregate giữ snapshot `setup.mem_available_before_kb` riêng ở mức matrix.

Sau khi cả hai matrix PASS, so sánh evidence offline mà không chạy inference:

```bash
python -m integrations.kaggle.compare_matrix_evidence \
  --q8-aggregate  /kaggle/working/mageflow-q8-matrix-paired/output/qualification-matrix-q8-reference-cpu.json \
  --bf16-aggregate /kaggle/working/mageflow-bf16-matrix-paired/output/qualification-matrix-bf16-high-memory-cpu-cpu.json \
  --output /kaggle/working/mageflow-q8-vs-bf16-comparison/comparison-q8-vs-bf16-cpu.json
```

Utility so sánh xác minh tính có thể so sánh (cùng source HEAD, backend CPU, SHA và commit runtime, dải resolution, prompt, seed, steps, CFG, threads, SHA text encoder và SHA VAE, cùng identity diffusion Q8 và BF16) trước khi tính tỷ lệ elapsed và RSS. Nếu bất kỳ gate nào fail nó báo `COMPARABILITY=FAIL` và không xuất bản tỷ lệ hiệu năng gây hiểu lầm.

Kết quả này chỉ báo cáo **evidence nghiên cứu so sánh CPU cùng máy**. Profile BF16 thử nghiệm và profile Q8 reference chỉ được so sánh về latency và bộ nhớ; kết quả không phải release qualification, không khẳng định chất lượng hình ảnh vượt trội, và không được mô tả như bằng chứng yêu cầu tối thiểu 27 GiB cho matrix 1024 đầy đủ.

## Qualification paired 768×768 đã hoàn tất

Một qualification visual paired cùng máy đã hoàn tất và được đóng băng:

- 10 prompt × 2 profile = 20 canonical run;
- cùng một CPU host;
- cùng prompt/seed cho mỗi cặp;
- CPU, 768×768, 4 steps, CFG 1.0, 4 threads;
- runtime/model identity chính xác được đóng băng;
- 20/20 thành công, 10/10 cặp hoàn tất, 0 failure;
- benchmark `visual-q8-vs-bf16-768-10p`, benchmark source head `e84db748f8d140d4a781c85739a18da5a50c8d35`;
- SHA-256 archive evidence cuối `9036186260441fd353b8d32f7b584fdb8bdc4b345d18e8b4a9008983eb33e04d`;
- Q8 mean elapsed ≈ 640,0 s/hình ≈ 10,67 phút/hình; BF16 mean elapsed ≈ 1013,9 s/hình ≈ 16,90 phút/hình;
- Q8 peak `sd-cli` RSS ≈ 8,89 GB; BF16 peak `sd-cli` RSS ≈ 12,54 GB;
- blind visual review: BF16 4 win, Q8 3 win, 3 tie (1 win vật chất rõ ràng của BF16);
- quyết định: Q8 vẫn canonical/default; BF16 vẫn là profile opt-in experimental high-memory được support.

Đây là kết quả trên Kaggle CPU host đã test với `threads=4`, không phải con số hiệu năng phổ quát.

Đây không phải qualification scaling 512→1024.
Không khẳng định scaling 8/16/32/64 vCPU.
Không khẳng định resolution cao hơn làm BF16 thắng Q8.

Người dùng có nhiều CPU/RAM hơn có thể thử BF16 ở resolution cao hơn, nhưng phải tự benchmark host của mình.
