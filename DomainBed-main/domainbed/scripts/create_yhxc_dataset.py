import os
import h5py
import numpy as np

import h5py
import numpy as np

def read_h5(file_path, load_data=True):
    """
    Parameters
    ----------
    file_path : str
        Đường dẫn tới file .h5

    Returns
    -------
    result : dict
        {
            "file_path": path_to_file,
            "meta": {...},
            "data": {"y": ..., "h": ..., "x": ..., "c": ...}
        }
    """
    with h5py.File(file_path, "r") as f:
        meta = {}
        meta["Channel model"] = f.attrs["Channel model"]
        meta["Delay spread (ns)"] = f.attrs["Delay spread (ns)"].item()
        meta["Speed (m/s)"] = f.attrs["Speed (m/s)"].item()
        meta["SNR (dB)"] = f.attrs["SNR (dB)"].tolist()

        data = {}
        for key in ["y", "h", "x", "c"]:
            if key in f:
                data[key] = f[key][...]

    return {
        "file_path": file_path,
        "meta": meta,
        "data": data
    }


def create_dummy_pusch_dataset(
    output_dir="pusch_dataset_demo",
    num_groups=3,          # số nhóm demo, có thể đổi thành 36
    samples_per_group=10,  # số mẫu mỗi nhóm, demo ít cho nhẹ
    n_rx=4,                # số port thu
    n_layer=1,             # số layer phát
    n_sym=14,              # số OFDM symbol
    n_sc=12,               # số subcarrier
    mod_order=4,           # QAM16 -> 4 bit/symbol
    seed=42
):
    """
    Tạo dataset HDF5 có cấu trúc:
      - Mỗi nhóm là 1 file: __00.h5, __01.h5, ...
      - Mỗi file chứa:
          + attrs: mô tả channel
          + datasets: y, h, x, c

    Shape:
      y: [N, n_rx, n_sym, n_sc]                  complex64
      h: [N, n_rx, n_layer, n_sym, n_sc]         complex64
      x: [N, n_layer, n_sym, n_sc]               complex64
      c: [N, n_layer * n_sc * 12 * mod_order]    uint8

    Ghi chú:
      - symbol DMRS là 3 và 11 (index từ 0)
      - số symbol data = 12
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    dmrs_symbols = [3, 11]
    data_symbols = [i for i in range(n_sym) if i not in dmrs_symbols]
    n_data_sym = len(data_symbols)  # sẽ là 12 nếu n_sym = 14
    c_len = n_layer * n_sc * n_data_sym * mod_order

    channel_models = ["TDL-A", "TDL-B", "TDL-C", "CDL-A", "CDL-B"]

    for g in range(num_groups):
        file_name = f"__{g:02d}.h5"
        file_path = os.path.join(output_dir, file_name)

        # ===== Meta =====
        channel_model = channel_models[g % len(channel_models)]
        delay_spread_ns = float(rng.choice([30.0, 100.0, 300.0, 1000.0]))
        speed_mps = float(rng.choice([0.0, 3.0, 30.0, 120.0]))
        snr_db = sorted(rng.choice(np.arange(-5, 21, 5), size=2, replace=False).tolist())

        # ===== Tạo dữ liệu giả =====
        # X: IQ phát miền tần số [N, n_layer, n_sym, n_sc]
        x_real = rng.standard_normal((samples_per_group, n_layer, n_sym, n_sc), dtype=np.float32)
        x_imag = rng.standard_normal((samples_per_group, n_layer, n_sym, n_sc), dtype=np.float32)
        x = (x_real + 1j * x_imag).astype(np.complex64)

        # H: kênh truyền miền tần số [N, n_rx, n_layer, n_sym, n_sc]
        h_real = rng.standard_normal((samples_per_group, n_rx, n_layer, n_sym, n_sc), dtype=np.float32)
        h_imag = rng.standard_normal((samples_per_group, n_rx, n_layer, n_sym, n_sc), dtype=np.float32)
        h = (h_real + 1j * h_imag).astype(np.complex64)

        # Y: IQ nhận miền tần số [N, n_rx, n_sym, n_sc]
        # Với n_layer=1, lấy h[:,:,0,:,:] * x[:,0,:,:] + noise
        noise_real = 0.05 * rng.standard_normal((samples_per_group, n_rx, n_sym, n_sc), dtype=np.float32)
        noise_imag = 0.05 * rng.standard_normal((samples_per_group, n_rx, n_sym, n_sc), dtype=np.float32)
        noise = (noise_real + 1j * noise_imag).astype(np.complex64)

        if n_layer == 1:
            y = h[:, :, 0, :, :] * x[:, 0, :, :][:, np.newaxis, :, :] + noise
        else:
            # Trường hợp nhiều layer: cộng chồng theo layer
            y = np.zeros((samples_per_group, n_rx, n_sym, n_sc), dtype=np.complex64)
            for l in range(n_layer):
                y += h[:, :, l, :, :] * x[:, l, :, :][:, np.newaxis, :, :]
            y += noise

        y = y.astype(np.complex64)

        # C: bit mã hóa trước QAM Mod [N, c_len]
        c = rng.integers(0, 2, size=(samples_per_group, c_len), dtype=np.uint8)

        # ===== Lưu file .h5 =====
        with h5py.File(file_path, "w") as f:
            # attrs để hàm read_h5 đọc được
            f.attrs["Channel model"] = channel_model
            f.attrs["Delay spread (ns)"] = delay_spread_ns
            f.attrs["Speed (m/s)"] = speed_mps
            f.attrs["SNR (dB)"] = np.array(snr_db, dtype=np.float32)

            # datasets
            f.create_dataset("y", data=y, compression="gzip")
            f.create_dataset("h", data=h, compression="gzip")
            f.create_dataset("x", data=x, compression="gzip")
            f.create_dataset("c", data=c, compression="gzip")

        print(f"Đã tạo: {file_path}")

    print("\nHoàn tất tạo dataset demo.")
    print(f"Thư mục output: {output_dir}")
    print("Các shape chuẩn:")
    print(f"  y: [N, {n_rx}, {n_sym}, {n_sc}]")
    print(f"  h: [N, {n_rx}, {n_layer}, {n_sym}, {n_sc}]")
    print(f"  x: [N, {n_layer}, {n_sym}, {n_sc}]")
    print(f"  c: [N, {c_len}]")
    print(f"DMRS symbols: {dmrs_symbols}")
    print(f"Data symbols: {data_symbols}")


if __name__ == "__main__":
    create_dummy_pusch_dataset(
        output_dir="pusch_dataset_demo",
        num_groups=3,
        samples_per_group=10,
        n_rx=4,
        n_layer=1,
        n_sym=14,
        n_sc=12,
        mod_order=4,  # QAM16
        seed=42
    )


    # Đọc thử 1 file
    file_path = os.path.join("pusch_dataset_demo", "__00.h5")
    result = read_h5(file_path)

    print("=== THÔNG TIN FILE ===")
    print("file_path:", result["file_path"])

    print("\n=== META ===")
    for k, v in result["meta"].items():
        print(f"{k}: {v}")

    print("\n=== SHAPE / DTYPE ===")
    for key in ["y", "h", "x", "c"]:
        arr = result["data"][key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

    print("\n=== KIỂM TRA NHANH ===")
    print("DMRS symbol index:", [3, 11])
    print("y sample[0,0,0,0] =", result["data"]["y"][0, 0, 0, 0])
    print("h sample[0,0,0,0,0] =", result["data"]["h"][0, 0, 0, 0, 0])
    print("x sample[0,0,0,0] =", result["data"]["x"][0, 0, 0, 0])
    print("c first 20 bits =", result["data"]["c"][0, :20])