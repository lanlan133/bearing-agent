def matlab_string(value):
    """
    把 MATLAB 字符数组转换成普通 Python 字符串
    """
    arr = np.asarray(value).squeeze()

    if arr.ndim == 0:
        return str(arr.item())

    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.flatten().tolist()).strip()

    return str(arr)
import numpy as np
from scipy.io import loadmat
from pathlib import Path


def load_mat_dataset(mat_path):
    """
    读取 1753data_2.mat 数据集

    MAT结构:
        data: 1753 x 12 cell

    当前已确认:
        第1列: 测点名称，例如 M1H
        第2列: 转速，例如 2981 rpm
        第3列: 采样频率，例如 10240 Hz
        第4列: 振动信号，例如 1x4096
        第5列: 状态标签
        第6列: 设备编号
        第7列: 设备名称
        第9列: 时间
    """

    mat_path = Path(mat_path)

    if not mat_path.exists():
        raise FileNotFoundError(
            f"找不到 MAT 文件: {mat_path}"
        )

    mat = loadmat(mat_path)

    if "data" not in mat:
        raise KeyError(
            "MAT 文件中没有找到变量 'data'"
        )

    data = mat["data"]

    print("=" * 50)
    print("MAT 数据加载成功")
    print("=" * 50)
    print(f"文件: {mat_path}")
    print(f"data shape: {data.shape}")
    print(f"data dtype: {data.dtype}")

    return data


def load_single_sample(mat_path, sample_index=0):
    """
    读取一条振动样本

    sample_index:
        Python从0开始
        sample_index=0 表示MATLAB中的第1行
    """

    data = load_mat_dataset(mat_path)

    if sample_index < 0 or sample_index >= data.shape[0]:
        raise IndexError(
            f"sample_index 超出范围: {sample_index}"
        )

    row = data[sample_index]

    # -------------------------
    # 提取基本信息
    # -------------------------

    measurement_point = matlab_string(row[0])

    speed = float(
        np.asarray(row[1]).squeeze()
    )

    sampling_rate = float(
        np.asarray(row[2]).squeeze()
    )

    # 第4列：振动信号
    signal = np.asarray(
        row[3],
        dtype=float
    ).squeeze()

    label = matlab_string(row[4])

    equipment_id = matlab_string(row[5])

    equipment_name = matlab_string(row[6])

    value_8 = np.asarray(row[7]).squeeze()

    timestamp = matlab_string(row[8])

    value_10 = np.asarray(row[9]).squeeze()

    value_11 = np.asarray(row[10]).squeeze()

    source = matlab_string(row[11])

    # -------------------------
    # 基础检查
    # -------------------------

    if signal.ndim != 1:
        signal = signal.ravel()

    if len(signal) == 0:
        raise ValueError("振动信号为空")

    if sampling_rate <= 0:
        raise ValueError(
            f"采样频率异常: {sampling_rate}"
        )

    sample = {
        "sample_index": sample_index,

        "measurement_point": measurement_point,

        "speed": speed,

        "sampling_rate": sampling_rate,

        "signal": signal,

        "label": label,

        "equipment_id": equipment_id,

        "equipment_name": equipment_name,

        "timestamp": timestamp,

        # 暂时不知道确切含义的字段
        "value_8": value_8,
        "value_10": value_10,
        "value_11": value_11,

        "source": source,
    }

    return sample


if __name__ == "__main__":

    # 修改成你的实际MAT文件路径
    mat_path = "../data/raw/1753data_2.mat"

    sample = load_single_sample(
        mat_path,
        sample_index=0
    )

    print("\n" + "=" * 50)
    print("第一条样本")
    print("=" * 50)

    print(
        "测点:",
        sample["measurement_point"]
    )

    print(
        "转速:",
        sample["speed"],
        "rpm"
    )

    print(
        "采样频率:",
        sample["sampling_rate"],
        "Hz"
    )

    print(
        "振动信号长度:",
        len(sample["signal"])
    )

    print(
        "设备编号:",
        sample["equipment_id"]
    )

    print(
        "设备名称:",
        sample["equipment_name"]
    )

    print(
        "标签:",
        sample["label"]
    )

    print(
        "时间:",
        sample["timestamp"]
    )

    print(
        "信号前10个点:"
    )

    print(
        sample["signal"][:10]
    )