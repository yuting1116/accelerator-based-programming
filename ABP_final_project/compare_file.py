import numpy as np

file1 = "/home/yuhu4933/accelerator-based-programming/ABP_final_project/data2/frame_000001.bin"
file2 = "/home/yuhu4933/accelerator-based-programming/ABP_final_project/case2_frame10_gt/frame_000001.bin"

# 假設都是 float32
data1 = np.fromfile(file1, dtype=np.float32)
data2 = np.fromfile(file2, dtype=np.float32)

print(f"file1 size = {len(data1)}")
print(f"file2 size = {len(data2)}")

# 檢查長度
if len(data1) != len(data2):
    print("長度不同")
    print(f"差異: {len(data1) - len(data2)}")
else:
    # 完全相等
    print("完全相等:", np.array_equal(data1, data2))

    # 計算誤差
    diff = np.abs(data1 - data2)

    print("Max Error :", diff.max())
    print("Mean Error:", diff.mean())

    # 找出不相等的位置
    idx = np.where(data1 != data2)[0]

    print("不同數量:", len(idx))

    # 顯示前 10 個差異
    for i in idx[:10]:
        print(
            f"idx={i}, "
            f"file1={data1[i]}, "
            f"file2={data2[i]}, "
            f"diff={data1[i]-data2[i]}"
        )