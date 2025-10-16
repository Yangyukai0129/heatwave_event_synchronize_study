import os
import imageio
import re

# 圖片資料夾
img_dir = r"D:\\es\\visualization\\clusters_with_itemsets_pre30y"
output_gif = r"D:\\es\\visualization\\clusters_with_itemsets_pre30y.gif"

# 取出檔案並依數字排序
def extract_num(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1

files = sorted(
    [f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))],
    key=extract_num
)

frames = []
for f in files:
    img_path = os.path.join(img_dir, f)
    frames.append(imageio.imread(img_path))

# 儲存成 GIF
imageio.mimsave(output_gif, frames, duration=1000)  # duration 每張圖顯示秒數

print(f"GIF 已輸出到: {output_gif}")
