from src.inference.infer import YOLOInfer

infer = YOLOInfer("models/release.pt")

# 圖片
result = infer.run("test_fire.jpg")
print(result)

# 影片
result = infer.run("forest1.avi", save_path="outputs/out.mp4")

print(result[0])  