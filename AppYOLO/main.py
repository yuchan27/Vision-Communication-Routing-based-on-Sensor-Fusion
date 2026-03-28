# main.py（你原本的程式碼，完全不用改）
from src.inference.infer import YOLOInfer

infer = YOLOInfer("models/release.pt")

real_image_path = "test_fire.jpg"       # ← 確認這個檔案真的存在

yolo_results, decision, vision_temp = infer.run_with_decision(
    image_path=real_image_path,
    save=True,      
    frame_id=1042
)

print("\n=== 圖片整合決策結果 ===")
print("YOLO 結果:", yolo_results)
print("視覺估測火焰溫度:", vision_temp, "°C" if vision_temp is not None else "（無火）")
print("決策:", decision["decision"]["suggested_action"])
print("解釋:", decision["explainability"]["trace_message"])

# 影片部分仍正常運作
result = infer.run("forest1.avi", save_path="outputs/out.mp4")
print(result[0])