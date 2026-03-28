def convert_to_yolo_format(result):
    boxes = result.boxes

    if boxes is None:
        return []

    output = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xywh = box.xywhn[0].tolist()  # normalized

        output.append({
            "class_id": cls_id,
            "class_name": result.names[cls_id],
            "confidence": conf,
            "bbox": xywh
        })

    return output