from ultralytics import YOLO
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import json

TEST_DATA = Path("/home/etman/etman/DocVision/test_data")
PDF_DATA = TEST_DATA / "pdfs"
IMAGES_DATA = TEST_DATA / "images"
IMAGES_DATA.mkdir(exist_ok=True)
PREPROCESSED_DATA = TEST_DATA / "preprocessed"
PREPROCESSED_DATA.mkdir(exist_ok=True)


def convert_pdf_to_images(pdf_file, output_dir):
    # get the pdf name without extenion
    pdf_name = pdf_file.stem
    
    # create an output dir for this pdf
    images_output_dir = output_dir / pdf_name
    images_output_dir.mkdir(exist_ok=True)
    
    print(f"Converting {pdf_name}.pdf...")
    
    images = convert_from_path(pdf_file, dpi=200)
    for page_num, image in enumerate(images, start=1):
        # save the image with its num_page
        output_path = images_output_dir / f"page_{page_num:03d}.jpg"
        image.save(output_path, "JPEG")
        
    print("DONE converting")
    
    
def crop_objects(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    image = Image.open(image).convert("RGB")
    cropped_img = image.crop((x1, y1, x2, y2))
    
    return cropped_img

def save_objects(cropped_object, crop_id, object_metadata, output_dir):
    object_output_path = output_dir / object_metadata["doc_id"]
    object_output_path.mkdir(exist_ok=True)
    object_page_path = object_output_path / object_metadata["page_num"]
    object_page_path.mkdir(exist_ok=True)
    cropped_path = object_page_path / f"{crop_id}.jpg"
    cropped_object.save(cropped_path)
    with open(object_page_path / f"{crop_id}.json", 'w') as f:
        json.dump(object_metadata, f)

def run_model(model_path, data_dir, output_dir, conf=0.4):
    SKIP = {"Page-header", "Page-footer", "Section-header", "Footnote"}
    model = YOLO(model=model_path)
    doc_id = data_dir.name
    for image_path in data_dir.iterdir():
        page_num = Path(image_path).stem
        result = model.predict(image_path, verbose=False, conf=conf)[0]
        h, w = result.orig_shape
        cls_to_idx = result.names
        object_metadata = {}
        for crop_id, box in enumerate(result.boxes, start=1):
            object_cls_id = int(box.cls)
            object_cls_name = cls_to_idx[object_cls_id]
            if object_cls_name in SKIP:
                continue
            bbox = box.xyxy.tolist()[0]
            conf = float(box.conf)
            object_metadata = {
                "doc_id": doc_id,
                "page_num": page_num,
                "crop_id": crop_id,
                # "reading_order": order,
                "cls_name": object_cls_name,
                "cls_id": object_cls_id,
                "confidence": box.conf.item(),
                "bbox": bbox,
                "page_size": [w, h],
                "dpi": 200,
                "page_image": str(image_path),
            }
            cropped_object = crop_objects(image_path, bbox)
            crop_id = f"{crop_id:03d}_{object_cls_name}"
            save_objects(cropped_object, crop_id, object_metadata, output_dir)
