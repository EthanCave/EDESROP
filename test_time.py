import os
import time
import cv2
import numpy as np
import tensorflow as tf

# ---------------------------------
# Medical preprocessing (your method)
# ---------------------------------
def load_and_preprocess_image(image_path, target_size=(224, 224), enhance=True):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CLAHE enhancement
        if enhance:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Resize and normalize
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# ---------------------------------
# Folder benchmarking
# ---------------------------------
def benchmark_folder(image_dir, model, enhance=True):
    preprocess_times = []
    inference_times = []
    total_times = []

    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Benchmarking {len(image_files)} images...\n")

    for i, image_path in enumerate(image_files, 1):
        start_total = time.perf_counter()

        # Preprocessing timing
        start_pre = time.perf_counter()
        img = load_and_preprocess_image(image_path, enhance=enhance)
        end_pre = time.perf_counter()

        if img is None:
            continue

        # Inference timing
        start_inf = time.perf_counter()
        _ = model.predict(img, verbose=0)
        end_inf = time.perf_counter()

        end_total = time.perf_counter()

        preprocess_times.append(end_pre - start_pre)
        inference_times.append(end_inf - start_inf)
        total_times.append(end_total - start_total)

        if i % 25 == 0:
            print(f"Processed {i}/{len(image_files)} images")

    # Convert to milliseconds
    preprocess_ms = np.array(preprocess_times) * 1000
    inference_ms = np.array(inference_times) * 1000
    total_ms = np.array(total_times) * 1000

    print("\nAverage Timing Results")
    print("=" * 40)
    print(f"Preprocessing (CLAHE): {preprocess_ms.mean():.2f} ms")
    print(f"Inference:             {inference_ms.mean():.2f} ms")
    print(f"Total per image:       {total_ms.mean():.2f} ms")
    print(f"Preprocessing: {preprocess_ms.mean():.2f} ± {preprocess_ms.std():.2f} ms "
      f"(min={preprocess_ms.min():.2f}, max={preprocess_ms.max():.2f})")

    print(f"Inference:     {inference_ms.mean():.2f} ± {inference_ms.std():.2f} ms "
        f"(min={inference_ms.min():.2f}, max={inference_ms.max():.2f})")

    print(f"Total:         {total_ms.mean():.2f} ± {total_ms.std():.2f} ms "
        f"(min={total_ms.min():.2f}, max={total_ms.max():.2f})")
    print(f"Median total time: {np.median(total_ms):.2f} ms")
    return preprocess_ms, inference_ms, total_ms

# ---------------------------------
# Example usage
# ---------------------------------
if __name__ == "__main__":
    IMAGE_DIR = "kaggle/input/170/01/"
    MODEL_PATH = "fianl_model.keras"

    model = tf.keras.models.load_model(MODEL_PATH)
    benchmark_folder(IMAGE_DIR, model, enhance=True)
