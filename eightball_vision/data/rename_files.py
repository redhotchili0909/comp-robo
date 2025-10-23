import os
from PIL import Image

# === CONFIG ===
folder_path = "data/Photos-orignal"   # your folder
base_name = "pool_table"      # base name for renamed files

def is_jpeg_disguised_as_heic(path):
    """Check if a .HEIC file actually contains JPEG data."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
            return header.startswith(b"\xff\xd8\xff")  # JPEG magic bytes
    except Exception:
        return False

def rename_and_convert(folder, base_name):
    files = sorted(os.listdir(folder))
    count = 1

    for filename in files:
        old_path = os.path.join(folder, filename)
        if not os.path.isfile(old_path):
            continue

        name, ext = os.path.splitext(filename)
        ext = ext.lower()

        new_filename = f"{base_name}_{count}.jpg"
        new_path = os.path.join(folder, new_filename)

        try:
            # Handle HEIC files that are really JPEGs
            if ext == ".heic" and is_jpeg_disguised_as_heic(old_path):
                print(f"⚠️  {filename} is actually JPEG — renaming...")
                image = Image.open(old_path).convert("RGB")
                image.save(new_path, "JPEG")
                os.remove(old_path)

            # Handle normal image files
            elif ext in [".jpg", ".jpeg", ".png"]:
                image = Image.open(old_path).convert("RGB")
                image.save(new_path, "JPEG")
                os.remove(old_path)

            else:
                print(f"⚠️  Skipping unsupported file: {filename}")
                continue

            print(f"✅ Renamed and saved: {new_filename}")
            count += 1

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    rename_and_convert(folder_path, base_name)
