from PIL import Image
from pathlib import Path

cover = ['cover_falls.jpg', 'cover_boat.jpg', 'cover_girl.jpg', 'cover_house.jpg']

path = Path("dataGen")

for image in cover:
    cov_path = path / image
    img = Image.open(cov_path).convert("RGB")
    img.save(cov_path, "JPEG", quality=95, optimize=True, progressive=False)

# Re-encode to baseline JPEG
# img = Image.open(cover).convert("RGB")
# img.save(cover, "JPEG", quality=95, optimize=True, progressive=False)