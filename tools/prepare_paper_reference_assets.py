"""Crop original published/official figures without redrawing their contents."""
from pathlib import Path
from PIL import Image
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / '.publication_133_sources'
OUT = ROOT / '.paper_reference_revision' / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
im = Image.open(SRC / 'mmpose_wholebody_133.png').convert('RGBA')
white = Image.new('RGBA', im.size, 'white')
white.alpha_composite(im)
white = white.convert('RGB')
white.save(OUT / 'official_wholebody_133.png')
# These are crops from the official annotation figure, not regenerated keypoints.
white.crop((0, 0, 1020, 2330)).save(OUT / 'official_body_crop.png')
white.crop((1080, 0, 2910, 2310)).save(OUT / 'official_face_crop.png')
white.crop((0, 2320, 1020, 3353)).save(OUT / 'official_feet_crop.png')
white.crop((1030, 2320, 2910, 3353)).save(OUT / 'official_hands_crop.png')
white.crop((1080, 690, 2910, 2310)).save(OUT / 'official_face_nodes.png')
Image.open(SRC / 'mmpose_hand_21.png').convert('RGB').save(OUT / 'official_hand_21.png')
with pdfplumber.open(SRC / 'rtmw_2407.08634.pdf') as pdf:
    # Figure 1 including its caption, page 4 of arXiv:2407.08634.
    pdf.pages[3].crop((85, 62, 511, 175)).to_image(resolution=288).save(OUT / 'rtmw_figure1.png')
print(OUT)
