"""Synthetic level glyphs; no private battle images are stored in tests."""
from PIL import Image, ImageDraw


def badge_image(side=-1):
    image=Image.new('RGB',(15,15),(180,35,65) if side==-1 else (25,105,220))
    draw=ImageDraw.Draw(image)
    for x in (2,8):
        draw.rectangle((x+1,2,x+3,12),fill='white')
        draw.point((x,3),fill='white')
    return image


def scene_with_badge(x=.25,y=.40,side=-1,size=(540,960)):
    image=Image.new('RGB',size,(45,55,45))
    badge=badge_image(side)
    badge=badge.resize((round(15*size[0]/540),round(15*size[0]/540)),Image.Resampling.NEAREST)
    image.paste(badge,(round(x*size[0]-badge.width/2),round(y*size[1]-badge.height/2)))
    return image
