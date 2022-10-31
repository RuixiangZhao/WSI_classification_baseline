import openslide
import os
import xml.dom.minidom as xmldom
import numpy as np
from PIL import Image, ImageDraw

slide_path = '/data1/zrx/Camelyon16/'
xml_path = '/data1/zrx/Camelyon16_xml/'
thumbnail_save_path = 'thumbnail/camelyon16'
level = -2

if not os.path.exists(thumbnail_save_path):
    os.makedirs(thumbnail_save_path)

def GetAnnotatedThumbnail(sldie_id):
    slide = openslide.OpenSlide(os.path.join(slide_path, slide_id+'.tif'))
    document_obj = xmldom.parse(os.path.join(xml_path, slide_id+'.xml'))
    coordinates_obj = document_obj.getElementsByTagName('Coordinates')
    
    # 从xml中读坐标
    coordinates_list = [] # [[(x,y),(x,y)],[(),()]]
    for coordinates in coordinates_obj:
        coordinate_list = [] # [(x,y),(x,y),...]
        coordinate_obj = coordinates.getElementsByTagName('Coordinate')
        for coordinate in coordinate_obj:
            x = float(coordinate.getAttribute('X'))
            y = float(coordinate.getAttribute('Y'))
            coordinate_list.append((x,y))
        coordinates_list.append(coordinate_list)

    thumbnail = slide.get_thumbnail(slide.level_dimensions[level]) # (hight, width, channel)
    thumbnail_draw = ImageDraw.Draw(thumbnail)

    # 在缩略图上画标注线
    mag = slide.level_dimensions[0][0] / slide.level_dimensions[level][0]
    for coordinates in coordinates_list:
        coordinates.append(coordinates[0])
        coordinates = (np.array(coordinates) / mag).astype(int) # 从level0的坐标转化为target_level层的坐标
        
        thumbnail_draw.line(list(coordinates.flatten()), fill='red', width=1)
    thumbnail.save(os.path.join(thumbnail_save_path, sldie_id+'.png'))        

slide_id = 'test_092'
GetAnnotatedThumbnail(slide_id)