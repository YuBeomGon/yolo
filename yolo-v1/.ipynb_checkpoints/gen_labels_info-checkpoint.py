import os
import numpy as np
import glob
from paps import rej_size, rej_table, replace_table
from xml.etree.ElementTree import parse

def generate_labels_info(out_path="./data/labels_info.npy"):
    home_dir_path = "/home/Dataset/Papsmear/original/"
    file_types = ["SS/*/*.jpg", "SS2/*/*.jpg", "SS/*/*.png", "SS2/*/*.png"]

    img_path_list = []
    for ftype in file_types:
        img_path_list.extend(glob.glob(home_dir_path + ftype))
    
    print("Num of Images: {}".format(len(img_path_list)))

    labels_info = {}
    for idx, img_path in enumerate(img_path_list):
        xml_path = img_path[:-3] + "xml"
        if os.path.isfile(xml_path):
            parser = XMLParser(xml_path)
            if (parser.height, parser.width) not in rej_size:
                ID = img_path.split("original/")[-1]            
                labels = parser.objects
                new_labels = []
                for label in labels:
                    cname, xmin, ymin, xmax, ymax = label
                    if cname in rej_table:
                        continue
                    if xmin >= xmax or ymin >= ymax:
                        continue                        

                    if cname in replace_table:
                        cname = replace_table[cname]
#                     new_labels.append([xmin, ymin, xmax, ymax, cname])
                    new_labels.append([xmin, ymin, xmax, ymax, 1.0])
                
                # Add new refined labels
                labels_info[ID] = new_labels

        if idx % 1000 == 0:
            print(idx, img_path)

    np.save(out_path, labels_info)


class XMLParser(object):
    def __init__(self, xml_path):
        self.file_name = ''
        self.width = 0
        self.height = 0
        self.objects = []
        
        tree = parse(xml_path)
        root = tree.getroot()
        
        self.file_name = root.find('filename').text
        self.width = int(root.find('size').find('width').text)
        self.height = int(root.find('size').find('height').text)
        
        objs = root.findall('object')
        
        for obj in objs:
            class_name = obj.find('name').text
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            
            self.objects.append([class_name, xmin, ymin, xmax, ymax])    

if __name__ == '__main__':
    out_path = './data/labels_info.npy'
    generate_labels_info(out_path)

