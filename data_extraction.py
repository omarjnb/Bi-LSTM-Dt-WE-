import os
import pandas as pd
from constants import *
import xml.etree.ElementTree as ET


def aspect_type(row):
    if row=='Not Available':
        return 'implicit'
    else:
        return 'explicit'

def get_categ(row):
    categ_dict = row[0]
    return categ_dict['category']

def extract_from_xml(xml_data_path):

    tree = ET.parse(xml_data_path)
    root = tree.getroot()
    elements = ['text','aspectTerms','aspectCategories']
    data_dict = {}
    for element in elements:
        for subroot in root:
            children = subroot.getchildren()
            if len(children)==3:
                for child in children:
                    if child.tag=='text':
                        if child.tag not in data_dict.keys():
                            data_dict[child.tag] = []
                            data_dict[child.tag].append(child.text)
                        else:
                            data_dict[child.tag].append(child.text)

                    elif child.tag == 'aspectTerms':
                        aspects = child.getchildren()
                        aspect_list = []
                        for aspect in aspects:
                            aspect_list.append(aspect.attrib)

                        if child.tag not in data_dict.keys():
                            data_dict[child.tag] = []
                            data_dict[child.tag].append(aspect_list)
                        else:
                            data_dict[child.tag].append(aspect_list)

                    elif child.tag == 'aspectCategories':
                        categories = child.getchildren()
                        category_list = []
                        for category in categories:
                            category_list.append(category.attrib)

                        if child.tag not in data_dict.keys():
                            data_dict[child.tag] = []
                            data_dict[child.tag].append(category_list)
                        else:
                            data_dict[child.tag].append(category_list)
            else:
                for child in children:
                    if child.tag=='text':
                        if child.tag not in data_dict.keys():
                            data_dict[child.tag] = []
                            data_dict[child.tag].append(child.text)
                        else:
                            data_dict[child.tag].append(child.text)

                    elif child.tag == 'aspectCategories':
                        categories = child.getchildren()
                        category_list = []
                        for category in categories:
                            category_list.append(category.attrib)

                        if child.tag not in data_dict.keys():
                            data_dict[child.tag] = []
                            data_dict[child.tag].append(category_list)
                        else:
                            data_dict[child.tag].append(category_list)

                if 'aspectTerms' not in data_dict.keys():
                    data_dict['aspectTerms'] = []
                    data_dict['aspectTerms'].append('Not Available')
                else:
                    data_dict['aspectTerms'].append('Not Available')


    df = pd.DataFrame(data_dict)

    return df

def preprocess_df(df):

    df = df.copy()
    df['aspectType'] = df['aspectTerms'].apply(aspect_type)
    df['numberCategory'] = df['aspectCategories'].apply(lambda row:len(row))
    df = df[df['numberCategory']==1]
    df['aspectCategory'] = df['aspectCategories'].apply(get_categ)
    return df

def extract_df_from_xml(xml_data_path):

    df = extract_from_xml(xml_data_path)
    return df

#extract = extract_df_from_xml('C:/Users\Owner\OneDrive - Universiti Sains Malaysia\sentiment-analysis\coding 2021\project_aspect-implicit-aspect-extraction-(BiLSTM_GloVe) - R_15\project_aspect\data\Restaurants_Train.xml')
#print(extract)