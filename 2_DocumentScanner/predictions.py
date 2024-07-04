import numpy as np
import pandas as pd
import cv2
import pytesseract
import spacy
import re
import string
from glob import glob
import warnings

warnings.filterwarnings('ignore')

model_ner = spacy.load('output/model-best/')


def clean_text(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    table_whitespace = str.maketrans('', '', whitespace)
    table_punctuation = str.maketrans('', '', punctuation)

    text = str(txt)
    # text = text.lower()
    remove_whitespace = text.translate(table_whitespace)
    remove_punctuation = remove_whitespace.translate(table_punctuation)

    return str(remove_punctuation)


# group the label
class GroupGen:
    def __init__(self):
        self.id = 0
        self.text = ''

    def get_group(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id


ggen = GroupGen()


def parser(text, label):
    if label == 'PHONE':
        text = text.lower()
        text = re.sub(r'\D', '', text)
    elif label == 'EMAIL':
        text = text.lower()
        allow_chars = '@_.\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_chars), '', text)
    elif label == 'WEB':
        text = text.lower()
        allow_chars = ':/.%#\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_chars), '', text)
    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        text = text.title()
    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9]', '', text)
        text = text.title()
    return text


def get_predictions(image):
    tess_data = pytesseract.image_to_data(image)

    # Convert into df
    tess_list = list(map(lambda x: x.split('\t'), tess_data.split('\n')))
    df = pd.DataFrame(tess_list[1:], columns=tess_list[0])
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(clean_text)

    # Convert df into content
    df_clean = df.query('text != "" ')
    content = " ".join([w for w in df_clean['text']])

    # Get prediction from NER model
    doc = model_ner(content)

    # Convert doc into json
    doc_json = doc.to_json()
    doc_text = doc_json['text']

    # Creating tokens
    df_tokens = pd.DataFrame(doc_json['tokens'])
    df_tokens['token'] = df_tokens[['start', 'end']].apply(lambda x: doc_text[x[0]:x[1]], axis=1)

    right_table = pd.DataFrame(doc_json['ents'])[['start', 'label']]
    df_tokens = pd.merge(df_tokens, right_table, how='left', on='start')
    df_tokens.fillna('O', inplace=True)

    # Join label to df_clean
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
    df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)

    # Inner join with start
    df_info = pd.merge(df_clean, df_tokens[['start', 'token', 'label']], how='inner', on='start')

    # Bbox
    bb_df = df_info.query("label != 'O' ")
    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
    bb_df['group'] = bb_df['label'].apply(ggen.get_group)

    # Right and bottom of bbox
    bb_df[['left', 'top', 'width', 'height']] = bb_df[['left', 'top', 'width', 'height']].astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']

    # Tagging
    col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
    group_tag_img = bb_df[col_group].groupby(by='group')
    img_tagging = group_tag_img.agg({
        'left': min,
        'right': max,
        'top': min,
        'bottom': max,
        'label': np.unique,
        'token': lambda x: " ".join(x)
    })

    img_bb = image.copy()
    for l, r, t, b, label, token in img_tagging.values:
        cv2.rectangle(img_bb, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(img_bb, label[0], (l, t), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    # Entities
    info_arr = df_info[['token', 'label']].values
    entities = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
    previous = 'O'

    for token, label in info_arr:
        bio_tag = label[0]
        label_tag = label[2:]

        # Parse token
        text = parser(token, label_tag)
        if bio_tag in ('B', 'I'):
            if previous != label_tag:
                entities[label_tag].append(text)
            else:
                if bio_tag == 'B':
                    entities[label_tag].append(text)
                else:
                    if label_tag in ("NAME", "ORG", "DES"):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text
        previous = label_tag

    return img_bb, entities
