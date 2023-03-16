#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pydicom
from pydicom import dcmread
import dicom2jpg
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image


# ### Determine mapping between brixia score and severity score

# In[2]:


def convert_severity_score(input_score,convert_to_opacity=True):
    """
    Takes a brixia covid severity score and converts it to opacity score, or vice versa.
    Uses linear regression calculated from a set of 65 samples that have been scored both for brixia and in cohen ieee8023.
    """
    # Open brixia score annotations of cohen dataset
    brixia = pd.read_csv("./Datasets/Brixia/Brixia-score-COVID-19-master/Brixia-score-COVID-19-master/data/public-annotations.csv").set_index('filename')
    # Average Brixia scores of the two radiologists
    brixia['brixia_mean'] = brixia[['S-Global','J-Global']].mean(axis=1)

    # Open cohen opacity scores of cohen dataset
    cohen = pd.read_csv("./Datasets/ieee8023/covid-severity-scores.csv", skiprows=5).set_index('filename')
    
    # Merge datasets on 'filename'
    merged = pd.merge(brixia,cohen,on='filename',how='outer')
#     print(f'merged size: {len(merged)}')
    # Drop all rows that are not in common
    merged = merged.dropna(axis = 'index')
#     print(f'merged size after dropna: {len(merged)}')
#     print(merged)

    from sklearn.linear_model import LinearRegression

    ## Map brixia score to opacity ##

    # Initialize linear regressor
    brixia_to_opacity = LinearRegression()
    
    # Fit with brixia score as X and opacity score as y
    brixia_to_opacity.fit(merged['brixia_mean'].values.reshape(-1,1), 
                          merged['opacity_mean'].values.reshape(-1,1))

    # Get coefficient and intercept
#     print(f"b -> o | coef: {brixia_to_opacity.coef_}, intercept: {brixia_to_opacity.intercept_}")
    
    ## Map opacity to brixia score ##

    # Initialize linear regressor
    opacity_to_brixia = LinearRegression()

    # Fit with opacity score as X and brixia score as y
    opacity_to_brixia.fit(merged['opacity_mean'].values.reshape(-1,1),
                          merged['brixia_mean'].values.reshape(-1,1))

    # Get coefficient and intercept
#     print(f"o -> b | coef: {opacity_to_brixia.coef_}, intercept: {opacity_to_brixia.intercept_}")

    # Use coefficient and intercept to calculate converted score (y = mx+b)
    if convert_to_opacity:
        converted_score = brixia_to_opacity.coef_ * input_score + brixia_to_opacity.intercept_
    else:
        converted_score = opacity_to_brixia.coef_ * input_score + opacity_to_brixia.intercept_
        
    return round(float(converted_score),2)


# ### Extract info from .dcm files (brixia)

# In[3]:


def extract_dcm_info(folder_path, keywords, convert_to_jpg=False):
    """
    Extracts selected info from .dcm files in a given folder.
    If convert_to_jpg is True, also converts the file to .jpg format.
    
    Inputs: 
    folder_path: target directory containing .dcm files
    keywords: csv file containing keywords under the heading "Keywords"
    
    Returns:
    pandas dataframe containing the given info
    """
    # Get a list of files in the target directory
    files = os.listdir(folder_path)

    # Set up a dataframe to store filename + all keywords
    column_names = ['filename']+list(keywords['Keyword'])
    df = pd.DataFrame(columns = column_names)

    # Go through each .dcm file in target folder (folder_path)
    for filename in files:
        if filename[-4:] == '.dcm':
            # Set up a dictionary to add as a row to the dataframe
            df_row = {}
            # Set the filename column as the file name, depending on convert_to_jpg
            if convert_to_jpg:
                df_row['filename'] = f"{filename[:-4]}.jpg"
            else:
                df_row['filename'] = f"{filename}"
            # Read the file
            ds = dcmread(folder_path + filename)
            # Add each keyword value to the dataframe, if present
            for keyword in keywords['Keyword']:
                # Try-catch block in case one of the keyword fields is not encoded in the .dcm
                try:
                    # If no value is present, make it nan
                    if ds[keyword].value == "":
                        df_row[keyword] = np.nan
                    # Otherwise, give it the value
                    else: 
                        df_row[keyword] = ds[keyword].value
                # If there isn't that keyword in the .dcm image
                except:
                    pass
            # Add the row to the dataframe
            df = pd.concat([df, pd.DataFrame.from_records([df_row])])
            
            if convert_to_jpg:
                # Make folder for jpgs
                newpath = os.path.join(folder_path,"jpegs")
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                # Save the image as a jpg after scaling
                new_image = ds.pixel_array.astype(float)
                    
                scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
                scaled_image = np.uint8(scaled_image)
                final_image = Image.fromarray(scaled_image)
                final_image.save(os.path.join(newpath,f"{filename[:-4]}.jpg"))
   
    # Save a copy of the dataframe in the target directory
    df.to_csv(f"{folder_path}/images_info.csv")
    
    return df


# In[4]:


# Define the target directory
folder_path = "./Datasets/Brixia/dicom/dicom_clean/"
# Get a list of keywords to extract from the .dcm "image"
keywords = pd.read_csv("./Datasets/Brixia/dicom_codes.csv")
# Extract!
extracted = extract_dcm_info(folder_path, keywords, convert_to_jpg = True)

# Note: CR = Computed Radiography, DX = Digital Radiography

# Open Brixia metadata file
brixia_metadata = pd.read_csv('./Datasets/Brixia/metadata_global_v2.csv', sep=';')

# Merge extracted metadata with Brixia metadata file, based on filename
brixia_metadata['filename'] = brixia_metadata['Filename'].str[:-4] + '.jpg'
brixia_metadata = brixia_metadata.merge(extracted.drop(columns=["Modality"]), 
                                        how = 'outer', 
                                        on = 'filename')

# Drop columns where all elements are NaN, and the BrixiaScore column which has confounding info with the BrixiaGlobalScore
brixia_metadata = brixia_metadata.dropna(axis=1, how='all').drop(columns=['BrixiaScore'])

# Assign an opacity score to each sample based on brixia->opacity linear regression
brixia_metadata['OpacityScoreGlobal'] = brixia_metadata['BrixiaScoreGlobal'].apply(lambda x: convert_severity_score(x,convert_to_opacity=True))

# Give all items in the brixia dataset a "2"
brixia_metadata['Dataset'] = 2

# Rename columns to be more consistent with cohen data
brixia_metadata.rename(columns={'StudyDate':'date',
                                'Subject':'patientid',
                               'Modality': 'modality',
                               'ViewPosition': 'view',
                               'InstitutionName': 'location'}, inplace = True)


# In[5]:


brixia_metadata


# ### Process RALO metadata

# In[6]:


# Helper function to batch rename (add a name to) files
def prepend_info(folder_path,addition):
    """
    Adds a specified string onto the front of a filename for each file in a directory.
    """
    files = os.listdir(folder_path)
    # Touch each file
    for f in files:
        # Get its old name
        old_name = os.path.join(folder_path,f)
        # The new name is the old name with the addition tacked on the front
        new_name = os.path.join(folder_path,f'{addition}{f}')
        # Rename the file with the new name
        os.rename(old_name, new_name)


# In[7]:


# Process RALO images so they have a more descriptive name
folder_path = './Datasets/RALO/CXR_images_scored'
# prepend_info(folder_path, 'ralo_sbm_')

# Get ralo metadata
ralo_metadata = pd.read_csv('./Datasets/RALO/ralo-dataset-metadata.csv', skiprows=1)


# In[9]:


# Add filename to ralo metadata and average the total geographic and opacity scores
ralo_metadata['filename'] = [f'ralo_sbm_{i}.jpg' for i in range(len(ralo_metadata))]
ralo_metadata['StudyDate'] = ralo_metadata['Exam_DateTime'].str[:8]
ralo_metadata['GeographicScoreGlobal'] = ralo_metadata[['Total GEOGRAPHIC','Total GEOGRAPHIC.1']].mean(axis=1)
ralo_metadata['OpacityScoreGlobal'] = ralo_metadata[['Total OPACITY','Total OPACITY.1']].mean(axis=1)

# Give all items in the ralo dataset a "3"
ralo_metadata['Dataset'] = 3

## Problem: RALO opacity is scored (0-8) not (0-6)

# Get the ralo dataset (#3) opacity score values 
to_scale = ralo_metadata.loc[ralo_metadata['Dataset'] == 3, 'OpacityScoreGlobal']
# minmax scale them to (0-1)
to_scale = (to_scale - to_scale.min()) / (to_scale.max() - to_scale.min())
# Rescale to (0-6)
to_scale = to_scale*6
# Save values in appropriate column
ralo_metadata.loc[ralo_metadata['Dataset'] == 3, 'OpacityScoreGlobal'] = to_scale

# rename columns to be more consistent with cohen
ralo_metadata.rename(columns={'StudyDate':'date',
                                'Subject_ID':'patientid',
                               'Notes' : 'clinical_notes',
                               'Notes.1': 'other_notes'}, inplace = True)


# ### Process Cohen ieee8023 metadata

# In[10]:


# Get metadata file from Cohen ieee8023 dataset
cohen_metadata = pd.read_csv("./Datasets/ieee8023/metadata.csv")

# Get associated geo and opacity scores from cohen dataset
cohen_severity = pd.read_csv('./Datasets/ieee8023/covid-severity-scores.csv', skiprows = 5)

# Brixia study did brixia scores for a subset of cohen samples - load these
brixia_cohen_annotated_severity = pd.read_csv("./Datasets/Brixia/Brixia-score-COVID-19-master/Brixia-score-COVID-19-master/data/public-annotations.csv")

# Take the mean of the two brixia radiologists
brixia_cohen_annotated_severity['BrixiaScoreGlobal'] = brixia_cohen_annotated_severity[['S-Global','J-Global']].mean(axis=1)

# Give the brixia-annotated samples an opacity value from the logistic regression converter calculated above
brixia_cohen_annotated_severity['OpacityScoreGlobalFromBrixia'] = brixia_cohen_annotated_severity['BrixiaScoreGlobal'].apply(lambda x: convert_severity_score(x,convert_to_opacity=True))

# Rename the cohen severity geographic_mean and opacity_mean as scores
cohen_severity.rename(columns={'geographic_mean':'GeographicScoreGlobal',
                                'opacity_mean':'OpacityScoreGlobal'}, inplace = True)

# Merge cohen dataset with new brixia score info and converted opacity score
cohen_metadata = cohen_metadata.merge(cohen_severity, 
                                      how = 'outer', 
                                      on = 'filename')
cohen_metadata = cohen_metadata.merge(brixia_cohen_annotated_severity, 
                                      how = 'outer', 
                                      on = 'filename')

# Give all items in the cohen dataset a "1"
cohen_metadata['Dataset'] = 1


# ### Combine all three metadata files (Brixia, RALO, Cohen)

# In[11]:


metadata = pd.concat([cohen_metadata,ralo_metadata,brixia_metadata], ignore_index=True)


# In[13]:


# Drop superfluous columns
metadata = metadata.drop(columns = ['folder', 'doi', 'url',
       'license', 'Unnamed: 29',
       'S-A', 'S-B', 'S-C', 'S-D', 'S-E',
       'S-F', 'S-Global', 'J-A', 'J-B', 'J-C', 'J-D', 'J-E', 'J-F', 'J-Global',
          'Exam_DateTime', 'Right GEOGRAPHIC', 'Right OPACITY', 'Left GEOGRAPHIC',
       'Left OPACITY', 'Total GEOGRAPHIC', 'Total OPACITY',
       'Right GEOGRAPHIC.1', 'Right OPACITY.1', 'Left GEOGRAPHIC.1',
       'Left OPACITY.1', 'Total GEOGRAPHIC.1', 'Total OPACITY.1',
       'Delta Geo-total', 'Delta Opa total', 'Filename', 'Columns', 'Rows',  'StudyId', 
        'AcquisitionDate'])
# switch order of columns so we have filename, then scores, then other metadata
neworder = ['filename','OpacityScoreGlobal','GeographicScoreGlobal','BrixiaScoreGlobal',
       'OpacityScoreGlobalFromBrixia', 'Dataset', 'patientid', 'offset', 'sex', 'age', 'finding', 'RT_PCR_positive',
       'survival', 'intubated', 'intubation_present', 'went_icu',
       'in_icu', 'needed_supplemental_O2', 'extubated', 'temperature',
       'pO2_saturation', 'leukocyte_count', 'neutrophil_count',
       'lymphocyte_count', 'view', 'modality', 'date', 'location',
        'clinical_notes', 'other_notes',  'Manufacturer',
       'PhotometricInterpretation', 'ConsensusTestset', 'Sex',
       'AgeAtStudyDateFiveYear', 'TableMotion', 'TableAngle']
metadata = metadata.loc[:,neworder]

# Get only rows where we have an opacity score (drop rows that are na)
metadata = metadata[metadata['OpacityScoreGlobal'].notna()]


# In[14]:


# remove any images that we don't have data for from the Combined folder
to_keep = list(metadata['filename'])
folder = './Datasets/Combined/images'
for image in os.listdir(folder):
    if image not in to_keep:
        os.remove(os.path.join(folder,image))


# In[15]:


# Save metadata dataframe as a csv
metadata.to_csv('combined_cxr_metadata.csv')

