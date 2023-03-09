Steps:

Downloaded Brixia score dataset from https://brixia.github.io/ on 2023-03-03
Associated paper: https://www.sciencedirect.com/science/article/pii/S136184152100092X?via%3Dihub#tbl0003

Downloaded Ralo dataset from https://zenodo.org/record/4634000#.ZAJXjHbMKUk on 2023-03-03
Associated paper: https://www.mdpi.com/2076-3417/12/10/4861

Downloaded Cohen ieee8023 dataset from https://github.com/ieee8023/covid-chestxray-dataset on 2023-03-03
Associated paper: https://arxiv.org/abs/2006.11988

The Brixia paper has a subset of the Cohen data where they got their own radiologists (2x) to give Brixia scores. There is an overlap of 65 samples between these data and the original Cohen dataset that has severity scores. I used these 65 to make a linear regressor to convert between brixia and opacity scores.

The Brixia group provides their images in .dcm format, a common medical imaging format. These files have a lot of metadata associated with them and I attempted to extract metadata deemed relevant such as AdmittingDiagnosesDescription, PatientBirthDate, PatientSex and PatientAge but most fields were blank. They also provide a metadata file but some of the columns are unclear (e.g. AgeAtStudyDateFiveYear doesn't seem to be Age).
Note that Brixia .dcm images were either taken in Monochrome1 or Monochrome2 but when extracting images I did not take that into account.

I used the linear regressor to convert Brixia scores into opacity scores.

I did little processing to the Ralo dataset other than to rename columns and combine the scores from the two radiologists into one Global score (GeographicScoreGlobal or OpacityScoreGlobal)

I used the Brixia score annotations of the Cohen dataset to assign "BrixiaScoreGlobal" to samples where applicable. These were not calculated using the linear regressor, and are Brixia "ground truth". I then converted those scores into opacity scores (OpacityScoreGlobalFromBrixia); can be used to check how close to ground truth the converter gets.

Note that the Cohen dataset has examples of other conditions than covid - check the "finding" column for items labeled 'Dataset 1'.


Last, I combined all three datasets, dropped superfluous columns. I then checked that all files had an opacity score associated with them, and made sure to remove images from my Combined images folder that didn't have an opacity score. Then, saved as 'combined_cxr_metadata.csv'.