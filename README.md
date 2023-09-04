# Synesthete: A Deep Learning Engine that Sees Sound
### Introduction
**synesthete** ***(sĭn′ĭs-thēt″) - noun***

***A person who experiences synesthesia, a perceptual phenomenon in which stimulation of one sensory or cognitive pathway
leads to involuntary experiences in a second sensory or cognitive pathway.***

There are many music recommendation systems out there and many of them utilize sophisticated algorithms or leverage data 
gathered from the user over time in order to provide realistic suggestions and high accuracy. Synesthete is a unique recommendation 
engine that will provide immediate feedback to a user based on an audio sample using no past information. The method by which 
it will do this is novel as well as it utilizes an audio encoder to change the data from auditory information to a spectrographic 
image, which in turn is used as the input for a convolutional neural network (CNN). A recommendations is then provided to 
the user based on image similarity. 

And thus, Synesthete, is born. A deep learning engine that sees sound.

### The Problem Area
My primary area of interest is entertainment and media related data and within this realm I wanted to investigate methods of 
recommending content to users. Specifically, I wanted to work on some kind of music recommendation engine for the purposes 
of providing users a means of discovering new artists and music. Content recommendation often requires the use of data gathered
over time or from other existing users to make recommendations to new users. Moreover, these methods also suffer from the fact
that statistical inference is difficult to perform for a new user as there is little to no information from which to make a 
prediction, which is also known as the cold start problem. 

Within this sphere, I think that Synesthete could help immeasurably as it could provide recommendations immediately to a single
user without requiring data to be gathered about that user beforehand. Moreover, the method that this engine will employ does not 
require other users to have been present on the platform beforehand to act as a basis for comparison with a new user.  As such, 
I think the groups that experience the biggest problems related to this idea are the consumers and the platform owners. Users 
want to find new artists to listen to or discover music that is similar to the songs that they already like. Meanwhile, platform 
owners and content creators want to maintain the engagement of the audience as that translates into more views, subscription 
renewals, new subscribers, and ad revenue.

### Proposed Solution
After performing some research, it seems that a method that is very well suited to this purpose would be a convolutional neural 
network (CNN). In order to setup this neural network the following method is outlined:
1.	Create spectrographic images of audio data via encoder/transformer
2.	Vectorize images into n-dimensional vectors
3.	Collate vectors into singular data set
4.	Train convolutional neural network model (CNN)
5.	Accept user input
    * i.	Audio recording captured from phone
    * ii.	Audio file provided to model
6.	Convert into image via encoder
7.	Perform similarity calculation and provide top five most similar matches
    * i.	Cosine similarity given that two vectors are being compared

While the above seems like a good generalized plan, at present it is not known what challenges or new information will come to 
light. As such, it should be noted that the above is more of a detailed guideline than a hard and fast plan of action at this 
present stage of development.

### The Impact
Looking at the impact from a business standpoint, the creation of a solid recommendation system could end up generating millions 
of dollars in ad revenue. Realistically though, I think this particular project will at best showcase a potential methodology that 
has not yet been considered within a larger organization and could lead to the development of a more optimized model for recommendation 
that is more responsive to and in tune with the users of the platform upon which it is implemented. Moreover, it could also potentially 
provide some additional functionality to an existing app already published by a larger organization, such as Spotify or Apple. Finally, 
I think that the users stand to gain the largest benefit from this engine as it would provide a novel approach to finding new artists 
and music that is not only informative, but also entertaining in and of itself.

### The Data
The data that is being used within this project will have to be created in order to train the model using actual music tracks. In order 
to get a large enough set of music tracks, the Spotify Kaggle dataset [1] provides a robust list to start with. The dataset contains 
232,725 rows with each row representing a single musical track. Each track has several different features including `duration`, `acousticness`, 
`danceability`, `energy` and so on. However, these fields are, at least presently, not of interest to this project. For this project, 
the main focus is using the `track_id` feature as a means of uniquely identifying and downloading a large enough sample set to train a 
CNN. Each of these track IDs is concatenated with a base Spotify URL to provide a URL to the `spotify_dl` package, which is a command 
line tool that can be run via python and executed in bash to download tracks. Looping through the IDs automates this process. Eventually, 
these MP3s will be vectorized into n-dimensional vectors, collated into a singular dataset, and used to train the CNN model.

### Project Flowchart

![Project Flowchart](https://github.com/Burn1n9m4n/brainstation_capstone/blob/598bbe84fba204d2490cb904db80194c0761d129/reports/images/20230905_project_flow.drawio.png)

The general code flow for this project is as follows. Note that this is not final and is subject to change.
1. Ingest Kaggle data
2. Randommally sample 30,000 track_ids from dataset
3. Set while loop with limit of 12,000 and feed each track_id into `spotify_dl` to download audio as `.mp3`
    1. Obtain at least 10,000 tracks for model training
4. Use `librosa` package to extract Mel Spectrum, MFCC, Chroma, and Tonnetz features
5. Extract features for models
    1. Extract mean, min, and max to encode mp3 data into 1x498 vector for pairwise cosine similarity
    2. Stack Mel Spectrum, Chroma, and Tonnetz into image and use ResNet50 to create 1x1000 embedding
6. Calculate pairwise cosine similarity using `pairwise` from `sklearn.metrics`
7. Sort resulting arrays and extract individual similarity vector for a given track (one row within array)
8. Use those numerical indicies within the Kaggle DataFrame to obtain the song and its top 10 matches
9. Create a playlist using those track_ids within an API call for the user
10. With new tracks, repeat steps 3-8.

### Project Organization
The project is current organized so that all code is within Jupyter Notebooks. This may change in the future 
if the app requires a shift to `.py` files. All notebooks are contained within the notebooks directory. Data is 
stored within the data directory. However, at present the mp3 audio, vectorized mp3s, and complete vectorized data 
are not present in the repo due to potential space limitations. Credits and references are compiled within this 
README.md. DEV_NOTES.md is a collection of raw development notes from planning meetings. A tree diagram of the 
project's structure can be found below (Note: Data directories that contain more than 20 files will not be shown):

```
brainstation_capstone
├── CHANGELOG.md
├── DEV_NOTES.md
├── LICENSE
├── README.md
├── data
│   ├── 20230903_model_means_df.parquet
│   ├── 20230903_unique_track_ids_for_training.parquet
│   ├── SpotifyFeatures.csv
│   ├── mp3s  [11922 entries exceeds filelimit, not opening dir]
│   └── vectorized_mp3s
│       ├── cnn_complete_parquets
│       │   └── 20230901_complete_cnn_data.parquet
│       ├── cnn_parquets  [11578 entries exceeds filelimit, not opening dir]
│       ├── pairwise_complete_parquets
│       │   ├── 20230830_complete_vectorized_data_pairwise.parquet
│       │   ├── 20230831_complete_vectorized_data.parquet
│       │   └── 20230901_complete_pairwise_data.parquet
│       ├── pairwise_parquets  [11578 entries exceeds filelimit, not opening dir]
│       └── raw  [46312 entries exceeds filelimit, not opening dir]
├── notebooks
│   ├── 20230905_sprint3_checkpoint.ipynb
│   └── development  [23 entries exceeds filelimit, not opening dir]
├── reports
│   ├── 20230728_synesthete_brainstation_sprint1_vk.pdf
│   └── 20230821_synesthete_brainstation_sprint2_vk.pdf
└── requirements.txt
```

### Walkthrough Demo

...
...
...

* `data` 
    - contains link to copy of the dataset (stored in a publicly accessible Google Drive folder)
    - saved copy of aggregated / processed data as long as those are not too large (> 10 MB)

* `model`
    - joblib dump of final model / model object

* `notebooks`
    - contains all final notebooks involved in the project

* `reports`
    - contains final report which summarises the project

* `references`
    - contains papers / tutorials used in the project

* `src`
    - Contains the project source code (refactored from the notebooks)

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `capstine_env.yml`
    - Conda environment specification

* `Makefile`
    - Automation script for the project

* `README.md`
    - Project landing page (this page)

* `LICENSE`
    - Project license

### Dataset

...
...
...

### Credits & References
1. “Chroma Feature.” Wikipedia, Wikimedia Foundation, 27 July 2023, en.wikipedia.org/wiki/Chroma_feature.
2. Hamidani, Zaheen. “Spotify Tracks DB.” Kaggle, 23 July 2019, www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db?resource=download.
3. katiagilligan888. “Spotify-Discover-Weekly.” GitHub, github.com/katiagilligan888/Spotify-Discover-Weekly. Accessed 21 Aug. 2023. 
4. Lheureux, Adil. “Music Genre Classification Using Librosa and Tensorflow/Keras.” Paperspace Blog, Paperspace Blog, 11 Aug. 2022, blog.paperspace.com/music-genre-classification-using-librosa-and-pytorch/.
5. “librosa.feature.chroma_stft.” Librosa.Feature.Chroma_stft - Librosa 0.10.1 Documentation, librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html. Accessed 21 Aug. 2023.
6. “librosa.feature.melspectrogram.” Librosa.Feature.Melspectrogram - Librosa 0.10.1 Documentation, librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html. Accessed 20 Aug. 2023.
7. “Librosa.Feature.Tonnetz.” Librosa.Feature.Tonnetz - Librosa 0.10.1 Documentation, librosa.org/doc/latest/generated/librosa.feature.tonnetz.html. Accessed 21 Aug. 2023.
8. “Mel-Frequency Cepstrum.” Wikipedia, Wikimedia Foundation, 14 Aug. 2023, en.wikipedia.org/wiki/Mel-frequency_cepstrum.
9. Roberts, Leland. “Understanding the Mel Spectrogram.” Medium, Analytics Vidhya, 17 Aug. 2022, medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53.
10. “Tonnetz.” Wikipedia, Wikimedia Foundation, 25 July 2023, en.wikipedia.org/wiki/Tonnetz.
11. YouTube, YouTube, 8 Apr. 2023, https://www.youtube.com/watch?v=mBycigbJQzA. Accessed 21 Aug. 2023. 

--------