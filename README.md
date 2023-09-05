# Synesthete: A Deep Learning Engine that Sees Sound

![Visualization of Mel Spectrum, Chroma, and Tonnetz from Audio Data](https://github.com/Burn1n9m4n/brainstation_capstone/blob/48a4685fbf96c189c132e42e276c41f2692c6f6a/reports/images/20230904_readme_splash.png)

### Introduction
**synesthete** ***(sĭn′ĭs-thēt″) - noun***

***A person who experiences synesthesia, a perceptual phenomenon in which stimulation of one sensory or cognitive pathway
leads to involuntary experiences in a second sensory or cognitive pathway.***

There are many music recommendation systems out there and many of them utilize sophisticated algorithms or leverage data 
gathered from the user over time in order to provide realistic suggestions and high accuracy. Synesthete is a unique recommendation 
engine that will provide immediate feedback to a user based on an audio sample using no past information. The method by which 
it will do this is novel as well as it utilizes an audio encoder to change the data from auditory information to a series of 
spectrographic images, which in turn are used as the input for a convolutional neural network (CNN). Track recommendations are 
then provided to the user based on image similarity. 

And thus, Synesthete, is born. A deep learning engine that sees sound.

### The Problem Area
My primary area of interest is entertainment and media related data and within this realm I wanted to investigate methods of 
recommending content to users. Specifically, I wanted to work on some kind of music recommendation engine for the purposes 
of providing users a means of discovering new artists and music. Content recommendation often requires the use of data gathered
over time or from other existing users to make recommendations to new users. However, these methods suffer from the fact
that statistical inference is difficult, if not impossible, to perform for a new user as there is little to no information from 
which to make a prediction, which is also known as the cold start problem. 

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
1.  Gather audio data
2.  Create spectrographic images of audio data via encoder/transformer
3.	Vectorize images into n-dimensional vectors
4.	Collate vectors into singular data set
5.	Train models 
    * Convolutional neural network model (CNN)
    * Standard pairwise cosine similarity
6.	Accept user input
    * i.	Audio recording captured from phone
    * ii.	Audio file provided to model
7.	Convert into image via encoder
    * Or vectorized data
8.	Perform similarity calculation and provide top ten most similar matches
    * i.	Cosine similarity given that two vectors are being compared

The above acted as a detailed guideline rather than a hard and fast plan of action during development. The finalized procedure 
is showing in the Project Flowchart section and is vieweable within the latest Jupyter Notebook in the `notebooks` directory.

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
to get a large enough set of music tracks, the Spotify Kaggle dataset [2] provides a robust list to start with. The dataset contains 
232,725 rows with each row representing a single musical track. Each track has several different features including `duration`, `acousticness`, `danceability`, `energy` and so on. However, these fields are, at least presently, not of interest to this project 
with respect to model training. For this project, the main focus is using the `track_id` feature as a means of uniquely identifying 
and downloading a large enough sample set to train models (one of which is a CNN). Each of these track IDs is concatenated with a base Spotify URL to provide a URL to the `spotify_dl` package, which is a command line tool that can be run via python and executed in bash 
to download tracks. Looping through the IDs automates this process. Eventually, these MP3s will be vectorized into n-dimensional vectors, collated into a singular dataset, and used to train recommendation models.

### Project Flowchart

![Project Flowchart](https://github.com/Burn1n9m4n/brainstation_capstone/blob/0c17f599a4d83926313a69da2ce7353decf85032/reports/images/20230905_project_flow_drawio_light.png)

The general code flow follows the diagram shown above. Further details are described below:
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
The project is currently organized so that all code is within Jupyter Notebooks. This may change in the future 
if the app requires a shift to `.py` files. All notebooks are contained within the `notebooks` directory. Data is 
stored within the `data` directory. However, at present the mp3 audio, vectorized mp3s, and raw numpy files are 
not present in the repo due to potential space limitations. A 'Credits & References' section is compiled within this 
README.md. DEV_NOTES.md is a collection of raw development notes from planning meetings. A tree diagram of the 
project's structure can be found below. It was obtained by running `tree --filelimit 20 -I "*cache*"` in a bash 
prompt. Note that data directories that contain more than 20 files will not be shown nor will any file contianing
the word 'cache', which will remove any `.cache` files or `__pycache__` folders.

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
More information on the individual components within the project can be found below:

* `data` 
    - contains all data generated by code within this project
    - `SpotifyFeatures.csv` is required to get track information [2]
    - `unique_track_ids_for_training.parquet` is a list of unique `track_id` used to train models and can also be 
    used to ensure reproducibility
        - list contains 11,578 unique `track_id`
    - `model_means_df.parquet`contains information on mean similarities for each track for each of the models
    - `cnn_complete_parquets` and `pairwise_complete_parquets` contain all vectorized audio data for use in models.

* `notebooks`
    - contains all final notebooks involved in the project
    - outdated notebooks or notebooks used in development are within the `development` folder

* `reports`
    - contains all sprint reports for this project
    - Also contains `images` directory which houses all images for this repo along with editor files 
    (eg: `.drawio` files)

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `README.md`
    - Project landing page (this page)

* `LICENSE`
    - Project license

### Dataset
The dataset for this project consists of 11,578 unique audio tracks that were obtained through a random sample of the 
`SpotifyFeatures.csv` data obtained from Kaggle. Features were then obtained from the audio data using the `librosa` 
package. The features generated for each model are listed below:
* Normal Pairwise Model
    * MFCC
    * Mel Spectrum
    * Chroma
    * Tonnetz
        * Obtained the mean, max, and min for each
        * Concatenated this information into 1x498 vector
* CNN Model
    * Mel Spectrum
    * Chroma
    * Tonnetz
        * Converted each to `float32`` and stacked to form an image
        * Ran each image as input to ResNet50 model
        * ResNet50 outputs 1x1000 vector
        * Concatenated vectors to create complete dataset

### Credits & References
1. “Chroma Feature.” Wikipedia, Wikimedia Foundation, 27 July 2023, https://en.wikipedia.org/wiki/Chroma_feature.
2. Hamidani, Zaheen. “Spotify Tracks DB.” Kaggle, 23 July 2019, https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db?resource=download.
3. katiagilligan888. “Spotify-Discover-Weekly.” GitHub, https://github.com/katiagilligan888/Spotify-Discover-Weekly. Accessed 21 Aug. 2023. 
4. Lheureux, Adil. “Music Genre Classification Using Librosa and Tensorflow/Keras.” Paperspace Blog, Paperspace Blog, 11 Aug. 2022, https://blog.paperspace.com/music-genre-classification-using-librosa-and-pytorch/.
5. “librosa.feature.chroma_stft.” Librosa.Feature.Chroma_stft - Librosa 0.10.1 Documentation, https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html. Accessed 21 Aug. 2023.
6. “librosa.feature.melspectrogram.” Librosa.Feature.Melspectrogram - Librosa 0.10.1 Documentation, https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html. Accessed 20 Aug. 2023.
7. “Librosa.Feature.Tonnetz.” Librosa.Feature.Tonnetz - Librosa 0.10.1 Documentation, https://librosa.org/doc/latest/generated/librosa.feature.tonnetz.html. Accessed 21 Aug. 2023.
8. “Mel-Frequency Cepstrum.” Wikipedia, Wikimedia Foundation, 14 Aug. 2023, https://en.wikipedia.org/wiki/Mel-frequency_cepstrum.
9. Roberts, Leland. “Understanding the Mel Spectrogram.” Medium, Analytics Vidhya, 17 Aug. 2022, https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53.
10. "Spotify OAuth: Automating Discover Weekly Playlist - Full Tutorial", YouTube, YouTube, 8 Apr. 2023, https://www.youtube.com/watch?v=mBycigbJQzA. Accessed 21 Aug. 2023. 
11. “Tonnetz.” Wikipedia, Wikimedia Foundation, 25 July 2023, https://en.wikipedia.org/wiki/Tonnetz.

--------