# Synesthete: A Music Recommendation System
============================================

### Introduction
**synesthete** ***(sĭn′ĭs-thēt″) - noun***

***A person who experiences synesthesia, a perceptual phenomenon in which stimulation of one sensory or cognitive pathway
leads to involuntary experiences in a second sensory or cognitive pathway.***

There are many music recommendation systems out there and many of them utilize sophisticated algorithms to achieve high 
accuracy or leverage data gathered from the user over time in order to provide realistic suggestions. Synesthete is a
unique recommendation engine that will provide immediate feedback to a user based on a audio sample provided immediately
using no past information. The method by which it will do this is novel as well as it utilizes an audio encoder to change
the data from auditory information to a spectrographic image. This image is then used as the input for a convolutional neural
network (CNN) to provide recommendations to the user based on image similarity. 

And thus, Synesthete, is born. A deep learning engine that sees sound.

### The Problem Area
My primary area of interest is entertainment and media related data and within this realm I wanted to investigate methods of recommending content to users. Specifically, I wanted to work on some kind of music recommendation engine for the purposes of providing users a means of discovering new artists and music. Content recommendation often requires the use of data gathered over time or from other existing users to make recommendations to new users. Moreover, these methods also suffer from the fact that inference is difficult to perform if the user is new which is also known as the cold start problem. 

Within this sphere, I think that Synesthete could help immeasurably as it could provide recommendations immediately to a single user without requiring data to be gathered about that user beforehand. Moreover, the method that this engine will employ also does not require other users to have been present on the platform beforehand to act as a basis for comparison with a new user.  As such, I think the groups that experience the biggest problems related to this idea are the consumers and the platform owners. Users want to find new artists to listen to or discover music that is similar to the songs that they already like. Meanwhile, platform owners and content creators want to maintain the engagement of the audience as that translates into more views, subscription renewals, new subscribers, and ad revenue.

### Proposed Solution
After performing some research, it seems that a method that is very well suited to this purpose would be a convolutional neural network (CNN). In order to setup this neural network the following method is outlined:
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

While the above seems like a good generalized plan, at present it is not known what challenges or new information will come to light. As such, it should be noted that the above is more of a detailed guideline than a hard and fast plan of action at this present stage of development.

### The Impact
Looking at the impact from a business standpoint, the creation of a solid recommendation system could end up generating millions of dollars in ad revenue. Realistically though, I think this particular project will at best showcase a potential methodology that has not yet been considered within a larger organization and could lead to the development of a more optimized model for recommendation that is more responsive to and in tune with the users of the platform upon which it is implemented. Moreover, it could also potentially provide some additional functionality to an existing app already published by a larger organization, such as Spotify or Apple. Finally, I think that the users stand to gain the largest benefit from this engine as it would provide a novel approach to finding new artists and music that is not only informative, but also entertaining in and of itself.

### The Data
The data that is being used within this project will have to be created in order to train the model from actual music tracks. In order to get a large enough list of music tracks, the Spotify Kaggle data set [1] provides a robust list to start with. The data set contains 232,725 rows with each row representing a single musical track. Each track has several different features including duration, acousticness, danceability, energy and so on. However, these fields are, at least presently, not of interest to this project. For this project, the main focus is using the track_id feature as a means of uniquely identifying and downloading a large enough sample set to train a CNN. Each of these track IDs is concatenated with a base Spotify URL and then into the spotify_dl package. This provides a command line tool that can be run via python and executed in bash to download tracks. Looping through the IDs automates this process. Eventually, these MP3s will be vectorized into n-dimensional vectors, collated into a singular dataset, and used to train the CNN model.

### References
1. Hamidani, Z. (2019, July 23). Spotify tracks DB. Kaggle. https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db?resource=download 



### Walkthrough Demo

...
...
...

### Project Flowchart

...
...
...

### Project Organization

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

...
...
...

--------