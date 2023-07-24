## Meeting with Paul 7/13/2023
* Concept of distance between two elements
    * Distance between items or users
    * What would make my Spotify related recommender different?
* Things that are more interesting
    * Word embeddings - arithmatic on words and sentences
    * Recommendation engine - Based on cover art (How would we validate this?)
        * Could use a CNN
        * How many playlists do I find these songs embedded on? - Expected value vs Actual
            * Co-occurences
        * Could also do a CNN for audio embedding
* Check if API has the ability to recommend ideas as a point of comparison
* They would send course material to me early so that I could get to work
* Data would end up in a latent space defined entirely by me
* Hugging face transformers for sound and video libraries
* Next Steps: 
    * If we wanted to stick to Spotify
    * Create a DataFrame with the links to audio data or album art
    * 10,000 songs to start for some Deep Learning/CNN applications
    * Check Spotify dataset on Kaggle to check what information is already out there
        * Data set of 10,000 most popular songs on Spotify, for example
        * datasetsearch on Google

## Meeting with Paul 7/21/2023
* Possible questions to ask?
    * Does cover art predict tempo?
    * Using song intros as a means of recommending other songs?
        * "Give me a track that sounds like this without it being the same song."
        * Take data
        * Find embeddings within the audio and see how they match to dataset
        * Embedding meaning that you break down the audio into a n-dimensional vector (50 for example)
            * Could use auto encoding
        * Then try to map the encoding of a song to another song
        * Could then use cosine similarity and then match to closest approximation
        * We could pass in the audio data using either a link to the audio data or via someone playing the sound via a phone
        * Transfer learning - could check what CNNs exist for audio
        * Auto Encoder for a data - it would take a while
        * Could look for pre-trained transformer 
            * Hugging face libraries - they might have something for music recognition or genre recognition
                * Maps audio to image
        * I could consult the dataset I have to check what was recommended vs what was passed in.
        * Because of the use of vectors you can add and subtract
        * LAME or FFMPEG for python
            * Encoder
            * curl | mmpeg...etc
* **Action Items**: Try to get about 100 of the audio clips from Spotify API
    * This would involve getting the links to the audio
    * https://huggingface.co/learn/audio-course/chapter0/introduction
    * Try to go through that course and pull music for first sprint

## Sprint 1 Submission Details 7/24/2023
* README.md should have a description of the project at this stage including the following:
    * Info on the data set (rows, column explanations)
    * This could be relevant to my project: https://github.com/hjoneal/spotify-moodgrid-playlister/blob/main/README.md
* Jupyter Notebook
    * Make sure the notebook is reproducible
        * Include link to data if it is very big
    * Don't include digressions
    * Data to build preliminary model 
        * Even if it's just a preliminary model that's kind of lousy

## Meeting with Paul 7/24/2023
* Instructions to get audio:
    * python -m pip install spotify_dl
    * put in secrets and token
    * spotify_dl -l your_track_id
    * conda install -c conda-forge ffmpeg (installs ffmpeg)
    * Needed to use `export` to put in `SPOTIPY_CLIENT_ID` and `SPOTIPY_CLIENT_SECRET`
    * command line code to get track:
        * spotify_dl -l "https://open.spotify.com/track/28Ct4qwkQXY2W5yyNCLuVI" -o .
        * The hash here is a track ID. 
    * place master dir
    * come up with code to extract from directory and rename using trackid

