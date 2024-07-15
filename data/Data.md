### Dataset Download

**[MicroLens]** Please refer to this repo: https://github.com/westlake-repl/MicroLens, select **MicroLens-100k-Dataset** and download the **raw videos** with their **titles (en)**.

### Data Preprocess

#### Create data.pkl

Create a dataframe in pickle format, named data.pkl, with the following columns:

- item_id : This columns is the id of each micro-video, which can be the name of each video without '.mp4' suffix.
- text: This columns is the title of each micro-video.
- label: The label popularity here is defined as the number of total comments for a micro-video.

#### Data preprocess

##### Feature extraction

Run the **video_frame_capture.py** for each raw micro-videos obtain video frames, here the default number of frames is 10 for each video.

Then run the **textual_engineering.py**, **visual_engineering.py** to extract the features and add these two features to data.pkl named **visual_feature_embedding_cls** and **textual_feature_embedding**, respectively.

##### Retrieval Preprocess

Run the **image_to_text_multi_threads.py** to obtain the image caption for each frame of a micro-video. Then concat all frame captions of a micro-video as a simple "video caption" for each micro-video.

Then run the **text_semantic_embedding.py** to obtain the retrieval vector for each micro-video. Here the text for each micro-video is the concatenation of the "title" and the "video caption". Then add this to the data.pkl with column name retrieval_feature.

#### Split the dataset

Run the **dataset_split.py** to obtain train, test and valid pickle files.

#### Retrieval

Run the retriever.py to do retrieval process. After this step, all the data preprocessing is finished.

