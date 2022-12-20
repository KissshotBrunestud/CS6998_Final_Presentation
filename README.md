# CS6998_Final_Presentation

## Project Description
----------------------------------------------------------------------------
This project implement the DALL-E 2 pytorch model locally based on:\
https://github.com/lucidrains/DALLE2-pytorch \
https://github.com/LAION-AI/dalle2-laion

![DALL-E 2,  OpenAI's updated text-to-image synthesis neural network](https://user-images.githubusercontent.com/27121819/208495737-702a0eaf-5af5-46e8-9836-e0a0d91ea5eb.png)



## Repository Description
----------------------------------------------------------------------------
#### dalle2-laion : this dir contains an implemented inference pipeline and the model pretrained on LAION dataset
#### web-scraper-south-park-images : this dir download images from internet to form a small dataset we prepared to train on. (Reference : https://github.com/maria-ilie/web-scraper-south-park-images)


## Example Commands        
----------------------------------------------------------------------------
### For Inference: 
cd dalle2-laion \
python dream_gradio_inference.py \
python variation_gradio_inference.py \
python inpaint_gradio_inference.py

## Results And Observations  
----------------------------------------------------------------------------
![The user interface example](https://user-images.githubusercontent.com/27121819/208494611-330da5b6-328a-467e-b528-e18df404e7db.png)

We generated several groups of images and here are what we observed:
![Dream1](https://user-images.githubusercontent.com/27121819/208494915-8e55c0c0-21ba-4096-a733-b994a308c727.png)

We can see that the model doesn't perform well on realistic images like the OpenAI model does. For example, it dreams random text from the prompt in some images. And for the cube images, it introduce elements not described like a human. Also the cat dream shows that it sometimes has difficulty identifying numerical concepts like single and plural, and it generates bad image like a cat with 3 eyes.

But surprisingly it performs pretty good with artistic styled prompt, like a sketch of cathedral, an image in picasso style. Even the "realistic  human face" is kinda like from an art picture instead of real life. I guess there are 2 possible reasons for this: 

First, the bias from the LAION dataset. The LAION dataset contains a large portion of the art pictures, and there is even a subset LAION Aesthetics, which a lot of pre-trained model is trained on. It projects from the LAION dataset with a aesthetic threshold that results in most of the images in the dataset being art works. 
Second, the model is not trained enough. Like I mentioned, DALL-E 2 didn't release its model, so we can only find pre-trained model's latest checkpoint that's still under training.

![Dream2](https://user-images.githubusercontent.com/27121819/208494988-def2063e-9895-457e-a559-7394d58946fa.png)

Here we change the condition scale of the 3 parts of the model on the prompt. We find that the second decoder (upsampler) is significant for the AI to generate the image as an cat, and the prior condition scale may not impact that much and there's no need to setting it too high. In some cases, high prior condition scale leads to worse generation.

![Variation](https://user-images.githubusercontent.com/27121819/208495011-063e157a-5c2e-484f-afd4-dab6eb2d1784.png)

Here we can see a bunch of variation outputs. The AI seems to fail at understanding the image and changing the corresponding parts completely, and it maintains most of the elements in the original picture. But it indeed changes some details. Like for corgi the dogs' eyes are becoming more like cats', hedgehog itself becomes more like a rat/cat, and the facial look and height of the kids seem to grow more mature. 

Interestingly, there is no water in the hedgehog original image, but the AI automatically generates water for the image with a boat.

![Inpaint](https://user-images.githubusercontent.com/27121819/208495203-cf11d34e-643c-4215-b2f2-db04d60c466d.png)

The inplaint function performs extremely bad from our pre-trained model, i believe it's because of the model's insufficient training.

## Dataloaders

#### Decoder: Image Embedding Dataset

When training the decoder (and up samplers if training together) in isolation, you will need to load images and corresponding image embeddings. This dataset can read two similar types of datasets. First, it can read a [webdataset](https://github.com/webdataset/webdataset) that contains `.jpg` and `.npy` files in the `.tar`s that contain the images and associated image embeddings respectively. Alternatively, you can also specify a source for the embeddings outside of the webdataset. In this case, the path to the embeddings should contain `.npy` files with the same shard numbers as the webdataset and there should be a correspondence between the filename of the `.jpg` and the index of the embedding in the `.npy`. So, for example, `0001.tar` from the webdataset with image `00010509.jpg` (the first 4 digits are the shard number and the last 4 are the index) in it should be paralleled by a `img_emb_0001.npy` which contains a NumPy array with the embedding at index 509.

Generating a dataset of this type: 
1. Use [img2dataset](https://github.com/rom1504/img2dataset) to generate a webdataset.
2. Use [clip-retrieval](https://github.com/rom1504/clip-retrieval) to convert the images to embeddings.
3. Use [embedding-dataset-reordering](https://github.com/Veldrovive/embedding-dataset-reordering) to reorder the embeddings into the expected format.