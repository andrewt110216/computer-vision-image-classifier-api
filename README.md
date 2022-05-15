# Welcome to Computer Vision Image Classifier API!

## Introduction
This site represents a **simple** project with a few learning objectives, including:

 - Learn the basics of image classification using computer vision with Pytorch
 - Create a lightweight web application using Flask
 - Deploy an application to the web using Heroku

## How to Use the API
1. Download an image and save it as a JPEG file
2. Next, either:
    - Navigate to the <a href='https://api-cv-image-classifier.herokuapp.com/' target="_blank">Heroku app home page</a>, navigate to the <a href="https://api-cv-image-classifier.herokuapp.com/predict" target="_blank">Predict page</a> to upload your file to the API via your browser, or</li>
    - Send a POST request to https://api-cv-image-classifier.herokuapp.com/predict, using an application like Postman, with the JPEG in the body of the request with the key name: "file".
3. You will receive back a JSON object including the model's top 5 class predictions. For each prediction, you will receive the class name, class ID, and the model's confidence level in the prediction.
---
## Additional Comments
The API uses a pretrained Pytorch model (googlenet) that was trained on ImageNet. No additional training has been done. As a result, the model doesn't, umm...work at all really (maybe that was something I should have said at the top!). This is the reason that I decided to return the top 5 predictions from the model, instead of only the top prediction, along with the confidence levels. Obviously, if you see 5 predictions that don't seem like related classes, each with very low confidence levels, then the model didn't work very well. Sorry :(
    
As a future project, I hope to narrow the focus of the API to a particular subset of image classes (e.g. zoo animals or food items), and use transfer learning to get the model to actually work and return moderately accurate results. But, as much as I'd *love* to dive into that now!...it is outside of the scope of what I set out to do at this time.

Thanks to the Coursera Project Network for guidance on this project.
