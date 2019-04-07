## Image Captioning

The goal of image captioning is to convert a given input image into a sentence that is comprehensible
and natural in a language such as English. This is difficult because it combines two areas of AI resarch in which are 
at the cutting edge of technology; both CNNs (Convolutional Neural Networks) and RNNs (Recurrent Neural Networks). Both 
neural network architectures excel at different things; Convolutional Neural Networks excel and extracting complex
hierarchies of patterns from data (often, even if said patterns don't even exist), which is why they're the most common 
architectures used for image recognition and other complex tasks like segmentation (drawing 'bounding' boxes around
different objects in a given image or video), whereas Recurrent Neural Networks (sometimes referred to as LSTMs) form
from an architecture that deals exceptionally well with time-sequenced data like natural language. Often times speech 
recognition networks will take in the input as a 'spectograph' and pass it into such a network. Let's take a look at the 
architectures that you will find in this specific project.


### Recurrent Neural Network

The basic building block of the RNN is the memory cell - or LSTM. 

![title](/imgs/lstmblock.png)
