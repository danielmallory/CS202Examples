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

'Regular' or fully-connected neural netowrks have immense computing power, but they suffer from the 'memoryless' property;
by assuming each input and output is independent, we ignore and lose information that can be encoded in temporal/sequential 
data. For many tasks, like natural language processing, it is important that the network knows and keeps track of what words
came before it; natural language syntax is what distinguishes something from being spoken by a baby and an adult. RNNs are 
neural networks that keep track of just that; they perform the same computations on each element of their input, with each 
step depending on the preceding sequence for its specific weighting and calculi. You can think of RNNs as having a stored
'memory' in each cell which stores information about what has been calculated so far, or what information it has sent 
and received up until that point. The most widely used model of an RNN involves the 'basic' building block called an LSTM-
or the 'Long Short-Term Memory':


![title](https://github.com/danielmallory/CS202Examples/blob/master/CNN-LSTM/imgs/lstmblock.png)
