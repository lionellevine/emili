from ..abstract import SequentialProcessor
from .. import processors as pr
from . import PreprocessImage
from ..models.classification import MiniXception
from ..datasets import get_class_names
from .keypoints import MinimalHandPoseEstimation


# neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
EMOTION_COLORS = [[255, 0, 0], [45, 90, 45], [255, 0, 255], [255, 255, 0],
                  [0, 0, 255], [0, 255, 255], [0, 255, 0]]


class MiniXceptionFER(SequentialProcessor):
    """Mini Xception pipeline for classifying emotions from RGB faces.

    # Example
        ``` python
        from paz.pipelines import MiniXceptionFER

        classify = MiniXceptionFER()

        # apply directly to an image (numpy-array)
        inference = classify(image)
        ```

     # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``class_names`` and ``scores``.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)

    """
    def __init__(self):
        super(MiniXceptionFER, self).__init__()
        self.classifier = MiniXception((48, 48, 1), 7, weights='FER')
        self.class_names = get_class_names('FER')

        preprocess = PreprocessImage(self.classifier.input_shape[1:3], None)
        preprocess.insert(0, pr.ConvertColorSpace(pr.RGB2GRAY))
        preprocess.add(pr.ExpandDims(0))
        preprocess.add(pr.ExpandDims(-1))
        self.add(pr.Predict(self.classifier, preprocess))
        self.add(pr.CopyDomain([0], [1]))
        self.add(pr.ControlMap(pr.ToClassName(self.class_names), [0], [0]))
        self.add(pr.WrapOutput(['class_name', 'scores']))

    def get_last_hidden_state(self, image): # last hidden state before emotion scores

        # Initialize the preprocessing sequence
        preprocess = PreprocessImage(self.classifier.input_shape[1:3], None)
        preprocess.insert(0, pr.ConvertColorSpace(pr.RGB2GRAY))
        preprocess.add(pr.ExpandDims(0))
        preprocess.add(pr.ExpandDims(-1))

        # Apply the preprocessing steps to the image
        processed_image = preprocess(image)

        # Define the intermediate model to extract the output of the 'add_4' layer
        # This should be defined once, e.g., in the __init__ method for efficiency
        intermediate_layer_model = Model(inputs=self.classifier.input,
                                        outputs=self.classifier.get_layer('add_4').output)

        # Get the last hidden state for the processed image
        last_hidden_state = intermediate_layer_model.predict(processed_image)

        return last_hidden_state # shape [1,6,6,128]


class ClassifyHandClosure(SequentialProcessor):
    """Pipeline to classify minimal hand closure status.

    # Example
        ``` python
        from paz.pipelines import ClassifyHandClosure

        classify = ClassifyHandClosure()

        # apply directly to an image (numpy-array)
        inference = classify(image)
        ```

     # Returns
        A function that takes an RGB image and outputs an image with class
        status drawn on it.
    """
    def __init__(self, draw=True, right_hand=False):
        super(ClassifyHandClosure, self).__init__()
        self.add(MinimalHandPoseEstimation(draw, right_hand))
        self.add(pr.UnpackDictionary(['image', 'relative_angles']))
        self.add(pr.ControlMap(pr.IsHandOpen(), [1], [1]))
        self.add(pr.ControlMap(pr.BooleanToTextMessage('OPEN', 'CLOSE'),
                               [1], [1]))
        if draw:
            self.add(pr.ControlMap(pr.DrawText(), [0, 1], [0], {1: 1}))
        self.add(pr.WrapOutput(['image', 'status']))
