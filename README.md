# Mnist-ONNX-CSharp
Digit recognizer in C#/Javascript, with an ONNX Mnist Machine learning model created in Python.

1. Uses [Python code in a Jupyter notebook](MnistONNX/Onnx%20Mnist%20Generator.ipynb) to generate an Mnist machine learning model file in ONNX format, with training data from the Mnist Dataset and a simple LogisticRegression algorithm.
2. Uses [C# code](MnistONNX/Controllers/HomeController.cs) to load that ONNX model into an ML.NET ONNX InferenceSession variable.
3. Uses [Javascript](MnistONNX/Views/Home/Index.cshtml) to allow users to draw a number in an HTML canvas, and have an AJAX call the model and try to recognize the correct answer.

![](DigitPredict.gif)
