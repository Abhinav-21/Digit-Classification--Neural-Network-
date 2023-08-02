# Handwritten Digit Classification using Neural Networks

This repository contains a Python implementation of a neural network from scratch for classifying handwritten digits from the MNIST dataset. The neural network is designed to classify digits from 0 to 9, showcasing a fundamental example of image classification using deep learning techniques.

## Project Overview

The goal of this project is to demonstrate the process of building and training a neural network for image classification. The model is trained to accurately classify handwritten digits by learning from the MNIST dataset.

Key features of the project:

- Implementation of a neural network architecture from scratch.
- Utilization of activation functions like ReLU and softmax.
- Backpropagation for optimizing network weights and biases.
- Training using gradient descent optimization.
- Evaluation of the trained model's accuracy on the test dataset.

## Requirements

To run the project, you will need the following:

- Python 3.x
- Numpy
- Pandas
- Matplotlib
- Keras (for dataset access)
- Pickle (for saving and loading trained parameters)
```
pip install -r requirements.txt
```

## How to Use

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the `neural_network_mnist.py` script to train the neural network.
4. The trained parameters (weights and biases) will be saved in `trained_params.pkl`.
5. Use the `make_predictions` and `show_prediction` functions to make predictions and visualize the results.

## Results

The neural network's performance and accuracy are tracked during training. The accuracy is plotted over iterations to showcase the model's learning progress.

## Example Predictions

You can use the trained model to make predictions on the MNIST test dataset. The `show_prediction` function displays images from the test dataset along with their corresponding predicted labels.

## Conclusion

This project demonstrates the implementation of a neural network for handwritten digit classification. It provides insights into the process of building and training neural networks from scratch and showcases the power of deep learning in image recognition tasks.

Feel free to explore, modify, and expand upon this project to further develop your understanding of neural networks and their applications.

## Acknowledgments

The MNIST dataset is a product of the National Institute of Standards and Technology (NIST) and is available through the Keras library.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
