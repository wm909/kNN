import java.util.Arrays;

// Basic neuron class
class NeuronN {
    double[] weights;
    double bias;

    public NeuronN(int inputSize) {
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = Math.random();
        }
        bias = Math.random();
    }

    public double activate(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sigmoid(sum);
    }

    public void updateWeights(double[] inputs, double error, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * error * inputs[i];
        }
        bias += learningRate * error;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public String toString() {
        return Arrays.toString(weights) + " " + bias;
    }
}

// Simple neural network with 2 input neurons, 2 hidden neurons, 1 output neuron
class SimpleNeuralNetwork {
    NeuronN hidden1;
    NeuronN hidden2;
    NeuronN output;

    public SimpleNeuralNetwork(int inputSize) {
        hidden1 = new NeuronN(inputSize);
        hidden2 = new NeuronN(inputSize);
        output = new NeuronN(2); // 2 Hidden-Ausgaben
    }

    public String neuronDisplay(){
       String weights = hidden1.toString();
       String weights2 = hidden1.toString();
       String output2 = output.toString();
       return "w1" + weights +
               " w2" + weights2 + " " +
               "OP" + output2;
    }


    public double forward(double[] input) {
        double out1 = hidden1.activate(input);
        double out2 = hidden2.activate(input);
        double[] hiddenOutputs = {out1, out2};
        return output.activate(hiddenOutputs);
    }

    public void train(double[][] inputs, double[] labels, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                // Step 1: Forward pass
                double out1 = hidden1.activate(inputs[i]);
                double out2 = hidden2.activate(inputs[i]);
                double[] hiddenOutputs = {out1, out2};
                double prediction = output.activate(hiddenOutputs);

                // Step 2: Calculate output error
                double error = labels[i] - prediction;

                // Step 3: Update weights of output neuron
                output.updateWeights(hiddenOutputs, error, learningRate);

                // Optional: Update hidden neurons (not implemented for simplicity)
            }
        }
    }
}

public class kNN {
    public static void main(String[] args) {
        SimpleNeuralNetwork net = new SimpleNeuralNetwork(2);

        System.out.println(net.neuronDisplay());

        // Training data: [pointy ears, fluffy tail] => 1.0 = cat, 0.0 = dog
        double[][] trainingData = {
                {1.0, 0.5}, // cat
                {0.0, 0.2}, // dog
                {0.9, 0.9}, // cat
                {0.2, 0.1}  // dog
        };

        double[] labels = { //was erwartet wird
                1.0,
                0.0,
                1.0,
                0.0
        };

        // Train the network
        net.train(trainingData, labels, 1000, 0.1);

        // Test the network
        double[] newInput = {1.0, 0.5}; // should be cat
        double prediction = net.forward(newInput);
        System.out.println("Prediction for [1.0, 0.5] (should be cat): " + prediction);

        newInput = new double[]{0.1, 0.3}; // should be dog
        prediction = net.forward(newInput);
        System.out.println("Prediction for [0.1, 0.3] (should be dog): " + prediction);
    }
}
