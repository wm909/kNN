public class NeuralNetwork {

}

class Neuron {
    private double[] weights;
    private double bias;

    public Neuron(int inputs) {
        weights = new double[inputs];
        for(int i = 0; i < inputs; i++) {
            weights[i] = Math.random();
        }
        bias = Math.random();
    }

    
}