use crate::layer::Layer;
use rand::{thread_rng, Rng};

pub struct NeuralNetwork {
    learning_rate: f32,
    layers: Vec<Layer>,
    activation: fn(f32) -> f32,
    derivative: fn(f32) -> f32,
}

impl NeuralNetwork {
    pub fn new(
        learning_rate: f32, activation: fn(f32) -> f32,
        derivative: fn(f32) -> f32, sizes: &[usize],
    ) -> NeuralNetwork {
        let mut rng = thread_rng();
        let sizes_len = sizes.len();
        let mut layers: Vec<Layer> = Vec::with_capacity(sizes_len);

        let mut random = || rng.gen_range(0.0..=1.0) * 2.0 - 1.0;

        for i in 0..sizes_len {
            let next_size = {
                if i < sizes_len - 1 { sizes[i + 1] } else { 0 }
            };
            layers.push(Layer::new(sizes[i], next_size));
            for j in 0..sizes[i] {
                layers[i].biases[j] = random();
                for k in 0..next_size {
                    layers[i].weights[j][k] = random();
                }
            }
        }
        NeuralNetwork {
            learning_rate,
            layers,
            activation,
            derivative,
        }
    }

    pub fn feed_forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        let layers_len = self.layers.len();
        self.layers[0].neurons = inputs.to_owned();

        for i in 0..layers_len - 1 {
            let l = &self.layers[i] as *const Layer;
            let next = &mut self.layers[i + 1] as *mut Layer;

            unsafe {
                for j in 0..(*next).size {
                    for k in 0..(*l).size {
                        (*next).neurons[j] += (*l).neurons[k] * (*l).weights[k][j];
                    }
                    (*next).neurons[j] += (*next).biases[j];
                    (*next).neurons[j] = (self.activation)((*next).neurons[j]);
                }
            }
        }

        let neurons = self.layers[layers_len - 1].neurons.to_owned();
        neurons
    }

    pub fn backpropagation(&mut self, targets: &[f32]) {
        let layers_len = self.layers.len();
        let last_layer = &self.layers[layers_len - 1];
        let mut errors = vec![0.0; last_layer.size];

        for i in 0..last_layer.size {
            errors[i] = targets[i] - last_layer.neurons[i];
        }

        for k in (0..layers_len - 1).rev() {
            let l = &mut self.layers[k] as *mut Layer;
            let l1 = &mut self.layers[k + 1] as *mut Layer;

            unsafe {
                let mut errors_next = vec![0.0; (*l).size];
                let mut gradients = vec![0.0; (*l1).size];

                for i in 0..(*l1).size {
                    gradients[i] = errors[i] * (self.derivative)((*l1).neurons[i]);
                    gradients[i] *= self.learning_rate;
                }
                let mut deltas = vec![vec![0.0; (*l).size]; (*l1).size];
                for i in 0..(*l1).size {
                    for j in 0..(*l).size {
                        deltas[i][j] = gradients[i] * (*l).neurons[j];
                    }
                }
                for i in 0..(*l).size {
                    for j in 0..(*l1).size {
                        errors_next[i] += (*l).weights[i][j] * errors[j];
                    }
                }
                errors = errors_next;
                let mut new_weights = vec![vec![0.0; (*l).weights[0].len()]; (*l).weights.len()];
                for i in 0..(*l1).size {
                    for j in 0..(*l).size {
                        new_weights[j][i] = (*l).weights[j][i] + deltas[i][j];
                    }
                }
                (*l).weights = new_weights;
                for i in 0..(*l1).size {
                    (*l1).biases[i] += gradients[i];
                }
            }
        }
    }
}