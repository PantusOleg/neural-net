pub struct Layer {
    pub size: usize,
    pub neurons: Vec<f32>,
    pub biases: Vec<f32>,
    pub weights: Vec<Vec<f32>>,
}

impl Layer {
    pub fn new(size: usize, next_size: usize) -> Layer {
        Layer {
            size,
            biases: vec![0.0; size],
            neurons: vec![0.0; size],
            weights: vec![vec![0.0; next_size]; size],
        }
    }
}