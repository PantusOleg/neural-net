use crate::net::{NeuralNetwork};
use std::{fs};
use std::path::PathBuf;
use image::{Pixel, RgbImage};
use image::io::Reader as ImageReader;
use rand::{thread_rng, Rng};

mod net;
mod layer;

fn main() {
    let sigmoid = |x: f32| -> f32 { 1.0 / (1.0 + (-x).exp()) };
    let dsigmoid = |y: f32| -> f32 { y * (1.0 - y) };
    let sizes = &[784, 512, 128, 32, 10];
    let mut net = NeuralNetwork::new(0.001, sigmoid, dsigmoid, sizes);

    let samples = 60000;
    let mut images: Vec<RgbImage> = Vec::with_capacity(samples);
    let mut digits: Vec<i8> = Vec::with_capacity(samples);
    let files: Vec<PathBuf> = fs::read_dir("../numbers").unwrap()
        .map(|f| f.unwrap().path()).collect();

    for path in files.iter() {
        let img = ImageReader::open(path).unwrap().decode().unwrap();
        let digit = &path.file_name().unwrap().to_str().unwrap()[10..11];
        images.push(img.into_rgb8());
        digits.push(digit.parse::<i8>().unwrap());
    }

    let mut inputs: Vec<Vec<f32>> = vec![vec![0.0; 784]; samples];

    for i in 0..samples {
        for (index, pixel) in images[i].pixels().enumerate() {
            inputs[i][index] = (pixel.channels()[2]) as f32 / 255.0;
        }
    }

    let mut rng = thread_rng();
    let mut random_img = || -> usize {
        (rng.gen_range(0.0..1.0) * (samples) as f32) as usize
    };

    let epochs = 1000;
    for i in 0..epochs {
        let mut correct = 0;
        let mut error_sum = 0.0;
        let batch_size = 100;

        for _ in 0..batch_size {
            let img_index = random_img();
            let mut targets: Vec<f32> = vec![0.0; 10];
            let digit = digits[img_index];
            targets[digit as usize] = 1.0;

            let outputs = net.feed_forward(&inputs[img_index]);
          
            let mut max_digit: i8 = 0;
            let mut max_digit_weight = -1.0;

            for k in 0..10 {
                if outputs[k] > max_digit_weight {
                    max_digit_weight = outputs[k];
                    max_digit = k as i8;
                }
            }

            if digit == max_digit { correct += 1 };
            for k in 0..10 {
                error_sum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
            }
            net.backpropagation(&targets);
        }
        println!("epoch: {}; correct: {}; errors: {}", i, correct, error_sum);
    }
}
