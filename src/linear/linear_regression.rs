struct LinearRegression {
    weights: Vec<f64>,
    bias: f64,
}
impl LinearRegression {
    fn new(n_features: usize) -> Self {
        LinearRegression {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        self.bias + features.iter().zip(&self.weights).map(|(f, w)| f * w).sum::<f64>()
    }

    fn fit(
        &mut self,
        features: &Vec<Vec<f64>>,
        labels: &Vec<f64>,
        learning_rate: f64,
        epochs: usize,
    ) {
        let num_samples = features.len();
        if num_samples == 0 {
            return;
        }
        let num_features = self.weights.len();
        if num_features == 0 || num_features != features[0].len() || num_samples != labels.len() {
            panic!("Mismatched dimensions between features, labels, and weights.");
        }

        for _ in 0..epochs {
            let mut dw = vec![0.0; num_features];
            let mut db = 0.0;

            for i in 0..num_samples {
                let prediction = self.predict(&features[i]);
                let error = prediction - labels[i];

                for j in 0..num_features {
                    dw[j] += error * features[i][j];
                }
                db += error;
            }

            // Update weights and bias
            for j in 0..num_features {
                self.weights[j] -= learning_rate * dw[j] / num_samples as f64;
            }
            self.bias -= learning_rate * db / num_samples as f64;
        }
    }
}