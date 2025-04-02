include!("linear/logistic_regression.rs");
fn main() {
    // Example usage of the fit function
    let mut model = LogisticRegression::new(2);
    let training_features = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 1.0],
        vec![4.0, 2.0],
    ];
    let training_labels = vec![false, false, true, true];

    println!("Initial weights: {:?}", model.weights);
    println!("Initial bias: {}", model.bias);

    model.fit(&training_features, &training_labels, 0.1, 1000);

    println!("Trained weights: {:?}", model.weights);
    println!("Trained bias: {}", model.bias);

    // Make a prediction after training
    let test_features = vec![3.5, 1.5];
    let probability = model.predict_proba(&test_features);
    let prediction = model.predict(&test_features);

    println!("Probability for {:?}: {}", test_features, probability);
    println!("Prediction for {:?}: {}", test_features, prediction);
}