use rand::Rng;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn d_sigmoid(x: f64) -> f64 {
    x * (1.0 - x)
}

fn init_weight() -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen::<f64>()
}

fn line() -> String {
    (0..61).map(|_| "-").collect::<String>()
}

fn main() {
    const N_INPUTS: usize = 2;
    const N_HIDDEN_NODES: usize = 2;
    const N_OUTPUTS: usize = 1;
    const EPOCHS: i32 = 10000000;
    const LR: f64 = 0.1; // learning rate

    let mut hidden_layer = vec![0.0; N_HIDDEN_NODES];
    let mut output_layer = vec![0.0; N_OUTPUTS];

    let mut hidden_layer_bias: Vec<f64> = vec![0.0; N_HIDDEN_NODES];
    let mut output_layer_bias = vec![0.0; N_OUTPUTS];

    let mut hidden_weights = vec![[0f64; N_HIDDEN_NODES]; N_INPUTS];
    let mut output_weights = vec![[0f64; N_OUTPUTS]; N_HIDDEN_NODES];

    let training_inputs = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let training_outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    // set initial weight values
    for i in 0..N_INPUTS {
        for j in 0..N_HIDDEN_NODES {
            hidden_weights[i][j] = init_weight();
        }
    }

    for i in 0..N_HIDDEN_NODES {
        hidden_layer_bias[i] = init_weight();
        for j in 0..N_OUTPUTS {
            output_weights[i][j] = init_weight();
        }
    }

    for i in 0..N_OUTPUTS {
        output_layer_bias[i] = init_weight();
    }

    // training
    let mut rng = rand::thread_rng();
    for n in 0..EPOCHS {
        let i = rng.gen_range(0..4);

        // hidden layer activation
        for j in 0..N_HIDDEN_NODES {
            let mut activation = hidden_layer_bias[j];
            for k in 0..N_INPUTS {
                activation += training_inputs[i][k] * hidden_weights[k][j];
            }
            hidden_layer[j] = sigmoid(activation);
        }

        // output layer activation
        for j in 0..N_OUTPUTS {
            let mut activation = output_layer_bias[j];
            for k in 0..N_HIDDEN_NODES {
                activation += hidden_layer[k] * output_weights[k][j];
            }
            output_layer[j] = sigmoid(activation);
        }

        if n % 1000 == 0 {
            println!(
                "Epoch: {}\t| Input: {} : {}\t| Output: {:.3}\t| Expected: {}",
                n,
                training_inputs[i][0],
                training_inputs[i][1],
                output_layer[0],
                training_outputs[i][0]
            );
        }

        // change output weights
        let mut delta_output = vec![0.0; N_OUTPUTS];
        for j in 0..N_OUTPUTS {
            let d_error = training_outputs[i][j] - output_layer[j];
            delta_output[j] = d_error * d_sigmoid(output_layer[j]);
        }

        // change hidden weights
        let mut delta_hidden = vec![0.0; N_HIDDEN_NODES];
        for j in 0..N_HIDDEN_NODES {
            let mut d_error = 0.0;
            for k in 0..N_OUTPUTS {
                d_error += delta_output[k] * output_weights[j][k];
            }
            delta_hidden[j] = d_error * d_sigmoid(hidden_layer[j]);
        }

        // apply change to output weights
        for j in 0..N_OUTPUTS {
            output_layer_bias[j] += delta_output[j] * LR;
            for k in 0..N_HIDDEN_NODES {
                output_weights[k][j] += hidden_layer[k] * delta_output[j] * LR;
            }
        }

        // apply change to output weights
        for j in 0..N_HIDDEN_NODES {
            hidden_layer_bias[j] += delta_hidden[j] * LR;
            for k in 0..N_INPUTS {
                hidden_weights[k][j] += training_inputs[i][k] * delta_hidden[j] * LR;
            }
        }
    }

    println!("{}", line());

    println!("Hidden weights: {:?}", hidden_weights);
    println!("Hidden biases: {:?}", hidden_layer_bias);

    println!("Output weights: {:?}", output_weights);
    println!("Output biases: {:?}", output_layer_bias);
}
