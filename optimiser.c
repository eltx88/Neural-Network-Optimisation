#include "optimiser.h"

#include <time.h>
#include <omp.h>

#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"
#include <omp.h>
#include <stdlib.h>

// Function declarations
void update_parameters(unsigned int batch_size);
void update_parameters_adagrad(unsigned int batch_size);
void free_adagrad();
void adagrad_init();
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy);
void update_layer_weights(int input_neurons, int output_neurons, weight_struct_t weights[input_neurons][output_neurons], unsigned int batch_size);
double validate_gradients(unsigned int sample);
// Optimisation parameters
unsigned int log_freq = 30000; // Compute and print accuracy every log_freq iterations

// Parameters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
double learning_rate;

// Optimisation technique flags, defaulted to 0 , set to 1 if required
unsigned int use_lr_decay = 0;
unsigned int use_momentum = 0;
unsigned int use_adaptive_lr = 0;

// Parameters for each technique above
double initial_learning_rate = 0.1;
double final_learning_rate = 0.00001;
double momentum_coefficient = 0.6;
double epsilon = 1e-8;

//Gradient Variables for Adagrad
double **G_L3_LO, **G_L2_L3, **G_L1_L2, **G_LI_L1;

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy){
    printf("Epoch: %u,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter, total_iter, mean_loss, test_accuracy);
}

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs){
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    total_epochs = cmd_line_total_epochs;

    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with parameters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\n",
           total_epochs, batch_size, num_batches, learning_rate);

    if (use_adaptive_lr) {
        adagrad_init();
    }
}

void adagrad_init() {
    printf("Initialising AdaGrad structures...\n");

    G_L3_LO = (double**)malloc(N_NEURONS_L3 * sizeof(double*));
    G_L2_L3 = (double**)malloc(N_NEURONS_L2 * sizeof(double*));
    G_L1_L2 = (double**)malloc(N_NEURONS_L1 * sizeof(double*));
    G_LI_L1 = (double**)malloc(N_NEURONS_LI * sizeof(double*));

    // Initialize output layer accumulators
    for (int i = 0; i < N_NEURONS_L3; i++) {
        G_L3_LO[i] = (double*)malloc(N_NEURONS_LO * sizeof(double));
        for (int j = 0; j < N_NEURONS_LO; j++) {
            G_L3_LO[i][j] = 0.0;
        }
    }

    // Initialize hidden layer 3 accumulators
    for (int i = 0; i < N_NEURONS_L2; i++) {
        G_L2_L3[i] = (double*)malloc(N_NEURONS_L3 * sizeof(double));
        for (int j = 0; j < N_NEURONS_L3; j++) {
            G_L2_L3[i][j] = 0.0;
        }
    }

    // Initialize hidden layer 2 accumulators
    for (int i = 0; i < N_NEURONS_L1; i++) {
        G_L1_L2[i] = (double*)malloc(N_NEURONS_L2 * sizeof(double));
        for (int j = 0; j < N_NEURONS_L2; j++) {
            G_L1_L2[i][j] = 0.0;
        }
    }

    // Initialize input layer accumulators
    for (int i = 0; i < N_NEURONS_LI; i++) {
        G_LI_L1[i] = (double*)malloc(N_NEURONS_L1 * sizeof(double));
        for (int j = 0; j < N_NEURONS_L1; j++) {
            G_LI_L1[i][j] = 0.0;
        }
    }

}

double validate_gradients(unsigned int sample) {
    double total_diff = 0.0;
    int count = 0;
    clock_t start, end;
    double analytical_time = 0.0, numerical_time = 0.0;
    FILE *grad_file = fopen("gradient_validation.txt", "w");

    if (grad_file == NULL) {
        printf("Error opening file for gradient validation.\n");
        return -1.0;
    }

    fprintf(grad_file, "Weight_Index,Analytical,Forward_Diff,Backward_Diff,Central_Diff\n");

    printf("Validating gradients using numerical differentiation...\n");

    // Generate a fresh sample for more realistic gradients
    unsigned int validation_sample = (sample + 5) % N_TRAINING_SET;

    // First get the original loss and analytical gradients
    start = clock();
    evaluate_forward_pass(training_data, validation_sample);
    double original_loss = compute_xent_loss(training_labels[validation_sample]);

    // Calculate analytical gradients
    evaluate_backward_pass_sparse(training_labels[validation_sample], validation_sample);
    end = clock();
    analytical_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Validate gradients for a subset of weights from L3-LO layer (output layer)
    start = clock();
    for (int i = 0; i < N_NEURONS_L3; i += 10) {  // Sample every 10th neuron to save time
        for (int j = 0; j < N_NEURONS_LO; j++) {
            double original_weight = w_L3_LO[i][j].w;
            double analytical_grad = dL_dW_L3_LO[0][i + (N_NEURONS_L3 * j)];

            // Forward difference
            w_L3_LO[i][j].w = original_weight + epsilon;
            evaluate_forward_pass(training_data, validation_sample);
            double loss_plus = compute_xent_loss(training_labels[validation_sample]);
            double forward_diff = (loss_plus - original_loss) / epsilon;

            // Backward difference
            w_L3_LO[i][j].w = original_weight - epsilon;
            evaluate_forward_pass(training_data, validation_sample);
            double loss_minus = compute_xent_loss(training_labels[validation_sample]);
            double backward_diff = (original_loss - loss_minus) / epsilon;

            // Central difference
            double central_diff = (loss_plus - loss_minus) / (2 * epsilon);

            // Restore original weight
            w_L3_LO[i][j].w = original_weight;

            // Calculate relative error using central difference
            double rel_error = 0.0;
            if (fabs(analytical_grad) > 1e-10) {
                rel_error = fabs(central_diff - analytical_grad) / fabs(analytical_grad);
                total_diff += rel_error;
                count++;
            }

            // Save to file for analysis
            fprintf(grad_file, "L3_LO_%d_%d,%e,%e,%e,%e\n",
                    i, j, analytical_grad, forward_diff, backward_diff, central_diff);

            // Print some samples for quick reference
            if ((i % 30 == 0) && (j % 3 == 0)) {
                printf("Weight (%d,%d): Analytical=%.8e, Central=%.8e, Rel Error=%.8e%%\n",
                       i, j, analytical_grad, central_diff, rel_error * 100);
            }
        }
    }
    end = clock();
    numerical_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    fclose(grad_file);

    // Reset all gradients after validation
    for (int i = 0; i < N_NEURONS_L3; i++) {
        for (int j = 0; j < N_NEURONS_LO; j++) {
            w_L3_LO[i][j].dw = 0.0;
        }
    }

    double avg_error = count > 0 ? total_diff / count : -1.0;
    printf("Gradient validation complete. Average relative error: %.8e\n", avg_error);
    printf("Time comparison - Analytical: %.6f sec, Numerical: %.6f sec\n",
           analytical_time, numerical_time);

    return avg_error;
}

void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;
    char filename[50];

    // Timing variables
    clock_t start_time, end_time;
    double execution_time;
    start_time = clock();

    // Initialize learning rate to initial value
    if (use_lr_decay && use_momentum) {
        learning_rate = initial_learning_rate;
        sprintf(filename, "results/mom_decay_metrics_lri%.3f_bs%d_lrf_%.3f.csv",
           learning_rate, batch_size, final_learning_rate);
    }
    else if (use_lr_decay) {
        printf("Using lr decay with initial learning rate %f \n", initial_learning_rate);
        learning_rate = initial_learning_rate;
        sprintf(filename, "results/decay_metrics_lri%.3f_bs%d_lrf_%.3f.csv",
           learning_rate, batch_size, final_learning_rate);
    }
    else if (use_momentum) {
        sprintf(filename, "results/momentum_metrics_lr%.3f_bs%d.csv",
           learning_rate, batch_size);
    }
    else if (use_adaptive_lr) {
        sprintf(filename, "results/adaptive_metrics_lr%.3f_bs%d.csv",
           learning_rate, batch_size);
    }
    else {
        sprintf(filename, "results/training_metrics_lr%.3f_bs%d.csv",
           learning_rate, batch_size);
    }

    FILE *metrics_file = fopen(filename, "w");
    fprintf(metrics_file, "Epoch,Iteration,Loss,Accuracy,BatchSize,LearningRate,ExecutionTime\n");

    // Run optimiser - update parameters after each minibatch
    for (int i=0; i < num_batches; i++){
        for (int j = 0; j < batch_size; j++){

            // Evaluate accuracy on testing set (expensive, evaluate infrequently)
            if (total_iter % log_freq == 0 || total_iter == 0){
                if (total_iter > 0){
                    mean_loss = mean_loss/((double) log_freq);
                }

                test_accuracy = evaluate_testing_accuracy();
                print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);

                // Calculate elapsed time so far
                double current_time = ((double)(clock() - start_time)) / CLOCKS_PER_SEC;

                // Save metrics to file with batch size, learning rate, and current time
                fprintf(metrics_file, "%d,%d,%f,%f,%d,%f,%f\n",
                        epoch_counter, total_iter, mean_loss, test_accuracy,
                        batch_size, learning_rate, current_time);

                // Reset mean_loss for next reporting period
                mean_loss = 0.0;
            }

            // Evaluate forward pass and calculate gradients
            obj_func = evaluate_objective_function(training_sample);
            mean_loss+=obj_func;

            // Only run validation once at the start of training
            if(total_iter == 0){
                validate_gradients(training_sample);
            }

            // Update iteration counters (reset at end of training set to allow multiple epochs)
            total_iter++;
            training_sample++;

            // On epoch completion:
            if (training_sample == N_TRAINING_SET){
                training_sample = 0;
                epoch_counter++;

                // Update learning rate at the beginning of each epoch
                if (use_lr_decay) {
                    double alpha = (double)epoch_counter / (double)total_epochs;
                    learning_rate = initial_learning_rate * (1.0 - alpha) + alpha * final_learning_rate;
                    printf("Epoch %d: Updated learning rate to %f\n", epoch_counter, learning_rate);
                }
            }
        }

        // Update weights on batch completion
        if (use_adaptive_lr) {
            update_parameters_adagrad(batch_size);
        }
        else {
            update_parameters(batch_size);
        }

    }

    // Get end time
    end_time = clock();
    execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Print final performance
    test_accuracy = evaluate_testing_accuracy();
    mean_loss = mean_loss/((double) (log_freq));

    print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);
    printf("Total execution time: %.2f seconds\n", execution_time);

    // Write final line with all metrics
    fprintf(metrics_file, "%d,%d,%f,%f,%d,%f,%f\n",
            epoch_counter, total_iter, mean_loss, test_accuracy,
            batch_size, learning_rate, execution_time);

    fclose(metrics_file);

    if (use_adaptive_lr) {
        free_adagrad();
    }

    // Print the filename for reference
    printf("Results saved to: %s\n", filename);
}

double evaluate_objective_function(unsigned int sample){

    // Compute network performance
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);

    // Evaluate gradients
    //evaluate_backward_pass(training_labels[sample], sample);
    evaluate_backward_pass_sparse(training_labels[sample], sample);

    // Evaluate parameter updates
    store_gradient_contributions();

    return loss;
}

void update_layer_weights(int input_neurons, int output_neurons, weight_struct_t weights_i_o [input_neurons][output_neurons], unsigned int batch_size) {
    for (int i = 0; i < input_neurons; i++) {
        for (int j = 0; j < output_neurons; j++) {
            if (use_momentum) {
                // With momentum
                double dw_current = momentum_coefficient * weights_i_o[i][j].prev_dw -
                                  learning_rate * weights_i_o[i][j].dw / (double)batch_size;
                weights_i_o[i][j].prev_dw = dw_current;
                weights_i_o[i][j].w += dw_current;
            }
            else{
                weights_i_o[i][j].w -= learning_rate * weights_i_o[i][j].dw / (double)batch_size;
            }
            weights_i_o[i][j].dw = 0;
        }
    }
}

void update_parameters(unsigned int batch_size){
        update_layer_weights(N_NEURONS_L3, N_NEURONS_LO, w_L3_LO, batch_size);
        update_layer_weights(N_NEURONS_L2, N_NEURONS_L3, w_L2_L3, batch_size);
        update_layer_weights(N_NEURONS_L1, N_NEURONS_L2, w_L1_L2, batch_size);
        update_layer_weights(N_NEURONS_LI, N_NEURONS_L1, w_LI_L1, batch_size);
}

void update_layer_weights_adagrad(int input_neurons, int output_neurons,
                                 weight_struct_t weights[input_neurons][output_neurons],
                                 double** G,
                                 unsigned int batch_size) {
    for (int i = 0; i < input_neurons; i++) {
        for (int j = 0; j < output_neurons; j++) {
            // Accumulate squared gradient and normalize by batch size
            G[i][j] += (weights[i][j].dw * weights[i][j].dw) / (double)batch_size;

            // Compute adaptive learning rate
            double adaptive_learning_rate = learning_rate / (sqrt(G[i][j]) + epsilon);

            weights[i][j].w -= adaptive_learning_rate * weights[i][j].dw;
            weights[i][j].dw = 0;
        }
    }
}

void update_parameters_adagrad(unsigned int batch_size) {
    update_layer_weights_adagrad(N_NEURONS_L3, N_NEURONS_LO, w_L3_LO, G_L3_LO, batch_size);
    update_layer_weights_adagrad(N_NEURONS_L2, N_NEURONS_L3, w_L2_L3, G_L2_L3, batch_size);
    update_layer_weights_adagrad(N_NEURONS_L1, N_NEURONS_L2, w_L1_L2, G_L1_L2, batch_size);
    update_layer_weights_adagrad(N_NEURONS_LI, N_NEURONS_L1, w_LI_L1, G_LI_L1, batch_size);
}

void free_adagrad() {
    for (int i = 0; i < N_NEURONS_L3; i++) {
        free(G_L3_LO[i]);
    }
    free(G_L3_LO);

    for (int i = 0; i < N_NEURONS_L2; i++) {
        free(G_L2_L3[i]);
    }
    free(G_L2_L3);

    for (int i = 0; i < N_NEURONS_L1; i++) {
        free(G_L1_L2[i]);
    }
    free(G_L1_L2);

    for (int i = 0; i < N_NEURONS_LI; i++) {
        free(G_LI_L1[i]);
    }
    free(G_LI_L1);
}