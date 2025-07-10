from time_series_maa import MAA_time_series
import torch

# Initialize parameters
args = None  # Pass parameters based on actual situation
N_pairs = 3  # Number of generators or discriminators
batch_size = 32
num_epochs = 100
generators_names = ['gru', 'lstm', 'transformer']
discriminators_names = None
ckpt_dir = 'ckpt/20250425_102916' #NEEDED CKPT PATH
output_dir = 'output'
window_sizes = [5, 10, 15]
initial_learning_rate = 2e-5
train_split = 0.7
do_distill_epochs = 1
cross_finetune_epochs = 5
precise = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 3407
ckpt_path = "./ckpt/20250425_102916"
gan_weights = None

# Instantiate GCA_time_series class
gca = MAA_time_series(args, N_pairs, batch_size, num_epochs,
                      generators_names, discriminators_names,
                      ckpt_dir, output_dir,
                      window_sizes,
                      initial_learning_rate,
                      train_split,
                      do_distill_epochs, cross_finetune_epochs,
                      precise,
                      device,
                      seed,
                      ckpt_path,
                      gan_weights)

# Process data
data_path = 'database/processed_PTA_day.csv'
start_row = 1060
end_row = 4284
target_columns = [1]
feature_columns_list = [list(range(1, 19)),list(range(1, 19)),list(range(1, 19)),]
log_diff = True
gca.process_data(data_path, start_row, end_row, target_columns, feature_columns_list, log_diff)

# Initialize data loader
gca.init_dataloader()


# Initialize model
num_cls = 3  # Number of classes
gca.init_model(num_cls)

# Perform prediction
results = gca.pred()

# Extract predictions for training and testing sets
train_preds = results["train_mse_per_target"]
test_preds = results["test_mse_per_target"]

# Print predictions
print("Train Predictions:", train_preds)
print("Test Predictions:", test_preds)