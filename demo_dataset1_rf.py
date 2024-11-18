import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def read_cfg_file(cfg_path):
    with open(cfg_path, 'r') as file:
        lines = file.readlines()
    tol_samp_num = int(lines[1].strip())
    port_num = int(lines[3].strip())
    ant_num = int(lines[4].strip())
    sc_num = int(lines[5].strip())
    return tol_samp_num, port_num, ant_num, sc_num

def load_channel_data(file_path, indices, sc_num, ant_num, port_num):
    num_values_per_sample = 2 * sc_num * ant_num * port_num
    data_values = []
    with open(file_path, 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx in indices:
                data_values.append(list(map(float, line.strip().split())))
    data_values = np.array(data_values, dtype=np.float32)
    if data_values.size != len(indices) * num_values_per_sample:
        raise ValueError(f"Expected {len(indices) * num_values_per_sample} values, but got {data_values.size}")
    Htmp = data_values.reshape((len(indices), 2, sc_num, ant_num, port_num))
    Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
    Htmp = Htmp.transpose((0, 3, 2, 1))
    return Htmp.astype(np.complex64)

def preprocess_channel_data(H_batch, scaler=None, pca=None, fit=False):
    H_real = H_batch.real
    H_imag = H_batch.imag
    H_combined = np.concatenate([H_real, H_imag], axis=-1)
    H_flattened = H_combined.reshape(H_combined.shape[0], -1)
    if fit:
        H_scaled = scaler.fit_transform(H_flattened)
        H_pca = pca.fit_transform(H_scaled)
    else:
        H_scaled = scaler.transform(H_flattened)
        H_pca = pca.transform(H_scaled)
    return H_pca

if __name__ == "__main__":
    base_path = '/Users/chunz/Downloads/hack/'
    test_cfg_files = [f'{base_path}Dataset1/Dataset1CfgData{i}.txt' for i in range(1, 4)]
    test_input_data_files = [f'{base_path}Dataset1/Dataset1InputData{i}.txt' for i in range(1, 4)]
    test_input_pos_files = [f'{base_path}Dataset1/Dataset1InputPos{i}.txt' for i in range(1, 4)]
    result_folder = f'{base_path}Dataset1/result/'
    os.makedirs(result_folder, exist_ok=True)

    slice_samp_num = 1000
    print("Initializing scaler, PCA, and model...")
    scaler = StandardScaler()
    pca = PCA(n_components=50)
    #model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)

    print("Training model with known points from Dataset1...")
    for i in range(len(test_cfg_files)):
        # Load configuration
        tol_samp_num, port_num, ant_num, sc_num = read_cfg_file(test_cfg_files[i])

        # Load known points and their indices
        known_points = np.loadtxt(test_input_pos_files[i])
        indices = known_points[:, 0].astype(int) - 1  # Convert to zero-based index
        y_train = known_points[:, 1:]

        # Load corresponding features for known points
        H_train = load_channel_data(test_input_data_files[i], indices, sc_num, ant_num, port_num)

        # Preprocess features
        X_train = preprocess_channel_data(H_train, scaler, pca, fit=True)

        # Train model
        model.fit(X_train, y_train)
        print(f"Model training completed for Dataset1InputPos{i+1}.")

        # Predict on the entire Dataset1
        y_pred_list = []
        for batch_idx in range(tol_samp_num // slice_samp_num):
            start_sample = batch_idx * slice_samp_num
            end_sample = start_sample + slice_samp_num
            H_batch = load_channel_data(
                test_input_data_files[i], range(start_sample, end_sample), sc_num, ant_num, port_num
            )
            X_batch = preprocess_channel_data(H_batch, scaler, pca)
            y_batch_pred = model.predict(X_batch)
            y_pred_list.append(y_batch_pred)
        y_pred = np.vstack(y_pred_list)

        # Validate on known points
        y_known_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_known_pred)
        print(f"Validation Mean Squared Error for Dataset1InputPos{i+1}: {mse:.4f}")

        # Save predictions
        output_file = f'{result_folder}Dataset1OutputPos{i+1}.txt'
        np.savetxt(output_file, y_pred, fmt='%.6f')
        print(f"Predictions saved to {output_file}")

    print("Prediction completed.")
