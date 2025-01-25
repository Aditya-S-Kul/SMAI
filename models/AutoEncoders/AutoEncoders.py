import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.MLP.MLP import MLP_Regression

class AutoEncoder:
    def __init__(self, input_size, latent_size, hidden_layers, learning_rate=0.01, activation='relu', optimizer='sgd', epochs=100, early_stopping=False, patience=15):
        """
        Initializes the autoencoder model by setting up the encoder and decoder using the MLP_Regression class.
        """
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        
        # Define the encoder: Compresses input to latent space
        self.encoder = MLP_Regression(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=latent_size,
            learning_rate=learning_rate,
            activation=activation,
            optimizer=optimizer,
            epochs=50,
            task='regression',
            early_stopping=True,
            patience=patience
        )
        
        # Define the decoder: Reconstructs original input from latent space
        self.decoder = MLP_Regression(
            input_size=latent_size,
            hidden_layers=hidden_layers[::-1],  # Reversed hidden layers for symmetry
            output_size=input_size,
            learning_rate=learning_rate,
            activation=activation,
            optimizer=optimizer,
            epochs=epochs,
            task='regression',
            early_stopping=early_stopping,
            patience=patience
        )

    def fit(self, X_train, X_val=None):
        """
        Trains the autoencoder by first training the encoder and decoder.
        The training target for both encoder and decoder is to minimize reconstruction loss.
        """
        # Train the encoder-decoder as a full pipeline.
        # Forward pass: Encoder + Decoder
        def autoencoder_predict(X):
            latent_rep = self.encoder.predict(X)  # Compress data
            reconstructed_X = self.decoder.predict(latent_rep)  # Reconstruct original data
            return reconstructed_X
        
        # Custom fit loop for autoencoder
        n_samples = X_train.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            print("epoch: ",epoch)
            # Pass the input through encoder and decoder
            latent_rep = self.encoder.predict(X_train)
            reconstructed_X = self.decoder.predict(latent_rep)
            
            # Calculate loss (reconstruction error)
            loss = np.mean((X_train - reconstructed_X) ** 2)
            
            # Update encoder and decoder with backpropagation
            self.encoder.fit(X_train, latent_rep)
            self.decoder.fit(latent_rep, X_train)
            
            if X_val is not None:
                val_reconstruction = autoencoder_predict(X_val)
                val_loss = np.mean((X_val - val_reconstruction) ** 2)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if self.early_stopping and patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def get_latent(self, X):
        """
        Returns the latent space representation (compressed data) of the input X.
        """
        latent_rep = self.encoder.predict(X)
        return latent_rep
