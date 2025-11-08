import torch
import torch.nn as nn
import torch.optim as optim
from .data import DataLoader
from .models import HRMModel
from .utils import setup_logging, load_config

def main():
    # Load config
    config = load_config('config.yaml')

    # Logging setup
    logger = setup_logging(config['logging']['level'])

    # Data loader setup
    data_loader = DataLoader(config['data'])

    # Initialize model
    model = HRMModel(config['model'])

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        for inputs, labels in data_loader.train_loader():
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        logger.info(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader.val_loader():
            outputs = model(inputs)
            # Calculate metrics here

if __name__ == '__main__':
    main()