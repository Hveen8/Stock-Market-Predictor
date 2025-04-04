import os
import numpy as np
from src.trainer import Trainer
from src.predictor import Predictor

def embed():
    # File paths and configuration
    excel_path = "news_articles.xlsx"      # Input Excel file with articles per day/column
    sheet_name = "Sheet1"
    model_filepath = "saved_models/article_weight_predictor.h5"
    output_excel = "daily_predictions.xlsx"
    epochs = 50

    # Check if a saved model exists
    if os.path.exists(model_filepath):
        print("Found saved model. Running predictor...")
        # Create Predictor instance and load the saved model
        predictor = Predictor(excel_path, model_filepath, sheet_name)
        # Determine input dimension by processing a small batch
        _, x_temp = predictor.processor.build_daily_dataset()
        input_dim = x_temp.shape[1]
        predictor.load_model(input_dim)
        dates, predictions = predictor.predict_daily()
        predictor.save_predictions(dates, predictions, output_excel)
    else:
        print("No saved model found. Running trainer to train a new model...")
        # Create Trainer instance
        trainer = Trainer(excel_path, sheet_name)
        dates, x = trainer.prepare_data()
        # Prepare training labels (dummy values as placeholders)
        y = np.random.rand(len(x), 4)
        
        # Train the model
        predictor_model = trainer.train_model(x, y, epochs=epochs)
        
        # Ensure the saved models folder exists
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
        trainer.save_model(model_filepath)
        
        # Now, run predictions with the newly trained model
        predictor = Predictor(excel_path, model_filepath, sheet_name)
        predictor.load_model(input_dim=x.shape[1])
        dates, predictions = predictor.predict_daily()
        predictor.save_predictions(dates, predictions, output_excel)

if __name__ == "__main__":
    embed()