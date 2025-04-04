import numpy as np
import pandas as pd
from src.article_weigher import ArticleWeightPredictor
from src.article_processor import ArticleProcessor, get_article_embedding


class Predictor:
    def __init__(self, excel_path, model_filepath, sheet_name=None):
        self.processor = ArticleProcessor(excel_path, sheet_name)
        self.model_filepath = model_filepath
        self.model = None  # Will be loaded in load_model()
    
    def load_model(self, input_dim):
        """
        Loads the saved model from disk.
        """
        self.model = ArticleWeightPredictor.load_model(self.model_filepath, input_dim)
    
    def predict_daily(self):
        """
        Processes the Excel file row by row. For each day (row), it:
          - Loops through each article (columns 2+)
          - Gets the embedding via the Gemini API for each article
          - Uses the neural network to predict the 4 outputs for each article
          - Averages the 4-dimensional outputs across all articles for that day
        Returns:
          - dates: List of dates (from the first column)
          - daily_predictions: A NumPy array with one 4-d vector per day.
        """
        # Use the raw DataFrame from the processor
        data = self.processor.data
        dates = []
        daily_predictions = []
        
        for index, row in data.iterrows():
            # The first column is assumed to be the date
            date = row[0]
            article_preds = []
            # Loop through each article (columns 2+)
            for article in row[1:]:
                if isinstance(article, str) and article.strip():
                    try:
                        # Get the embedding for this article
                        embedding = get_article_embedding(article)
                        # Get the neural network's 4-d prediction
                        pred = self.model.predict(np.array([embedding]))[0]
                        article_preds.append(pred)
                    except Exception as e:
                        print(f"Error processing article: {e}")
            # If there are predictions for the day, average them; otherwise use zeros.
            if article_preds:
                daily_avg = np.mean(article_preds, axis=0)
            else:
                daily_avg = np.zeros(4)
            dates.append(date)
            daily_predictions.append(daily_avg)
        
        return dates, np.array(daily_predictions)
    
    def save_predictions(self, dates, predictions, output_excel):
        """
        Saves the daily predictions to an Excel file.
        Each row in the Excel file corresponds to a day with 4 prediction values.
        """
        df = pd.DataFrame(predictions, columns=['relevance', 'up', 'down', 'unchanged'])
        df.insert(0, "date", dates)
        df.to_excel(output_excel, index=False)
        print(f"Predictions saved to {output_excel}")
