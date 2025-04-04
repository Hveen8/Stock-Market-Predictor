from src.article_weigher import ArticleWeightPredictor
from src.article_processor import ArticleProcessor

# I can add Bayesian Optimization for hyperparameters, later
class Trainer:
    def __init__(self, excel_path, sheet_name=None):
        self.processor = ArticleProcessor(excel_path, sheet_name)
        self.model = None
    
    def prepare_data(self):
        """
        Process the Excel file to build the daily dataset.
        Returns dates and aggregated daily embeddings.
        """
        dates, x = self.processor.build_daily_dataset()
        return dates, x
    
    def train_model(self, x, y, epochs=50):
        """
        Trains the ArticleWeightPredictor using the provided data.
        Returns the trained model.
        """
        input_dim = x.shape[1]
        predictor = ArticleWeightPredictor(input_dim)
        predictor.train(x, y, epochs=epochs)
        self.model = predictor
        return predictor
    
    def save_model(self, model_filepath):
        """
        Saves the trained model to the given filepath.
        """
        if self.model:
            self.model.save_model(model_filepath)
        else:
            print("No model has been trained yet.")