from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    model_path = pipeline.run_pipeline()
    print(f"\nPipeline complete. Model saved at: {model_path}")