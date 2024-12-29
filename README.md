# Stock Price Movement Prediction Using Financial News Headlines

This project implements a prediction of rise/fall of stocks based on news headlines using machine and deep learning techniques.

## Files

- `preprocess.py`: This file contains the preprocessing of the data - the headlines are fitted to dates.
- `extracting_features_updated.ipynb`: This notebook reads the preprocessed data. It then generates embeddings and sentiment analysis columns for the data.
- `model.py`: This file contains the LSTM prediction model of the data

## Data
We use financial news headlines data from the following Kaggle source: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests/data?select=raw_partner_headlines.csv

## Parameters
The main simulation logic is managed by `my_ground.py`, which offers several configurable parameters:

```python
battery_mode = False  # Enable battery mode simulation
run_simulations_mode = True  # Enable series of simulations
save_mode = False  # Save simulation results

# Parameters for simulation mode
min_robots = 1  # Minimum number of robots
max_robots = 5  # Maximum number of robots

# Parameters for single simulation mode
robot_number = 1  # Number of robots
seed = 42  # Random seed
patch_number = 7  # Patch number
```

Modify these parameters to customize the simulation’s behavior, including the number of robots, whether to run in battery mode, and saving the simulation results.

### Prerequisites

	•	Webots simulator installed on your system.
	•	Python 3.x for running controllers and data analysis notebooks.
	•	Required Python libraries: numpy, matplotlib, jupyter, pandas, controller (Webots)

### Running the Code

	1.	Open Webots and load the create4.wbt world file from /worlds.
	2.	Click the play button to start the simulation.

This project is currently not open to external contributions. Thank you for your interest!

## License

[MIT](https://choosealicense.com/licenses/mit/)
