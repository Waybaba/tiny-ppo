from ray.air import session
from ray import tune


def objective(x, a, b):  # Define an objective function.
	return a * (x**0.5) + b


def trainable(config):  # Pass a "config" dictionary into your trainable.
	# sleep for random time 1-10 seconds
	import time
	import random
	time.sleep(random.randint(1, 10))
	for x in range(20):  # "Train" for 20 iterations and compute intermediate scores.
		score = objective(x, config["a"], config["b"])

		session.report({"score": score})  # Send the score to Tune.
from ray.tune.search.bayesopt import BayesOptSearch
algo = BayesOptSearch(random_search_steps=4)

tuner = tune.Tuner(
	trainable, 
	param_space={"a": 2, "b": 4},
	tune_config=tune.TuneConfig(
		num_samples=10, 
		# time_budget_s=10,
		metric="score",
		mode="max",
		)
	)
results = tuner.fit()

best_result = results.get_best_result()  # Get best result object
best_config = best_result.config  # Get best trial's hyperparameters
best_logdir = best_result.log_dir  # Get best trial's logdir
best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
best_metrics = best_result.metrics  # Get best trial's last results
best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe