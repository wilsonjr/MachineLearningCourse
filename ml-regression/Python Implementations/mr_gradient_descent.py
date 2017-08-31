import graphlab
import numpy as np
from math import sqrt

class GradientDescent:

	def execute(self, feature_matrix, output, initial_weights, step_size, tolerance):

		converged = False
		weights = np.array(initial_weights)

		while not converged:

			predictions = predict_output(feature_matrix, weights)

			errors = np.subtract(predictions, output)

			gradient_sum_squares = 0

			for i in range(len(weights)):
				derivate = feature_derivative(errors, feature_matrix[:, i])
				gradient_sum_squares = gradient_sum_squares + derivate*derivate
				weights[i] = weights[i] - (step_size*derivate)

			gradient_magnitude = sqrt(gradient_sum_squares)
			if gradient_magnitude < tolerance:
				converged = True

		return (weights)



def feature_derivative(errors, feature):
	return 2*np.dot(errors, feature)

def predict_output(feature_matrix, weights):
	predictions = np.dot(feature_matrix, weights)
	return (predictions)


def get_numpy_data(data_sframe, features, output):

	data_sframe['constant'] = 1
	features = ['constant'] + features
	features_sframe = data_sframe[features]

	feature_matrix = features_sframe.to_numpy()
	output_sarray = data_sframe[output]
	output_array = output_sarray.to_numpy()

	return (feature_matrix, output_array)


def main(datafilename):

	sales = graphlab.SFrame(datafilename)

	(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
	#print example_features[0, :]
	#print example_output[0]


	#my_weights = np.array([1., 1.])
	#test_predictions = predict_output(example_features, my_weights)
	#print test_predictions[0]
	#print test_predictions[1]


	gd = GradientDescent()
	train_data, test_data = sales.random_split(.8, seed=0)

	
	print("Running the Gradient Descent as Simple Regression")
	
	simple_features = ['sqft_living']
	my_output = 'price'
	(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
	initial_weights = np.array([-47000., 1.])
	step_size = 7e-12
	tolerance = 2.5e7

	weights = gd.execute(simple_feature_matrix, output, initial_weights, step_size, tolerance)
	print(weights)

	(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

	predicted = predict_output(test_simple_feature_matrix, weights)
	print(predicted)

	errors = predicted-test_data['price']
	rss = (errors*errors).sum()
	print('rss train: '+str(rss))

	print("Running a multiple regression")

	model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors
	my_output = 'price'
	(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
	initial_weights = np.array([-100000., 1., 1.])
	step_size = 4e-12
	tolerance = 1e9

	weights_multipler = gd.execute(feature_matrix, output, initial_weights, step_size, tolerance)
	print(weights_multipler)

	(test_feature_matrix, output) = get_numpy_data(test_data, model_features, my_output)
	predictedm = predict_output(test_feature_matrix, weights_multipler)
	print(predictedm)

	print(predictedm[0])

	print(test_data['price'][0])	

	errors = predictedm-output
	rss = (errors*errors).sum()
	print('rss test: '+str(rss))












if __name__ == '__main__':
	main('kc_house_data.gl/')