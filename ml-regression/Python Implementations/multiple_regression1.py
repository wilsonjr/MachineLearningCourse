import graphlab
from math import log

class MultipleRegression:

	def create(self, data, target, features, validation_set):
		return graphlab.linear_regression.create(data, target, features, validation_set)

	def compute_rss(self, model, data, outcome):
		predictions = model.predict(data)
		residuals = outcome-predictions
		return (residuals*residuals).sum()


def main(datafile):

	sales = graphlab.SFrame(datafile)
	train_data, test_data = sales.random_split(.8, seed = 0)

	m_regression = MultipleRegression()
	example_model = m_regression.create(train_data, 'price', ['sqft_living', 'bedrooms', 'bathrooms'], None)

	example_weight_summary = example_model.get('coefficients')
	print example_weight_summary

	example_predictions = example_model.predict(train_data)
	print example_predictions[0]

	rss_example_train = m_regression.compute_rss(example_model, test_data, test_data['price'])


	"""
		Creating new features
	"""

	train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
	test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)


	train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
	train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
	train_data['lat_plus_long'] = train_data['lat']+train_data['long']

	test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']
	test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))
	test_data['lat_plus_long'] = test_data['lat']+test_data['long']
		

	"""
		Learning multiple models
	"""

	model1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
	model2_features = model1_features + ['bed_bath_rooms']
	model3_features = model2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

	model1 = m_regression.create(train_data, 'price', model1_features, None)
	model2 = m_regression.create(train_data, 'price', model2_features, None)
	model3 = m_regression.create(train_data, 'price', model3_features, None)

	model1_weight_summary = model1.get('coefficients')
	model2_weight_summary = model2.get('coefficients')
	model3_weight_summary = model3.get('coefficients')

	print(model1_weight_summary)
	print(model2_weight_summary)
	print(model3_weight_summary)

	"""
		Comparing multiple models
	"""
	rss_model1 = m_regression.compute_rss(model1, train_data, train_data['price'])
	rss_model2 = m_regression.compute_rss(model2, train_data, train_data['price'])
	rss_model3 = m_regression.compute_rss(model3, train_data, train_data['price'])

	print('RSS of model 1 on training data: '+str(rss_model1))
	print('RSS of model 2 on training data: '+str(rss_model2))
	print('RSS of model 3 on training data: '+str(rss_model3))

	rss_model1 = m_regression.compute_rss(model1, test_data, test_data['price'])
	rss_model2 = m_regression.compute_rss(model2, test_data, test_data['price'])
	rss_model3 = m_regression.compute_rss(model3, test_data, test_data['price'])

	print('RSS of model 1 on test data: '+str(rss_model1))
	print('RSS of model 2 on test data: '+str(rss_model2))
	print('RSS of model 3 on test data: '+str(rss_model3))




if __name__ == '__main__':
	main('kc_house_data.gl/')