import graphlab
import matplotlib.pyplot as plt


def import_csv(filename):

	data = graphlab.SFrame.read_csv(filename)
	return data

def linear_regression(data):

	model = graphlab.linear_regression.create(data, target='HousePrice', features=['CrimeRate'], validation_set=None, verbose=False)
	return model

def main():

	sales = import_csv('Philadelphia_Crime_Rate_noNA.csv/')
	print(sales)


	# perform linear regression using all data
	crime_model = linear_regression(sales)
	plt.plot(sales['CrimeRate'], sales['HousePrice'], '.', sales['CrimeRate'], crime_model.predict(sales), '-')
	plt.show()

	# remove high leverage point
	sales_noCC = sales[sales['MilesPhila'] != 0.0]
	crime_model_noCC = linear_regression(sales_noCC)
	plt.plot(sales_noCC['CrimeRate'], sales_noCC['HousePrice'], '.', sales_noCC['CrimeRate'], crime_model_noCC.predict(sales_noCC), '-')
	plt.show()


	print(crime_model.get('coefficients'))
	print(crime_model_noCC.get('coefficients'))


if __name__ == "__main__":
	main()

