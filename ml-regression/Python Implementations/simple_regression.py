import graphlab


class SimpleRegression:

    dataset_path = ''
    dataset = None
    train_data = None
    test_data = None

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        self.dataset = graphlab.SFrame(self.dataset_path)
        self.train_data, self.test_data = self.dataset.random_split(0.8, seed=0)

    def get_regression_predictions(self, input_feature, intercept, slope):
        return slope*input_feature + intercept

    def get_residual_sum_of_squares(self, input_feature, output, intercept, slope):
        predicted = slope*input_feature + intercept
        residuals = output-predicted

        return (residuals*residuals).sum()

    def inverse_regression_predictions(self, output, intercept, slope):
        estimated_feature = (output-intercept)/slope
        return estimated_feature

    """
        This function performs Simple Linear Regression using 'input_feature' and 'output'    
    """
    def execute(self, input_feature, output):
        # compute the sum of input_feature and output
        input_feature = input_feature+1
        sum_input = input_feature.sum()
        sum_output = output.sum()


        # compute the product of the output and the input feature and its sum
        prod_io = input_feature*output
        sum_prod_io = prod_io.sum()

        # compute the squared value of the input_feature and its sum
        squared_input = input_feature*input_feature
        sum_squared_input = squared_input.sum()

        # use the formula for the slope
        slope = (sum_prod_io - (sum_input*sum_output)/len(output)) / (sum_squared_input - (sum_input*sum_input)/len(output))

        # use the formula fot the inpercept
        intercept = sum_output/len(output) - slope*(sum_input/len(output))

        return (intercept, slope)



if __name__ == '__main__':
    sr = SimpleRegression('data/kc_house_data.gl')

    test_feature = graphlab.SArray(range(5))
    test_output = graphlab.SArray(1 + 1*test_feature)
    (test_intercept, test_slope) = sr.execute(test_feature, test_output)

    #print('Intercept: '+ str(test_intercept))
    #print('Slope: ' +str(test_slope))

    sr.load_data()


    sqft_intercept, sqft_slope = sr.execute(sr.train_data['sqft_living'], sr.train_data['price'])
    print 'Intercept: ' + str(sqft_intercept)
    print 'Slope: ' + str(sqft_slope)

    print '-------------------------------------------\n'

    my_house_sqft = 2650
    estimated_price = sr.get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
    print 'The estimated price for a house with %d squarefeed is $%.2f' % (my_house_sqft, estimated_price)

    print '-------------------------------------------\n'

    print sr.get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope)

    print '-------------------------------------------\n'

    rss_prices_on_sqft = sr.get_residual_sum_of_squares(sr.train_data['sqft_living'], sr.train_data['price'], sqft_intercept, sqft_slope)
    print 'The RSS of predicting Prices based on Square Feet is: ' + str(rss_prices_on_sqft)

    print '-------------------------------------------\n'

    my_house_price = 800000
    estimated_squarefeet = sr.inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
    print 'The estimated squarefeet for a house worth $%.2f is %d ' % (my_house_price, estimated_squarefeet)

    print '-------------------------------------------\n'

    sqft_intercept_bedrooms, sqft_slope_bedrooms = sr.execute(sr.train_data['bedrooms'], sr.train_data['price'])
    print 'Intercept (bedrooms): ' + str(sqft_intercept_bedrooms)
    print 'Slope (bedrooms): ' + str(sqft_slope_bedrooms)

    print '-------------------------------------------\n'

    rss_prices_on_bedrooms = sr.get_residual_sum_of_squares(sr.test_data['bedrooms'], sr.test_data['price'], sqft_intercept_bedrooms, sqft_slope_bedrooms)
    print 'The RSS of predicting Prices based on Bedrooms is: ' + str(rss_prices_on_bedrooms)

    rss_prices_on_sqft = sr.get_residual_sum_of_squares(sr.test_data['sqft_living'], sr.test_data['price'], sqft_intercept, sqft_slope)
    print 'The RSS of predicting Prices based on Square Feet is: ' + str(rss_prices_on_sqft)



