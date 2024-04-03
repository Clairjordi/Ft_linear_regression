import argparse
import pandas as pd
from matplotlib import pyplot as plt
import json 

def data_normalization(data, base_data):
    max = base_data.max()
    min = base_data.min()
    data_norm = (data - min) / (max - min)
    return data_norm

def data_denormalization(data, y):
    max = y.max()
    min = y.min()
    data_denorm = data * (max - min) + min
    return data_denorm


class LinearRegression():
    """
    Linear Regression with :
    Prediction : estimatePrice(mileage) = θ0 + (θ1 * mileage)

    theta0 : tmpθ0 = learningRate * ((estimateP rice(mileage[i]) - price[i]).sum() / m)
    theta1 : tmpθ1 = learningRate * (((estimateP rice(mileage[i]) - price[i]).sum() * mileage[i]) / m)
    """
    def __init__(self, data, display_plot, learning_rate, iteration, theta0, theta1, expected_columns, out_file):
        self.data = data
        self.display_plot = display_plot
        self.x = data.iloc[:, 0].values
        self.y = data.iloc[:, 1].values
        self.X = data_normalization(self.x, self.x)
        self.Y = data_normalization(self.y, self.y)
        self.lr = learning_rate
        self.iter = iteration
        self.theta0 = theta0
        self.theta1 = theta1
        self.expected_columns = expected_columns
        self.out_file = out_file
        self.m = len(self.x)
        self.loss = []
        if display_plot:
            self.fig = plt.figure(figsize=(10,9), layout='tight')

    def __making_plots(self,pred_data_train_denorm, pred_data_train):

        gs = self.fig.add_gridspec(2, 2)
        
        ## plot 1
        fig_plot1 = self.fig.add_subplot(gs[0, 0])
        fig_plot1.scatter(self.x, self.y)
        fig_plot1.set_title("Data repartition", fontdict={'fontsize' : 14, 'fontweight' : 'bold'}, color="green")
        fig_plot1.set_xlabel(self.expected_columns[0])
        fig_plot1.set_ylabel(self.expected_columns[1])
       
        ## plot 2
        fig_plot2 = self.fig.add_subplot(gs[0, 1])
        fig_plot2.scatter(self.x, self.y)
        fig_plot2.plot(self.x, pred_data_train_denorm, color='red')
        fig_plot2.set_title("Data repartition\nwith regression line", fontdict={'fontsize' : 14, 'fontweight' : 'bold'}, color="green")
        fig_plot2.set_xlabel(self.expected_columns[0])
        fig_plot2.set_ylabel(self.expected_columns[1])

        ## plot 3
        fig_plot3 = self.fig.add_subplot(gs[1, 0])
        fig_plot3.scatter(self.X, self.Y)
        fig_plot3.plot(self.X, pred_data_train, color='red')
        fig_plot3.set_title("Data repartition normalize\nwith regression line", fontdict={'fontsize' : 14, 'fontweight' : 'bold'}, color="green")
        fig_plot3.set_xlabel(self.expected_columns[0])
        fig_plot3.set_ylabel(self.expected_columns[1])

        ## plot 4
        fig_plot4 = self.fig.add_subplot(gs[1, 1])
        fig_plot4.plot([i for i in range(self.iter)], self.loss)
        fig_plot4.set_title("Loss", fontdict={'fontsize' : 14, 'fontweight' : 'bold'}, color="green")
        fig_plot4.set_xlabel("Iterations")
        fig_plot4.set_ylabel("Cost")

        self.fig.savefig('./plot_metrics_train.png')
        plt.close()

    def train(self):
        ## training loop with n iterations
        for _ in range(self.iter):
            temp0 = 0
            temp1 = 0
            mse = 0

            ## calculating theta 0 is theta 1 on each training data and calculating the MSE
            for i in range(self.m):
                T = self.theta0 + self.theta1 * self.X[i] - self.Y[i]
                temp0 += T
                temp1 += T * self.X[i]
                mse += T **2

            self.theta0 -= self.lr * (temp0 /self.m)
            self.theta1 -= self.lr * (temp1 / self.m)
            cost = mse / self.m
            self.loss.append(cost)

        ## recovery of theta 0 and theta 1
        data_thetas_and_norm = {
            "theta0" : self.theta0,
            "theta1" : self.theta1,
            "min_x" : float(self.x.min()),
            "max_x" : float(self.x.max()),
            "min_y" : float(self.y.min()),
            "max_y" : float(self.y.max()),
        }
        file = open(self.out_file, 'w')
        json.dump(data_thetas_and_norm, file, indent=4) 
        file.close()

        if self.display_plot:

            ## calculating predictions on the training dataset
            pred_data_train = self.theta0 + (self.theta1 * self.X)
            pred_data_train_denorm = data_denormalization(pred_data_train, self.y)
            
            ## calculation of the coefficient of determination (there is no precision in linear regression)
            u = ((self.Y - pred_data_train)**2).sum()
            v = ((self.Y - self.Y.mean())**2).sum()
            coef_det = 1 - u/v
            print(f"coefficient of determination (precision for a regression linear) =\n {coef_det * 100:.2f}%")
            
            ## plots
            self.__making_plots(pred_data_train_denorm, pred_data_train)


def get_args():
    parser = argparse.ArgumentParser(description='Train linear regression with data')
    parser.add_argument('--file', '-f', type=str, default='data.csv', help='source of data file')
    parser.add_argument('--plot', '-p', action="store_true", default=False, help='view data and tracing results')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print(args)
    try:
        df_data = pd.read_csv(args.file)
        expected_columns = ['km', 'price']
        ## check if the name of columns is correct : 'km','price'
        if list(df_data.columns) != expected_columns:
            raise Exception("Error : the name of columns must be 'km,price'")
        ## check if the value is int or float
        df_verif_NaN = pd.DataFrame({col :df_data[col].astype(float)  for col in expected_columns})

        ## check if there isn't value NaN
        if any(df_verif_NaN[col].isna().any() for col in expected_columns):
            raise Exception("Error : value NaN isn't accepted")
           
        model = LinearRegression(data=df_data,
                                 display_plot=args.plot,
                                 learning_rate=0.1,
                                 iteration=1000,
                                 theta0=0,
                                 theta1=0,
                                 expected_columns=expected_columns,
                                 out_file="thetas_datas.json")
        model.train()


    except Exception as e:
        print(f"Error : {e}")