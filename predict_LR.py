import os
import sys
import json 

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Error : the programm hasn't arguments")
    else:
        try:
            km_input = input("Mileage to estimate : ")
            km_int = int(km_input)
            if km_int <= 0:
                raise Exception("negative values or zero aren't accepted")

            input_file = "thetas_datas.json"
            if os.path.isfile(input_file):
                file = open(input_file, 'r')
                data_thetas_and_norm = json.load(file)
                ## To estimate the price, it is necessary to normalize the input data (mileage) and denormalize the estimated output price
                km_int_norm = (km_int - data_thetas_and_norm['min_x']) / (data_thetas_and_norm['max_x'] - data_thetas_and_norm['min_x'])
                estimatePrice = data_thetas_and_norm['theta0'] + (data_thetas_and_norm['theta1'] * km_int_norm)
                estimatePrice_denorm = estimatePrice * (data_thetas_and_norm['max_y'] - data_thetas_and_norm['min_y']) + data_thetas_and_norm['min_y']
                print(f"For a mileage of {km_int}km,\nThe estimate price is : {round(estimatePrice_denorm)}€")
                file.close()
            else:
                raise Exception("no existing file with thetas's data, run 'train' before predict")
        except Exception as e:
            print(f"Error : {e}")