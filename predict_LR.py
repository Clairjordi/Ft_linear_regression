import os
import sys
import json 

def main():
    try:
        km_input = input("Mileage to estimate : ")
        km_int = int(km_input)
        if km_int <= 0:
            raise Exception("negative values or zero aren't accepted")

        input_file = "thetas_datas.json"
        if os.path.isfile(input_file):
            file = open(input_file, 'r')
            data_thetas = json.load(file)
            file.close()

            estimatePrice = data_thetas['theta0'] + (data_thetas['theta1'] * km_int)
            print(f"The estimate price is : {round(estimatePrice)}€")
        else:
            theta0 = 0
            theta1 = 0

            estimatePrice = theta0 + (theta1 * km_int)
            print(f"The estimate price is : {round(estimatePrice)}€")

    except Exception as e:
        print(f"Error : {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("Error : the programm hasn't arguments")
    else:
        main()
