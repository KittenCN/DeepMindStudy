import matplotlib.pyplot as plt
def show_data(x_data=[1, 2, 3], y_data=[1, 2, 3]):
    plt.scatter(x=x_data, y=y_data, color='k')
    plt.show()